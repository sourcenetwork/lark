//! Lark: A pure Rust LSM-tree key-value store.
//!
//! Lark provides a fast, embedded key-value store with:
//! - **Snapshot isolation** via MVCC sequence numbers
//! - **Crash recovery** via write-ahead logging (WAL)
//! - **LZ4 compression** for data blocks
//! - **Bloom filters** for fast negative lookups
//! - **Level-based compaction** on a dedicated OS thread
//! - **Lock-free reads** via crossbeam skip list memtable
//!
//! # Quick Start
//!
//! ```no_run
//! use lark_kv::{Db, Options};
//!
//! let db = Db::open("/tmp/my_db", Options::default()).unwrap();
//!
//! // Write
//! db.put(b"hello", b"world").unwrap();
//!
//! // Read
//! let value = db.get(b"hello").unwrap();
//! assert_eq!(value, Some(b"world".to_vec()));
//!
//! // Delete
//! db.delete(b"hello").unwrap();
//!
//! // Batch write
//! let mut batch = lark_kv::WriteBatch::new();
//! batch.put(b"key1", b"val1");
//! batch.put(b"key2", b"val2");
//! batch.delete(b"key3");
//! db.write(batch).unwrap();
//!
//! // Snapshot reads
//! let snap = db.snapshot();
//! db.put(b"key1", b"val_new").unwrap();
//! // Snapshot still sees old value
//! assert_eq!(snap.get(b"key1").unwrap(), Some(b"val1".to_vec()));
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod engine;
mod error;
mod options;

pub use error::Error;
pub use options::{DurabilityMode, Options};

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use engine::LarkEngine;

/// Result type for lark operations.
pub type Result<T> = std::result::Result<T, Error>;

/// A key-value database backed by an LSM-tree.
pub struct Db {
    engine: Arc<LarkEngine>,
    durability: engine::DurabilityMode,
}

impl std::fmt::Debug for Db {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Db")
            .field("durability", &self.durability)
            .finish_non_exhaustive()
    }
}

impl Db {
    /// Open or create a database at the given path.
    pub fn open<P: AsRef<Path>>(path: P, opts: Options) -> Result<Self> {
        let durability = match opts.durability {
            DurabilityMode::Immediate => engine::DurabilityMode::Immediate,
            DurabilityMode::Eventual => engine::DurabilityMode::Eventual,
        };
        let engine = LarkEngine::open(path.as_ref(), opts.to_engine_options())?;
        Ok(Self { engine, durability })
    }

    /// Get the value for a key. Returns `None` if the key doesn't exist.
    pub fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        let seq = self.engine.snapshot_seq();
        self.engine.get(key, seq).map_err(Error::Io)
    }

    /// Set a key-value pair.
    pub fn put(&self, key: &[u8], value: &[u8]) -> Result<()> {
        let mut batch = BTreeMap::new();
        batch.insert(key.to_vec(), Some(value.to_vec()));
        self.engine
            .apply_batch(batch, self.durability)
            .map_err(Error::Io)
    }

    /// Delete a key.
    pub fn delete(&self, key: &[u8]) -> Result<()> {
        let mut batch = BTreeMap::new();
        batch.insert(key.to_vec(), None);
        self.engine
            .apply_batch(batch, self.durability)
            .map_err(Error::Io)
    }

    /// Apply a batch of writes atomically.
    pub fn write(&self, batch: WriteBatch) -> Result<()> {
        if batch.ops.is_empty() {
            return Ok(());
        }
        self.engine
            .apply_batch(batch.ops, self.durability)
            .map_err(Error::Io)
    }

    /// Apply a batch of writes atomically with explicit durability mode.
    pub fn write_with_durability(
        &self,
        batch: WriteBatch,
        durability: DurabilityMode,
    ) -> Result<()> {
        if batch.ops.is_empty() {
            return Ok(());
        }
        let dm = match durability {
            DurabilityMode::Immediate => engine::DurabilityMode::Immediate,
            DurabilityMode::Eventual => engine::DurabilityMode::Eventual,
        };
        self.engine.apply_batch(batch.ops, dm).map_err(Error::Io)
    }

    /// Create a point-in-time snapshot for consistent reads.
    pub fn snapshot(&self) -> Snapshot {
        let seq = self.engine.snapshot_seq();
        Snapshot {
            engine: Arc::clone(&self.engine),
            seq,
        }
    }

    /// Scan a key range. Returns all key-value pairs where `start <= key < end`.
    pub fn scan(
        &self,
        start: Option<&[u8]>,
        end: Option<&[u8]>,
    ) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let seq = self.engine.snapshot_seq();
        self.engine.scan(start, end, seq).map_err(Error::Io)
    }

    /// Delete all data in the database.
    pub fn drop_all(&self) -> Result<()> {
        self.engine.drop_all().map_err(Error::Io)
    }

    /// Flush all data to disk and shut down background threads.
    pub fn close(&self) -> Result<()> {
        self.engine.close().map_err(Error::Io)
    }
}

/// A point-in-time snapshot for consistent reads.
pub struct Snapshot {
    engine: Arc<LarkEngine>,
    seq: u64,
}

impl std::fmt::Debug for Snapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Snapshot")
            .field("seq", &self.seq)
            .finish_non_exhaustive()
    }
}

impl Snapshot {
    /// Get the value for a key at this snapshot.
    pub fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        self.engine.get(key, self.seq).map_err(Error::Io)
    }

    /// Scan a key range at this snapshot.
    pub fn scan(
        &self,
        start: Option<&[u8]>,
        end: Option<&[u8]>,
    ) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        self.engine.scan(start, end, self.seq).map_err(Error::Io)
    }
}

/// A batch of write operations to apply atomically.
#[derive(Debug, Default)]
pub struct WriteBatch {
    ops: BTreeMap<Vec<u8>, Option<Vec<u8>>>,
}

impl WriteBatch {
    /// Create an empty write batch.
    pub fn new() -> Self {
        Self {
            ops: BTreeMap::new(),
        }
    }

    /// Add a put operation to the batch.
    pub fn put(&mut self, key: &[u8], value: &[u8]) {
        self.ops.insert(key.to_vec(), Some(value.to_vec()));
    }

    /// Add a delete operation to the batch.
    pub fn delete(&mut self, key: &[u8]) {
        self.ops.insert(key.to_vec(), None);
    }

    /// Number of operations in the batch.
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Whether the batch is empty.
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn open_tmp() -> (Db, TempDir) {
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), Options::default()).unwrap();
        (db, dir)
    }

    /// Options that force flushes early so tests can exercise the SSTable path.
    fn tiny_flush_opts() -> Options {
        Options {
            write_buffer_size: 4 * 1024,
            ..Options::default()
        }
    }

    /// Write enough filler bytes to push the active memtable past
    /// `write_buffer_size`, forcing a flush to L0.
    fn force_flush(db: &Db, tag: &str) {
        let payload = vec![0u8; 512];
        for i in 0..32 {
            let key = format!("__flush_{}_{:04}", tag, i);
            db.put(key.as_bytes(), &payload).unwrap();
        }
    }

    #[test]
    fn test_basic_crud() {
        let (db, _dir) = open_tmp();

        db.put(b"key1", b"value1").unwrap();
        assert_eq!(db.get(b"key1").unwrap(), Some(b"value1".to_vec()));

        db.put(b"key1", b"value2").unwrap();
        assert_eq!(db.get(b"key1").unwrap(), Some(b"value2".to_vec()));

        db.delete(b"key1").unwrap();
        assert_eq!(db.get(b"key1").unwrap(), None);

        assert_eq!(db.get(b"nonexistent").unwrap(), None);
    }

    #[test]
    fn test_write_batch() {
        let (db, _dir) = open_tmp();

        let mut batch = WriteBatch::new();
        batch.put(b"a", b"1");
        batch.put(b"b", b"2");
        batch.put(b"c", b"3");
        db.write(batch).unwrap();

        assert_eq!(db.get(b"a").unwrap(), Some(b"1".to_vec()));
        assert_eq!(db.get(b"b").unwrap(), Some(b"2".to_vec()));
        assert_eq!(db.get(b"c").unwrap(), Some(b"3".to_vec()));
    }

    #[test]
    fn test_snapshot_isolation() {
        let (db, _dir) = open_tmp();

        db.put(b"key", b"v1").unwrap();
        let snap = db.snapshot();

        db.put(b"key", b"v2").unwrap();

        assert_eq!(snap.get(b"key").unwrap(), Some(b"v1".to_vec()));
        assert_eq!(db.get(b"key").unwrap(), Some(b"v2".to_vec()));
    }

    #[test]
    fn test_scan() {
        let (db, _dir) = open_tmp();

        db.put(b"a", b"1").unwrap();
        db.put(b"b", b"2").unwrap();
        db.put(b"c", b"3").unwrap();
        db.put(b"d", b"4").unwrap();

        let results = db.scan(Some(b"b"), Some(b"d")).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], (b"b".to_vec(), b"2".to_vec()));
        assert_eq!(results[1], (b"c".to_vec(), b"3".to_vec()));
    }

    #[test]
    fn test_drop_all() {
        let (db, _dir) = open_tmp();

        db.put(b"key1", b"val1").unwrap();
        db.put(b"key2", b"val2").unwrap();
        db.drop_all().unwrap();

        assert_eq!(db.get(b"key1").unwrap(), None);
        assert_eq!(db.get(b"key2").unwrap(), None);

        db.put(b"key3", b"val3").unwrap();
        assert_eq!(db.get(b"key3").unwrap(), Some(b"val3".to_vec()));
    }

    #[test]
    fn test_snapshot_isolation_across_flush() {
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), tiny_flush_opts()).unwrap();

        db.put(b"key", b"v1").unwrap();
        let snap = db.snapshot();

        db.put(b"key", b"v2").unwrap();
        force_flush(&db, "snap");

        assert_eq!(snap.get(b"key").unwrap(), Some(b"v1".to_vec()));
        assert_eq!(db.get(b"key").unwrap(), Some(b"v2".to_vec()));
    }

    #[test]
    fn test_delete_persists_across_flush() {
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), tiny_flush_opts()).unwrap();

        db.put(b"key", b"v1").unwrap();
        force_flush(&db, "a");

        db.delete(b"key").unwrap();
        force_flush(&db, "b");

        assert_eq!(db.get(b"key").unwrap(), None);
    }

    #[test]
    fn test_crash_recovery_without_close() {
        let dir = TempDir::new().unwrap();

        {
            let db = Db::open(dir.path(), Options::default()).unwrap();
            db.put(b"a", b"1").unwrap();
            db.put(b"b", b"2").unwrap();
            db.delete(b"a").unwrap();
            db.put(b"c", b"3").unwrap();
            // Drop without close() — simulates a crash; the WAL must be replayed.
        }

        let db = Db::open(dir.path(), Options::default()).unwrap();
        assert_eq!(db.get(b"a").unwrap(), None);
        assert_eq!(db.get(b"b").unwrap(), Some(b"2".to_vec()));
        assert_eq!(db.get(b"c").unwrap(), Some(b"3".to_vec()));
    }

    #[test]
    fn test_persistence() {
        let dir = TempDir::new().unwrap();

        {
            let db = Db::open(dir.path(), Options::default()).unwrap();
            db.put(b"persist", b"data").unwrap();
            db.close().unwrap();
        }

        {
            let db = Db::open(dir.path(), Options::default()).unwrap();
            assert_eq!(db.get(b"persist").unwrap(), Some(b"data".to_vec()));
        }
    }
}
