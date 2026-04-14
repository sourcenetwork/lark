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
mod iter;
mod options;

pub use error::Error;
pub use iter::Iter;
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
        collect_range(&self.engine, start, end, seq)
    }

    /// Create a streaming iterator over the current database state.
    ///
    /// The iterator captures a consistent view at the moment it is created
    /// — later writes are invisible to this iterator, and concurrent
    /// background compaction cannot invalidate it.
    ///
    /// A fresh iterator is not positioned; call one of
    /// [`Iter::seek_to_first`], [`Iter::seek`], or
    /// [`Iter::seek_for_prev`] before reading.
    pub fn iter(&self) -> Iter<'_> {
        let seq = self.engine.snapshot_seq();
        Iter::from_internal(self.engine.new_iter(seq))
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
        collect_range(&self.engine, start, end, self.seq)
    }

    /// Create a streaming iterator anchored at this snapshot.
    pub fn iter(&self) -> Iter<'_> {
        Iter::from_internal(self.engine.new_iter(self.seq))
    }
}

/// Collect a bounded range of `(user_key, value)` pairs via the streaming
/// iterator. This is the engine of `Db::scan` / `Snapshot::scan`; the
/// dedicated method exists so both callers share one merge implementation.
fn collect_range(
    engine: &LarkEngine,
    start: Option<&[u8]>,
    end: Option<&[u8]>,
    snapshot_seq: u64,
) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
    let mut iter = engine.new_iter(snapshot_seq);
    match start {
        Some(s) => iter.seek(s),
        None => iter.seek_to_first(),
    }
    iter.status().map_err(Error::Io)?;

    let mut out = Vec::new();
    while iter.valid() {
        let (Some(k), Some(v)) = (iter.key(), iter.value()) else {
            break;
        };
        if let Some(e) = end {
            if k >= e {
                break;
            }
        }
        out.push((k.to_vec(), v.to_vec()));
        iter.next();
    }
    iter.status().map_err(Error::Io)?;
    Ok(out)
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

    // ─── Streaming iterator tests ────────────────────────────────────────

    fn collect_iter(db: &Db) -> Vec<(Vec<u8>, Vec<u8>)> {
        let mut it = db.iter();
        it.seek_to_first();
        let mut out = Vec::new();
        while it.valid() {
            out.push((it.key().unwrap().to_vec(), it.value().unwrap().to_vec()));
            it.next();
        }
        it.status().unwrap();
        out
    }

    #[test]
    fn test_iter_empty_db() {
        let (db, _dir) = open_tmp();
        let mut it = db.iter();
        it.seek_to_first();
        assert!(!it.valid());
        it.seek(b"anything");
        assert!(!it.valid());
        assert!(it.status().is_ok());
    }

    #[test]
    fn test_iter_basic_forward() {
        let (db, _dir) = open_tmp();
        for i in 0..10 {
            let k = format!("k{:02}", i);
            let v = format!("v{}", i);
            db.put(k.as_bytes(), v.as_bytes()).unwrap();
        }
        let items = collect_iter(&db);
        assert_eq!(items.len(), 10);
        for (i, (k, v)) in items.iter().enumerate() {
            assert_eq!(k, format!("k{:02}", i).as_bytes());
            assert_eq!(v, format!("v{}", i).as_bytes());
        }
    }

    #[test]
    fn test_iter_seek_exact_and_between() {
        let (db, _dir) = open_tmp();
        db.put(b"a", b"1").unwrap();
        db.put(b"c", b"3").unwrap();
        db.put(b"e", b"5").unwrap();

        let mut it = db.iter();

        it.seek(b"a");
        assert!(it.valid());
        assert_eq!(it.key(), Some(b"a".as_ref()));

        it.seek(b"b");
        assert_eq!(it.key(), Some(b"c".as_ref()));

        it.seek(b"c");
        assert_eq!(it.key(), Some(b"c".as_ref()));

        it.seek(b"f");
        assert!(!it.valid());
    }

    #[test]
    fn test_iter_seek_for_prev() {
        let (db, _dir) = open_tmp();
        db.put(b"a", b"1").unwrap();
        db.put(b"c", b"3").unwrap();
        db.put(b"e", b"5").unwrap();

        let mut it = db.iter();

        it.seek_for_prev(b"e");
        assert_eq!(it.key(), Some(b"e".as_ref()));

        it.seek_for_prev(b"d");
        assert_eq!(it.key(), Some(b"c".as_ref()));

        it.seek_for_prev(b"a");
        assert_eq!(it.key(), Some(b"a".as_ref()));

        it.seek_for_prev(b"0");
        assert!(!it.valid());
    }

    #[test]
    fn test_iter_continues_after_seek() {
        let (db, _dir) = open_tmp();
        for c in b'a'..=b'j' {
            db.put(&[c], &[c]).unwrap();
        }

        let mut it = db.iter();
        it.seek(b"d");
        let mut keys = Vec::new();
        while it.valid() {
            keys.push(it.key().unwrap().to_vec());
            it.next();
        }
        assert_eq!(
            keys,
            vec![
                b"d".to_vec(),
                b"e".to_vec(),
                b"f".to_vec(),
                b"g".to_vec(),
                b"h".to_vec(),
                b"i".to_vec(),
                b"j".to_vec(),
            ]
        );
    }

    #[test]
    fn test_iter_across_memtable_and_l0() {
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), tiny_flush_opts()).unwrap();

        for i in 0..10 {
            let k = format!("old{:02}", i);
            db.put(k.as_bytes(), b"old").unwrap();
        }
        force_flush(&db, "to-l0");

        for i in 0..5 {
            let k = format!("new{:02}", i);
            db.put(k.as_bytes(), b"new").unwrap();
        }

        let items = collect_iter(&db);
        let olds = items.iter().filter(|(k, _)| k.starts_with(b"old")).count();
        let news = items.iter().filter(|(k, _)| k.starts_with(b"new")).count();
        assert_eq!(olds, 10);
        assert_eq!(news, 5);

        let sorted: Vec<_> = items.iter().map(|(k, _)| k.clone()).collect();
        let mut expected = sorted.clone();
        expected.sort();
        assert_eq!(sorted, expected);
    }

    #[test]
    fn test_iter_tombstone_hides_older_level_entry() {
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), tiny_flush_opts()).unwrap();

        db.put(b"kept", b"v1").unwrap();
        db.put(b"gone", b"v1").unwrap();
        force_flush(&db, "a");

        db.delete(b"gone").unwrap();

        let items = collect_iter(&db);
        let keys: Vec<_> = items.iter().map(|(k, _)| k.clone()).collect();
        assert!(keys.contains(&b"kept".to_vec()));
        assert!(!keys.contains(&b"gone".to_vec()));
    }

    #[test]
    fn test_iter_latest_version_wins_across_levels() {
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), tiny_flush_opts()).unwrap();

        db.put(b"k", b"v1").unwrap();
        force_flush(&db, "a");
        db.put(b"k", b"v2").unwrap();

        let mut it = db.iter();
        it.seek(b"k");
        assert_eq!(it.key(), Some(b"k".as_ref()));
        assert_eq!(it.value(), Some(b"v2".as_ref()));
    }

    #[test]
    fn test_iter_honors_snapshot_isolation() {
        let (db, _dir) = open_tmp();
        db.put(b"k", b"v1").unwrap();
        let snap = db.snapshot();
        db.put(b"k", b"v2").unwrap();

        let mut it = snap.iter();
        it.seek(b"k");
        assert_eq!(it.value(), Some(b"v1".as_ref()));
    }

    #[test]
    fn test_iter_snapshot_ignores_tombstone_newer_than_snap() {
        let (db, _dir) = open_tmp();
        db.put(b"k", b"v1").unwrap();
        let snap = db.snapshot();
        db.delete(b"k").unwrap();

        let mut it = snap.iter();
        it.seek(b"k");
        assert_eq!(it.value(), Some(b"v1".as_ref()));
    }

    #[test]
    fn test_iter_consistency_with_scan() {
        let (db, _dir) = open_tmp();
        for i in 0..100 {
            let k = format!("k{:03}", i);
            let v = format!("v{}", i);
            db.put(k.as_bytes(), v.as_bytes()).unwrap();
        }

        let scan = db.scan(Some(b"k020"), Some(b"k050")).unwrap();

        let mut it = db.iter();
        it.seek(b"k020");
        let mut from_iter = Vec::new();
        while it.valid() {
            let k = it.key().unwrap();
            if k >= b"k050".as_ref() {
                break;
            }
            from_iter.push((k.to_vec(), it.value().unwrap().to_vec()));
            it.next();
        }

        assert_eq!(scan, from_iter);
    }

    #[test]
    fn test_iter_large_scan_10k_keys_after_flush() {
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), tiny_flush_opts()).unwrap();

        const N: usize = 10_000;
        for i in 0..N {
            let k = format!("key_{:06}", i);
            db.put(k.as_bytes(), b"v").unwrap();
        }

        let mut it = db.iter();
        it.seek(b"key_");
        let mut count = 0;
        while it.valid() {
            let k = it.key().unwrap();
            if !k.starts_with(b"key_") {
                it.next();
                continue;
            }
            count += 1;
            it.next();
        }
        assert_eq!(count, N);
    }

    // ─── Reverse iteration tests ─────────────────────────────────────────

    fn collect_reverse(db: &Db) -> Vec<(Vec<u8>, Vec<u8>)> {
        let mut it = db.iter();
        it.seek_to_last();
        let mut out = Vec::new();
        while it.valid() {
            out.push((it.key().unwrap().to_vec(), it.value().unwrap().to_vec()));
            it.prev();
        }
        it.status().unwrap();
        out
    }

    #[test]
    fn test_iter_seek_to_last_empty() {
        let (db, _dir) = open_tmp();
        let mut it = db.iter();
        it.seek_to_last();
        assert!(!it.valid());
    }

    #[test]
    fn test_iter_reverse_walk_basic() {
        let (db, _dir) = open_tmp();
        for i in 0..10 {
            let k = format!("k{:02}", i);
            db.put(k.as_bytes(), b"v").unwrap();
        }
        let items = collect_reverse(&db);
        assert_eq!(items.len(), 10);
        for (i, (k, _)) in items.iter().enumerate() {
            assert_eq!(k, format!("k{:02}", 9 - i).as_bytes());
        }
    }

    #[test]
    fn test_iter_prev_latest_version() {
        let (db, _dir) = open_tmp();
        db.put(b"a", b"a1").unwrap();
        db.put(b"b", b"b1").unwrap();
        db.put(b"b", b"b2").unwrap();
        db.put(b"c", b"c1").unwrap();

        let mut it = db.iter();
        it.seek_to_last();
        assert_eq!(it.key(), Some(b"c".as_ref()));
        it.prev();
        assert_eq!(it.key(), Some(b"b".as_ref()));
        assert_eq!(it.value(), Some(b"b2".as_ref()));
        it.prev();
        assert_eq!(it.key(), Some(b"a".as_ref()));
        it.prev();
        assert!(!it.valid());
    }

    #[test]
    fn test_iter_seek_for_prev_then_prev() {
        let (db, _dir) = open_tmp();
        db.put(b"a", b"1").unwrap();
        db.put(b"c", b"3").unwrap();
        db.put(b"e", b"5").unwrap();
        db.put(b"g", b"7").unwrap();

        let mut it = db.iter();
        it.seek_for_prev(b"f");
        assert_eq!(it.key(), Some(b"e".as_ref()));
        it.prev();
        assert_eq!(it.key(), Some(b"c".as_ref()));
        it.prev();
        assert_eq!(it.key(), Some(b"a".as_ref()));
        it.prev();
        assert!(!it.valid());
    }

    #[test]
    fn test_iter_reverse_across_flush_levels() {
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), tiny_flush_opts()).unwrap();

        for i in 0..20 {
            let k = format!("k{:02}", i);
            db.put(k.as_bytes(), b"v").unwrap();
        }
        force_flush(&db, "a");
        for i in 20..30 {
            let k = format!("k{:02}", i);
            db.put(k.as_bytes(), b"v").unwrap();
        }

        let items = collect_reverse(&db);
        let k_count = items.iter().filter(|(k, _)| k.starts_with(b"k")).count();
        assert_eq!(k_count, 30);
        let mut prev_k: Option<Vec<u8>> = None;
        for (k, _) in items.iter().filter(|(k, _)| k.starts_with(b"k")) {
            if let Some(p) = &prev_k {
                assert!(k < p, "not descending: {:?} after {:?}", k, p);
            }
            prev_k = Some(k.clone());
        }
    }

    #[test]
    fn test_iter_reverse_hides_tombstoned_user_key() {
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), tiny_flush_opts()).unwrap();

        db.put(b"keep", b"v").unwrap();
        db.put(b"gone", b"v").unwrap();
        force_flush(&db, "a");
        db.delete(b"gone").unwrap();

        let items = collect_reverse(&db);
        let keys: Vec<_> = items.iter().map(|(k, _)| k.clone()).collect();
        assert!(keys.contains(&b"keep".to_vec()));
        assert!(!keys.contains(&b"gone".to_vec()));
    }

    #[test]
    fn test_iter_reverse_honors_snapshot_isolation() {
        let (db, _dir) = open_tmp();
        db.put(b"k", b"v1").unwrap();
        let snap = db.snapshot();
        db.put(b"k", b"v2").unwrap();

        let mut it = snap.iter();
        it.seek_to_last();
        assert_eq!(it.key(), Some(b"k".as_ref()));
        assert_eq!(it.value(), Some(b"v1".as_ref()));
    }

    #[test]
    fn test_iter_direction_flip_forward_to_reverse() {
        let (db, _dir) = open_tmp();
        for c in b'a'..=b'e' {
            db.put(&[c], &[c]).unwrap();
        }

        let mut it = db.iter();
        it.seek_to_first();
        assert_eq!(it.key(), Some(b"a".as_ref()));
        it.next();
        assert_eq!(it.key(), Some(b"b".as_ref()));
        it.next();
        assert_eq!(it.key(), Some(b"c".as_ref()));

        it.prev();
        assert_eq!(it.key(), Some(b"b".as_ref()));
        it.prev();
        assert_eq!(it.key(), Some(b"a".as_ref()));
        it.prev();
        assert!(!it.valid());
    }

    #[test]
    fn test_iter_direction_flip_reverse_to_forward() {
        let (db, _dir) = open_tmp();
        for c in b'a'..=b'e' {
            db.put(&[c], &[c]).unwrap();
        }

        let mut it = db.iter();
        it.seek_to_last();
        assert_eq!(it.key(), Some(b"e".as_ref()));
        it.prev();
        assert_eq!(it.key(), Some(b"d".as_ref()));
        it.prev();
        assert_eq!(it.key(), Some(b"c".as_ref()));

        it.next();
        assert_eq!(it.key(), Some(b"d".as_ref()));
        it.next();
        assert_eq!(it.key(), Some(b"e".as_ref()));
        it.next();
        assert!(!it.valid());
    }

    #[test]
    fn test_iter_reverse_scan_10k_keys_after_flush() {
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), tiny_flush_opts()).unwrap();

        const N: usize = 10_000;
        for i in 0..N {
            let k = format!("key_{:06}", i);
            db.put(k.as_bytes(), b"v").unwrap();
        }

        let mut it = db.iter();
        it.seek_for_prev(b"key_~"); // '~' sorts after digits
        let mut count = 0;
        let mut prev: Option<Vec<u8>> = None;
        while it.valid() {
            let k = it.key().unwrap().to_vec();
            if !k.starts_with(b"key_") {
                it.prev();
                continue;
            }
            if let Some(p) = &prev {
                assert!(k < *p, "not descending: {:?} after {:?}", k, p);
            }
            prev = Some(k);
            count += 1;
            it.prev();
        }
        assert_eq!(count, N);
        assert!(it.status().is_ok());
    }

    #[test]
    fn test_iter_seek_for_prev_on_tombstoned_key() {
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), tiny_flush_opts()).unwrap();

        db.put(b"a", b"a1").unwrap();
        db.put(b"b", b"b1").unwrap();
        db.put(b"c", b"c1").unwrap();
        force_flush(&db, "x");
        db.delete(b"b").unwrap();

        let mut it = db.iter();
        it.seek_for_prev(b"b");
        // `b` is tombstoned, so reverse-seek to `b` should skip past it
        // and land on `a`.
        assert_eq!(it.key(), Some(b"a".as_ref()));
    }

    #[test]
    fn test_iter_survives_drop_all() {
        // drop_all unlinks every SSTable file. An iterator captured before
        // drop_all holds its own Arc<SsTableReader>s (each with an open
        // File), so OS fd refcounting keeps the bytes alive and the
        // iterator continues to produce its original view.
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), tiny_flush_opts()).unwrap();

        for i in 0..20 {
            let k = format!("pin{:03}", i);
            db.put(k.as_bytes(), b"v").unwrap();
        }
        force_flush(&db, "pinned");

        let mut it = db.iter();
        it.seek_to_first();

        db.drop_all().unwrap();

        let mut seen_pinned = 0;
        while it.valid() {
            if it.key().unwrap().starts_with(b"pin") {
                seen_pinned += 1;
            }
            it.next();
        }
        assert_eq!(seen_pinned, 20);
        assert!(it.status().is_ok());
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
