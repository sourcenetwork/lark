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

    /// Look up a batch of keys in one call. Returns a vector with one
    /// entry per input key (preserving order and duplicates); each entry
    /// is `None` if the key does not exist or is tombstoned.
    ///
    /// All keys in a single call see the **same** consistent view — a
    /// concurrent writer cannot make two keys disagree about visibility.
    pub fn multi_get(&self, keys: &[&[u8]]) -> Result<Vec<Option<Vec<u8>>>> {
        let seq = self.engine.snapshot_seq();
        self.engine.multi_get(keys, seq).map_err(Error::Io)
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
    ///
    /// Snapshots also pin the compaction GC horizon: as long as at
    /// least one `Snapshot` at seq `S` is alive, the compaction
    /// thread will not drop any version needed to read at seq `S`.
    /// Dropping the returned `Snapshot` releases the pin and may
    /// allow subsequent compactions to reclaim space.
    pub fn snapshot(&self) -> Snapshot {
        let seq = self.engine.snapshot_seq();
        self.engine.register_snapshot(seq);
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

    /// Synchronously compact every SSTable overlapping the user-key
    /// range `[start, end)` down to the bottommost non-empty level.
    ///
    /// Passing `None` for either bound means "unbounded" on that side,
    /// so `compact_range(None, None)` compacts the entire database.
    ///
    /// Active memtable contents that fall in the range are flushed to
    /// L0 first. The call blocks until the requested compaction work
    /// is finished and is serialized with the background compaction
    /// scheduler so the two paths can't fight over the same inputs.
    pub fn compact_range(&self, start: Option<&[u8]>, end: Option<&[u8]>) -> Result<()> {
        self.engine.compact_range(start, end).map_err(Error::Io)
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

impl Drop for Snapshot {
    fn drop(&mut self) {
        // Release the pin this snapshot held in the engine's
        // compaction GC registry. Compaction is now free to drop any
        // version it was keeping alive for this snapshot's sake,
        // subject to other live snapshots that may still pin older
        // seqs.
        self.engine.release_snapshot(self.seq);
    }
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

    /// Batched point lookup anchored at this snapshot.
    pub fn multi_get(&self, keys: &[&[u8]]) -> Result<Vec<Option<Vec<u8>>>> {
        self.engine.multi_get(keys, self.seq).map_err(Error::Io)
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

    // ─── Snapshot-pinning GC tests ──────────────────────────────────────

    /// Thin wrapper around the engine's test-only persisted-versions
    /// accessor. Returns `(seq, value_type)` for every copy of
    /// `user_key` currently sitting in an SSTable at any level.
    fn all_versions_of(db: &Db, user_key: &[u8]) -> Vec<(u64, u8)> {
        db.engine.all_persisted_versions_of(user_key).unwrap()
    }

    #[test]
    fn test_gc_drops_old_versions_without_snapshot() {
        // With no live snapshot, compact_range(None, None) should
        // leave only the newest version of each user key on disk.
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), tiny_flush_opts()).unwrap();

        for v in 0..10 {
            db.put(b"k", format!("v{}", v).as_bytes()).unwrap();
        }

        db.compact_range(None, None).unwrap();

        let versions = all_versions_of(&db, b"k");
        assert_eq!(
            versions.len(),
            1,
            "expected a single surviving version, found {:?}",
            versions
        );
        assert_eq!(db.get(b"k").unwrap(), Some(b"v9".to_vec()));
    }

    #[test]
    fn test_gc_preserves_versions_pinned_by_snapshot() {
        // Take a snapshot at seq 5, then write more versions. After
        // compaction the snapshot must still read its view, which
        // requires preserving the version it pinned.
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), tiny_flush_opts()).unwrap();

        db.put(b"k", b"v1").unwrap();
        db.put(b"k", b"v2").unwrap();
        db.put(b"k", b"v3").unwrap();
        let snap = db.snapshot();
        // `snap` now pins seq=3 — the snapshot sees v3.

        for v in 4..10 {
            db.put(b"k", format!("v{}", v).as_bytes()).unwrap();
        }

        db.compact_range(None, None).unwrap();

        assert_eq!(snap.get(b"k").unwrap(), Some(b"v3".to_vec()));
        assert_eq!(db.get(b"k").unwrap(), Some(b"v9".to_vec()));
    }

    #[test]
    fn test_gc_releases_pin_when_snapshot_drops() {
        // Pinning a snapshot and then dropping it should fully
        // release the horizon so the next compaction can collapse
        // the key to a single surviving version.
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), tiny_flush_opts()).unwrap();

        for v in 0..5 {
            db.put(b"k", format!("v{}", v).as_bytes()).unwrap();
        }

        {
            let _snap = db.snapshot();
            assert_eq!(db.engine.oldest_live_seq(), 5);
        }
        // Pin released.
        assert_eq!(db.engine.oldest_live_seq(), u64::MAX);

        for v in 5..10 {
            db.put(b"k", format!("v{}", v).as_bytes()).unwrap();
        }

        db.compact_range(None, None).unwrap();

        let versions = all_versions_of(&db, b"k");
        assert_eq!(versions.len(), 1);
        assert_eq!(db.get(b"k").unwrap(), Some(b"v9".to_vec()));
    }

    #[test]
    fn test_gc_with_multiple_live_snapshots_uses_oldest() {
        // When two snapshots are live, the older one's seq is the
        // GC horizon. Every version newer than (or at) the older
        // snapshot's seq must be preserved so the newer snapshot
        // can still read its own view too.
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), tiny_flush_opts()).unwrap();

        db.put(b"k", b"v1").unwrap();
        db.put(b"k", b"v2").unwrap();
        let old_snap = db.snapshot(); // pins seq 2
        db.put(b"k", b"v3").unwrap();
        db.put(b"k", b"v4").unwrap();
        let new_snap = db.snapshot(); // pins seq 4
        db.put(b"k", b"v5").unwrap();
        db.put(b"k", b"v6").unwrap();

        db.compact_range(None, None).unwrap();

        // Both snapshots must still return their respective versions.
        assert_eq!(old_snap.get(b"k").unwrap(), Some(b"v2".to_vec()));
        assert_eq!(new_snap.get(b"k").unwrap(), Some(b"v4".to_vec()));
        assert_eq!(db.get(b"k").unwrap(), Some(b"v6".to_vec()));
    }

    #[test]
    fn test_gc_preserves_tombstone_hiding_older_entries() {
        // A tombstone newer than any live snapshot still needs to
        // survive compaction — it's the newest version and reads
        // must resolve to "deleted".
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), tiny_flush_opts()).unwrap();

        for v in 0..5 {
            db.put(b"k", format!("v{}", v).as_bytes()).unwrap();
        }
        db.delete(b"k").unwrap();

        db.compact_range(None, None).unwrap();

        assert_eq!(db.get(b"k").unwrap(), None);

        // The newest surviving version is a tombstone — look for it
        // on disk.
        let versions = all_versions_of(&db, b"k");
        assert!(!versions.is_empty());
        // Highest seq is the tombstone.
        let (_, vt) = *versions.iter().max_by_key(|(seq, _)| *seq).unwrap();
        const VALUE_TYPE_DELETION: u8 = 0;
        assert_eq!(vt, VALUE_TYPE_DELETION);
    }

    #[test]
    fn test_gc_across_many_user_keys() {
        // Stress the multi-group path: many distinct user keys each
        // with several versions. No snapshot is live so each key
        // should collapse to exactly one surviving version.
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), tiny_flush_opts()).unwrap();

        for i in 0..200 {
            for v in 0..3 {
                db.put(
                    format!("k{:03}", i).as_bytes(),
                    format!("v{}_{}", i, v).as_bytes(),
                )
                .unwrap();
            }
        }

        db.compact_range(None, None).unwrap();

        for i in 0..200 {
            let k = format!("k{:03}", i);
            let versions = all_versions_of(&db, k.as_bytes());
            assert_eq!(versions.len(), 1, "key {} survived with {:?}", k, versions);
            assert_eq!(
                db.get(k.as_bytes()).unwrap(),
                Some(format!("v{}_2", i).into_bytes())
            );
        }
    }

    // ─── compact_range tests ────────────────────────────────────────────

    fn level_file_count(db: &Db, level: usize) -> usize {
        db.engine.level_file_count(level)
    }

    fn total_file_count(db: &Db) -> usize {
        db.engine.total_file_count()
    }

    #[test]
    fn test_compact_range_empty_db() {
        let (db, _dir) = open_tmp();
        // No data, no files. compact_range is a no-op and must succeed.
        db.compact_range(None, None).unwrap();
        assert_eq!(total_file_count(&db), 0);
    }

    #[test]
    fn test_compact_range_full_preserves_reads() {
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), tiny_flush_opts()).unwrap();

        for i in 0..500 {
            let k = format!("k{:04}", i);
            db.put(k.as_bytes(), format!("v{}", i).as_bytes()).unwrap();
        }

        db.compact_range(None, None).unwrap();

        // Every key is still readable after the compaction.
        for i in 0..500 {
            let k = format!("k{:04}", i);
            assert_eq!(
                db.get(k.as_bytes()).unwrap(),
                Some(format!("v{}", i).into_bytes())
            );
        }
    }

    #[test]
    fn test_compact_range_flushes_active_memtable() {
        // Writes that are still in the memtable when compact_range is
        // called must be flushed to L0 before the walk, so the active
        // memtable is empty afterwards.
        let (db, _dir) = open_tmp();
        for i in 0..10 {
            let k = format!("m{:02}", i);
            db.put(k.as_bytes(), b"v").unwrap();
        }
        assert!(!db.engine.active_memtable_is_empty());

        db.compact_range(None, None).unwrap();

        assert!(db.engine.active_memtable_is_empty());
        // And data is still readable through the SSTable path.
        for i in 0..10 {
            let k = format!("m{:02}", i);
            assert_eq!(db.get(k.as_bytes()).unwrap(), Some(b"v".to_vec()));
        }
    }

    #[test]
    fn test_compact_range_drains_l0() {
        // After a full compact_range, nothing should remain at L0 —
        // every file must have been pushed down to L1+.
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), tiny_flush_opts()).unwrap();

        for i in 0..200 {
            let k = format!("k{:04}", i);
            db.put(k.as_bytes(), b"v").unwrap();
        }

        db.compact_range(None, None).unwrap();

        assert_eq!(level_file_count(&db, 0), 0);
        // Some higher level must hold the data.
        assert!(total_file_count(&db) > 0);
    }

    #[test]
    fn test_compact_range_bounded_preserves_all_data() {
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), tiny_flush_opts()).unwrap();

        // Three disjoint ranges: low (a*), mid (m*), high (z*).
        for i in 0..100 {
            db.put(format!("a{:03}", i).as_bytes(), b"a").unwrap();
        }
        for i in 0..100 {
            db.put(format!("m{:03}", i).as_bytes(), b"m").unwrap();
        }
        for i in 0..100 {
            db.put(format!("z{:03}", i).as_bytes(), b"z").unwrap();
        }

        // Only compact the mid range.
        db.compact_range(Some(b"m"), Some(b"n")).unwrap();

        // Every key must still be readable regardless of the range.
        for i in 0..100 {
            assert_eq!(
                db.get(format!("a{:03}", i).as_bytes()).unwrap(),
                Some(b"a".to_vec())
            );
            assert_eq!(
                db.get(format!("m{:03}", i).as_bytes()).unwrap(),
                Some(b"m".to_vec())
            );
            assert_eq!(
                db.get(format!("z{:03}", i).as_bytes()).unwrap(),
                Some(b"z".to_vec())
            );
        }
    }

    #[test]
    fn test_compact_range_reclaims_space_after_overwrite() {
        // Write N keys, overwrite them, force flush, then compact_range.
        // The number of distinct entries after compaction should be N
        // (one per user key) — the old overwritten versions got merged
        // away by deduplication during compaction.
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), tiny_flush_opts()).unwrap();

        for i in 0..200 {
            let k = format!("k{:03}", i);
            db.put(k.as_bytes(), b"v1").unwrap();
        }
        for i in 0..200 {
            let k = format!("k{:03}", i);
            db.put(k.as_bytes(), b"v2").unwrap();
        }

        db.compact_range(None, None).unwrap();

        for i in 0..200 {
            let k = format!("k{:03}", i);
            assert_eq!(db.get(k.as_bytes()).unwrap(), Some(b"v2".to_vec()));
        }
    }

    #[test]
    fn test_compact_range_runs_alongside_background_compaction() {
        // Write enough to trigger background compactions, then while
        // the engine is still churning, fire a foreground compact_range.
        // Both must complete without corruption.
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), tiny_flush_opts()).unwrap();

        const N: usize = 2_000;
        for i in 0..N {
            let k = format!("key_{:05}", i);
            db.put(k.as_bytes(), b"v").unwrap();
        }

        db.compact_range(None, None).unwrap();

        // After the foreground compaction, every key is still there.
        for i in 0..N {
            let k = format!("key_{:05}", i);
            assert_eq!(db.get(k.as_bytes()).unwrap(), Some(b"v".to_vec()));
        }
    }

    #[test]
    fn test_compact_range_iterator_still_correct() {
        // compact_range shouldn't perturb an iterator built after it.
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), tiny_flush_opts()).unwrap();

        for i in 0..300 {
            let k = format!("k{:04}", i);
            db.put(k.as_bytes(), b"v").unwrap();
        }

        db.compact_range(None, None).unwrap();

        let mut it = db.iter();
        it.seek_to_first();
        let mut count = 0;
        while it.valid() {
            if it.key().unwrap().starts_with(b"k") {
                count += 1;
            }
            it.next();
        }
        assert_eq!(count, 300);
    }

    #[test]
    fn test_compact_range_tombstones_are_preserved() {
        // Tombstones must survive compaction until the bottommost level
        // drops them — for now compaction preserves all versions, so a
        // deleted key is still absent to reads.
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), tiny_flush_opts()).unwrap();

        for i in 0..50 {
            let k = format!("k{:02}", i);
            db.put(k.as_bytes(), b"v").unwrap();
        }
        // Delete half of them.
        for i in (0..50).step_by(2) {
            let k = format!("k{:02}", i);
            db.delete(k.as_bytes()).unwrap();
        }

        db.compact_range(None, None).unwrap();

        for i in 0..50 {
            let k = format!("k{:02}", i);
            let expected = if i % 2 == 0 {
                None
            } else {
                Some(b"v".to_vec())
            };
            assert_eq!(db.get(k.as_bytes()).unwrap(), expected);
        }
    }

    // ─── MultiGet tests ─────────────────────────────────────────────────

    #[test]
    fn test_multi_get_empty_batch() {
        let (db, _dir) = open_tmp();
        db.put(b"x", b"y").unwrap();
        let results = db.multi_get(&[]).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_multi_get_all_hit() {
        let (db, _dir) = open_tmp();
        db.put(b"a", b"1").unwrap();
        db.put(b"b", b"2").unwrap();
        db.put(b"c", b"3").unwrap();

        let keys: &[&[u8]] = &[b"a", b"b", b"c"];
        let results = db.multi_get(keys).unwrap();
        assert_eq!(
            results,
            vec![
                Some(b"1".to_vec()),
                Some(b"2".to_vec()),
                Some(b"3".to_vec())
            ]
        );
    }

    #[test]
    fn test_multi_get_all_miss() {
        let (db, _dir) = open_tmp();
        db.put(b"a", b"1").unwrap();

        let keys: &[&[u8]] = &[b"x", b"y", b"z"];
        let results = db.multi_get(keys).unwrap();
        assert_eq!(results, vec![None, None, None]);
    }

    #[test]
    fn test_multi_get_mixed_hit_miss() {
        let (db, _dir) = open_tmp();
        db.put(b"a", b"1").unwrap();
        db.put(b"c", b"3").unwrap();

        let keys: &[&[u8]] = &[b"a", b"b", b"c", b"d"];
        let results = db.multi_get(keys).unwrap();
        assert_eq!(
            results,
            vec![Some(b"1".to_vec()), None, Some(b"3".to_vec()), None]
        );
    }

    #[test]
    fn test_multi_get_preserves_input_order() {
        let (db, _dir) = open_tmp();
        db.put(b"a", b"1").unwrap();
        db.put(b"b", b"2").unwrap();
        db.put(b"c", b"3").unwrap();

        // Reverse order input.
        let keys: &[&[u8]] = &[b"c", b"a", b"b"];
        let results = db.multi_get(keys).unwrap();
        assert_eq!(
            results,
            vec![
                Some(b"3".to_vec()),
                Some(b"1".to_vec()),
                Some(b"2".to_vec())
            ]
        );
    }

    #[test]
    fn test_multi_get_duplicates_in_input() {
        let (db, _dir) = open_tmp();
        db.put(b"a", b"1").unwrap();
        db.put(b"b", b"2").unwrap();

        let keys: &[&[u8]] = &[b"a", b"b", b"a", b"missing", b"a"];
        let results = db.multi_get(keys).unwrap();
        assert_eq!(
            results,
            vec![
                Some(b"1".to_vec()),
                Some(b"2".to_vec()),
                Some(b"1".to_vec()),
                None,
                Some(b"1".to_vec()),
            ]
        );
    }

    #[test]
    fn test_multi_get_honors_tombstones() {
        let (db, _dir) = open_tmp();
        db.put(b"a", b"1").unwrap();
        db.put(b"b", b"2").unwrap();
        db.put(b"c", b"3").unwrap();
        db.delete(b"b").unwrap();

        let keys: &[&[u8]] = &[b"a", b"b", b"c"];
        let results = db.multi_get(keys).unwrap();
        assert_eq!(
            results,
            vec![Some(b"1".to_vec()), None, Some(b"3".to_vec())]
        );
    }

    #[test]
    fn test_multi_get_tombstone_hides_older_level_entry() {
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), tiny_flush_opts()).unwrap();

        db.put(b"keep", b"v").unwrap();
        db.put(b"gone", b"v").unwrap();
        force_flush(&db, "x");
        db.delete(b"gone").unwrap();

        let keys: &[&[u8]] = &[b"keep", b"gone"];
        let results = db.multi_get(keys).unwrap();
        assert_eq!(results, vec![Some(b"v".to_vec()), None]);
    }

    #[test]
    fn test_multi_get_spans_memtable_and_l0() {
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), tiny_flush_opts()).unwrap();

        db.put(b"from_l0_1", b"v1").unwrap();
        db.put(b"from_l0_2", b"v2").unwrap();
        force_flush(&db, "x");

        db.put(b"from_mem_1", b"v3").unwrap();
        db.put(b"from_mem_2", b"v4").unwrap();

        let keys: &[&[u8]] = &[b"from_mem_1", b"from_l0_1", b"from_mem_2", b"from_l0_2"];
        let results = db.multi_get(keys).unwrap();
        assert_eq!(
            results,
            vec![
                Some(b"v3".to_vec()),
                Some(b"v1".to_vec()),
                Some(b"v4".to_vec()),
                Some(b"v2".to_vec())
            ]
        );
    }

    #[test]
    fn test_multi_get_snapshot_isolation() {
        let (db, _dir) = open_tmp();
        db.put(b"a", b"a1").unwrap();
        db.put(b"b", b"b1").unwrap();

        let snap = db.snapshot();

        db.put(b"a", b"a2").unwrap();
        db.put(b"c", b"c1").unwrap();
        db.delete(b"b").unwrap();

        let keys: &[&[u8]] = &[b"a", b"b", b"c"];
        let results = snap.multi_get(keys).unwrap();
        assert_eq!(
            results,
            vec![Some(b"a1".to_vec()), Some(b"b1".to_vec()), None],
        );
    }

    #[test]
    fn test_multi_get_consistency_with_get() {
        // For any batch, multi_get must return the same results as a
        // loop of individual get calls at the same snapshot.
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), tiny_flush_opts()).unwrap();

        for i in 0..500 {
            let k = format!("k{:04}", i);
            db.put(k.as_bytes(), format!("v{}", i).as_bytes()).unwrap();
        }
        // Delete some.
        for i in (0..500).step_by(7) {
            let k = format!("k{:04}", i);
            db.delete(k.as_bytes()).unwrap();
        }

        // Snapshot so individual gets and multi_get see the same thing.
        let snap = db.snapshot();

        let keys_owned: Vec<String> = (0..500)
            .step_by(3)
            .map(|i| format!("k{:04}", i))
            .chain(std::iter::once("missing_key".to_string()))
            .collect();
        let keys: Vec<&[u8]> = keys_owned.iter().map(|s| s.as_bytes()).collect();

        let individual: Vec<_> = keys.iter().map(|k| snap.get(k).unwrap()).collect();
        let batched = snap.multi_get(&keys).unwrap();

        assert_eq!(individual, batched);
        assert_eq!(individual.len(), keys.len());
    }

    #[test]
    fn test_multi_get_large_batch_after_flush() {
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), tiny_flush_opts()).unwrap();

        const N: usize = 2_000;
        for i in 0..N {
            let k = format!("key_{:05}", i);
            db.put(k.as_bytes(), b"v").unwrap();
        }

        let keys_owned: Vec<String> = (0..N).map(|i| format!("key_{:05}", i)).collect();
        let keys: Vec<&[u8]> = keys_owned.iter().map(|s| s.as_bytes()).collect();
        let results = db.multi_get(&keys).unwrap();
        assert_eq!(results.len(), N);
        for r in &results {
            assert_eq!(r.as_deref(), Some(b"v".as_ref()));
        }
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
    fn test_iter_reverse_seek_past_end_of_multi_block_sst() {
        // Regression: SsTableLevelIter::seek_for_prev used to fall back
        // to block 0 when the target exceeded every entry in the SST.
        // The correct fallback is the *last* block, so reverse walks
        // that start past the end actually visit every user key.
        //
        // Forces a multi-block SSTable with a small `block_size`, flushes
        // to L0 via `close()` so the data is guaranteed to be on disk,
        // then reopens and runs `seek_for_prev` with a target larger
        // than every key.
        let dir = TempDir::new().unwrap();
        let opts = Options {
            block_size: 128,
            write_buffer_size: 64 * 1024,
            ..Options::default()
        };
        {
            let db = Db::open(dir.path(), opts.clone()).unwrap();
            for i in 0..60u32 {
                let k = format!("k{:03}", i);
                db.put(k.as_bytes(), b"v").unwrap();
            }
            db.close().unwrap();
        }

        let db = Db::open(dir.path(), opts).unwrap();
        let mut it = db.iter();
        it.seek_for_prev(b"~"); // '~' sorts after 'k'

        let mut seen = Vec::new();
        while it.valid() {
            seen.push(it.key().unwrap().to_vec());
            it.prev();
        }
        assert_eq!(seen.len(), 60);
        assert_eq!(seen.first().map(|k| k.as_slice()), Some(&b"k059"[..]));
        assert_eq!(seen.last().map(|k| k.as_slice()), Some(&b"k000"[..]));
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
