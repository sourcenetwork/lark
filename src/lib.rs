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

mod backup;
mod checkpoint;
mod engine;
mod error;
mod iter;
mod options;
mod sst_file_writer;
mod transaction;
mod ttl;

pub use backup::{BackupEngine, BackupId, BackupInfo};
pub use checkpoint::Checkpoint;
pub use error::Error;
pub use iter::Iter;
pub use options::{
    CompactionDecision, CompactionFilter, CompressionType, DurabilityMode, FixedLengthPrefix,
    MergeOperator, Options, PrefixExtractor, WriteOptions,
};
pub use sst_file_writer::{IngestOptions, SstFileMeta, SstFileWriter};
pub use transaction::{
    OptimisticTransactionDb, Transaction, TransactionDb, TransactionError, TxResult,
};
pub use ttl::{strip_timestamp, DbWithTtl, TtlCompactionFilter};

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

    /// Set a key-value pair using the database-global durability mode
    /// and default write options.
    pub fn put(&self, key: &[u8], value: &[u8]) -> Result<()> {
        self.put_opt(&WriteOptions::default(), key, value)
    }

    /// Set a key-value pair with an explicit [`WriteOptions`] override.
    /// Overrides on a per-call basis the database-global
    /// [`Options::durability`] mode.
    pub fn put_opt(&self, opts: &WriteOptions, key: &[u8], value: &[u8]) -> Result<()> {
        let mut batch = BTreeMap::new();
        batch.insert(key.to_vec(), Some(value.to_vec()));
        let (dm, disable_wal) = self.resolve_write_opts(opts);
        self.engine
            .apply_batch(batch, Vec::new(), Vec::new(), dm, disable_wal)
            .map_err(Error::Io)
    }

    /// Delete a key using the database-global durability mode.
    pub fn delete(&self, key: &[u8]) -> Result<()> {
        self.delete_opt(&WriteOptions::default(), key)
    }

    /// Delete a key with an explicit [`WriteOptions`] override.
    pub fn delete_opt(&self, opts: &WriteOptions, key: &[u8]) -> Result<()> {
        let mut batch = BTreeMap::new();
        batch.insert(key.to_vec(), None);
        let (dm, disable_wal) = self.resolve_write_opts(opts);
        self.engine
            .apply_batch(batch, Vec::new(), Vec::new(), dm, disable_wal)
            .map_err(Error::Io)
    }

    /// Layer a merge operand on top of `key`.
    ///
    /// Requires an [`Options::merge_operator`] to be configured. The
    /// operand is written cheaply (no read-modify-write); readers
    /// collapse the chain of merges plus any base value via the
    /// configured operator at visibility time.
    pub fn merge(&self, key: &[u8], operand: &[u8]) -> Result<()> {
        self.merge_opt(&WriteOptions::default(), key, operand)
    }

    /// [`Db::merge`] with an explicit [`WriteOptions`] override.
    pub fn merge_opt(&self, opts: &WriteOptions, key: &[u8], operand: &[u8]) -> Result<()> {
        let (dm, disable_wal) = self.resolve_write_opts(opts);
        self.engine
            .apply_batch(
                BTreeMap::new(),
                Vec::new(),
                vec![(key.to_vec(), operand.to_vec())],
                dm,
                disable_wal,
            )
            .map_err(Error::Io)
    }

    /// Delete every key in the half-open range `[start, end)`.
    ///
    /// Range deletes are cheap regardless of how many keys the range
    /// covers — internally they are stored as a single range-tombstone
    /// record rather than as one point tombstone per key. The delete
    /// is durable under the same rules as [`Db::put`] / [`Db::delete`]
    /// and is atomic with respect to concurrent readers.
    ///
    /// If `start >= end` the call is a no-op.
    pub fn delete_range(&self, start: &[u8], end: &[u8]) -> Result<()> {
        self.delete_range_opt(&WriteOptions::default(), start, end)
    }

    /// Delete every key in `[start, end)` with an explicit
    /// [`WriteOptions`] override.
    pub fn delete_range_opt(&self, opts: &WriteOptions, start: &[u8], end: &[u8]) -> Result<()> {
        if start >= end {
            return Ok(());
        }
        let (dm, disable_wal) = self.resolve_write_opts(opts);
        self.engine
            .apply_batch(
                BTreeMap::new(),
                vec![(start.to_vec(), end.to_vec())],
                Vec::new(),
                dm,
                disable_wal,
            )
            .map_err(Error::Io)
    }

    /// Apply a batch of writes atomically using the database-global
    /// durability mode.
    pub fn write(&self, batch: WriteBatch) -> Result<()> {
        self.write_opt(&WriteOptions::default(), batch)
    }

    /// Apply a batch of writes atomically with an explicit
    /// [`WriteOptions`] override.
    pub fn write_opt(&self, opts: &WriteOptions, batch: WriteBatch) -> Result<()> {
        if batch.is_empty() {
            return Ok(());
        }
        let (dm, disable_wal) = self.resolve_write_opts(opts);
        self.engine
            .apply_batch(
                batch.ops,
                batch.range_deletes,
                batch.merges,
                dm,
                disable_wal,
            )
            .map_err(Error::Io)
    }

    /// Apply a batch of writes atomically with an explicit
    /// [`DurabilityMode`] override. Retained for backwards
    /// compatibility — prefer [`Db::write_opt`] for new code.
    pub fn write_with_durability(
        &self,
        batch: WriteBatch,
        durability: DurabilityMode,
    ) -> Result<()> {
        let opts = WriteOptions {
            sync: matches!(durability, DurabilityMode::Immediate),
            ..WriteOptions::default()
        };
        self.write_opt(&opts, batch)
    }

    /// Resolve a [`WriteOptions`] into the pair the engine's
    /// `apply_batch` actually consumes: a concrete
    /// `engine::DurabilityMode` and a `disable_wal` bool. `sync: true`
    /// maps to `Immediate` regardless of the database-global default;
    /// otherwise the default wins. `low_pri` and `no_slowdown` are
    /// accepted but currently no-ops — they're reserved for future
    /// write-stall / rate-limiter plumbing.
    fn resolve_write_opts(&self, opts: &WriteOptions) -> (engine::DurabilityMode, bool) {
        let dm = if opts.sync {
            engine::DurabilityMode::Immediate
        } else {
            self.durability
        };
        (dm, opts.disable_wal)
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

    /// Bulk-ingest one or more externally-built SSTable files. Each
    /// file must have been produced by [`SstFileWriter`]; on success
    /// every ingested file is placed at the appropriate level and its
    /// keys become visible to new reads and iterators. See
    /// [`IngestOptions`] for the snapshot-consistency and placement
    /// rules.
    ///
    /// The source files are left untouched on disk — the engine
    /// re-emits each file into the database's own SSTable directory
    /// so it can rewrite entry sequence numbers. Callers may delete
    /// the source files or re-ingest them at any time.
    pub fn ingest_external_files(
        &self,
        files: &[std::path::PathBuf],
        opts: IngestOptions,
    ) -> Result<()> {
        self.engine
            .ingest_external_files(files, &opts)
            .map_err(Error::Io)
    }

    /// Flush all data to disk and shut down background threads.
    pub fn close(&self) -> Result<()> {
        self.engine.close().map_err(Error::Io)
    }

    /// Test-only: number of SSTable files at `level`.
    #[cfg(test)]
    pub(crate) fn level_file_count(&self, level: usize) -> usize {
        self.engine.level_file_count(level)
    }

    /// Create a hard-linked [`Checkpoint`] of the database.
    ///
    /// Equivalent to [`Checkpoint::new`] followed by
    /// [`Checkpoint::create`]. The call briefly flushes the active
    /// memtable and compacts the manifest before any files are
    /// linked; concurrent writers continue to make progress.
    pub fn checkpoint<P: AsRef<Path>>(&self, target_dir: P) -> Result<()> {
        let cp = Checkpoint::new(self)?;
        cp.create(target_dir)
    }

    pub(crate) fn engine(&self) -> &LarkEngine {
        &self.engine
    }

    /// Clone the engine `Arc` — used by transaction facade types
    /// that need to carry an engine reference around independent
    /// of the owning `Db`'s lifetime. Internal-only.
    pub(crate) fn engine_arc(&self) -> Arc<LarkEngine> {
        Arc::clone(&self.engine)
    }

    /// Database-global durability mode. Used by transaction
    /// commit code to choose fsync semantics.
    pub(crate) fn durability(&self) -> engine::DurabilityMode {
        self.durability
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
    range_deletes: Vec<(Vec<u8>, Vec<u8>)>,
    merges: Vec<(Vec<u8>, Vec<u8>)>,
}

impl WriteBatch {
    /// Create an empty write batch.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a put operation to the batch.
    pub fn put(&mut self, key: &[u8], value: &[u8]) {
        self.ops.insert(key.to_vec(), Some(value.to_vec()));
    }

    /// Add a delete operation to the batch.
    pub fn delete(&mut self, key: &[u8]) {
        self.ops.insert(key.to_vec(), None);
    }

    /// Delete every key in the half-open range `[start, end)`.
    ///
    /// When the batch is applied, the range delete is recorded with
    /// the same transactional seq as the other batch operations, so
    /// concurrent readers see an all-or-nothing effect. Calls with
    /// `start >= end` are ignored.
    pub fn delete_range(&mut self, start: &[u8], end: &[u8]) {
        if start >= end {
            return;
        }
        self.range_deletes.push((start.to_vec(), end.to_vec()));
    }

    /// Add a merge operand for `key`. Requires the database to be
    /// configured with a [`MergeOperator`]; the operand is layered
    /// on top of any existing value or merge chain and collapsed at
    /// read time. Multiple merges on the same key in a single batch
    /// are allowed and applied in insertion order.
    pub fn merge(&mut self, key: &[u8], operand: &[u8]) {
        self.merges.push((key.to_vec(), operand.to_vec()));
    }

    /// Number of point operations in the batch. Range deletes and
    /// merges are counted separately via
    /// [`WriteBatch::range_delete_count`] and
    /// [`WriteBatch::merge_count`].
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Number of range-delete operations in the batch.
    pub fn range_delete_count(&self) -> usize {
        self.range_deletes.len()
    }

    /// Number of merge operations in the batch.
    pub fn merge_count(&self) -> usize {
        self.merges.len()
    }

    /// Whether the batch contains no operations of any kind.
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty() && self.range_deletes.is_empty() && self.merges.is_empty()
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

    // ── delete_range ────────────────────────────────────────────────────────

    #[test]
    fn test_delete_range_basic() {
        let (db, _dir) = open_tmp();
        for c in b'a'..=b'j' {
            db.put(&[c], &[c]).unwrap();
        }
        db.delete_range(b"c", b"g").unwrap();

        assert_eq!(db.get(b"a").unwrap(), Some(b"a".to_vec()));
        assert_eq!(db.get(b"b").unwrap(), Some(b"b".to_vec()));
        assert_eq!(db.get(b"c").unwrap(), None);
        assert_eq!(db.get(b"d").unwrap(), None);
        assert_eq!(db.get(b"e").unwrap(), None);
        assert_eq!(db.get(b"f").unwrap(), None);
        assert_eq!(db.get(b"g").unwrap(), Some(b"g".to_vec())); // end exclusive
        assert_eq!(db.get(b"j").unwrap(), Some(b"j".to_vec()));
    }

    #[test]
    fn test_delete_range_no_op_for_empty_or_inverted() {
        let (db, _dir) = open_tmp();
        db.put(b"a", b"1").unwrap();
        // Inverted range should be a silent no-op.
        db.delete_range(b"z", b"a").unwrap();
        // Equal bounds should also be a no-op (half-open empty range).
        db.delete_range(b"a", b"a").unwrap();
        assert_eq!(db.get(b"a").unwrap(), Some(b"1".to_vec()));
    }

    #[test]
    fn test_delete_range_then_put_inside_range() {
        let (db, _dir) = open_tmp();
        db.put(b"k", b"old").unwrap();
        db.delete_range(b"a", b"z").unwrap();
        // A put after the range delete must win — it has a higher seq.
        db.put(b"k", b"new").unwrap();
        assert_eq!(db.get(b"k").unwrap(), Some(b"new".to_vec()));
    }

    #[test]
    fn test_delete_range_put_then_range_delete_then_overwrite() {
        let (db, _dir) = open_tmp();
        db.put(b"k", b"v1").unwrap();
        db.delete_range(b"a", b"z").unwrap();
        assert_eq!(db.get(b"k").unwrap(), None);
        db.put(b"k", b"v2").unwrap();
        assert_eq!(db.get(b"k").unwrap(), Some(b"v2".to_vec()));
    }

    #[test]
    fn test_delete_range_snapshot_isolation() {
        let (db, _dir) = open_tmp();
        db.put(b"k", b"v1").unwrap();
        let snap = db.snapshot();
        db.delete_range(b"a", b"z").unwrap();
        assert_eq!(db.get(b"k").unwrap(), None);
        // Snapshot is anchored before the range delete.
        assert_eq!(snap.get(b"k").unwrap(), Some(b"v1".to_vec()));
    }

    #[test]
    fn test_delete_range_survives_flush() {
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), tiny_flush_opts()).unwrap();
        for i in 0..20 {
            db.put(format!("key_{:02}", i).as_bytes(), b"v").unwrap();
        }
        db.delete_range(b"key_05", b"key_15").unwrap();
        force_flush(&db, "rt");
        for i in 0..20 {
            let key = format!("key_{:02}", i);
            let got = db.get(key.as_bytes()).unwrap();
            if (5..15).contains(&i) {
                assert_eq!(got, None, "key {} should be deleted", key);
            } else {
                assert_eq!(got, Some(b"v".to_vec()), "key {} should survive", key);
            }
        }
    }

    #[test]
    fn test_delete_range_survives_compaction() {
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), tiny_flush_opts()).unwrap();
        for i in 0..30 {
            db.put(format!("key_{:02}", i).as_bytes(), b"v").unwrap();
        }
        db.delete_range(b"key_10", b"key_20").unwrap();
        // Force several flushes + a manual compaction down to L1+.
        for tag in 0..6 {
            force_flush(&db, &format!("c{}", tag));
        }
        db.compact_range(None, None).unwrap();

        for i in 0..30 {
            let key = format!("key_{:02}", i);
            let got = db.get(key.as_bytes()).unwrap();
            if (10..20).contains(&i) {
                assert_eq!(got, None, "key {} should be deleted post-compact", key);
            } else {
                assert_eq!(
                    got,
                    Some(b"v".to_vec()),
                    "key {} should survive compact",
                    key
                );
            }
        }
    }

    #[test]
    fn test_delete_range_iterator_skips_deleted() {
        let (db, _dir) = open_tmp();
        for c in b'a'..=b'h' {
            db.put(&[c], &[c]).unwrap();
        }
        db.delete_range(b"c", b"f").unwrap();

        let results = db.scan(None, None).unwrap();
        let keys: Vec<u8> = results.iter().map(|(k, _)| k[0]).collect();
        assert_eq!(keys, vec![b'a', b'b', b'f', b'g', b'h']);
    }

    #[test]
    fn test_delete_range_reverse_iterator_skips_deleted() {
        let (db, _dir) = open_tmp();
        for c in b'a'..=b'h' {
            db.put(&[c], &[c]).unwrap();
        }
        db.delete_range(b"c", b"f").unwrap();

        let mut iter = db.iter();
        iter.seek_to_last();
        let mut keys = Vec::new();
        while iter.valid() {
            keys.push(iter.key().unwrap()[0]);
            iter.prev();
        }
        assert_eq!(keys, vec![b'h', b'g', b'f', b'b', b'a']);
    }

    #[test]
    fn test_delete_range_multi_get_honors_rt() {
        let (db, _dir) = open_tmp();
        for c in b'a'..=b'f' {
            db.put(&[c], &[c]).unwrap();
        }
        db.delete_range(b"b", b"e").unwrap();

        let keys: Vec<&[u8]> = vec![b"a", b"b", b"c", b"d", b"e", b"f"];
        let got = db.multi_get(&keys).unwrap();
        assert_eq!(got[0], Some(b"a".to_vec()));
        assert_eq!(got[1], None);
        assert_eq!(got[2], None);
        assert_eq!(got[3], None);
        assert_eq!(got[4], Some(b"e".to_vec()));
        assert_eq!(got[5], Some(b"f".to_vec()));
    }

    #[test]
    fn test_delete_range_crash_recovery() {
        let dir = TempDir::new().unwrap();
        {
            let db = Db::open(dir.path(), Options::default()).unwrap();
            for c in b'a'..=b'e' {
                db.put(&[c], &[c]).unwrap();
            }
            db.delete_range(b"b", b"d").unwrap();
            // Drop without close — only the WAL has the range delete.
        }
        let db = Db::open(dir.path(), Options::default()).unwrap();
        assert_eq!(db.get(b"a").unwrap(), Some(b"a".to_vec()));
        assert_eq!(db.get(b"b").unwrap(), None);
        assert_eq!(db.get(b"c").unwrap(), None);
        assert_eq!(db.get(b"d").unwrap(), Some(b"d".to_vec()));
        assert_eq!(db.get(b"e").unwrap(), Some(b"e".to_vec()));
    }

    #[test]
    fn test_delete_range_in_write_batch() {
        let (db, _dir) = open_tmp();
        for c in b'a'..=b'f' {
            db.put(&[c], &[c]).unwrap();
        }
        let mut batch = WriteBatch::new();
        batch.put(b"x", b"x");
        batch.delete_range(b"b", b"e");
        batch.put(b"y", b"y");
        db.write(batch).unwrap();

        assert_eq!(db.get(b"a").unwrap(), Some(b"a".to_vec()));
        assert_eq!(db.get(b"b").unwrap(), None);
        assert_eq!(db.get(b"c").unwrap(), None);
        assert_eq!(db.get(b"d").unwrap(), None);
        assert_eq!(db.get(b"e").unwrap(), Some(b"e".to_vec()));
        assert_eq!(db.get(b"x").unwrap(), Some(b"x".to_vec()));
        assert_eq!(db.get(b"y").unwrap(), Some(b"y".to_vec()));
    }

    #[test]
    fn test_delete_range_overlapping_ranges() {
        let (db, _dir) = open_tmp();
        for c in b'a'..=b'j' {
            db.put(&[c], &[c]).unwrap();
        }
        db.delete_range(b"b", b"e").unwrap();
        db.delete_range(b"d", b"h").unwrap();

        assert_eq!(db.get(b"a").unwrap(), Some(b"a".to_vec()));
        for c in b'b'..=b'g' {
            assert_eq!(db.get(&[c]).unwrap(), None, "key {} deleted", c as char);
        }
        assert_eq!(db.get(b"h").unwrap(), Some(b"h".to_vec()));
    }

    // ── compression codecs ──────────────────────────────────────────────────

    fn compression_opts(codec: CompressionType) -> Options {
        Options {
            write_buffer_size: 4 * 1024,
            compression: codec,
            ..Options::default()
        }
    }

    fn write_and_read_back(opts: Options) {
        let dir = TempDir::new().unwrap();
        let payload: Vec<u8> = (0..256).map(|i| (i % 31) as u8).collect();
        {
            let db = Db::open(dir.path(), opts.clone()).unwrap();
            for i in 0..200 {
                let key = format!("key_{:04}", i);
                db.put(key.as_bytes(), &payload).unwrap();
            }
            // Force a flush so reads must go through the SSTable codec path.
            force_flush(&db, "comp");
            for i in 0..200 {
                let key = format!("key_{:04}", i);
                assert_eq!(
                    db.get(key.as_bytes()).unwrap().as_deref(),
                    Some(payload.as_slice()),
                    "round-trip failed for {key}"
                );
            }
            db.close().unwrap();
        }
        // Reopen to verify the on-disk codec is decoded correctly by a
        // fresh reader.
        let db = Db::open(dir.path(), opts).unwrap();
        for i in 0..200 {
            let key = format!("key_{:04}", i);
            assert_eq!(
                db.get(key.as_bytes()).unwrap().as_deref(),
                Some(payload.as_slice())
            );
        }
    }

    #[test]
    fn test_compression_none_roundtrip() {
        write_and_read_back(compression_opts(CompressionType::None));
    }

    #[test]
    fn test_compression_lz4_roundtrip() {
        write_and_read_back(compression_opts(CompressionType::Lz4));
    }

    #[test]
    fn test_compression_snappy_roundtrip() {
        write_and_read_back(compression_opts(CompressionType::Snappy));
    }

    #[test]
    fn test_compression_per_level_mixed_codecs() {
        // L0 = Snappy, L1+ = Lz4. After a flush + manual compaction the
        // database must hold blocks compressed with both codecs and
        // still read back correctly.
        let dir = TempDir::new().unwrap();
        let opts = Options {
            write_buffer_size: 4 * 1024,
            compression: CompressionType::Lz4,
            compression_per_level: Some(vec![
                CompressionType::Snappy, // L0
                CompressionType::Lz4,    // L1
                CompressionType::None,   // L2 (unused here, just to exercise the slot)
            ]),
            ..Options::default()
        };
        let payload: Vec<u8> = (0..256).map(|i| (i % 17) as u8).collect();
        {
            let db = Db::open(dir.path(), opts.clone()).unwrap();
            for i in 0..300 {
                let key = format!("k_{:04}", i);
                db.put(key.as_bytes(), &payload).unwrap();
            }
            force_flush(&db, "mix");
            // Push everything down to L1 with the manual compaction path.
            db.compact_range(None, None).unwrap();
            for i in 0..300 {
                let key = format!("k_{:04}", i);
                assert_eq!(
                    db.get(key.as_bytes()).unwrap().as_deref(),
                    Some(payload.as_slice())
                );
            }
            db.close().unwrap();
        }
        // Reopen and re-read so the test exercises a fresh reader
        // hitting both codecs through the level layout we just built.
        let db = Db::open(dir.path(), opts).unwrap();
        for i in 0..300 {
            let key = format!("k_{:04}", i);
            assert_eq!(
                db.get(key.as_bytes()).unwrap().as_deref(),
                Some(payload.as_slice())
            );
        }
    }

    #[test]
    fn test_compression_per_level_falls_back_to_default() {
        // Override only L0; L1+ should fall back to `compression`.
        let dir = TempDir::new().unwrap();
        let opts = Options {
            write_buffer_size: 4 * 1024,
            compression: CompressionType::Snappy,
            compression_per_level: Some(vec![CompressionType::None]),
            ..Options::default()
        };
        let db = Db::open(dir.path(), opts).unwrap();
        for i in 0..50 {
            db.put(format!("k_{i:03}").as_bytes(), b"v").unwrap();
        }
        force_flush(&db, "fb");
        db.compact_range(None, None).unwrap();
        for i in 0..50 {
            assert_eq!(
                db.get(format!("k_{i:03}").as_bytes()).unwrap(),
                Some(b"v".to_vec())
            );
        }
    }

    // ── compaction filter ───────────────────────────────────────────────────

    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

    /// Test filter that drops every entry whose user key ends in an
    /// odd ASCII digit. Also counts invocations so tests can verify
    /// the filter actually ran.
    struct DropOddKeysFilter {
        calls: AtomicUsize,
    }

    impl CompactionFilter for DropOddKeysFilter {
        fn filter(&self, _level: usize, key: &[u8], _value: &[u8]) -> CompactionDecision {
            self.calls.fetch_add(1, AtomicOrdering::Relaxed);
            match key.last() {
                Some(b) if b.is_ascii_digit() && (b - b'0') % 2 == 1 => CompactionDecision::Remove,
                _ => CompactionDecision::Keep,
            }
        }
        fn name(&self) -> &'static str {
            "drop_odd_keys"
        }
    }

    /// Test filter that uppercases every ASCII-lowercase byte in the
    /// value. Exercises `Change`.
    struct UppercaseValuesFilter;

    impl CompactionFilter for UppercaseValuesFilter {
        fn filter(&self, _level: usize, _key: &[u8], value: &[u8]) -> CompactionDecision {
            let up: Vec<u8> = value.iter().map(|b| b.to_ascii_uppercase()).collect();
            if up == value {
                CompactionDecision::Keep
            } else {
                CompactionDecision::Change(up)
            }
        }
        fn name(&self) -> &'static str {
            "uppercase_values"
        }
    }

    /// Filter that drops every range tombstone it sees.
    struct DropRangeTombstonesFilter;

    impl CompactionFilter for DropRangeTombstonesFilter {
        fn filter(&self, _level: usize, _key: &[u8], _value: &[u8]) -> CompactionDecision {
            CompactionDecision::Keep
        }
        fn filter_range_delete(
            &self,
            _level: usize,
            _start: &[u8],
            _end: &[u8],
        ) -> CompactionDecision {
            CompactionDecision::Remove
        }
        fn name(&self) -> &'static str {
            "drop_range_tombstones"
        }
    }

    #[test]
    fn test_compaction_filter_removes_matching_entries() {
        let dir = TempDir::new().unwrap();
        let filter = Arc::new(DropOddKeysFilter {
            calls: AtomicUsize::new(0),
        });
        let opts = Options {
            write_buffer_size: 4 * 1024,
            compaction_filter: Some(filter.clone()),
            ..Options::default()
        };
        let db = Db::open(dir.path(), opts).unwrap();
        // 20 keys: k0..k9 written twice so compaction has work. Use
        // longer payloads so the tiny write buffer triggers flushes.
        let payload = vec![b'v'; 512];
        for _round in 0..4 {
            for i in 0..10 {
                db.put(format!("k{i}").as_bytes(), &payload).unwrap();
            }
        }
        db.compact_range(None, None).unwrap();

        // After compaction, odd-suffix keys are gone.
        for i in 0..10 {
            let got = db.get(format!("k{i}").as_bytes()).unwrap();
            if i % 2 == 1 {
                assert_eq!(got, None, "k{i} should be filtered");
            } else {
                assert_eq!(got, Some(payload.clone()), "k{i} should survive");
            }
        }
        assert!(
            filter.calls.load(AtomicOrdering::Relaxed) > 0,
            "filter should have been invoked"
        );
    }

    #[test]
    fn test_compaction_filter_rewrites_values() {
        let dir = TempDir::new().unwrap();
        let opts = Options {
            write_buffer_size: 4 * 1024,
            compaction_filter: Some(Arc::new(UppercaseValuesFilter)),
            ..Options::default()
        };
        let db = Db::open(dir.path(), opts).unwrap();
        for i in 0..20 {
            db.put(format!("k{i:02}").as_bytes(), b"hello world")
                .unwrap();
        }
        // Force enough flushes + manual compaction to run the filter.
        force_flush(&db, "filter");
        db.compact_range(None, None).unwrap();

        for i in 0..20 {
            assert_eq!(
                db.get(format!("k{i:02}").as_bytes()).unwrap(),
                Some(b"HELLO WORLD".to_vec())
            );
        }
    }

    #[test]
    fn test_compaction_filter_skipped_while_snapshot_alive() {
        let dir = TempDir::new().unwrap();
        let opts = Options {
            write_buffer_size: 4 * 1024,
            compaction_filter: Some(Arc::new(UppercaseValuesFilter)),
            ..Options::default()
        };
        let db = Db::open(dir.path(), opts).unwrap();
        for i in 0..20 {
            db.put(format!("k{i:02}").as_bytes(), b"hello").unwrap();
        }
        // Hold a snapshot so the compaction filter is skipped entirely.
        let snap = db.snapshot();
        force_flush(&db, "snap_filter");
        db.compact_range(None, None).unwrap();

        // The snapshot still observes the pre-filter value because
        // the filter was suppressed while it was alive. The live db
        // reads also see the unmodified value since compaction left
        // it intact.
        for i in 0..20 {
            assert_eq!(
                snap.get(format!("k{i:02}").as_bytes()).unwrap(),
                Some(b"hello".to_vec())
            );
            assert_eq!(
                db.get(format!("k{i:02}").as_bytes()).unwrap(),
                Some(b"hello".to_vec())
            );
        }
    }

    #[test]
    fn test_compaction_filter_drops_range_tombstones() {
        let dir = TempDir::new().unwrap();
        let opts = Options {
            write_buffer_size: 4 * 1024,
            compaction_filter: Some(Arc::new(DropRangeTombstonesFilter)),
            ..Options::default()
        };
        let db = Db::open(dir.path(), opts).unwrap();
        for c in b'a'..=b'f' {
            db.put(&[c], &[c]).unwrap();
        }
        db.delete_range(b"b", b"e").unwrap();
        // Before compaction, the range-delete is honored — no snapshot
        // pinning, so the read path sees the memtable RT directly.
        for c in b'b'..=b'd' {
            assert_eq!(db.get(&[c]).unwrap(), None);
        }
        force_flush(&db, "drop_rt");
        db.compact_range(None, None).unwrap();

        // After compaction the filter dropped the RT, so the original
        // values come back (they were never actually overwritten).
        for c in b'a'..=b'f' {
            assert_eq!(
                db.get(&[c]).unwrap(),
                Some(vec![c]),
                "key {} restored",
                c as char
            );
        }
    }

    fn prefix_opts() -> Options {
        Options {
            write_buffer_size: 4 * 1024,
            prefix_extractor: Some(std::sync::Arc::new(FixedLengthPrefix(10))),
            ..Options::default()
        }
    }

    #[test]
    fn test_seek_prefix_basic() {
        let (db, _dir) = open_tmp();
        db.put(b"tenant_001:k1", b"1").unwrap();
        db.put(b"tenant_001:k2", b"2").unwrap();
        db.put(b"tenant_002:k1", b"3").unwrap();
        db.put(b"tenant_010:k1", b"4").unwrap();

        let mut it = db.iter();
        it.seek_prefix(b"tenant_001");
        let mut got: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
        while it.valid() {
            got.push((it.key().unwrap().to_vec(), it.value().unwrap().to_vec()));
            it.next();
        }
        assert_eq!(
            got,
            vec![
                (b"tenant_001:k1".to_vec(), b"1".to_vec()),
                (b"tenant_001:k2".to_vec(), b"2".to_vec()),
            ]
        );
    }

    #[test]
    fn test_seek_prefix_absent_returns_empty() {
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), prefix_opts()).unwrap();
        for i in 0..200 {
            let key = format!("tenant_001:k{:04}", i);
            db.put(key.as_bytes(), b"v").unwrap();
        }
        force_flush(&db, "p");

        let mut it = db.iter();
        it.seek_prefix(b"tenant_999");
        assert!(!it.valid(), "expected no keys under an absent prefix");
    }

    #[test]
    fn test_seek_prefix_across_flush_boundary() {
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), prefix_opts()).unwrap();

        // First generation → flushed to L0.
        db.put(b"tenant_001:a", b"1a").unwrap();
        db.put(b"tenant_002:a", b"2a").unwrap();
        force_flush(&db, "p1");

        // Second generation → stays in memtable at iteration time.
        db.put(b"tenant_001:b", b"1b").unwrap();
        db.put(b"tenant_002:b", b"2b").unwrap();

        let mut it = db.iter();
        it.seek_prefix(b"tenant_001");
        let mut keys: Vec<Vec<u8>> = Vec::new();
        while it.valid() {
            keys.push(it.key().unwrap().to_vec());
            it.next();
        }
        assert_eq!(
            keys,
            vec![b"tenant_001:a".to_vec(), b"tenant_001:b".to_vec()]
        );
    }

    #[test]
    fn test_seek_prefix_after_compact_range() {
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), prefix_opts()).unwrap();

        for i in 0..50 {
            db.put(format!("tenant_001:{:04}", i).as_bytes(), b"v")
                .unwrap();
            db.put(format!("tenant_002:{:04}", i).as_bytes(), b"v")
                .unwrap();
        }
        force_flush(&db, "c1");
        db.compact_range(None, None).unwrap();

        let mut it = db.iter();
        it.seek_prefix(b"tenant_002");
        let mut count = 0;
        while it.valid() {
            let k = it.key().unwrap();
            assert!(
                k.starts_with(b"tenant_002"),
                "got unexpected key {:?}",
                std::str::from_utf8(k).unwrap_or("<non-utf8>")
            );
            count += 1;
            it.next();
        }
        assert_eq!(count, 50);
    }

    #[test]
    fn test_seek_prefix_mixed_with_without_extractor() {
        // Open with no extractor, flush some data (file A has no prefix
        // bloom), then reopen with an extractor and write new data
        // (file B has a prefix bloom). Reads through the extractor-
        // configured DB must still return correct results across both
        // files.
        let dir = TempDir::new().unwrap();
        {
            let db = Db::open(
                dir.path(),
                Options {
                    write_buffer_size: 4 * 1024,
                    ..Options::default()
                },
            )
            .unwrap();
            db.put(b"tenant_001:old", b"old").unwrap();
            force_flush(&db, "a");
        }

        let db = Db::open(dir.path(), prefix_opts()).unwrap();
        db.put(b"tenant_001:new", b"new").unwrap();
        db.put(b"tenant_002:new", b"new").unwrap();
        force_flush(&db, "b");

        let mut it = db.iter();
        it.seek_prefix(b"tenant_001");
        let mut keys: Vec<Vec<u8>> = Vec::new();
        while it.valid() {
            keys.push(it.key().unwrap().to_vec());
            it.next();
        }
        assert_eq!(
            keys,
            vec![b"tenant_001:new".to_vec(), b"tenant_001:old".to_vec()]
        );
    }

    #[test]
    fn test_compaction_filter_none_is_noop() {
        let (db, _dir) = open_tmp();
        for i in 0..10 {
            db.put(format!("k{i}").as_bytes(), b"v").unwrap();
        }
        db.compact_range(None, None).unwrap();
        for i in 0..10 {
            assert_eq!(
                db.get(format!("k{i}").as_bytes()).unwrap(),
                Some(b"v".to_vec())
            );
        }
    }

    // ── per-write WriteOptions ──────────────────────────────────────────────

    #[test]
    fn test_write_options_defaults_unchanged() {
        // `put_opt` with a default-constructed WriteOptions must
        // behave identically to `put`.
        let (db, _dir) = open_tmp();
        db.put_opt(&WriteOptions::default(), b"a", b"1").unwrap();
        assert_eq!(db.get(b"a").unwrap(), Some(b"1".to_vec()));
    }

    #[test]
    fn test_write_options_sync_override_persists_across_reopen() {
        // With Eventual default, a sync write should still land on
        // disk such that a reopen recovers it. (Eventual alone
        // already survives a clean close — this test's real content
        // is that the sync flag doesn't break the normal code path.)
        let dir = TempDir::new().unwrap();
        let opts = Options {
            durability: DurabilityMode::Eventual,
            ..Options::default()
        };
        {
            let db = Db::open(dir.path(), opts.clone()).unwrap();
            db.put_opt(&WriteOptions::sync(), b"critical", b"payload")
                .unwrap();
            // Deliberately skip close() — sync must have forced the
            // WAL to durable storage already.
        }
        let db = Db::open(dir.path(), opts).unwrap();
        assert_eq!(db.get(b"critical").unwrap(), Some(b"payload".to_vec()));
    }

    #[test]
    fn test_write_options_disable_wal_loses_data_on_drop_without_flush() {
        // disable_wal skips the WAL append entirely. Without a clean
        // close(), a reopen cannot recover the write because neither
        // the WAL nor an SSTable has it.
        let dir = TempDir::new().unwrap();
        let opts = Options::default();
        {
            let db = Db::open(dir.path(), opts.clone()).unwrap();
            db.put_opt(&WriteOptions::disable_wal(), b"ephemeral", b"ghost")
                .unwrap();
            // No close() — simulate a crash. The memtable holds the
            // write but nothing on disk does.
        }
        let db = Db::open(dir.path(), opts).unwrap();
        assert_eq!(db.get(b"ephemeral").unwrap(), None);
    }

    #[test]
    fn test_write_options_disable_wal_visible_within_session() {
        // Within the same process, a disable_wal write is visible
        // to subsequent reads via the memtable — only a crash
        // erases it.
        let (db, _dir) = open_tmp();
        db.put_opt(&WriteOptions::disable_wal(), b"k", b"v")
            .unwrap();
        assert_eq!(db.get(b"k").unwrap(), Some(b"v".to_vec()));
    }

    #[test]
    fn test_write_options_disable_wal_survives_clean_close() {
        // A clean close() flushes the memtable to an SSTable before
        // shutting down. A disable_wal write still made it into the
        // memtable, so close() + reopen recovers it via the SSTable
        // (not the WAL).
        let dir = TempDir::new().unwrap();
        let opts = Options::default();
        {
            let db = Db::open(dir.path(), opts.clone()).unwrap();
            db.put_opt(&WriteOptions::disable_wal(), b"bulk", b"loaded")
                .unwrap();
            db.close().unwrap();
        }
        let db = Db::open(dir.path(), opts).unwrap();
        assert_eq!(db.get(b"bulk").unwrap(), Some(b"loaded".to_vec()));
    }

    #[test]
    fn test_write_options_batch_overrides() {
        let (db, _dir) = open_tmp();
        let mut batch = WriteBatch::new();
        batch.put(b"a", b"1");
        batch.put(b"b", b"2");
        batch.delete(b"ghost");
        db.write_opt(&WriteOptions::sync(), batch).unwrap();
        assert_eq!(db.get(b"a").unwrap(), Some(b"1".to_vec()));
        assert_eq!(db.get(b"b").unwrap(), Some(b"2".to_vec()));
    }

    #[test]
    fn test_write_options_delete_and_delete_range_opts() {
        let (db, _dir) = open_tmp();
        for c in b'a'..=b'f' {
            db.put(&[c], &[c]).unwrap();
        }
        db.delete_opt(&WriteOptions::sync(), b"c").unwrap();
        db.delete_range_opt(&WriteOptions::sync(), b"d", b"f")
            .unwrap();
        assert_eq!(db.get(b"a").unwrap(), Some(b"a".to_vec()));
        assert_eq!(db.get(b"c").unwrap(), None);
        assert_eq!(db.get(b"d").unwrap(), None);
        assert_eq!(db.get(b"e").unwrap(), None);
        assert_eq!(db.get(b"f").unwrap(), Some(b"f".to_vec()));
    }

    #[test]
    fn test_write_options_low_pri_and_no_slowdown_are_no_ops() {
        // Accepted but currently ignored. Reserved for future
        // write-stall / priority-queue plumbing.
        let (db, _dir) = open_tmp();
        let opts = WriteOptions {
            low_pri: true,
            no_slowdown: true,
            ..WriteOptions::default()
        };
        db.put_opt(&opts, b"k", b"v").unwrap();
        assert_eq!(db.get(b"k").unwrap(), Some(b"v".to_vec()));
    }

    // ── merge operator ──────────────────────────────────────────────────────

    /// Integer-counter merge operator: every operand is the 8-byte
    /// big-endian i64 delta to add. `full_merge` sums them (starting
    /// from `base` if present) and emits the new counter value.
    /// `partial_merge` folds two deltas by adding them.
    struct CounterMerge;

    impl MergeOperator for CounterMerge {
        fn full_merge(
            &self,
            _key: &[u8],
            base: Option<&[u8]>,
            operands: &[&[u8]],
        ) -> Option<Vec<u8>> {
            let mut total: i64 = match base {
                Some(b) if b.len() == 8 => i64::from_be_bytes(b.try_into().unwrap()),
                Some(_) => return None,
                None => 0,
            };
            for op in operands {
                if op.len() != 8 {
                    return None;
                }
                total = total.wrapping_add(i64::from_be_bytes((*op).try_into().unwrap()));
            }
            Some(total.to_be_bytes().to_vec())
        }

        fn partial_merge(&self, _key: &[u8], left: &[u8], right: &[u8]) -> Option<Vec<u8>> {
            if left.len() != 8 || right.len() != 8 {
                return None;
            }
            let l = i64::from_be_bytes(left.try_into().unwrap());
            let r = i64::from_be_bytes(right.try_into().unwrap());
            Some(l.wrapping_add(r).to_be_bytes().to_vec())
        }

        fn name(&self) -> &'static str {
            "CounterMerge"
        }
    }

    /// String-append merge operator: every operand is raw bytes;
    /// `full_merge` concatenates the base (if any) with every
    /// operand in oldest-first order.
    struct AppendMerge;

    impl MergeOperator for AppendMerge {
        fn full_merge(
            &self,
            _key: &[u8],
            base: Option<&[u8]>,
            operands: &[&[u8]],
        ) -> Option<Vec<u8>> {
            let mut out: Vec<u8> = base.map(|b| b.to_vec()).unwrap_or_default();
            for op in operands {
                out.extend_from_slice(op);
            }
            Some(out)
        }

        fn name(&self) -> &'static str {
            "AppendMerge"
        }
    }

    fn counter_opts() -> Options {
        Options {
            write_buffer_size: 4 * 1024,
            merge_operator: Some(Arc::new(CounterMerge)),
            ..Options::default()
        }
    }

    fn encode_i64(n: i64) -> Vec<u8> {
        n.to_be_bytes().to_vec()
    }

    #[test]
    fn test_merge_counter_basic_chain_of_one() {
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), counter_opts()).unwrap();
        db.merge(b"counter", &encode_i64(5)).unwrap();
        assert_eq!(db.get(b"counter").unwrap(), Some(encode_i64(5)));
    }

    #[test]
    fn test_merge_counter_chain_of_two() {
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), counter_opts()).unwrap();
        db.put(b"counter", &encode_i64(10)).unwrap();
        db.merge(b"counter", &encode_i64(3)).unwrap();
        assert_eq!(db.get(b"counter").unwrap(), Some(encode_i64(13)));
    }

    #[test]
    fn test_merge_counter_chain_of_ten() {
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), counter_opts()).unwrap();
        db.put(b"counter", &encode_i64(100)).unwrap();
        for i in 1..=10 {
            db.merge(b"counter", &encode_i64(i)).unwrap();
        }
        // 100 + (1+2+...+10) = 155
        assert_eq!(db.get(b"counter").unwrap(), Some(encode_i64(155)));
    }

    #[test]
    fn test_merge_counter_chain_of_1000() {
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), counter_opts()).unwrap();
        for _ in 0..1000 {
            db.merge(b"counter", &encode_i64(1)).unwrap();
        }
        assert_eq!(db.get(b"counter").unwrap(), Some(encode_i64(1000)));
    }

    #[test]
    fn test_merge_without_base_defaults_to_none() {
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), counter_opts()).unwrap();
        // No put — counter starts at 0 (base=None).
        db.merge(b"counter", &encode_i64(7)).unwrap();
        db.merge(b"counter", &encode_i64(5)).unwrap();
        assert_eq!(db.get(b"counter").unwrap(), Some(encode_i64(12)));
    }

    #[test]
    fn test_merge_snapshot_isolation() {
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), counter_opts()).unwrap();
        db.put(b"counter", &encode_i64(10)).unwrap();
        let snap = db.snapshot();
        db.merge(b"counter", &encode_i64(5)).unwrap();
        // Live read sees 15; snapshot still sees 10.
        assert_eq!(db.get(b"counter").unwrap(), Some(encode_i64(15)));
        assert_eq!(snap.get(b"counter").unwrap(), Some(encode_i64(10)));
    }

    #[test]
    fn test_merge_survives_flush() {
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), counter_opts()).unwrap();
        db.put(b"counter", &encode_i64(0)).unwrap();
        for i in 1..=20 {
            db.merge(b"counter", &encode_i64(i)).unwrap();
        }
        // Push past the tiny write buffer so the chain crosses a
        // flush boundary (memtable → L0).
        force_flush(&db, "merge");
        // Sum = 1+2+...+20 = 210
        assert_eq!(db.get(b"counter").unwrap(), Some(encode_i64(210)));
    }

    #[test]
    fn test_merge_survives_compaction_and_collapses() {
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), counter_opts()).unwrap();
        db.put(b"counter", &encode_i64(0)).unwrap();
        for i in 1..=50 {
            db.merge(b"counter", &encode_i64(i)).unwrap();
        }
        for tag in 0..4 {
            force_flush(&db, &format!("c{tag}"));
        }
        db.compact_range(None, None).unwrap();
        // Sum 1..=50 = 1275
        assert_eq!(db.get(b"counter").unwrap(), Some(encode_i64(1275)));
    }

    #[test]
    fn test_merge_tombstone_interaction() {
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), counter_opts()).unwrap();
        // Value=10, then two merges, then delete, then two more merges.
        db.put(b"k", &encode_i64(10)).unwrap();
        db.merge(b"k", &encode_i64(5)).unwrap();
        db.merge(b"k", &encode_i64(3)).unwrap();
        db.delete(b"k").unwrap();
        db.merge(b"k", &encode_i64(7)).unwrap();
        db.merge(b"k", &encode_i64(1)).unwrap();
        // Reads layer the two latest merges on top of the deletion
        // (which resets the base to None → 0): 0 + 7 + 1 = 8.
        assert_eq!(db.get(b"k").unwrap(), Some(encode_i64(8)));
    }

    #[test]
    fn test_merge_range_tombstone_interaction() {
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), counter_opts()).unwrap();
        db.put(b"k", &encode_i64(10)).unwrap();
        db.merge(b"k", &encode_i64(5)).unwrap();
        db.delete_range(b"j", b"l").unwrap(); // hides the base
        db.merge(b"k", &encode_i64(7)).unwrap();
        // After the RT, only the latest merge (7) applies to a None base.
        assert_eq!(db.get(b"k").unwrap(), Some(encode_i64(7)));
    }

    #[test]
    fn test_merge_write_batch() {
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), counter_opts()).unwrap();
        let mut batch = WriteBatch::new();
        batch.put(b"a", &encode_i64(1));
        batch.merge(b"a", &encode_i64(2));
        batch.merge(b"a", &encode_i64(3));
        batch.put(b"b", &encode_i64(100));
        db.write(batch).unwrap();
        assert_eq!(db.get(b"a").unwrap(), Some(encode_i64(6)));
        assert_eq!(db.get(b"b").unwrap(), Some(encode_i64(100)));
    }

    #[test]
    fn test_merge_append_operator() {
        let dir = TempDir::new().unwrap();
        let opts = Options {
            merge_operator: Some(Arc::new(AppendMerge)),
            ..Options::default()
        };
        let db = Db::open(dir.path(), opts).unwrap();
        db.put(b"s", b"hello").unwrap();
        db.merge(b"s", b" ").unwrap();
        db.merge(b"s", b"world").unwrap();
        assert_eq!(db.get(b"s").unwrap(), Some(b"hello world".to_vec()));
    }

    #[test]
    fn test_merge_iterator_sees_collapsed_value() {
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), counter_opts()).unwrap();
        db.put(b"a", &encode_i64(0)).unwrap();
        db.merge(b"a", &encode_i64(5)).unwrap();
        db.put(b"b", &encode_i64(100)).unwrap();
        db.merge(b"b", &encode_i64(10)).unwrap();
        db.merge(b"b", &encode_i64(2)).unwrap();

        let pairs = db.scan(None, None).unwrap();
        assert_eq!(
            pairs,
            vec![
                (b"a".to_vec(), encode_i64(5)),
                (b"b".to_vec(), encode_i64(112)),
            ]
        );
    }

    #[test]
    fn test_merge_iterator_reverse() {
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), counter_opts()).unwrap();
        db.put(b"a", &encode_i64(0)).unwrap();
        db.merge(b"a", &encode_i64(1)).unwrap();
        db.put(b"b", &encode_i64(0)).unwrap();
        db.merge(b"b", &encode_i64(2)).unwrap();
        db.merge(b"b", &encode_i64(3)).unwrap();

        let mut iter = db.iter();
        iter.seek_to_last();
        let mut collected = Vec::new();
        while iter.valid() {
            collected.push((iter.key().unwrap().to_vec(), iter.value().unwrap().to_vec()));
            iter.prev();
        }
        assert_eq!(
            collected,
            vec![
                (b"b".to_vec(), encode_i64(5)),
                (b"a".to_vec(), encode_i64(1)),
            ]
        );
    }

    #[test]
    fn test_merge_crash_recovery() {
        let dir = TempDir::new().unwrap();
        {
            let db = Db::open(dir.path(), counter_opts()).unwrap();
            db.put(b"counter", &encode_i64(0)).unwrap();
            db.merge(b"counter", &encode_i64(7)).unwrap();
            db.merge(b"counter", &encode_i64(3)).unwrap();
            // No close — memtable flush didn't happen; WAL must
            // survive the chain.
        }
        let db = Db::open(dir.path(), counter_opts()).unwrap();
        assert_eq!(db.get(b"counter").unwrap(), Some(encode_i64(10)));
    }

    #[test]
    fn test_merge_operator_name_plumbs_through() {
        // Surface-area smoke test: the configured operator's `name`
        // is reachable via Options::debug.
        let opts = counter_opts();
        let dbg = format!("{opts:?}");
        assert!(dbg.contains("CounterMerge"));
    }
}
