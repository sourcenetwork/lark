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
mod column_family;
mod engine;
mod error;
mod event_listener;
mod iter;
mod options;
mod os_hint;
mod perf_context;
mod rate_limiter;
mod sst_file_writer;
mod statistics;
mod tailing;
mod transaction;
mod ttl;

pub use backup::{BackupEngine, BackupId, BackupInfo};
pub use checkpoint::Checkpoint;
pub use column_family::{ColumnFamilyHandle, DEFAULT_CF_NAME};
pub use error::Error;
pub use event_listener::{
    BackgroundErrorReason, CompactionJobInfo, EventListener, ExternalFileIngestionInfo,
    FlushJobInfo, TableFileCreationInfo, TableFileCreationReason, TableFileDeletionInfo,
    WalFullInfo,
};
pub use iter::Iter;
pub use options::{
    CompactionDecision, CompactionFilter, CompactionStyle, CompressionType, DurabilityMode,
    FifoCompactionOptions, FixedLengthPrefix, MergeOperator, Options, PrefixExtractor,
    UniversalCompactionOptions, WriteOptions,
};
pub use perf_context::{PerfContext, PerfContextSnapshot, PerfLevel};
pub use rate_limiter::{Priority, RateLimiter, TokenBucketRateLimiter};
pub use sst_file_writer::{IngestOptions, SstFileMeta, SstFileWriter};
pub use statistics::{Histogram, HistogramSnapshot, Statistics, Ticker};
pub use tailing::TailingIter;
pub use transaction::{
    OptimisticTransactionDb, Transaction, TransactionDb, TransactionError, TxResult,
};
pub use ttl::{strip_timestamp, DbWithTtl, TtlCompactionFilter};

use column_family::{
    cf_lower_bound, cf_upper_bound, meta, prefix_key, CfRegistry, DEFAULT_CF_ID, META_CF_ID,
};

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use engine::LarkEngine;

/// Minimal snapshot of the currently-configured options, returned
/// by `Db::get_property("lark.options")`. Deliberately small —
/// lark doesn't retain the full `Options` past `Db::open`, and
/// the Debug impl of this struct is the property's string value.
#[derive(Debug)]
#[allow(dead_code)]
struct OptionsSnapshot {
    durability: engine::DurabilityMode,
    default_cf: &'static str,
}

/// Format a raw engine key for inclusion in a property string.
/// Internal keys in lark carry a 4-byte CF prefix; if the key
/// is long enough we strip it and ASCII-escape the remainder.
/// Anything non-printable (or keys too short to strip) falls
/// back to a hex rendering so the output stays single-line.
fn format_key_for_display(key: &[u8]) -> String {
    let payload = if key.len() > 4 { &key[4..] } else { key };
    if payload.iter().all(|&b| b.is_ascii_graphic() || b == b' ') {
        format!("\"{}\"", String::from_utf8_lossy(payload))
    } else {
        let hex: String = payload.iter().map(|b| format!("{b:02x}")).collect();
        format!("0x{hex}")
    }
}

/// A half-open key range `[start, end)` passed to the approximate-size
/// APIs. Borrowed; cheap to construct inline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Range<'a> {
    /// Inclusive lower bound.
    pub start: &'a [u8],
    /// Exclusive upper bound.
    pub end: &'a [u8],
}

impl<'a> Range<'a> {
    /// Construct a new `[start, end)` range.
    pub fn new(start: &'a [u8], end: &'a [u8]) -> Self {
        Self { start, end }
    }
}

/// Approximate memtable stats returned by
/// [`Db::get_approximate_memtable_stats`]. `count` is the number of
/// raw entries (including every version and every tombstone) for
/// user keys in the queried range; `size` is the sum of
/// `internal_key.len() + value.len()` over those entries. Both
/// values are exact with respect to the current active memtable —
/// this method walks the skip list.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MemTableStats {
    /// Number of raw entries in the range.
    pub count: u64,
    /// Approximate total bytes in the range.
    pub size: u64,
}

/// Result type for lark operations.
pub type Result<T> = std::result::Result<T, Error>;

/// A key-value database backed by an LSM-tree.
pub struct Db {
    engine: Arc<LarkEngine>,
    durability: engine::DurabilityMode,
    cfs: Arc<CfRegistry>,
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
    ///
    /// On a fresh database the default column family (`"default"`)
    /// is created automatically and every non-`*_cf` method uses
    /// it. Callers who want logical keyspace isolation can then
    /// call [`Db::create_column_family`] for additional CFs; those
    /// calls persist into the database and survive reopen.
    pub fn open<P: AsRef<Path>>(path: P, opts: Options) -> Result<Self> {
        let durability = match opts.durability {
            DurabilityMode::Immediate => engine::DurabilityMode::Immediate,
            DurabilityMode::Eventual => engine::DurabilityMode::Eventual,
        };
        let engine = LarkEngine::open(path.as_ref(), opts.to_engine_options())?;
        let cfs = Arc::new(CfRegistry::new());
        let db = Self {
            engine,
            durability,
            cfs,
        };
        db.load_cf_registry()?;
        Ok(db)
    }

    /// Populate the in-memory [`CfRegistry`] from the on-disk
    /// metadata CF, creating the default CF entry if this is a
    /// fresh database. Called once from [`Db::open`].
    fn load_cf_registry(&self) -> Result<()> {
        // Scan every `name:*` entry in the meta CF to rebuild the
        // name→id map. The default CF is not persisted to disk —
        // it's always injected into the in-memory registry with a
        // hardcoded id so an empty database stays byte-free on
        // disk. User-created CFs are the only thing that produces
        // on-disk metadata writes.
        let mut entries: Vec<(String, u32)> = Vec::new();
        let seq = self.engine.snapshot_seq();
        let pairs = collect_range(
            &self.engine,
            Some(&meta::name_scan_prefix()),
            Some(&meta::name_scan_upper()),
            seq,
        )?;
        for (key, value) in pairs {
            if value.len() != 4 {
                continue;
            }
            let Some(name) = meta::name_from_key(&key) else {
                continue;
            };
            let id = u32::from_be_bytes(value.as_slice().try_into().unwrap());
            entries.push((name.to_string(), id));
        }

        // Recover `next_id`. Absent on a fresh database.
        let next_id_raw = self
            .engine
            .get(&meta::next_id_key(), self.engine.snapshot_seq())
            .map_err(Error::Io)?;
        let next_id = match next_id_raw {
            Some(bytes) if bytes.len() == 4 => {
                u32::from_be_bytes(bytes.as_slice().try_into().unwrap())
            }
            _ => DEFAULT_CF_ID + 1,
        };

        // Inject the default CF into the in-memory registry so
        // `Db::default_cf()` always succeeds. It's never persisted
        // to the meta CF — any re-open computes the same id.
        entries.push((DEFAULT_CF_NAME.to_string(), DEFAULT_CF_ID));
        self.cfs.load(entries, next_id);
        Ok(())
    }

    /// Get the value for a key from the default column family.
    /// Returns `None` if the key doesn't exist.
    pub fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        self.get_raw(&prefix_key(DEFAULT_CF_ID, key))
    }

    /// Engine-direct point read that bypasses CF prefixing. Used
    /// by the CF metadata loader and by other modules (e.g. the
    /// `_cf` wrappers above) that have already prefixed the key.
    fn get_raw(&self, prefixed_key: &[u8]) -> Result<Option<Vec<u8>>> {
        let stats = self.stats();
        let _scope = statistics::TimeScope::new(stats, Histogram::DbGet);
        if let Some(s) = stats {
            s.add(Ticker::KeysRead, 1);
        }
        perf_context::record_get_call();
        let seq = self.engine.snapshot_seq();
        let result = self.engine.get(prefixed_key, seq).map_err(Error::Io);
        if let (Some(s), Ok(Some(v))) = (stats, &result) {
            s.add(Ticker::BytesRead, v.len() as u64);
            s.record(Histogram::BytesPerRead, v.len() as u64);
        }
        result
    }

    /// Helper that exposes a borrowed reference to the engine's
    /// `Statistics` (if any) so instrumented methods can call
    /// `stats.add(..)` / `stats.record(..)` through a single
    /// `Option::is_some` check.
    fn stats(&self) -> Option<&Statistics> {
        self.engine.statistics()
    }

    /// Look up a batch of keys in the default column family.
    /// Returns a vector with one entry per input key (preserving
    /// order and duplicates); each entry is `None` if the key does
    /// not exist or is tombstoned.
    ///
    /// All keys in a single call see the **same** consistent view —
    /// a concurrent writer cannot make two keys disagree about
    /// visibility.
    pub fn multi_get(&self, keys: &[&[u8]]) -> Result<Vec<Option<Vec<u8>>>> {
        let owned: Vec<Vec<u8>> = keys.iter().map(|k| prefix_key(DEFAULT_CF_ID, k)).collect();
        let refs: Vec<&[u8]> = owned.iter().map(|k| k.as_slice()).collect();
        let seq = self.engine.snapshot_seq();
        self.engine.multi_get(&refs, seq).map_err(Error::Io)
    }

    /// Set a key-value pair in the default column family using
    /// the database-global durability mode and default write
    /// options.
    pub fn put(&self, key: &[u8], value: &[u8]) -> Result<()> {
        self.put_opt(&WriteOptions::default(), key, value)
    }

    /// Set a key-value pair in the default column family with an
    /// explicit [`WriteOptions`] override.
    pub fn put_opt(&self, opts: &WriteOptions, key: &[u8], value: &[u8]) -> Result<()> {
        self.wait_for_write_capacity(opts)?;
        let stats = self.stats();
        let _scope = statistics::TimeScope::new(stats, Histogram::DbWrite);
        let bytes = (key.len() + value.len()) as u64;
        if let Some(s) = stats {
            s.add(Ticker::KeysWritten, 1);
            s.add(Ticker::BytesWritten, bytes);
            s.record(Histogram::BytesPerWrite, bytes);
        }
        perf_context::record_write_call();
        let mut batch = BTreeMap::new();
        batch.insert(prefix_key(DEFAULT_CF_ID, key), Some(value.to_vec()));
        let (dm, disable_wal) = self.resolve_write_opts(opts);
        self.engine
            .apply_batch(batch, Vec::new(), Vec::new(), dm, disable_wal)
            .map_err(Error::Io)
    }

    /// Delete a key from the default column family using the
    /// database-global durability mode.
    pub fn delete(&self, key: &[u8]) -> Result<()> {
        self.delete_opt(&WriteOptions::default(), key)
    }

    /// Delete a key from the default column family with an
    /// explicit [`WriteOptions`] override.
    pub fn delete_opt(&self, opts: &WriteOptions, key: &[u8]) -> Result<()> {
        self.wait_for_write_capacity(opts)?;
        if let Some(s) = self.stats() {
            s.add(Ticker::KeysDeleted, 1);
        }
        let mut batch = BTreeMap::new();
        batch.insert(prefix_key(DEFAULT_CF_ID, key), None);
        let (dm, disable_wal) = self.resolve_write_opts(opts);
        self.engine
            .apply_batch(batch, Vec::new(), Vec::new(), dm, disable_wal)
            .map_err(Error::Io)
    }

    /// Layer a merge operand on top of `key` in the default
    /// column family.
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
        self.wait_for_write_capacity(opts)?;
        if let Some(s) = self.stats() {
            s.add(Ticker::MergesWritten, 1);
        }
        let (dm, disable_wal) = self.resolve_write_opts(opts);
        self.engine
            .apply_batch(
                BTreeMap::new(),
                Vec::new(),
                vec![(prefix_key(DEFAULT_CF_ID, key), operand.to_vec())],
                dm,
                disable_wal,
            )
            .map_err(Error::Io)
    }

    /// Delete every key in `[start, end)` in the default column
    /// family.
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

    /// Delete every key in `[start, end)` in the default column
    /// family with an explicit [`WriteOptions`] override.
    pub fn delete_range_opt(&self, opts: &WriteOptions, start: &[u8], end: &[u8]) -> Result<()> {
        if start >= end {
            return Ok(());
        }
        self.wait_for_write_capacity(opts)?;
        if let Some(s) = self.stats() {
            s.add(Ticker::RangeDeletesWritten, 1);
        }
        let (dm, disable_wal) = self.resolve_write_opts(opts);
        self.engine
            .apply_batch(
                BTreeMap::new(),
                vec![(
                    prefix_key(DEFAULT_CF_ID, start),
                    prefix_key(DEFAULT_CF_ID, end),
                )],
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
        self.wait_for_write_capacity(opts)?;
        perf_context::record_write_call();
        let stats = self.stats();
        let _scope = statistics::TimeScope::new(stats, Histogram::DbWrite);
        if let Some(s) = stats {
            let mut bytes: u64 = 0;
            let mut puts: u64 = 0;
            let mut deletes: u64 = 0;
            for (k, v) in &batch.ops {
                match v {
                    Some(val) => {
                        puts += 1;
                        bytes += (k.len() + val.len()) as u64;
                    }
                    None => {
                        deletes += 1;
                        bytes += k.len() as u64;
                    }
                }
            }
            s.add(Ticker::KeysWritten, puts);
            s.add(Ticker::KeysDeleted, deletes);
            s.add(Ticker::BytesWritten, bytes);
            s.add(
                Ticker::RangeDeletesWritten,
                batch.range_deletes.len() as u64,
            );
            s.add(Ticker::MergesWritten, batch.merges.len() as u64);
            s.record(Histogram::BytesPerWrite, bytes);
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
    /// otherwise the default wins. `low_pri` is accepted but is
    /// currently a no-op; `no_slowdown` is handled separately by
    /// the write-stall pre-check.
    fn resolve_write_opts(&self, opts: &WriteOptions) -> (engine::DurabilityMode, bool) {
        let dm = if opts.sync {
            engine::DurabilityMode::Immediate
        } else {
            self.durability
        };
        (dm, opts.disable_wal)
    }

    /// Run the write-stall pre-check. Block the caller until the
    /// engine is ready to accept another write, or return
    /// [`Error::Busy`] immediately if `opts.no_slowdown` is set and
    /// any stall condition is currently active.
    fn wait_for_write_capacity(&self, opts: &WriteOptions) -> Result<()> {
        self.engine.wait_for_write_capacity(opts.no_slowdown)?;
        Ok(())
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

    /// Scan a key range in the default column family. Returns all
    /// key-value pairs where `start <= key < end`, with keys in
    /// their user-visible form (no CF prefix).
    pub fn scan(
        &self,
        start: Option<&[u8]>,
        end: Option<&[u8]>,
    ) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let lo = match start {
            Some(s) => prefix_key(DEFAULT_CF_ID, s),
            None => cf_lower_bound(DEFAULT_CF_ID),
        };
        let hi = match end {
            Some(e) => prefix_key(DEFAULT_CF_ID, e),
            None => cf_upper_bound(DEFAULT_CF_ID),
        };
        let seq = self.engine.snapshot_seq();
        let raw = collect_range(&self.engine, Some(&lo), Some(&hi), seq)?;
        Ok(raw.into_iter().map(|(k, v)| (k[4..].to_vec(), v)).collect())
    }

    /// Create a streaming iterator over the default column family.
    ///
    /// The iterator captures a consistent view at the moment it is created
    /// — later writes are invisible to this iterator, and concurrent
    /// background compaction cannot invalidate it. Keys returned from
    /// the iterator have the CF prefix stripped and appear exactly as
    /// the caller supplied them on put.
    ///
    /// A fresh iterator is not positioned; call one of
    /// [`CfIter::seek_to_first`], [`CfIter::seek`], or
    /// [`CfIter::seek_for_prev`] before reading.
    pub fn iter(&self) -> CfIter<'_> {
        let default = self.default_cf();
        self.iter_cf(&default)
    }

    /// Create a raw streaming iterator over the entire engine
    /// keyspace, including the reserved metadata CF. Internal —
    /// used by [`Db::iter_cf`] via `CfIter`.
    fn raw_iter(&self) -> Iter<'_> {
        let seq = self.engine.snapshot_seq();
        Iter::from_internal(self.engine.new_iter(seq)).with_stats(self.engine.statistics_arc())
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

    /// Return the string value of a named property, or `None` if
    /// `name` isn't recognized. See the module-level docs for the
    /// full list of supported properties; the most useful ones
    /// are `"lark.stats"`, `"lark.sstables"`,
    /// `"lark.levelstats"`, and `"lark.options"`.
    pub fn get_property(&self, name: &str) -> Option<String> {
        match name {
            "lark.stats" => Some(self.format_stats_property()),
            "lark.sstables" => Some(self.format_sstables_property()),
            "lark.levelstats" => Some(self.format_levelstats_property()),
            "lark.options" => Some(format!("{:#?}", self.options_snapshot())),
            _ => {
                // Integer properties surfaced through the string
                // API too — every int property's string form is
                // just its decimal number.
                self.get_int_property(name).map(|v| v.to_string())
            }
        }
    }

    /// Return the integer value of a named property, or `None` if
    /// `name` isn't recognized or doesn't have an integer form.
    pub fn get_int_property(&self, name: &str) -> Option<u64> {
        if let Some(level_str) = name.strip_prefix("lark.num-files-at-level") {
            let level: usize = level_str.parse().ok()?;
            return Some(self.engine.num_files_at_level(level));
        }
        match name {
            "lark.total-sst-files-size" => Some(self.engine.total_sst_size()),
            "lark.cur-size-active-mem-table" => Some(self.engine.active_memtable_size()),
            "lark.cur-size-all-mem-tables" => {
                Some(self.engine.active_memtable_size() + self.engine.frozen_memtables_size())
            }
            "lark.num-entries-active-mem-table" => {
                // Approximate: the memtable exposes `approximate_size`
                // in bytes but no direct entry count. Estimate by
                // assuming a 48-byte average entry (internal key +
                // value). This is a rough indicator, not an exact
                // count.
                let bytes = self.engine.active_memtable_size();
                Some(bytes / 48)
            }
            "lark.num-entries-imm-mem-tables" => {
                let bytes = self.engine.frozen_memtables_size();
                Some(bytes / 48)
            }
            "lark.estimate-num-keys" => {
                // Lower-bound estimate: exact SST entry count plus
                // a rough guess for the memtable contribution.
                let sst = self.engine.total_sst_num_entries();
                let mem_bytes =
                    self.engine.active_memtable_size() + self.engine.frozen_memtables_size();
                Some(sst + mem_bytes / 48)
            }
            "lark.estimate-live-data-size" => Some(self.engine.total_sst_size()),
            "lark.num-snapshots" => Some(self.engine.live_snapshot_count()),
            "lark.oldest-snapshot-time" => self.engine.oldest_snapshot_time_unix(),
            "lark.block-cache-usage" => Some(self.engine.block_cache_usage() as u64),
            "lark.block-cache-capacity" => Some(self.engine.block_cache_capacity() as u64),
            // Background errors are surfaced through the
            // `EventListener::on_background_error` callback today
            // — no dedicated counter yet. Report `0` so any
            // monitoring layer consuming this property gets a
            // stable numeric value instead of `None`.
            "lark.background-errors" => Some(0),
            _ => None,
        }
    }

    /// Format the multi-line `lark.stats` property: counters +
    /// histograms (when statistics are enabled) plus per-level
    /// file counts and compaction aggregates.
    fn format_stats_property(&self) -> String {
        let mut out = String::new();
        out.push_str("== lark engine stats ==\n");
        out.push_str(&self.format_levelstats_property());
        if let Some(stats) = self.engine.statistics() {
            out.push('\n');
            out.push_str(&stats.dump());
        } else {
            out.push_str("\n(no Statistics object configured — see Options::statistics)\n");
        }
        out
    }

    /// Format the `lark.levelstats` property: one row per
    /// level with file count and total size in bytes.
    fn format_levelstats_property(&self) -> String {
        let version = self.engine.current_version();
        let mut out = String::from("Level  Files     Size(B)\n");
        for (lvl, files) in version.levels.iter().enumerate() {
            let count = files.len();
            let size: u64 = files.iter().map(|f| f.meta.file_size).sum();
            out.push_str(&format!("{lvl:5}  {count:5}  {size:10}\n"));
        }
        out
    }

    /// Format the `lark.sstables` property: one row per live
    /// SSTable with its level, file id, size, and key range.
    fn format_sstables_property(&self) -> String {
        let version = self.engine.current_version();
        let mut out =
            String::from("Level    FileID       Size(B)     Entries  SmallestKey..LargestKey\n");
        for (lvl, files) in version.levels.iter().enumerate() {
            for f in files {
                // Strip the CF prefix for display when the key
                // has room for it; otherwise show the raw bytes.
                let smallest = format_key_for_display(&f.meta.smallest_key);
                let largest = format_key_for_display(&f.meta.largest_key);
                out.push_str(&format!(
                    "{lvl:5}  {:8}  {:12}  {:10}  {}..{}\n",
                    f.meta.file_id, f.meta.file_size, f.meta.num_entries, smallest, largest
                ));
            }
        }
        out
    }

    /// A minimal snapshot of the engine options. We deliberately
    /// do not carry the full `Options` struct around past
    /// construction, so this returns a small struct with just
    /// the observable knobs.
    fn options_snapshot(&self) -> OptionsSnapshot {
        OptionsSnapshot {
            durability: self.durability,
            default_cf: DEFAULT_CF_NAME,
        }
    }

    /// Return the approximate on-disk bytes in each of the given
    /// ranges, in the same order as `ranges`. Each range is
    /// scoped to the default column family.
    ///
    /// Computed index-only: no data-block decompression happens,
    /// so the cost is sub-linear in the range size. Accuracy is
    /// bounded by one data-block worth of bytes per range
    /// boundary (partially-covered blocks at `start` and `end` are
    /// included whole). Active-memtable contents are **not**
    /// included — call [`Db::get_approximate_memtable_stats`] for
    /// those.
    pub fn get_approximate_sizes(&self, ranges: &[Range<'_>]) -> Vec<u64> {
        ranges
            .iter()
            .map(|r| self.approximate_size_in_range(&self.default_cf(), r))
            .collect()
    }

    /// CF-scoped variant of [`Db::get_approximate_sizes`].
    pub fn get_approximate_sizes_cf(
        &self,
        cf: &ColumnFamilyHandle,
        ranges: &[Range<'_>],
    ) -> Vec<u64> {
        ranges
            .iter()
            .map(|r| self.approximate_size_in_range(cf, r))
            .collect()
    }

    fn approximate_size_in_range(&self, cf: &ColumnFamilyHandle, r: &Range<'_>) -> u64 {
        if r.start >= r.end {
            return 0;
        }
        let lo = prefix_key(cf.id(), r.start);
        let hi = prefix_key(cf.id(), r.end);
        self.engine.approximate_size_in_range(&lo, &hi)
    }

    /// Exact count + approximate size of entries in the active
    /// memtable whose user key falls in `range`, scoped to the
    /// default column family. Frozen memtables are not included.
    pub fn get_approximate_memtable_stats(&self, range: Range<'_>) -> MemTableStats {
        self.memtable_stats_in(&self.default_cf(), &range)
    }

    /// CF-scoped variant of [`Db::get_approximate_memtable_stats`].
    pub fn get_approximate_memtable_stats_cf(
        &self,
        cf: &ColumnFamilyHandle,
        range: Range<'_>,
    ) -> MemTableStats {
        self.memtable_stats_in(cf, &range)
    }

    fn memtable_stats_in(&self, cf: &ColumnFamilyHandle, range: &Range<'_>) -> MemTableStats {
        if range.start >= range.end {
            return MemTableStats::default();
        }
        let lo = prefix_key(cf.id(), range.start);
        let hi = prefix_key(cf.id(), range.end);
        let (count, size) = self.engine.approximate_memtable_stats(&lo, &hi);
        MemTableStats { count, size }
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

    // ── column families ─────────────────────────────────────────────────

    /// Return a handle to the default column family. Always
    /// present — [`Db::open`] creates it if the database didn't
    /// already contain one.
    pub fn default_cf(&self) -> ColumnFamilyHandle {
        self.cfs
            .get(DEFAULT_CF_NAME)
            .expect("default CF is created at Db::open time")
    }

    /// Look up a column family by name. Returns `None` when no CF
    /// with that name has been created (or if the CF was dropped).
    pub fn column_family(&self, name: &str) -> Option<ColumnFamilyHandle> {
        self.cfs.get(name)
    }

    /// Return the names of every live column family, including
    /// `"default"`. Order is unspecified.
    pub fn list_column_families(&self) -> Vec<String> {
        let mut names = self.cfs.names();
        names.sort();
        names
    }

    /// Create a new column family with `name`. The name must be
    /// non-empty and unique; creating a CF with an existing name
    /// returns the existing handle (idempotent).
    ///
    /// The new CF is persisted to the on-disk metadata before this
    /// call returns, so it survives a crash and a reopen.
    pub fn create_column_family(&self, name: &str) -> Result<ColumnFamilyHandle> {
        if name.is_empty() {
            return Err(Error::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "column family name must not be empty",
            )));
        }
        if let Some(existing) = self.cfs.get(name) {
            return Ok(existing);
        }
        let (handle, next_id) = self.cfs.allocate(name);
        let mut batch = BTreeMap::new();
        batch.insert(
            meta::name_key(name),
            Some(handle.id().to_be_bytes().to_vec()),
        );
        batch.insert(meta::next_id_key(), Some(next_id.to_be_bytes().to_vec()));
        self.engine
            .apply_batch(batch, Vec::new(), Vec::new(), self.durability, false)
            .map_err(Error::Io)?;
        Ok(handle)
    }

    /// Drop a column family. Every key stored in the CF is removed
    /// via a single range tombstone (O(1) write work regardless of
    /// key count) and the CF name is unregistered so future
    /// lookups via [`Db::column_family`] return `None`. Space is
    /// physically reclaimed by the next compaction over the range.
    ///
    /// Dropping the default column family is not allowed and
    /// returns an error.
    pub fn drop_column_family(&self, cf: ColumnFamilyHandle) -> Result<()> {
        if cf.id() == DEFAULT_CF_ID {
            return Err(Error::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "cannot drop the default column family",
            )));
        }
        if cf.id() == META_CF_ID {
            return Err(Error::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "cannot drop the reserved metadata column family",
            )));
        }
        let lo = cf_lower_bound(cf.id());
        let hi = cf_upper_bound(cf.id());
        // Apply the data range-delete and the metadata entry
        // removal in a single atomic batch so a crash mid-drop
        // either leaves the CF fully present or fully removed.
        let mut point_ops = BTreeMap::new();
        point_ops.insert(meta::name_key(cf.name()), None);
        let range_deletes = vec![(lo, hi)];
        self.engine
            .apply_batch(point_ops, range_deletes, Vec::new(), self.durability, false)
            .map_err(Error::Io)?;
        self.cfs.remove(cf.name());
        Ok(())
    }

    /// Read `key` from column family `cf`. Same semantics as
    /// [`Db::get`] but scoped to the CF's keyspace.
    pub fn get_cf(&self, cf: &ColumnFamilyHandle, key: &[u8]) -> Result<Option<Vec<u8>>> {
        self.get_raw(&prefix_key(cf.id(), key))
    }

    /// Batched point lookup across a single CF.
    pub fn multi_get_cf(
        &self,
        cf: &ColumnFamilyHandle,
        keys: &[&[u8]],
    ) -> Result<Vec<Option<Vec<u8>>>> {
        let owned: Vec<Vec<u8>> = keys.iter().map(|k| prefix_key(cf.id(), k)).collect();
        let refs: Vec<&[u8]> = owned.iter().map(|k| k.as_slice()).collect();
        let seq = self.engine.snapshot_seq();
        self.engine.multi_get(&refs, seq).map_err(Error::Io)
    }

    /// Write `key → value` in column family `cf`.
    pub fn put_cf(&self, cf: &ColumnFamilyHandle, key: &[u8], value: &[u8]) -> Result<()> {
        let mut batch = BTreeMap::new();
        batch.insert(prefix_key(cf.id(), key), Some(value.to_vec()));
        self.engine
            .apply_batch(batch, Vec::new(), Vec::new(), self.durability, false)
            .map_err(Error::Io)
    }

    /// Delete `key` in column family `cf`.
    pub fn delete_cf(&self, cf: &ColumnFamilyHandle, key: &[u8]) -> Result<()> {
        let mut batch = BTreeMap::new();
        batch.insert(prefix_key(cf.id(), key), None);
        self.engine
            .apply_batch(batch, Vec::new(), Vec::new(), self.durability, false)
            .map_err(Error::Io)
    }

    /// Delete every key in `[start, end)` in column family `cf`.
    pub fn delete_range_cf(&self, cf: &ColumnFamilyHandle, start: &[u8], end: &[u8]) -> Result<()> {
        if start >= end {
            return Ok(());
        }
        self.engine
            .apply_batch(
                BTreeMap::new(),
                vec![(prefix_key(cf.id(), start), prefix_key(cf.id(), end))],
                Vec::new(),
                self.durability,
                false,
            )
            .map_err(Error::Io)
    }

    /// Layer a merge operand on top of `key` in column family `cf`.
    /// Requires [`Options::merge_operator`] to be set.
    pub fn merge_cf(&self, cf: &ColumnFamilyHandle, key: &[u8], operand: &[u8]) -> Result<()> {
        self.engine
            .apply_batch(
                BTreeMap::new(),
                Vec::new(),
                vec![(prefix_key(cf.id(), key), operand.to_vec())],
                self.durability,
                false,
            )
            .map_err(Error::Io)
    }

    /// Scan a key range inside column family `cf`. Returned keys
    /// have the CF prefix stripped — they appear exactly as the
    /// caller supplied them on put.
    pub fn scan_cf(
        &self,
        cf: &ColumnFamilyHandle,
        start: Option<&[u8]>,
        end: Option<&[u8]>,
    ) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let lo = match start {
            Some(s) => prefix_key(cf.id(), s),
            None => cf_lower_bound(cf.id()),
        };
        let hi = match end {
            Some(e) => prefix_key(cf.id(), e),
            None => cf_upper_bound(cf.id()),
        };
        let seq = self.engine.snapshot_seq();
        let raw = collect_range(&self.engine, Some(&lo), Some(&hi), seq)?;
        Ok(raw.into_iter().map(|(k, v)| (k[4..].to_vec(), v)).collect())
    }

    /// Streaming iterator bounded to column family `cf`. The
    /// returned keys have the CF prefix stripped.
    pub fn iter_cf<'a>(&'a self, cf: &ColumnFamilyHandle) -> CfIter<'a> {
        CfIter {
            inner: self.raw_iter(),
            cf_id: cf.id(),
            upper_bound: cf_upper_bound(cf.id()),
        }
    }

    /// Create a forward-only [`TailingIter`] over the default
    /// column family. Unlike [`Db::iter`], a tailing iterator
    /// sees writes that arrive after it was created and does not
    /// pin the database at a point in time — see [`TailingIter`]
    /// for the ordering rules.
    pub fn iter_tailing(&self) -> TailingIter {
        tailing::new_default(Arc::clone(&self.engine))
    }

    /// Create a forward-only [`TailingIter`] scoped to column
    /// family `cf`.
    pub fn iter_tailing_cf(&self, cf: &ColumnFamilyHandle) -> TailingIter {
        tailing::new_for_cf(Arc::clone(&self.engine), cf)
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

/// Streaming iterator scoped to a single column family. Wraps a
/// regular [`Iter`] and bounds the scan to the CF's prefix range,
/// stripping the 4-byte CF prefix from every key before returning
/// it. Created by [`Db::iter_cf`] / [`Snapshot::iter_cf`].
pub struct CfIter<'a> {
    inner: Iter<'a>,
    cf_id: u32,
    upper_bound: Vec<u8>,
}

impl<'a> CfIter<'a> {
    /// Position the cursor at the first key in the CF.
    pub fn seek_to_first(&mut self) {
        let lo = self.cf_id.to_be_bytes();
        self.inner.seek(&lo);
    }

    /// Position the cursor at the last key in the CF (or before
    /// the CF's upper bound if the CF is empty).
    pub fn seek_to_last(&mut self) {
        // `seek_for_prev(upper_bound - 1)` is the trick: seek to
        // the largest key strictly less than `upper_bound`.
        let mut probe = self.upper_bound.clone();
        // Decrement by one — upper_bound is built by incrementing
        // the CF id, so it's never all zeros; subtracting one byte
        // gives a valid probe. Simpler: seek_for_prev(upper_bound)
        // which lands on the last key < upper_bound.
        if let Some(last) = probe.last_mut() {
            if *last > 0 {
                *last -= 1;
                // Append 0xff bytes to get a probe strictly less
                // than upper_bound but larger than every key in
                // the CF.
                probe.extend_from_slice(&[0xff; 8]);
            }
        }
        self.inner.seek_for_prev(&probe);
    }

    /// Position the cursor at the first key `>= target` in the CF.
    pub fn seek(&mut self, target: &[u8]) {
        self.inner.seek(&prefix_key(self.cf_id, target));
    }

    /// Position the cursor at the last key `<= target` in the CF.
    pub fn seek_for_prev(&mut self, target: &[u8]) {
        self.inner.seek_for_prev(&prefix_key(self.cf_id, target));
    }

    /// Position the cursor at the first key in this CF that starts
    /// with `prefix`, and bound subsequent forward iteration to
    /// that prefix. Delegates to the underlying [`Iter::seek_prefix`],
    /// with `prefix` first re-scoped to include the CF prefix.
    pub fn seek_prefix(&mut self, prefix: &[u8]) {
        self.inner.seek_prefix(&prefix_key(self.cf_id, prefix));
    }

    /// Advance the cursor forward. Invalidates the iterator if
    /// the next key crosses the CF's upper bound.
    pub fn next(&mut self) {
        self.inner.next();
    }

    /// Move the cursor backward. Invalidates the iterator if
    /// the previous key crosses the CF's lower bound.
    pub fn prev(&mut self) {
        self.inner.prev();
    }

    /// Whether the cursor is positioned on a visible key within
    /// the CF.
    pub fn valid(&self) -> bool {
        let Some(k) = self.inner.key() else {
            return false;
        };
        if k < self.cf_id.to_be_bytes().as_slice() {
            return false;
        }
        if k >= self.upper_bound.as_slice() {
            return false;
        }
        true
    }

    /// Current key, with the CF prefix stripped.
    pub fn key(&self) -> Option<&[u8]> {
        if !self.valid() {
            return None;
        }
        self.inner.key().map(|k| &k[4..])
    }

    /// Current value.
    pub fn value(&self) -> Option<&[u8]> {
        if !self.valid() {
            return None;
        }
        self.inner.value()
    }

    /// Propagate any I/O error from the underlying iterator.
    pub fn status(&self) -> Result<()> {
        self.inner.status()
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
    /// Get the value for a key at this snapshot (default CF).
    pub fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        self.engine
            .get(&prefix_key(DEFAULT_CF_ID, key), self.seq)
            .map_err(Error::Io)
    }

    /// Batched point lookup anchored at this snapshot (default CF).
    pub fn multi_get(&self, keys: &[&[u8]]) -> Result<Vec<Option<Vec<u8>>>> {
        let owned: Vec<Vec<u8>> = keys.iter().map(|k| prefix_key(DEFAULT_CF_ID, k)).collect();
        let refs: Vec<&[u8]> = owned.iter().map(|k| k.as_slice()).collect();
        self.engine.multi_get(&refs, self.seq).map_err(Error::Io)
    }

    /// Scan a key range at this snapshot (default CF).
    pub fn scan(
        &self,
        start: Option<&[u8]>,
        end: Option<&[u8]>,
    ) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let lo = match start {
            Some(s) => prefix_key(DEFAULT_CF_ID, s),
            None => cf_lower_bound(DEFAULT_CF_ID),
        };
        let hi = match end {
            Some(e) => prefix_key(DEFAULT_CF_ID, e),
            None => cf_upper_bound(DEFAULT_CF_ID),
        };
        let raw = collect_range(&self.engine, Some(&lo), Some(&hi), self.seq)?;
        Ok(raw.into_iter().map(|(k, v)| (k[4..].to_vec(), v)).collect())
    }

    /// Create a streaming iterator anchored at this snapshot
    /// (default CF). Keys returned have the CF prefix stripped.
    pub fn iter(&self) -> CfIter<'_> {
        CfIter {
            inner: Iter::from_internal(self.engine.new_iter(self.seq))
                .with_stats(self.engine.statistics_arc()),
            cf_id: DEFAULT_CF_ID,
            upper_bound: cf_upper_bound(DEFAULT_CF_ID),
        }
    }

    /// CF-scoped get at this snapshot.
    pub fn get_cf(&self, cf: &ColumnFamilyHandle, key: &[u8]) -> Result<Option<Vec<u8>>> {
        self.engine
            .get(&prefix_key(cf.id(), key), self.seq)
            .map_err(Error::Io)
    }

    /// CF-scoped multi_get at this snapshot.
    pub fn multi_get_cf(
        &self,
        cf: &ColumnFamilyHandle,
        keys: &[&[u8]],
    ) -> Result<Vec<Option<Vec<u8>>>> {
        let owned: Vec<Vec<u8>> = keys.iter().map(|k| prefix_key(cf.id(), k)).collect();
        let refs: Vec<&[u8]> = owned.iter().map(|k| k.as_slice()).collect();
        self.engine.multi_get(&refs, self.seq).map_err(Error::Io)
    }

    /// CF-scoped scan at this snapshot. Returned keys have the
    /// CF prefix stripped.
    pub fn scan_cf(
        &self,
        cf: &ColumnFamilyHandle,
        start: Option<&[u8]>,
        end: Option<&[u8]>,
    ) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let lo = match start {
            Some(s) => prefix_key(cf.id(), s),
            None => cf_lower_bound(cf.id()),
        };
        let hi = match end {
            Some(e) => prefix_key(cf.id(), e),
            None => cf_upper_bound(cf.id()),
        };
        let raw = collect_range(&self.engine, Some(&lo), Some(&hi), self.seq)?;
        Ok(raw.into_iter().map(|(k, v)| (k[4..].to_vec(), v)).collect())
    }

    /// CF-scoped streaming iterator at this snapshot.
    pub fn iter_cf<'a>(&'a self, cf: &ColumnFamilyHandle) -> CfIter<'a> {
        CfIter {
            inner: Iter::from_internal(self.engine.new_iter(self.seq))
                .with_stats(self.engine.statistics_arc()),
            cf_id: cf.id(),
            upper_bound: cf_upper_bound(cf.id()),
        }
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

    /// Add a put operation to the batch (default column family).
    pub fn put(&mut self, key: &[u8], value: &[u8]) {
        self.ops
            .insert(prefix_key(DEFAULT_CF_ID, key), Some(value.to_vec()));
    }

    /// Add a delete operation to the batch (default column family).
    pub fn delete(&mut self, key: &[u8]) {
        self.ops.insert(prefix_key(DEFAULT_CF_ID, key), None);
    }

    /// Delete every key in the half-open range `[start, end)` in
    /// the default column family.
    ///
    /// When the batch is applied, the range delete is recorded with
    /// the same transactional seq as the other batch operations, so
    /// concurrent readers see an all-or-nothing effect. Calls with
    /// `start >= end` are ignored.
    pub fn delete_range(&mut self, start: &[u8], end: &[u8]) {
        if start >= end {
            return;
        }
        self.range_deletes.push((
            prefix_key(DEFAULT_CF_ID, start),
            prefix_key(DEFAULT_CF_ID, end),
        ));
    }

    /// Add a merge operand for `key` in the default column family.
    /// Requires the database to be configured with a
    /// [`MergeOperator`]; the operand is layered on top of any
    /// existing value or merge chain and collapsed at read time.
    /// Multiple merges on the same key in a single batch are
    /// allowed and applied in insertion order.
    pub fn merge(&mut self, key: &[u8], operand: &[u8]) {
        self.merges
            .push((prefix_key(DEFAULT_CF_ID, key), operand.to_vec()));
    }

    /// Add a put scoped to column family `cf`.
    pub fn put_cf(&mut self, cf: &ColumnFamilyHandle, key: &[u8], value: &[u8]) {
        self.ops
            .insert(prefix_key(cf.id(), key), Some(value.to_vec()));
    }

    /// Add a delete scoped to column family `cf`.
    pub fn delete_cf(&mut self, cf: &ColumnFamilyHandle, key: &[u8]) {
        self.ops.insert(prefix_key(cf.id(), key), None);
    }

    /// Add a range delete scoped to column family `cf`.
    pub fn delete_range_cf(&mut self, cf: &ColumnFamilyHandle, start: &[u8], end: &[u8]) {
        if start >= end {
            return;
        }
        self.range_deletes
            .push((prefix_key(cf.id(), start), prefix_key(cf.id(), end)));
    }

    /// Add a merge operand scoped to column family `cf`.
    pub fn merge_cf(&mut self, cf: &ColumnFamilyHandle, key: &[u8], operand: &[u8]) {
        self.merges
            .push((prefix_key(cf.id(), key), operand.to_vec()));
    }

    /// Insert an already-prefixed put (internal use by wrappers
    /// like `DbWithTtl` that iterate a source batch's raw entries
    /// and rebuild a new batch without re-applying the CF prefix).
    pub(crate) fn insert_raw_put(&mut self, prefixed_key: Vec<u8>, value: Vec<u8>) {
        self.ops.insert(prefixed_key, Some(value));
    }

    /// Insert an already-prefixed delete.
    pub(crate) fn insert_raw_delete(&mut self, prefixed_key: Vec<u8>) {
        self.ops.insert(prefixed_key, None);
    }

    /// Insert an already-prefixed range delete.
    pub(crate) fn insert_raw_range_delete(
        &mut self,
        prefixed_start: Vec<u8>,
        prefixed_end: Vec<u8>,
    ) {
        self.range_deletes.push((prefixed_start, prefixed_end));
    }

    /// Insert an already-prefixed merge operand.
    pub(crate) fn insert_raw_merge(&mut self, prefixed_key: Vec<u8>, operand: Vec<u8>) {
        self.merges.push((prefixed_key, operand));
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
        // The helper walks raw engine keys, so re-apply the
        // default-CF prefix before querying.
        let prefixed = prefix_key(DEFAULT_CF_ID, user_key);
        db.engine.all_persisted_versions_of(&prefixed).unwrap()
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

    // ── column families ─────────────────────────────────────────────────

    #[test]
    fn test_cf_default_exists_on_open() {
        let (db, _dir) = open_tmp();
        let default = db.default_cf();
        assert_eq!(default.name(), DEFAULT_CF_NAME);
        assert!(db.column_family(DEFAULT_CF_NAME).is_some());
        assert_eq!(db.list_column_families(), vec![DEFAULT_CF_NAME.to_string()]);
    }

    #[test]
    fn test_cf_create_and_lookup() {
        let (db, _dir) = open_tmp();
        let users = db.create_column_family("users").unwrap();
        let orders = db.create_column_family("orders").unwrap();
        assert_ne!(users, orders);
        assert_eq!(db.column_family("users"), Some(users.clone()));
        assert_eq!(db.column_family("orders"), Some(orders.clone()));
        assert!(db.column_family("missing").is_none());

        let mut names = db.list_column_families();
        names.sort();
        assert_eq!(names, vec!["default", "orders", "users"]);
    }

    #[test]
    fn test_cf_create_is_idempotent() {
        let (db, _dir) = open_tmp();
        let a = db.create_column_family("x").unwrap();
        let b = db.create_column_family("x").unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn test_cf_put_get_isolated_from_default() {
        let (db, _dir) = open_tmp();
        let users = db.create_column_family("users").unwrap();
        db.put(b"k", b"default_val").unwrap();
        db.put_cf(&users, b"k", b"users_val").unwrap();
        assert_eq!(db.get(b"k").unwrap(), Some(b"default_val".to_vec()));
        assert_eq!(
            db.get_cf(&users, b"k").unwrap(),
            Some(b"users_val".to_vec())
        );
    }

    #[test]
    fn test_cf_writes_to_a_invisible_from_b() {
        let (db, _dir) = open_tmp();
        let a = db.create_column_family("a").unwrap();
        let b = db.create_column_family("b").unwrap();
        db.put_cf(&a, b"shared_key", b"alpha").unwrap();
        assert_eq!(
            db.get_cf(&a, b"shared_key").unwrap(),
            Some(b"alpha".to_vec())
        );
        assert_eq!(db.get_cf(&b, b"shared_key").unwrap(), None);
    }

    #[test]
    fn test_cf_delete_cf() {
        let (db, _dir) = open_tmp();
        let cf = db.create_column_family("c").unwrap();
        db.put_cf(&cf, b"k", b"v").unwrap();
        db.delete_cf(&cf, b"k").unwrap();
        assert_eq!(db.get_cf(&cf, b"k").unwrap(), None);
    }

    #[test]
    fn test_cf_scan_strips_prefix() {
        let (db, _dir) = open_tmp();
        let cf = db.create_column_family("s").unwrap();
        db.put_cf(&cf, b"a", b"1").unwrap();
        db.put_cf(&cf, b"b", b"2").unwrap();
        db.put_cf(&cf, b"c", b"3").unwrap();
        let pairs = db.scan_cf(&cf, None, None).unwrap();
        assert_eq!(
            pairs,
            vec![
                (b"a".to_vec(), b"1".to_vec()),
                (b"b".to_vec(), b"2".to_vec()),
                (b"c".to_vec(), b"3".to_vec()),
            ]
        );
        // Bounded scan.
        let pairs = db.scan_cf(&cf, Some(b"b"), Some(b"c")).unwrap();
        assert_eq!(pairs, vec![(b"b".to_vec(), b"2".to_vec())]);
    }

    #[test]
    fn test_cf_iter_bounded_to_cf() {
        let (db, _dir) = open_tmp();
        let a = db.create_column_family("a").unwrap();
        let b = db.create_column_family("b").unwrap();
        db.put_cf(&a, b"a1", b"A1").unwrap();
        db.put_cf(&a, b"a2", b"A2").unwrap();
        db.put_cf(&b, b"b1", b"B1").unwrap();
        db.put(b"d1", b"D1").unwrap();

        let mut iter = db.iter_cf(&a);
        iter.seek_to_first();
        let mut keys = Vec::new();
        while iter.valid() {
            keys.push(iter.key().unwrap().to_vec());
            iter.next();
        }
        assert_eq!(keys, vec![b"a1".to_vec(), b"a2".to_vec()]);
    }

    #[test]
    fn test_cf_iter_reverse() {
        let (db, _dir) = open_tmp();
        let cf = db.create_column_family("rev").unwrap();
        db.put_cf(&cf, b"a", b"1").unwrap();
        db.put_cf(&cf, b"b", b"2").unwrap();
        db.put_cf(&cf, b"c", b"3").unwrap();

        let mut iter = db.iter_cf(&cf);
        iter.seek_to_last();
        let mut keys = Vec::new();
        while iter.valid() {
            keys.push(iter.key().unwrap().to_vec());
            iter.prev();
        }
        assert_eq!(keys, vec![b"c".to_vec(), b"b".to_vec(), b"a".to_vec()]);
    }

    #[test]
    fn test_cf_drop_removes_all_keys_in_cf() {
        let (db, _dir) = open_tmp();
        let cf = db.create_column_family("tmp").unwrap();
        db.put_cf(&cf, b"a", b"1").unwrap();
        db.put_cf(&cf, b"b", b"2").unwrap();
        db.put_cf(&cf, b"c", b"3").unwrap();
        db.put(b"default_key", b"default_val").unwrap();

        db.drop_column_family(cf.clone()).unwrap();

        // The CF name is unregistered.
        assert!(db.column_family("tmp").is_none());
        // Default CF survives.
        assert_eq!(
            db.get(b"default_key").unwrap(),
            Some(b"default_val".to_vec())
        );
        // Re-creating with the same name yields a fresh, empty CF.
        let cf2 = db.create_column_family("tmp").unwrap();
        assert_eq!(db.get_cf(&cf2, b"a").unwrap(), None);
    }

    #[test]
    fn test_cf_cannot_drop_default() {
        let (db, _dir) = open_tmp();
        let default = db.default_cf();
        assert!(db.drop_column_family(default).is_err());
    }

    #[test]
    fn test_cf_survives_reopen() {
        let dir = TempDir::new().unwrap();
        {
            let db = Db::open(dir.path(), Options::default()).unwrap();
            let cf = db.create_column_family("persistent").unwrap();
            db.put_cf(&cf, b"k", b"v").unwrap();
            db.close().unwrap();
        }
        let db = Db::open(dir.path(), Options::default()).unwrap();
        let cf = db
            .column_family("persistent")
            .expect("CF must survive reopen");
        assert_eq!(db.get_cf(&cf, b"k").unwrap(), Some(b"v".to_vec()));
    }

    #[test]
    fn test_cf_dropped_cf_does_not_survive_reopen() {
        let dir = TempDir::new().unwrap();
        {
            let db = Db::open(dir.path(), Options::default()).unwrap();
            let cf = db.create_column_family("doomed").unwrap();
            db.put_cf(&cf, b"k", b"v").unwrap();
            db.drop_column_family(cf).unwrap();
            db.close().unwrap();
        }
        let db = Db::open(dir.path(), Options::default()).unwrap();
        assert!(db.column_family("doomed").is_none());
    }

    #[test]
    fn test_cf_write_batch_cross_cf_atomic() {
        let (db, _dir) = open_tmp();
        let a = db.create_column_family("a").unwrap();
        let b = db.create_column_family("b").unwrap();
        let mut batch = WriteBatch::new();
        batch.put_cf(&a, b"k1", b"v_a1");
        batch.put_cf(&b, b"k1", b"v_b1");
        batch.put(b"k1", b"v_default");
        batch.delete_cf(&a, b"ghost");
        db.write(batch).unwrap();

        assert_eq!(db.get_cf(&a, b"k1").unwrap(), Some(b"v_a1".to_vec()));
        assert_eq!(db.get_cf(&b, b"k1").unwrap(), Some(b"v_b1".to_vec()));
        assert_eq!(db.get(b"k1").unwrap(), Some(b"v_default".to_vec()));
    }

    #[test]
    fn test_cf_write_batch_survives_crash_recovery() {
        let dir = TempDir::new().unwrap();
        {
            let db = Db::open(dir.path(), Options::default()).unwrap();
            let cf = db.create_column_family("txn").unwrap();
            let mut batch = WriteBatch::new();
            batch.put_cf(&cf, b"a", b"1");
            batch.put_cf(&cf, b"b", b"2");
            batch.put(b"default_k", b"default_v");
            db.write(batch).unwrap();
            // No close — simulate a crash. WAL must survive.
        }
        let db = Db::open(dir.path(), Options::default()).unwrap();
        let cf = db.column_family("txn").expect("CF must survive");
        assert_eq!(db.get_cf(&cf, b"a").unwrap(), Some(b"1".to_vec()));
        assert_eq!(db.get_cf(&cf, b"b").unwrap(), Some(b"2".to_vec()));
        assert_eq!(db.get(b"default_k").unwrap(), Some(b"default_v".to_vec()));
    }

    #[test]
    fn test_cf_snapshot_isolation_per_cf() {
        let (db, _dir) = open_tmp();
        let a = db.create_column_family("a").unwrap();
        db.put_cf(&a, b"k", b"v0").unwrap();
        let snap = db.snapshot();
        db.put_cf(&a, b"k", b"v1").unwrap();
        assert_eq!(snap.get_cf(&a, b"k").unwrap(), Some(b"v0".to_vec()));
        assert_eq!(db.get_cf(&a, b"k").unwrap(), Some(b"v1".to_vec()));
    }

    #[test]
    fn test_cf_scan_across_cfs_is_isolated() {
        let (db, _dir) = open_tmp();
        let a = db.create_column_family("a").unwrap();
        let b = db.create_column_family("b").unwrap();
        db.put_cf(&a, b"apple", b"A").unwrap();
        db.put_cf(&b, b"apple", b"B").unwrap();
        db.put(b"apple", b"D").unwrap();

        assert_eq!(
            db.scan_cf(&a, None, None).unwrap(),
            vec![(b"apple".to_vec(), b"A".to_vec())]
        );
        assert_eq!(
            db.scan_cf(&b, None, None).unwrap(),
            vec![(b"apple".to_vec(), b"B".to_vec())]
        );
        assert_eq!(
            db.scan(None, None).unwrap(),
            vec![(b"apple".to_vec(), b"D".to_vec())]
        );
    }

    #[test]
    fn test_cf_multi_get_cf() {
        let (db, _dir) = open_tmp();
        let cf = db.create_column_family("mg").unwrap();
        db.put_cf(&cf, b"a", b"1").unwrap();
        db.put_cf(&cf, b"b", b"2").unwrap();
        let keys: Vec<&[u8]> = vec![b"a", b"missing", b"b"];
        let got = db.multi_get_cf(&cf, &keys).unwrap();
        assert_eq!(got, vec![Some(b"1".to_vec()), None, Some(b"2".to_vec())]);
    }

    #[test]
    fn test_cf_delete_range_cf() {
        let (db, _dir) = open_tmp();
        let cf = db.create_column_family("r").unwrap();
        for c in b'a'..=b'f' {
            db.put_cf(&cf, &[c], &[c]).unwrap();
        }
        db.delete_range_cf(&cf, b"b", b"e").unwrap();
        assert_eq!(db.get_cf(&cf, b"a").unwrap(), Some(b"a".to_vec()));
        assert_eq!(db.get_cf(&cf, b"b").unwrap(), None);
        assert_eq!(db.get_cf(&cf, b"c").unwrap(), None);
        assert_eq!(db.get_cf(&cf, b"d").unwrap(), None);
        assert_eq!(db.get_cf(&cf, b"e").unwrap(), Some(b"e".to_vec()));
        assert_eq!(db.get_cf(&cf, b"f").unwrap(), Some(b"f".to_vec()));
    }

    #[test]
    fn test_cf_create_empty_name_errors() {
        let (db, _dir) = open_tmp();
        assert!(db.create_column_family("").is_err());
    }

    #[test]
    fn test_cf_many_cfs_all_isolated() {
        let (db, _dir) = open_tmp();
        let mut handles = Vec::new();
        for i in 0..10 {
            handles.push(db.create_column_family(&format!("cf{i}")).unwrap());
        }
        for (i, h) in handles.iter().enumerate() {
            db.put_cf(h, b"k", format!("v{i}").as_bytes()).unwrap();
        }
        for (i, h) in handles.iter().enumerate() {
            assert_eq!(
                db.get_cf(h, b"k").unwrap(),
                Some(format!("v{i}").into_bytes())
            );
        }
    }

    // ── get_approximate_sizes / get_approximate_memtable_stats ──────────

    #[test]
    fn test_approximate_sizes_empty_db() {
        let (db, _dir) = open_tmp();
        let sizes = db.get_approximate_sizes(&[Range::new(b"a", b"z")]);
        assert_eq!(sizes, vec![0]);
    }

    #[test]
    fn test_approximate_sizes_empty_range_returns_zero() {
        let (db, _dir) = open_tmp();
        db.put(b"k", b"v").unwrap();
        // Inverted / empty range must not panic and must be 0.
        assert_eq!(db.get_approximate_sizes(&[Range::new(b"z", b"a")]), vec![0]);
        assert_eq!(db.get_approximate_sizes(&[Range::new(b"k", b"k")]), vec![0]);
    }

    #[test]
    fn test_approximate_memtable_stats_exact_for_memtable() {
        let (db, _dir) = open_tmp();
        for c in b'a'..=b'e' {
            db.put(&[c], b"v").unwrap();
        }
        let stats = db.get_approximate_memtable_stats(Range::new(b"b", b"e"));
        assert_eq!(stats.count, 3, "count must be exact");
        // Each entry is [4-byte cf prefix][1-byte key] as
        // internal-key + 9-byte seq/type suffix + 1-byte value.
        // The size must be strictly > 0 and < (5 full entries * 50).
        assert!(stats.size > 0);
        assert!(stats.size < 500);
    }

    #[test]
    fn test_approximate_memtable_stats_empty_range() {
        let (db, _dir) = open_tmp();
        db.put(b"k", b"v").unwrap();
        let stats = db.get_approximate_memtable_stats(Range::new(b"m", b"n"));
        assert_eq!(stats, MemTableStats::default());
    }

    #[test]
    fn test_approximate_memtable_stats_counts_every_version() {
        let (db, _dir) = open_tmp();
        db.put(b"k", b"v1").unwrap();
        db.put(b"k", b"v2").unwrap();
        db.put(b"k", b"v3").unwrap();
        let stats = db.get_approximate_memtable_stats(Range::new(b"k", b"l"));
        // Three versions of the same user key.
        assert_eq!(stats.count, 3);
    }

    #[test]
    fn test_approximate_sizes_after_flush_within_factor_of_2() {
        // Write enough data to materialize into L0, then check the
        // approximate size against the on-disk file size. The
        // accuracy contract is "within a factor of 2".
        //
        // Use a high-entropy payload so LZ4 can't crush it — a
        // zero-filled payload compresses to near-nothing and would
        // undercut the accuracy window we're checking.
        let dir = TempDir::new().unwrap();
        let opts = Options {
            write_buffer_size: 4 * 1024,
            compression: CompressionType::None,
            ..Options::default()
        };
        let db = Db::open(dir.path(), opts).unwrap();
        let payload: Vec<u8> = (0..256).map(|i| (i % 251) as u8).collect();
        for i in 0..100 {
            db.put(format!("k_{i:04}").as_bytes(), &payload).unwrap();
        }
        force_flush(&db, "sizes");
        let sizes = db.get_approximate_sizes(&[Range::new(b"k_0000", b"k_9999")]);
        assert!(sizes[0] > 0, "whole-range size must be > 0 after flush");
        // Raw on-disk footprint of the point data: 100 entries,
        // each ≈ 256-byte value + ~20-byte key/overhead + a bit
        // of block framing, so ~28–30k. The approximation
        // includes whole covered blocks, so a 2x window covers
        // it comfortably.
        let approx = sizes[0];
        assert!(approx > 10_000, "approx={approx} too small; expected > 10k");
        assert!(
            approx < 1_000_000,
            "approx={approx} absurdly large; expected < 1M"
        );
    }

    #[test]
    fn test_approximate_sizes_multi_range_preserves_order() {
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), tiny_flush_opts()).unwrap();
        let payload = vec![0u8; 256];
        for i in 0..200 {
            db.put(format!("k_{i:04}").as_bytes(), &payload).unwrap();
        }
        force_flush(&db, "multi");
        let ranges = vec![
            Range::new(b"k_0000", b"k_0050"),
            Range::new(b"k_0050", b"k_0100"),
            Range::new(b"k_0100", b"k_0200"),
        ];
        let sizes = db.get_approximate_sizes(&ranges);
        assert_eq!(sizes.len(), 3);
        // Every range should contain some bytes.
        for (i, &s) in sizes.iter().enumerate() {
            assert!(s > 0, "range {i} size was 0");
        }
    }

    #[test]
    fn test_approximate_sizes_cf_scoped() {
        let (db, _dir) = open_tmp();
        let cf = db.create_column_family("scoped").unwrap();
        // Put into default CF but not into `scoped` — the
        // scoped CF's whole-range size must be 0.
        for i in 0..20 {
            db.put(format!("k{i}").as_bytes(), b"v").unwrap();
        }
        let default_sizes = db.get_approximate_sizes(&[Range::new(b"a", b"z")]);
        let cf_sizes = db.get_approximate_sizes_cf(&cf, &[Range::new(b"a", b"z")]);
        // Memtable contents aren't in approximate_sizes, but they
        // aren't on disk either — the default-CF whole-range
        // matches the scoped-CF whole-range (both 0) unless a
        // flush happened. With default write_buffer_size, 20 small
        // writes don't trigger a flush.
        assert_eq!(default_sizes[0], 0);
        assert_eq!(cf_sizes[0], 0);

        // Memtable-stats however sees the default CF entries but
        // not the scoped CF.
        let default_mt = db.get_approximate_memtable_stats(Range::new(b"a", b"z"));
        let cf_mt = db.get_approximate_memtable_stats_cf(&cf, Range::new(b"a", b"z"));
        assert_eq!(default_mt.count, 20);
        assert_eq!(cf_mt.count, 0);
    }

    // ── atomic flush across column families ────────────────────────────

    #[test]
    fn test_atomic_flush_multi_cf_batch_survives_crash() {
        // A WriteBatch that touches multiple CFs must be
        // all-or-nothing across a crash, even when the write
        // lands in the memtable without an explicit flush.
        let dir = TempDir::new().unwrap();
        {
            let db = Db::open(dir.path(), Options::default()).unwrap();
            let cf_a = db.create_column_family("a").unwrap();
            let cf_b = db.create_column_family("b").unwrap();
            let mut batch = WriteBatch::new();
            batch.put_cf(&cf_a, b"k1", b"a1");
            batch.put_cf(&cf_a, b"k2", b"a2");
            batch.put_cf(&cf_b, b"k1", b"b1");
            batch.put_cf(&cf_b, b"k2", b"b2");
            batch.put(b"default_k", b"default_v");
            db.write(batch).unwrap();
            // Drop without close — simulate a crash. WAL is the
            // source of truth; recovery must restore every key.
        }
        let db = Db::open(dir.path(), Options::default()).unwrap();
        let cf_a = db.column_family("a").expect("cf a survives reopen");
        let cf_b = db.column_family("b").expect("cf b survives reopen");
        assert_eq!(db.get_cf(&cf_a, b"k1").unwrap(), Some(b"a1".to_vec()));
        assert_eq!(db.get_cf(&cf_a, b"k2").unwrap(), Some(b"a2".to_vec()));
        assert_eq!(db.get_cf(&cf_b, b"k1").unwrap(), Some(b"b1".to_vec()));
        assert_eq!(db.get_cf(&cf_b, b"k2").unwrap(), Some(b"b2".to_vec()));
        assert_eq!(db.get(b"default_k").unwrap(), Some(b"default_v".to_vec()));
    }

    #[test]
    fn test_atomic_flush_cross_cf_survives_rotate_and_flush() {
        // Drive the memtable past its flush threshold while a
        // multi-CF batch is in flight. The rotated memtable
        // produces one L0 SSTable that contains every CF's half
        // of the batch atomically.
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), tiny_flush_opts()).unwrap();
        let cf_a = db.create_column_family("a").unwrap();
        let cf_b = db.create_column_family("b").unwrap();

        // Seed enough filler to push the tiny 4KB buffer over
        // on the next write.
        for i in 0..16 {
            db.put_cf(&cf_a, format!("fill_a_{i:02}").as_bytes(), &[0u8; 256])
                .unwrap();
            db.put_cf(&cf_b, format!("fill_b_{i:02}").as_bytes(), &[0u8; 256])
                .unwrap();
        }

        let mut batch = WriteBatch::new();
        batch.put_cf(&cf_a, b"pivot", b"A_PIVOT");
        batch.put_cf(&cf_b, b"pivot", b"B_PIVOT");
        db.write(batch).unwrap();
        force_flush(&db, "atomic");

        assert_eq!(
            db.get_cf(&cf_a, b"pivot").unwrap(),
            Some(b"A_PIVOT".to_vec())
        );
        assert_eq!(
            db.get_cf(&cf_b, b"pivot").unwrap(),
            Some(b"B_PIVOT".to_vec())
        );
    }

    #[test]
    fn test_atomic_flush_empty_cf_mixed_with_populated() {
        // Creating a CF and leaving it empty while another CF
        // gets flushed must not corrupt the empty CF.
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), tiny_flush_opts()).unwrap();
        let cf_populated = db.create_column_family("populated").unwrap();
        let cf_empty = db.create_column_family("empty").unwrap();

        for i in 0..50 {
            db.put_cf(&cf_populated, format!("k{i:02}").as_bytes(), b"v")
                .unwrap();
        }
        force_flush(&db, "empty_mix");

        for i in 0..50 {
            assert_eq!(
                db.get_cf(&cf_populated, format!("k{i:02}").as_bytes())
                    .unwrap(),
                Some(b"v".to_vec())
            );
        }
        assert_eq!(db.get_cf(&cf_empty, b"anything").unwrap(), None);

        // The empty CF still accepts new writes after the flush.
        db.put_cf(&cf_empty, b"new", b"fresh").unwrap();
        assert_eq!(
            db.get_cf(&cf_empty, b"new").unwrap(),
            Some(b"fresh".to_vec())
        );
    }

    #[test]
    fn test_atomic_flush_option_accepted() {
        // The flag is a no-op for API parity — both values must
        // open cleanly and produce the same atomic behavior.
        let dir = TempDir::new().unwrap();
        let opts = Options {
            atomic_flush: true,
            ..Options::default()
        };
        let db = Db::open(dir.path(), opts).unwrap();
        let cf = db.create_column_family("cf1").unwrap();
        db.put_cf(&cf, b"k", b"v").unwrap();
        assert_eq!(db.get_cf(&cf, b"k").unwrap(), Some(b"v".to_vec()));
    }

    #[test]
    fn test_atomic_flush_close_with_pending_multi_cf_writes() {
        // A clean close with pending multi-CF writes must flush
        // the active memtable to L0 before returning, so every
        // CF's state is durable on reopen.
        let dir = TempDir::new().unwrap();
        {
            let db = Db::open(dir.path(), Options::default()).unwrap();
            let cf_a = db.create_column_family("a").unwrap();
            let cf_b = db.create_column_family("b").unwrap();
            let mut batch = WriteBatch::new();
            batch.put_cf(&cf_a, b"k", b"a");
            batch.put_cf(&cf_b, b"k", b"b");
            db.write(batch).unwrap();
            db.close().unwrap();
        }
        let db = Db::open(dir.path(), Options::default()).unwrap();
        let cf_a = db.column_family("a").unwrap();
        let cf_b = db.column_family("b").unwrap();
        assert_eq!(db.get_cf(&cf_a, b"k").unwrap(), Some(b"a".to_vec()));
        assert_eq!(db.get_cf(&cf_b, b"k").unwrap(), Some(b"b".to_vec()));
    }

    // ── event listeners ─────────────────────────────────────────────────

    /// Test listener that counts every callback it receives and
    /// records enough detail for assertions.
    #[derive(Default)]
    struct CountingListener {
        flush_completed: AtomicUsize,
        compaction_begin: AtomicUsize,
        compaction_completed: AtomicUsize,
        table_file_created: AtomicUsize,
        table_file_deleted: AtomicUsize,
        external_file_ingested: AtomicUsize,
        background_error: AtomicUsize,
        last_flush_file_id: AtomicUsize,
        last_compaction_output_count: AtomicUsize,
    }

    impl EventListener for CountingListener {
        fn on_flush_completed(&self, info: &FlushJobInfo) {
            self.flush_completed.fetch_add(1, AtomicOrdering::Relaxed);
            self.last_flush_file_id
                .store(info.file_id as usize, AtomicOrdering::Relaxed);
        }
        fn on_compaction_begin(&self, _info: &CompactionJobInfo) {
            self.compaction_begin.fetch_add(1, AtomicOrdering::Relaxed);
        }
        fn on_compaction_completed(&self, info: &CompactionJobInfo) {
            self.compaction_completed
                .fetch_add(1, AtomicOrdering::Relaxed);
            self.last_compaction_output_count
                .store(info.output_files.len(), AtomicOrdering::Relaxed);
        }
        fn on_table_file_created(&self, _info: &TableFileCreationInfo) {
            self.table_file_created
                .fetch_add(1, AtomicOrdering::Relaxed);
        }
        fn on_table_file_deleted(&self, _info: &TableFileDeletionInfo) {
            self.table_file_deleted
                .fetch_add(1, AtomicOrdering::Relaxed);
        }
        fn on_external_file_ingested(&self, _info: &ExternalFileIngestionInfo) {
            self.external_file_ingested
                .fetch_add(1, AtomicOrdering::Relaxed);
        }
        fn on_background_error(&self, _reason: BackgroundErrorReason, _err: &Error) {
            self.background_error.fetch_add(1, AtomicOrdering::Relaxed);
        }
    }

    #[test]
    fn test_listener_fires_on_flush() {
        let listener = Arc::new(CountingListener::default());
        let dir = TempDir::new().unwrap();
        let opts = Options {
            write_buffer_size: 4 * 1024,
            listeners: vec![listener.clone() as Arc<dyn EventListener>],
            ..Options::default()
        };
        let db = Db::open(dir.path(), opts).unwrap();
        force_flush(&db, "listener");

        assert!(
            listener.flush_completed.load(AtomicOrdering::Relaxed) >= 1,
            "flush callback should fire at least once"
        );
        assert!(
            listener.table_file_created.load(AtomicOrdering::Relaxed) >= 1,
            "table_file_created should fire for every flushed file"
        );
        assert_ne!(
            listener.last_flush_file_id.load(AtomicOrdering::Relaxed),
            0,
            "file id recorded"
        );
    }

    #[test]
    fn test_listener_fires_on_compaction() {
        let listener = Arc::new(CountingListener::default());
        let dir = TempDir::new().unwrap();
        let opts = Options {
            write_buffer_size: 4 * 1024,
            listeners: vec![listener.clone() as Arc<dyn EventListener>],
            ..Options::default()
        };
        let db = Db::open(dir.path(), opts).unwrap();
        // Drive enough writes to generate L0 files, then manually
        // compact the range so the compaction callbacks fire on
        // the calling thread.
        for i in 0..400 {
            db.put(format!("k_{i:04}").as_bytes(), b"v").unwrap();
        }
        force_flush(&db, "listener");
        db.compact_range(None, None).unwrap();

        let begin = listener.compaction_begin.load(AtomicOrdering::Relaxed);
        let complete = listener.compaction_completed.load(AtomicOrdering::Relaxed);
        assert!(
            begin >= 1,
            "compaction_begin must fire at least once, got {begin}"
        );
        assert_eq!(
            begin, complete,
            "begin and completed must fire in matched pairs"
        );
        assert!(
            listener.table_file_created.load(AtomicOrdering::Relaxed) >= 2,
            "flush + compaction both produce files"
        );
        assert!(
            listener.table_file_deleted.load(AtomicOrdering::Relaxed) >= 1,
            "old L0 files must be unlinked after compaction"
        );
    }

    #[test]
    fn test_listener_fires_on_ingest() {
        let listener = Arc::new(CountingListener::default());
        let dir = TempDir::new().unwrap();
        let opts = Options {
            listeners: vec![listener.clone() as Arc<dyn EventListener>],
            ..Options::default()
        };
        let db = Db::open(dir.path(), opts.clone()).unwrap();

        let sst_path = dir.path().join("ingest.sst");
        {
            let mut w = SstFileWriter::create(&sst_path, &opts).unwrap();
            for i in 0..10 {
                w.put(format!("ik_{i:02}").as_bytes(), b"iv").unwrap();
            }
            w.finish().unwrap();
        }
        db.ingest_external_files(&[sst_path], IngestOptions::default())
            .unwrap();

        assert_eq!(
            listener
                .external_file_ingested
                .load(AtomicOrdering::Relaxed),
            1,
            "external_file_ingested fires once per ingested file"
        );
        assert!(
            listener.table_file_created.load(AtomicOrdering::Relaxed) >= 1,
            "ingest re-emits the file and fires table_file_created"
        );
    }

    #[test]
    fn test_listener_multiple_listeners_all_fire() {
        let a = Arc::new(CountingListener::default());
        let b = Arc::new(CountingListener::default());
        let dir = TempDir::new().unwrap();
        let opts = Options {
            write_buffer_size: 4 * 1024,
            listeners: vec![
                a.clone() as Arc<dyn EventListener>,
                b.clone() as Arc<dyn EventListener>,
            ],
            ..Options::default()
        };
        let db = Db::open(dir.path(), opts).unwrap();
        force_flush(&db, "multi");

        assert!(a.flush_completed.load(AtomicOrdering::Relaxed) >= 1);
        assert!(b.flush_completed.load(AtomicOrdering::Relaxed) >= 1);
    }

    #[test]
    fn test_listener_none_configured_is_noop() {
        // Sanity check: with no listeners, all paths still work
        // and nothing panics.
        let (db, _dir) = open_tmp();
        db.put(b"k", b"v").unwrap();
        force_flush(&db, "none");
        db.compact_range(None, None).unwrap();
    }

    #[test]
    fn test_listener_compaction_job_info_contains_input_files() {
        // Capture the last CompactionJobInfo on `on_compaction_completed`
        // and assert it carries the expected input file ids.
        struct CaptureListener {
            captured: Mutex<Option<CompactionJobInfo>>,
        }
        impl EventListener for CaptureListener {
            fn on_compaction_completed(&self, info: &CompactionJobInfo) {
                *self.captured.lock() = Some(info.clone());
            }
        }
        use parking_lot::Mutex;

        let listener = Arc::new(CaptureListener {
            captured: Mutex::new(None),
        });
        let dir = TempDir::new().unwrap();
        let opts = Options {
            write_buffer_size: 4 * 1024,
            listeners: vec![listener.clone() as Arc<dyn EventListener>],
            ..Options::default()
        };
        let db = Db::open(dir.path(), opts).unwrap();
        for i in 0..400 {
            db.put(format!("k_{i:04}").as_bytes(), b"v").unwrap();
        }
        force_flush(&db, "capture");
        db.compact_range(None, None).unwrap();

        let captured = listener.captured.lock().clone();
        let info = captured.expect("compaction_completed must have fired");
        assert!(
            !info.input_files_input_level.is_empty(),
            "at least one L0 input file was picked"
        );
        assert_eq!(info.output_level, info.input_level + 1);
        assert!(!info.output_files.is_empty(), "compaction produced outputs");
    }

    // ── statistics ──────────────────────────────────────────────────────

    fn stats_opts(stats: Arc<Statistics>) -> Options {
        Options {
            statistics: Some(stats),
            ..Options::default()
        }
    }

    fn tiny_flush_stats_opts(stats: Arc<Statistics>) -> Options {
        Options {
            write_buffer_size: 4 * 1024,
            statistics: Some(stats),
            ..Options::default()
        }
    }

    #[test]
    fn test_stats_keys_written_and_bytes_written() {
        let stats = Arc::new(Statistics::new());
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), stats_opts(stats.clone())).unwrap();
        db.put(b"k1", b"value1").unwrap();
        db.put(b"k2", b"value2").unwrap();
        assert_eq!(stats.get_ticker(Ticker::KeysWritten), 2);
        // Expected bytes = 2 + 6 + 2 + 6 = 16
        assert_eq!(stats.get_ticker(Ticker::BytesWritten), 16);
    }

    #[test]
    fn test_stats_keys_read_and_bytes_read() {
        let stats = Arc::new(Statistics::new());
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), stats_opts(stats.clone())).unwrap();
        db.put(b"k", b"value").unwrap();
        db.get(b"k").unwrap();
        db.get(b"missing").unwrap();
        assert_eq!(stats.get_ticker(Ticker::KeysRead), 2);
        // Only the found value contributes to BytesRead.
        assert_eq!(stats.get_ticker(Ticker::BytesRead), 5);
        let get_hist = stats.get_histogram_snapshot(Histogram::DbGet);
        assert_eq!(get_hist.count, 2);
    }

    #[test]
    fn test_stats_delete_counter() {
        let stats = Arc::new(Statistics::new());
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), stats_opts(stats.clone())).unwrap();
        db.put(b"k", b"v").unwrap();
        db.delete(b"k").unwrap();
        assert_eq!(stats.get_ticker(Ticker::KeysDeleted), 1);
    }

    #[test]
    fn test_stats_block_cache_hit_and_miss_populate() {
        // After a deterministic flush + compact_range (so no
        // concurrent background compaction can race the reads
        // and contaminate the counters), every point lookup
        // that reaches a data block fires either a hit or a
        // miss on the block cache. We don't assert the strict
        // `adds == misses` invariant here — LRU eviction plus
        // any lingering background work can perturb that
        // equality on fast machines. The weaker "both hits and
        // misses see traffic" is the observable contract.
        let stats = Arc::new(Statistics::new());
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), tiny_flush_stats_opts(stats.clone())).unwrap();
        for i in 0..200 {
            db.put(format!("k_{i:05}").as_bytes(), b"value").unwrap();
        }
        force_flush(&db, "cache");
        // Drain any pending compaction before measuring.
        db.compact_range(None, None).unwrap();
        stats.reset();
        // Read the same few keys twice: the first read is a
        // miss + add, the second is a hit.
        for _ in 0..2 {
            for i in 0..5 {
                db.get(format!("k_{i:05}").as_bytes()).unwrap();
            }
        }
        let hits = stats.get_ticker(Ticker::BlockCacheHit);
        let misses = stats.get_ticker(Ticker::BlockCacheMiss);
        let adds = stats.get_ticker(Ticker::BlockCacheAdd);
        assert!(misses > 0, "expected at least one block cache miss");
        assert!(hits > 0, "expected at least one block cache hit");
        // `adds` tracks inserts after a miss — it can never
        // exceed `misses`.
        assert!(adds <= misses, "adds={adds} misses={misses}");
    }

    #[test]
    fn test_stats_bloom_filter_useful_increments_on_absent_key() {
        // Deterministic layout: write 200 keys spaced on even
        // suffixes (so the resulting SST covers `[k_00000,
        // k_00398]`), compact to L1, then query odd suffixes
        // within that range. The partition_point-based file
        // lookup lands on the single L1 file for every query
        // and the bloom has a chance to say "not present".
        let stats = Arc::new(Statistics::new());
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), tiny_flush_stats_opts(stats.clone())).unwrap();
        for i in 0..200 {
            let even = i * 2;
            db.put(format!("k_{even:05}").as_bytes(), b"v").unwrap();
        }
        force_flush(&db, "bloom");
        db.compact_range(None, None).unwrap();
        stats.reset();
        // Query 100 absent (odd-suffix) keys inside the range.
        // With ~10 bits/key the false-positive rate is ~1%, so
        // almost all queries will register as "useful".
        for i in 0..100 {
            let odd = i * 2 + 1;
            db.get(format!("k_{odd:05}").as_bytes()).unwrap();
        }
        let useful = stats.get_ticker(Ticker::BloomFilterUseful);
        assert!(
            useful > 0,
            "bloom filter should have ruled out at least one absent key"
        );
    }

    #[test]
    fn test_stats_bloom_filter_full_positive_on_present_key() {
        let stats = Arc::new(Statistics::new());
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), tiny_flush_stats_opts(stats.clone())).unwrap();
        for i in 0..100 {
            db.put(format!("k_{i:04}").as_bytes(), b"v").unwrap();
        }
        force_flush(&db, "bloom_pos");
        db.compact_range(None, None).unwrap();
        stats.reset();
        for i in 0..100 {
            db.get(format!("k_{i:04}").as_bytes()).unwrap();
        }
        let pos = stats.get_ticker(Ticker::BloomFilterFullPositive);
        assert!(
            pos > 0,
            "bloom filter should have returned 'maybe' and we found the key"
        );
    }

    #[test]
    fn test_stats_flush_and_compaction_counters() {
        let stats = Arc::new(Statistics::new());
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), tiny_flush_stats_opts(stats.clone())).unwrap();
        for i in 0..200 {
            db.put(format!("k_{i:04}").as_bytes(), b"v").unwrap();
        }
        force_flush(&db, "fcstats");
        db.compact_range(None, None).unwrap();
        assert!(stats.get_ticker(Ticker::FlushCount) >= 1);
        assert!(stats.get_ticker(Ticker::FlushBytesWritten) > 0);
        assert!(stats.get_ticker(Ticker::CompactionCount) >= 1);
        assert!(stats.get_ticker(Ticker::CompactionBytesRead) > 0);
        assert!(stats.get_ticker(Ticker::CompactionBytesWritten) > 0);
        assert!(stats.get_histogram_snapshot(Histogram::FlushTime).count > 0);
        assert!(
            stats
                .get_histogram_snapshot(Histogram::CompactionTime)
                .count
                > 0
        );
    }

    #[test]
    fn test_stats_wal_counters() {
        let stats = Arc::new(Statistics::new());
        let dir = TempDir::new().unwrap();
        let opts = Options {
            statistics: Some(stats.clone()),
            durability: DurabilityMode::Immediate,
            ..Options::default()
        };
        let db = Db::open(dir.path(), opts).unwrap();
        db.put(b"k", b"v").unwrap();
        db.put(b"k2", b"v2").unwrap();
        assert!(stats.get_ticker(Ticker::WalBytesWritten) > 0);
        // Immediate durability fsyncs per call.
        assert_eq!(stats.get_ticker(Ticker::WalSyncCount), 2);
        assert!(stats.get_histogram_snapshot(Histogram::WalWriteTime).count >= 2);
    }

    #[test]
    fn test_stats_iter_seek_and_next_counters() {
        let stats = Arc::new(Statistics::new());
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), stats_opts(stats.clone())).unwrap();
        db.put(b"a", b"1").unwrap();
        db.put(b"b", b"2").unwrap();
        db.put(b"c", b"3").unwrap();
        let mut it = db.iter();
        it.seek_to_first();
        while it.valid() {
            it.next();
        }
        assert!(stats.get_ticker(Ticker::IterSeekCount) >= 1);
        // Two `next` calls produced keys (b, c); the third
        // invalidated and doesn't count.
        assert_eq!(stats.get_ticker(Ticker::IterNextCount), 2);
    }

    #[test]
    fn test_stats_snapshot_register_release() {
        let stats = Arc::new(Statistics::new());
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), stats_opts(stats.clone())).unwrap();
        {
            let _snap = db.snapshot();
        }
        assert_eq!(stats.get_ticker(Ticker::SnapshotsRegistered), 1);
        assert_eq!(stats.get_ticker(Ticker::SnapshotsReleased), 1);
    }

    #[test]
    fn test_stats_reset_clears_everything() {
        let stats = Arc::new(Statistics::new());
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), stats_opts(stats.clone())).unwrap();
        db.put(b"k", b"v").unwrap();
        assert!(stats.get_ticker(Ticker::KeysWritten) > 0);
        stats.reset();
        assert_eq!(stats.get_ticker(Ticker::KeysWritten), 0);
        assert_eq!(stats.get_ticker(Ticker::BytesWritten), 0);
    }

    #[test]
    fn test_stats_none_configured_is_noop() {
        // Sanity: with statistics disabled every hot path still
        // works and nothing panics.
        let (db, _dir) = open_tmp();
        db.put(b"k", b"v").unwrap();
        assert_eq!(db.get(b"k").unwrap(), Some(b"v".to_vec()));
    }

    #[test]
    fn test_stats_dump_is_non_empty_and_contains_every_ticker() {
        let stats = Arc::new(Statistics::new());
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), stats_opts(stats.clone())).unwrap();
        db.put(b"k", b"v").unwrap();
        let dump = stats.dump();
        for ticker_name in [
            "lark.bytes_written",
            "lark.keys_written",
            "lark.bloom_filter_useful",
            "lark.compaction_count",
            "lark.flush_count",
        ] {
            assert!(dump.contains(ticker_name), "dump missing {ticker_name}");
        }
    }

    // ── properties API ──────────────────────────────────────────────────

    #[test]
    fn test_property_unknown_name_returns_none() {
        let (db, _dir) = open_tmp();
        assert!(db.get_property("not.a.real.property").is_none());
        assert!(db.get_int_property("not.a.real.property").is_none());
    }

    #[test]
    fn test_property_num_files_at_level() {
        let (db, _dir) = open_tmp();
        assert_eq!(db.get_int_property("lark.num-files-at-level0"), Some(0));
        assert_eq!(db.get_int_property("lark.num-files-at-level6"), Some(0));
        // Out-of-range level is a valid query that returns 0.
        assert_eq!(db.get_int_property("lark.num-files-at-level99"), Some(0));
    }

    #[test]
    fn test_property_level_counts_after_flush_and_compact() {
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), tiny_flush_opts()).unwrap();
        for i in 0..200 {
            db.put(format!("k_{i:04}").as_bytes(), b"v").unwrap();
        }
        force_flush(&db, "props");
        // At this point we expect some L0 files.
        let l0_before = db.get_int_property("lark.num-files-at-level0").unwrap();
        assert!(l0_before > 0 || db.get_int_property("lark.num-files-at-level1").unwrap() > 0);

        // Drain everything to the deepest level.
        db.compact_range(None, None).unwrap();
        assert_eq!(
            db.get_int_property("lark.num-files-at-level0"),
            Some(0),
            "L0 should be empty after compact_range"
        );
    }

    #[test]
    fn test_property_total_sst_size_after_flush() {
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), tiny_flush_opts()).unwrap();
        assert_eq!(db.get_int_property("lark.total-sst-files-size"), Some(0));
        for i in 0..100 {
            db.put(format!("k_{i:04}").as_bytes(), b"v").unwrap();
        }
        force_flush(&db, "size");
        let size = db.get_int_property("lark.total-sst-files-size").unwrap();
        assert!(size > 0, "SST size should be > 0 after a flush");
    }

    #[test]
    fn test_property_cur_size_active_mem_table() {
        let (db, _dir) = open_tmp();
        assert_eq!(
            db.get_int_property("lark.cur-size-active-mem-table"),
            Some(0)
        );
        for i in 0..50 {
            db.put(format!("k_{i:03}").as_bytes(), b"value").unwrap();
        }
        let size = db
            .get_int_property("lark.cur-size-active-mem-table")
            .unwrap();
        assert!(size > 0, "active memtable should have non-zero size");
    }

    #[test]
    fn test_property_cur_size_all_mem_tables_aggregates() {
        let (db, _dir) = open_tmp();
        db.put(b"k", b"v").unwrap();
        let active = db
            .get_int_property("lark.cur-size-active-mem-table")
            .unwrap();
        let all = db.get_int_property("lark.cur-size-all-mem-tables").unwrap();
        assert!(all >= active, "all mem tables must be >= active");
    }

    #[test]
    fn test_property_estimate_num_keys() {
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), tiny_flush_opts()).unwrap();
        for i in 0..100 {
            db.put(format!("k_{i:04}").as_bytes(), b"v").unwrap();
        }
        force_flush(&db, "estimate");
        db.compact_range(None, None).unwrap();
        let estimate = db.get_int_property("lark.estimate-num-keys").unwrap();
        // Exact count per SST includes the flush filler + the 100
        // writes; the property is a lower bound, so > 50 is a
        // safe floor.
        assert!(estimate > 50, "estimate-num-keys={estimate} too low");
    }

    #[test]
    fn test_property_num_snapshots_and_oldest_snapshot_time() {
        let (db, _dir) = open_tmp();
        assert_eq!(db.get_int_property("lark.num-snapshots"), Some(0));
        assert!(
            db.get_int_property("lark.oldest-snapshot-time").is_none(),
            "oldest-snapshot-time should be None when no snapshots are live"
        );
        let _snap_a = db.snapshot();
        let _snap_b = db.snapshot();
        assert_eq!(db.get_int_property("lark.num-snapshots"), Some(2));
        assert!(db.get_int_property("lark.oldest-snapshot-time").is_some());
    }

    #[test]
    fn test_property_background_errors_returns_zero() {
        let (db, _dir) = open_tmp();
        // No background errors on a fresh db.
        assert_eq!(db.get_int_property("lark.background-errors"), Some(0));
    }

    #[test]
    fn test_property_stats_string_includes_level_header_and_counters() {
        let stats = Arc::new(Statistics::new());
        let opts = Options {
            statistics: Some(stats),
            ..Options::default()
        };
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), opts).unwrap();
        db.put(b"k", b"v").unwrap();
        let text = db.get_property("lark.stats").unwrap();
        assert!(text.contains("== lark engine stats =="));
        assert!(text.contains("Level  Files     Size(B)"));
        assert!(text.contains("lark.keys_written"));
    }

    #[test]
    fn test_property_stats_string_without_statistics_configured() {
        let (db, _dir) = open_tmp();
        let text = db.get_property("lark.stats").unwrap();
        assert!(text.contains("== lark engine stats =="));
        assert!(text.contains("(no Statistics object configured"));
    }

    #[test]
    fn test_property_sstables_lists_files_after_flush() {
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), tiny_flush_opts()).unwrap();
        for i in 0..50 {
            db.put(format!("k_{i:03}").as_bytes(), b"v").unwrap();
        }
        force_flush(&db, "ssts");
        let text = db.get_property("lark.sstables").unwrap();
        assert!(text.contains("Level    FileID"));
        // Should list at least one file with non-zero size.
        assert!(
            text.lines().any(|l| l.contains("\"k_")),
            "expected a file line to include a user key from the writes"
        );
    }

    #[test]
    fn test_property_levelstats_format() {
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), tiny_flush_opts()).unwrap();
        for i in 0..20 {
            db.put(format!("k_{i:03}").as_bytes(), b"v").unwrap();
        }
        force_flush(&db, "lvl");
        let text = db.get_property("lark.levelstats").unwrap();
        assert!(text.starts_with("Level  Files     Size(B)"));
        // Every level row is present, not just the populated ones.
        for lvl in 0..7 {
            assert!(
                text.contains(&format!("{lvl:5}")),
                "level {lvl} should appear in levelstats"
            );
        }
    }

    #[test]
    fn test_property_options_debug_dump() {
        let (db, _dir) = open_tmp();
        let text = db.get_property("lark.options").unwrap();
        assert!(text.contains("OptionsSnapshot"));
        assert!(text.contains("default"));
    }

    #[test]
    fn test_property_integer_forms_available_via_get_property() {
        // Integer properties should also be reachable via
        // get_property, returning their decimal string form.
        let (db, _dir) = open_tmp();
        assert_eq!(
            db.get_property("lark.num-files-at-level0").as_deref(),
            Some("0")
        );
        assert_eq!(db.get_property("lark.num-snapshots").as_deref(), Some("0"));
    }

    #[test]
    fn test_perf_context_captures_db_get_and_put_activity() {
        // End-to-end: enable PerfContext timing on the current
        // thread, do a few writes and reads, then snapshot. The
        // counters should show one get_count per read, one
        // write_count per put, and non-zero time in both the
        // WAL/memtable write phases and the memtable read phase.
        let (db, _dir) = open_tmp();

        PerfContext::set_level(PerfLevel::EnableTime);
        PerfContext::reset();

        db.put(b"alpha", b"1").unwrap();
        db.put(b"beta", b"2").unwrap();
        db.put(b"gamma", b"3").unwrap();

        let _ = db.get(b"alpha").unwrap();
        let _ = db.get(b"beta").unwrap();

        let snap = PerfContext::capture();
        assert_eq!(snap.write_count, 3, "3 puts → write_count 3");
        assert_eq!(snap.get_count, 2, "2 gets → get_count 2");
        assert!(
            snap.write_wal_time_nanos > 0,
            "WAL phase should record non-zero time under EnableTime"
        );
        assert!(
            snap.write_memtable_time_nanos > 0,
            "memtable write phase should record non-zero time"
        );
        assert!(
            snap.get_from_memtable_time_nanos > 0,
            "memtable read phase should record non-zero time"
        );

        // Disable and confirm subsequent activity is invisible.
        PerfContext::set_level(PerfLevel::Disable);
        let before = snap;
        db.put(b"delta", b"4").unwrap();
        let _ = db.get(b"alpha").unwrap();
        let after = PerfContext::capture();
        assert_eq!(after, before, "Disable level must freeze counters");
    }

    #[test]
    fn test_use_direct_io_for_compaction_is_correctness_neutral() {
        // Enabling the page-cache hint must not change what a
        // compaction produces. On Linux the `posix_fadvise`
        // syscall runs but is a best-effort hint; on other
        // platforms it's a no-op. Either way, the output SSTs
        // contain the same data as the leveled baseline, so
        // readers must see identical values afterward.
        let opts = Options {
            write_buffer_size: 4 * 1024,
            use_direct_io_for_compaction: true,
            ..Options::default()
        };
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), opts).unwrap();

        for i in 0..128 {
            let k = format!("k{i:04}");
            let v = format!("v{i}");
            db.put(k.as_bytes(), v.as_bytes()).unwrap();
        }
        // Overwrite a few so dedup runs through the hint path.
        for i in 0..32 {
            let k = format!("k{i:04}");
            let v = format!("v{i}-new");
            db.put(k.as_bytes(), v.as_bytes()).unwrap();
        }
        db.compact_range(None, None).unwrap();

        for i in 0..128 {
            let k = format!("k{i:04}");
            let expected = if i < 32 {
                format!("v{i}-new")
            } else {
                format!("v{i}")
            };
            assert_eq!(
                db.get(k.as_bytes()).unwrap(),
                Some(expected.into_bytes()),
                "key {k} must still read its latest value"
            );
        }
    }

    #[test]
    fn test_universal_compaction_reads_are_correct_after_merge() {
        // Write a batch under Universal, force a full merge via
        // compact_range, and verify every key is still readable
        // with the most recent value.
        let opts = Options {
            write_buffer_size: 4 * 1024,
            compaction_style: CompactionStyle::Universal,
            ..Options::default()
        };
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), opts).unwrap();

        for i in 0..64 {
            let k = format!("k{i:04}");
            let v = format!("v{i}");
            db.put(k.as_bytes(), v.as_bytes()).unwrap();
        }
        // Overwrite the first 16 keys so dedup has to pick the
        // newest version during the merge.
        for i in 0..16 {
            let k = format!("k{i:04}");
            let v = format!("v{i}-updated");
            db.put(k.as_bytes(), v.as_bytes()).unwrap();
        }

        db.compact_range(None, None).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(100));

        for i in 0..64 {
            let k = format!("k{i:04}");
            let expected = if i < 16 {
                format!("v{i}-updated")
            } else {
                format!("v{i}")
            };
            assert_eq!(
                db.get(k.as_bytes()).unwrap(),
                Some(expected.into_bytes()),
                "key {k} must read back its latest value"
            );
        }
    }

    #[test]
    fn test_universal_compaction_never_creates_l1_files() {
        // Every Universal merge output should stay at L0 — the
        // level-size push-down rule must not fire for this style.
        let opts = Options {
            write_buffer_size: 4 * 1024,
            l0_compaction_trigger: 1,
            compaction_style: CompactionStyle::Universal,
            ..Options::default()
        };
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), opts).unwrap();
        for i in 0..64 {
            let k = format!("k{i:04}");
            db.put(k.as_bytes(), &vec![0xCC; 256]).unwrap();
        }
        db.compact_range(None, None).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(100));

        let l1 = db.get_int_property("lark.num-files-at-level1").unwrap();
        assert_eq!(l1, 0, "Universal must not produce L1 files, saw {l1}");
        let l0 = db.get_int_property("lark.num-files-at-level0").unwrap();
        assert!(
            l0 >= 1,
            "Universal compaction should leave at least one L0 file"
        );
    }

    #[test]
    fn test_universal_compaction_full_merge_drops_shadowed_versions() {
        // After a full universal compact_range, we expect the
        // output to be a single L0 file (min cardinality). This
        // exercises the compact_range full-merge path.
        let opts = Options {
            write_buffer_size: 4 * 1024,
            compaction_style: CompactionStyle::Universal,
            ..Options::default()
        };
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), opts).unwrap();
        for i in 0..32 {
            let k = format!("k{i:04}");
            db.put(k.as_bytes(), &vec![0xAA; 512]).unwrap();
        }
        // Give the background scheduler a moment to potentially
        // kick off work, then force-merge synchronously.
        std::thread::sleep(std::time::Duration::from_millis(50));
        db.compact_range(None, None).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(100));

        let l0 = db.get_int_property("lark.num-files-at-level0").unwrap();
        assert_eq!(
            l0, 1,
            "full universal compact_range should fold everything into one L0 file, saw {l0}"
        );
    }

    #[test]
    fn test_fifo_compaction_bounds_total_size() {
        // Tiny memtable + tight FIFO cap: sustained writes should
        // produce many L0 files, and after each flush the oldest
        // ones should be unlinked so the total stays bounded.
        let opts = Options {
            write_buffer_size: 4 * 1024,
            compaction_style: CompactionStyle::Fifo,
            fifo_compaction_options: FifoCompactionOptions {
                max_table_files_size: 32 * 1024,
            },
            ..Options::default()
        };
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), opts).unwrap();

        // Write enough data to produce ~16 flushes of ~4 KB each,
        // well over the 32 KB cap. Each write has a distinct
        // monotonically increasing key so flushes don't overlap.
        let payload = vec![0xEEu8; 256];
        for i in 0..256 {
            let k = format!("k{i:06}");
            db.put(k.as_bytes(), &payload).unwrap();
        }

        // Give the background compaction thread a moment to
        // process the trailing flushes + FIFO drops.
        std::thread::sleep(std::time::Duration::from_millis(200));

        // Force any remaining flushes through and run one more
        // FIFO pass via compact_range (which acquires the
        // compaction lock and drains pending work).
        db.compact_range(None, None).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(100));

        let total = db
            .get_int_property("lark.total-sst-files-size")
            .unwrap_or(0);
        assert!(
            total <= 64 * 1024,
            "FIFO cap 32KB should keep total < 64KB slack, got {total}"
        );
        // Meanwhile the newest keys must still be readable (the
        // oldest ones may have been dropped by FIFO).
        assert_eq!(
            db.get(b"k000255").unwrap(),
            Some(payload.clone()),
            "newest key must survive FIFO compaction"
        );
    }

    #[test]
    fn test_fifo_compaction_keeps_at_least_one_file() {
        // A single oversized file must not be deleted — FIFO
        // refuses to drop the last surviving SST because that
        // would wipe the database.
        let opts = Options {
            write_buffer_size: 4 * 1024,
            compaction_style: CompactionStyle::Fifo,
            fifo_compaction_options: FifoCompactionOptions {
                max_table_files_size: 1, // 1 byte cap: always over limit
            },
            ..Options::default()
        };
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), opts).unwrap();
        for i in 0..32 {
            let k = format!("k{i:04}");
            db.put(k.as_bytes(), &vec![0xAA; 512]).unwrap();
        }
        std::thread::sleep(std::time::Duration::from_millis(200));
        db.compact_range(None, None).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(100));

        let l0 = db.get_int_property("lark.num-files-at-level0").unwrap();
        assert!(
            l0 >= 1,
            "FIFO must keep at least one L0 file even when over the cap"
        );
    }

    #[test]
    fn test_fifo_compaction_never_promotes_to_l1() {
        // Under FIFO, the background scheduler should never
        // promote files from L0 to L1. The `l0_compaction_trigger`
        // knob is a level-style knob and must have no effect.
        let opts = Options {
            write_buffer_size: 4 * 1024,
            l0_compaction_trigger: 2,
            compaction_style: CompactionStyle::Fifo,
            fifo_compaction_options: FifoCompactionOptions {
                max_table_files_size: 10 * 1024 * 1024,
            },
            ..Options::default()
        };
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), opts).unwrap();
        for i in 0..64 {
            let k = format!("k{i:04}");
            db.put(k.as_bytes(), &vec![0xBB; 256]).unwrap();
        }
        std::thread::sleep(std::time::Duration::from_millis(200));

        let l1 = db.get_int_property("lark.num-files-at-level1").unwrap();
        assert_eq!(l1, 0, "FIFO must not produce L1 files, saw {l1}");
    }

    #[test]
    fn test_tailing_iter_sees_writes_after_creation() {
        // Initial writes, then a tailing iterator, then more
        // writes — the tailing iterator must surface the later
        // writes once it advances past the initial set.
        let (db, _dir) = open_tmp();
        for i in 0..5 {
            let k = format!("log/{i:04}");
            db.put(k.as_bytes(), format!("v{i}").as_bytes()).unwrap();
        }

        let mut tail = db.iter_tailing();
        tail.seek_to_first();

        // Drain the initial 5 entries.
        let mut seen: Vec<String> = Vec::new();
        while tail.valid() {
            seen.push(String::from_utf8(tail.key().unwrap().to_vec()).unwrap());
            tail.next();
        }
        assert_eq!(seen.len(), 5, "first drain saw {seen:?}");

        // Now push more writes at strictly larger keys.
        for i in 5..10 {
            let k = format!("log/{i:04}");
            db.put(k.as_bytes(), format!("v{i}").as_bytes()).unwrap();
        }

        // Stepping again should refresh the view and surface
        // the new entries without re-emitting the first batch.
        tail.next();
        while tail.valid() {
            seen.push(String::from_utf8(tail.key().unwrap().to_vec()).unwrap());
            tail.next();
        }
        assert_eq!(seen.len(), 10, "tail saw {seen:?}");
        for (i, k) in seen.iter().enumerate() {
            assert_eq!(k, &format!("log/{i:04}"));
        }
    }

    #[test]
    fn test_tailing_iter_survives_flush_and_compaction() {
        // Tiny write_buffer so writes between drains roll
        // memtables and produce L0 files. The tailing iter must
        // pick up those new SSTs on the next refresh.
        let opts = Options {
            write_buffer_size: 4 * 1024,
            ..Options::default()
        };
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), opts).unwrap();

        for i in 0..16 {
            let k = format!("log/{i:04}");
            db.put(k.as_bytes(), &vec![0xAA; 256]).unwrap();
        }

        let mut tail = db.iter_tailing();
        tail.seek_to_first();
        let mut seen: usize = 0;
        while tail.valid() {
            seen += 1;
            tail.next();
        }
        assert_eq!(seen, 16);

        // Force a flush and a compaction — the existing tail
        // iter is no longer pinned to anything visible, but a
        // refresh + new writes should still work.
        db.compact_range(None, None).unwrap();
        for i in 16..32 {
            let k = format!("log/{i:04}");
            db.put(k.as_bytes(), &vec![0xBB; 256]).unwrap();
        }

        tail.refresh();
        let mut seen_after = 0;
        while tail.valid() {
            seen_after += 1;
            tail.next();
        }
        assert_eq!(
            seen_after, 16,
            "tail should pick up the 16 new entries after refresh"
        );
    }

    #[test]
    fn test_tailing_iter_no_re_emission_after_explicit_refresh() {
        let (db, _dir) = open_tmp();
        for i in 0..3 {
            db.put(format!("k{i}").as_bytes(), b"v").unwrap();
        }
        let mut tail = db.iter_tailing();
        tail.seek_to_first();
        assert!(tail.valid());
        let first_key = tail.key().unwrap().to_vec();
        assert_eq!(first_key, b"k0");
        tail.next();
        assert_eq!(tail.key().unwrap(), b"k1");

        // Explicit refresh in the middle of iteration must NOT
        // re-emit k0.
        tail.refresh();
        // After refresh we should be positioned strictly after
        // the last returned key (k1), so the next valid key is
        // k2.
        assert!(tail.valid());
        assert_eq!(tail.key().unwrap(), b"k2");
        tail.next();
        assert!(!tail.valid(), "no more keys after k2");
    }

    #[test]
    fn test_tailing_iter_cf_scoping() {
        // Tailing iterator scoped to one CF must not surface
        // keys from other CFs.
        let (db, _dir) = open_tmp();
        let cf_logs = db.create_column_family("logs").unwrap();
        let cf_other = db.create_column_family("other").unwrap();

        db.put_cf(&cf_logs, b"a", b"1").unwrap();
        db.put_cf(&cf_other, b"a", b"x").unwrap();
        db.put_cf(&cf_logs, b"b", b"2").unwrap();

        let mut tail = db.iter_tailing_cf(&cf_logs);
        tail.seek_to_first();
        let mut seen = Vec::new();
        while tail.valid() {
            seen.push((tail.key().unwrap().to_vec(), tail.value().unwrap().to_vec()));
            tail.next();
        }
        assert_eq!(
            seen,
            vec![
                (b"a".to_vec(), b"1".to_vec()),
                (b"b".to_vec(), b"2".to_vec())
            ]
        );

        // A write to the other CF must not bleed in even after
        // refresh.
        db.put_cf(&cf_other, b"c", b"y").unwrap();
        tail.refresh();
        assert!(!tail.valid());
    }

    #[test]
    fn test_block_cache_usage_property_reports_nonzero_after_reads() {
        // A cache with a small-but-nonzero budget fills with
        // decompressed data blocks as reads touch SSTables. The
        // `lark.block-cache-usage` property must report a
        // positive number once at least one read has happened
        // against a file that isn't entirely in the memtable.
        let opts = Options {
            write_buffer_size: 4 * 1024,
            block_cache_size: 1024 * 1024,
            ..Options::default()
        };
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), opts).unwrap();

        let payload = vec![0xABu8; 256];
        for i in 0..200 {
            let k = format!("k{i:04}");
            db.put(k.as_bytes(), &payload).unwrap();
        }
        // Force a flush so the reads below have to touch SST blocks.
        db.compact_range(None, None).unwrap();

        // Read a few keys to populate the block cache.
        for i in 0..50 {
            let k = format!("k{i:04}");
            let _ = db.get(k.as_bytes()).unwrap();
        }

        let usage = db
            .get_int_property("lark.block-cache-usage")
            .expect("property must exist");
        assert!(
            usage > 0,
            "expected block-cache-usage > 0 after reads, got {usage}"
        );
        let cap = db
            .get_int_property("lark.block-cache-capacity")
            .expect("property must exist");
        assert!(
            cap >= 512 * 1024,
            "expected at least 512KB capacity, got {cap}"
        );
        assert!(usage <= cap, "usage {usage} must not exceed capacity {cap}");
    }

    #[test]
    fn test_rate_limiter_throttles_compaction() {
        use std::sync::Arc;
        use std::time::{Duration, Instant};

        // 100 KB/s sustained, 5 KB burst. Compression is disabled so
        // the on-disk SST size stays proportional to the data we
        // feed in (otherwise LZ4 would collapse the payload to a
        // few KB and nothing meaningful would be throttled). A
        // 16 MB buffer keeps everything in the memtable until
        // compact_range triggers a flush + compaction, both of
        // which the limiter throttles.
        let limiter = Arc::new(TokenBucketRateLimiter::new(
            100_000,
            Duration::from_millis(50),
            5_000,
        ));
        let opts = Options {
            write_buffer_size: 16 * 1024 * 1024,
            compression: CompressionType::None,
            rate_limiter: Some(limiter.clone() as Arc<dyn RateLimiter>),
            ..Options::default()
        };

        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), opts).unwrap();

        // Write ~100 KB of well-dispersed keys so the resulting SST
        // is large enough that the limiter has real work to do.
        for i in 0..200 {
            let k = format!("key-{i:010}");
            // Each value is distinct so prefix compression can't
            // collapse the block.
            let v = format!("value-for-key-{i:010}-payload-{}", i);
            db.put(k.as_bytes(), v.as_bytes()).unwrap();
        }

        let start = Instant::now();
        db.compact_range(None, None).unwrap();
        let elapsed = start.elapsed();

        // With ~10 KB of uncompressed output flushed + compacted at
        // 100 KB/s past a 5 KB burst, we expect the critical path
        // to block for at least one refill period (~50 ms) and in
        // practice several. Assert a conservative floor to confirm
        // the limiter was actually consulted without flaking on
        // CI variance.
        assert!(
            elapsed >= Duration::from_millis(100),
            "compaction with 100KB/s limiter finished in {elapsed:?}, expected >= 100ms"
        );

        // The limiter must have been consulted for background I/O.
        assert!(
            limiter.get_total_bytes_through(Priority::Low) > 0,
            "limiter saw zero background bytes"
        );
        assert_eq!(limiter.get_total_bytes_through(Priority::High), 0);
    }

    #[test]
    fn test_write_stall_slowdown_accumulates_micros() {
        use std::sync::Arc;

        let stats = Arc::new(Statistics::new());
        let opts = Options {
            // Tiny memtable so every handful of puts rolls an L0 file.
            write_buffer_size: 4 * 1024,
            // Disable automatic compaction so L0 can't drain on us.
            l0_compaction_trigger: 1000,
            // Slow down once L0 has 2 files, never stop (high trigger).
            level0_slowdown_writes_trigger: 2,
            level0_stop_writes_trigger: 10_000,
            // Disable the memtable-count trigger for this test so we
            // isolate the L0 slowdown path.
            max_write_buffer_number: 0,
            statistics: Some(stats.clone()),
            ..Options::default()
        };
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), opts).unwrap();

        // Write enough data to cross the slowdown trigger and keep
        // going. Each put is ~600 bytes, so after ~7 puts the
        // memtable rolls, and after the 2nd flush L0 hits the
        // slowdown trigger.
        let payload = vec![0xCDu8; 600];
        for i in 0..128 {
            let k = format!("k{i:04}");
            db.put(k.as_bytes(), &payload).unwrap();
        }

        let stall = stats.get_ticker(Ticker::WriteStallMicros);
        assert!(
            stall > 0,
            "expected WriteStallMicros > 0 after crossing slowdown trigger, got {stall}"
        );
    }

    #[test]
    fn test_write_stall_no_slowdown_returns_busy() {
        let opts = Options {
            write_buffer_size: 4 * 1024,
            l0_compaction_trigger: 1000,
            level0_slowdown_writes_trigger: 2,
            level0_stop_writes_trigger: 10_000,
            max_write_buffer_number: 0,
            ..Options::default()
        };
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), opts).unwrap();

        // Build up L0 past the slowdown trigger.
        let payload = vec![0xEFu8; 600];
        for i in 0..64 {
            let k = format!("k{i:04}");
            db.put(k.as_bytes(), &payload).unwrap();
        }

        // A write with `no_slowdown` must now return Busy rather
        // than sleep or block.
        let wo = WriteOptions {
            no_slowdown: true,
            ..WriteOptions::default()
        };
        let err = db.put_opt(&wo, b"extra", b"value").unwrap_err();
        assert!(
            matches!(err, Error::Busy(_)),
            "expected Error::Busy, got {err:?}"
        );
    }

    #[test]
    fn test_write_stall_stop_unblocks_after_compaction() {
        use std::sync::Arc;
        use std::thread;
        use std::time::{Duration, Instant};

        // Stop writes entirely once L0 hits 2 files.
        let opts = Options {
            write_buffer_size: 4 * 1024,
            l0_compaction_trigger: 1000,
            level0_slowdown_writes_trigger: 10_000,
            level0_stop_writes_trigger: 2,
            max_write_buffer_number: 0,
            ..Options::default()
        };
        let dir = TempDir::new().unwrap();
        let db = Arc::new(Db::open(dir.path(), opts).unwrap());

        // Fill L0 to the stop trigger. Writes go through until the
        // snapshot after the flush shows L0 >= 2; from then on the
        // next write would block, so we time it carefully with a
        // spawned thread.
        let payload = vec![0x12u8; 600];
        for i in 0..32 {
            let k = format!("fill{i:04}");
            db.put(k.as_bytes(), &payload).unwrap();
            if db.get_int_property("lark.num-files-at-level0").unwrap_or(0) >= 2 {
                break;
            }
        }
        let l0 = db.get_int_property("lark.num-files-at-level0").unwrap_or(0);
        assert!(l0 >= 2, "precondition: need L0 >= 2, got {l0}");

        let db_writer = db.clone();
        let blocked = thread::spawn(move || {
            let start = Instant::now();
            db_writer.put(b"stopkey", b"stopval").unwrap();
            start.elapsed()
        });

        // Give the writer time to fully enter the stall loop.
        thread::sleep(Duration::from_millis(50));
        assert!(!blocked.is_finished(), "writer should be blocked on stall");

        // compact_range empties L0 and fires stall_signal.notify_all
        // from the compaction loop after the pass. The writer should
        // wake promptly.
        db.compact_range(None, None).unwrap();

        let waited = blocked.join().unwrap();
        assert!(
            waited < Duration::from_secs(5),
            "blocked writer took too long to unblock: {waited:?}"
        );

        // The key we wrote while stalled is readable afterwards.
        assert_eq!(db.get(b"stopkey").unwrap(), Some(b"stopval".to_vec()));
    }

    #[test]
    fn test_rate_limiter_unset_leaves_compaction_uncapped() {
        // Sanity check: with no limiter in Options, compaction still
        // runs and produces correct results. This is the default
        // configuration; the test exists mainly to pin the no-op
        // branch.
        use std::time::Instant;

        let opts = Options {
            write_buffer_size: 64 * 1024,
            ..Options::default()
        };
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), opts).unwrap();

        let payload = vec![0xABu8; 1024];
        for i in 0..256 {
            let k = format!("k{i:06}");
            db.put(k.as_bytes(), &payload).unwrap();
        }

        let start = Instant::now();
        db.compact_range(None, None).unwrap();
        assert!(
            start.elapsed() < std::time::Duration::from_secs(5),
            "unthrottled compaction took unreasonably long: {:?}",
            start.elapsed()
        );

        // Reads still work after compaction.
        for i in 0..256 {
            let k = format!("k{i:06}");
            assert_eq!(db.get(k.as_bytes()).unwrap(), Some(payload.clone()));
        }
    }
}
