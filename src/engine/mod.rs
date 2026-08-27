pub(crate) mod arena;
pub(crate) mod block;
pub(crate) mod block_cache;
pub(crate) mod bloom;
pub(crate) mod checksum;
pub(crate) mod commit;
pub(crate) mod compaction;
mod db_lock;
pub(crate) mod durability;
pub(crate) mod filter_block;
pub(crate) mod index_block;
pub(crate) mod internal_key;
pub(crate) mod iterator;
pub(crate) mod lookup_key;
#[cfg(loom)]
pub mod loom_model;
pub(crate) mod manifest;
pub(crate) mod memtable;
pub(crate) mod range_tombstone;
pub(crate) mod read_horizon;
pub(crate) mod read_view;
pub(crate) mod skiplist;
pub(crate) mod snapshot_registry;
pub(crate) mod sstable;
pub(crate) mod sync;
pub(crate) mod wal;
pub(crate) mod wal_replay;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};

use kovan_queue::array_queue::ArrayQueue;
use parking_lot::{Mutex, RwLock};

use block_cache::BlockCache;
use commit::{Pipeline, StallSignal, WriteSlot};
use compaction::{CompactionOptions, CompactionScheduler};
use db_lock::DbDirectoryLock;
use lookup_key::{LookupKey, with_key_scratch};
use manifest::{VersionEdit, VersionSet};
use memtable::{MemTable, MemTableConfig};
use read_horizon::ReadHorizon;
use read_view::{ReadView, ReadViewCell, VersionStore};
use snapshot_registry::SnapshotRegistry;

use crate::{DbSlice, WriteBatchOp, event_listener};
use sstable::{
    LiveSst, LookupResult, Materialize, PointValue, SsTableMeta, SsTableReader, SsTableWriter,
    sst_filename,
};
use wal::{Wal, WalEntry, wal_filename};
use wal_replay::WalReplayIter;

/// Controls when data is flushed to disk after a commit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DurabilityMode {
    Immediate,
    Eventual,
}

/// Outcome of [`LarkEngine::commit_with_conflict_check`].
/// `Conflict` indicates that another writer changed one of the
/// tracked keys after the sequence the transaction observed it at;
/// the caller typically surfaces this as a retry-able error.
#[derive(Debug)]
pub(crate) enum CommitOutcome {
    Ok,
    Conflict {
        key: Vec<u8>,
        observed_seq: u64,
        latest_seq: u64,
    },
}

fn grouped_batch_ops(
    point_ops: BTreeMap<Vec<u8>, Option<Vec<u8>>>,
    range_deletes: Vec<(Vec<u8>, Vec<u8>)>,
    merges: Vec<(Vec<u8>, Vec<u8>)>,
) -> Vec<WriteBatchOp> {
    let mut ops = Vec::with_capacity(point_ops.len() + range_deletes.len() + merges.len());
    for (key, value) in point_ops {
        match value {
            Some(value) => ops.push(WriteBatchOp::Put { key, value }),
            None => ops.push(WriteBatchOp::Delete { key }),
        }
    }
    for (start, end) in range_deletes {
        ops.push(WriteBatchOp::DeleteRange { start, end });
    }
    for (key, operand) in merges {
        ops.push(WriteBatchOp::Merge { key, operand });
    }
    ops
}

fn batch_op_wal_bytes(op: &WriteBatchOp) -> u64 {
    match op {
        WriteBatchOp::Put { key, value } => (key.len() + value.len() + 8) as u64,
        WriteBatchOp::Delete { key } => (key.len() + 8) as u64,
        WriteBatchOp::DeleteRange { start, end } => (start.len() + end.len() + 8) as u64,
        WriteBatchOp::Merge { key, operand } => (key.len() + operand.len() + 8) as u64,
    }
}

fn memtable_needs_flush(memtable: &MemTable) -> bool {
    !memtable.is_empty() || !memtable.clone_range_tombstones().is_empty()
}

fn apply_batch_op_to_memtable(memtable: &MemTable, op: &WriteBatchOp, seq: u64) {
    match op {
        WriteBatchOp::Put { key, value } => memtable.put(key, value, seq),
        WriteBatchOp::Delete { key } => memtable.delete(key, seq),
        WriteBatchOp::DeleteRange { start, end } => memtable.delete_range(start, end, seq),
        WriteBatchOp::Merge { key, operand } => memtable.merge(key, operand, seq),
    }
}

struct MultiGetEntry {
    key: Vec<u8>,
    output_indexes: Vec<usize>,
    max_rt: u64,
    resolved: bool,
}

fn grouped_multi_get_entries(keys: &[&[u8]]) -> Vec<MultiGetEntry> {
    let mut grouped: BTreeMap<Vec<u8>, Vec<usize>> = BTreeMap::new();
    for (idx, key) in keys.iter().enumerate() {
        grouped.entry((*key).to_vec()).or_default().push(idx);
    }
    grouped
        .into_iter()
        .map(|(key, output_indexes)| MultiGetEntry {
            key,
            output_indexes,
            max_rt: 0,
            resolved: false,
        })
        .collect()
}

fn file_covers_key(file: &LiveSst, key: &[u8]) -> bool {
    file.meta.smallest_key.as_slice() <= key && key <= file.meta.largest_key.as_slice()
}

fn key_range_for_file(entries: &[MultiGetEntry], file: &LiveSst) -> std::ops::Range<usize> {
    let start =
        entries.partition_point(|entry| entry.key.as_slice() < file.meta.smallest_key.as_slice());
    let end = start
        + entries[start..]
            .partition_point(|entry| entry.key.as_slice() <= file.meta.largest_key.as_slice());
    start..end
}

fn resolve_multi_get_value(pseq: u64, popt: Option<DbSlice>, rt_seq: u64) -> Option<Vec<u8>> {
    if pseq > rt_seq {
        popt.map(DbSlice::into_vec)
    } else {
        None
    }
}

fn set_multi_get_result(
    entry: &mut MultiGetEntry,
    results: &mut [Option<Vec<u8>>],
    value: Option<Vec<u8>>,
) {
    for &output_idx in &entry.output_indexes {
        results[output_idx] = value.clone();
    }
    entry.resolved = true;
}

/// Configuration for the Lark engine.
#[derive(Clone)]
pub(crate) struct EngineOptions {
    pub(crate) write_buffer_size: usize,
    pub(crate) arena_profile: crate::options::ArenaProfile,
    pub(crate) block_size: usize,
    pub(crate) block_cache_size: usize,
    pub(crate) block_cache_num_shard_bits: u32,
    pub(crate) strict_capacity_limit: bool,
    pub(crate) bloom_bits_per_key: usize,
    pub(crate) compression: crate::options::CompressionType,
    pub(crate) compression_per_level: Option<Vec<crate::options::CompressionType>>,
    pub(crate) l0_compaction_trigger: usize,
    pub(crate) level_base_bytes: u64,
    pub(crate) level_size_multiplier: u64,
    pub(crate) target_file_size: u64,
    pub(crate) compaction_filter: Option<Arc<dyn crate::options::CompactionFilter>>,
    pub(crate) prefix_extractor: Option<Arc<dyn crate::options::PrefixExtractor>>,
    pub(crate) merge_operator: Option<Arc<dyn crate::options::MergeOperator>>,
    pub(crate) listeners: Vec<Arc<dyn crate::event_listener::EventListener>>,
    pub(crate) statistics: Option<Arc<crate::statistics::Statistics>>,
    pub(crate) rate_limiter: Option<Arc<dyn crate::rate_limiter::RateLimiter>>,
    pub(crate) level0_slowdown_writes_trigger: usize,
    pub(crate) level0_stop_writes_trigger: usize,
    pub(crate) soft_pending_compaction_bytes_limit: u64,
    pub(crate) hard_pending_compaction_bytes_limit: u64,
    pub(crate) max_write_buffer_number: usize,
    pub(crate) compaction_style: crate::options::CompactionStyle,
    pub(crate) fifo_compaction_options: crate::options::FifoCompactionOptions,
    pub(crate) universal_compaction_options: crate::options::UniversalCompactionOptions,
    pub(crate) evict_compaction_data_from_page_cache: bool,
    pub(crate) max_background_compactions: usize,
    pub(crate) partitioned_index: bool,
    pub(crate) metadata_block_size: usize,
    pub(crate) cache_index_and_filter_blocks: bool,
    pub(crate) read_only: bool,
    pub(crate) max_key_size: usize,
    pub(crate) max_value_size: usize,
}

impl EngineOptions {
    /// How readers this engine opens should hold their index and
    /// filter blocks.
    pub(crate) fn metadata_policy(&self) -> sstable::MetadataPolicy {
        if self.cache_index_and_filter_blocks {
            sstable::MetadataPolicy::Cached
        } else {
            sstable::MetadataPolicy::Pinned
        }
    }

    /// Resolve the codec to use when writing an SSTable destined for
    /// `level`. A per-level override (if any) wins; otherwise fall
    /// back to the default codec.
    pub(crate) fn compression_for_level(&self, level: usize) -> crate::options::CompressionType {
        match &self.compression_per_level {
            Some(per_level) if level < per_level.len() => per_level[level],
            _ => self.compression,
        }
    }
}

impl Default for EngineOptions {
    fn default() -> Self {
        Self {
            write_buffer_size: 64 * 1024 * 1024,
            arena_profile: crate::options::ArenaProfile::SERVER,
            block_size: 16 * 1024,
            block_cache_size: 512 * 1024 * 1024,
            block_cache_num_shard_bits: 6,
            strict_capacity_limit: false,
            bloom_bits_per_key: 10,
            compression: crate::options::CompressionType::Lz4,
            compression_per_level: None,
            l0_compaction_trigger: compaction::L0_COMPACTION_TRIGGER,
            level_base_bytes: compaction::DEFAULT_LEVEL_BASE_BYTES,
            level_size_multiplier: compaction::LEVEL_SIZE_MULTIPLIER,
            target_file_size: compaction::DEFAULT_TARGET_FILE_SIZE,
            compaction_filter: None,
            prefix_extractor: None,
            merge_operator: None,
            listeners: Vec::new(),
            statistics: None,
            rate_limiter: None,
            level0_slowdown_writes_trigger: 20,
            level0_stop_writes_trigger: 36,
            soft_pending_compaction_bytes_limit: 64 * 1024 * 1024 * 1024,
            hard_pending_compaction_bytes_limit: 256 * 1024 * 1024 * 1024,
            max_write_buffer_number: 2,
            compaction_style: crate::options::CompactionStyle::Level,
            fifo_compaction_options: crate::options::FifoCompactionOptions::default(),
            universal_compaction_options: crate::options::UniversalCompactionOptions::default(),
            evict_compaction_data_from_page_cache: false,
            max_background_compactions: 1,
            partitioned_index: false,
            metadata_block_size: 4096,
            cache_index_and_filter_blocks: false,
            read_only: false,
            max_key_size: crate::options::DEFAULT_MAX_KEY_SIZE,
            max_value_size: crate::options::DEFAULT_MAX_VALUE_SIZE,
        }
    }
}

const CLOSE_STATE_OPEN: u8 = 0;
const CLOSE_STATE_CLOSING: u8 = 1;
const CLOSE_STATE_CLOSED: u8 = 2;

/// The core LSM-tree engine.
pub(crate) struct LarkEngine {
    /// The published read view: the active memtable, the frozen
    /// memtables and the version every read resolves against. Loading
    /// it is the whole of a read's source acquisition - one shared lock
    /// acquisition and one `Arc` clone - and the three sources it hands
    /// back are consistent with each other by construction.
    view: Arc<ReadViewCell>,
    /// The engine's one memtable arena pool and its sizing policy. Every
    /// memtable this engine builds recycles the others' chunks through it.
    memtable_config: MemTableConfig,
    /// The version set, together with the publication of every version
    /// it installs into `view`.
    versions: Arc<VersionStore>,
    cache: Arc<BlockCache>,
    /// Sequence-number allocator. Advanced up front (before a write's
    /// data lands) so WAL and memtable entries can be stamped, and used
    /// as the durable "last sequence" marker for WAL replay.
    latest_seq: AtomicU64,
    /// Published read horizon: the highest sequence whose data is fully
    /// applied and durable. Snapshots read this, never `latest_seq`, so a
    /// snapshot taken mid-commit cannot observe a sequence whose WAL and
    /// memtable writes have not landed yet. The ordering that makes that
    /// true lives in [`ReadHorizon`], which is where it is model-checked.
    visible_seq: ReadHorizon,
    close_state: AtomicU8,
    close_lock: Mutex<()>,
    active_wal: Mutex<Option<Wal>>,
    wal_id: AtomicU64,
    sst_dir: PathBuf,
    wal_dir: PathBuf,
    compaction: Mutex<CompactionScheduler>,
    /// Engine-wide RwLock that coordinates foreground and background
    /// compaction. Background workers each hold a read lock so they
    /// can run concurrently; foreground callers (`compact_range`,
    /// `ingest_external_files`, `checkpoint_capture`) hold the write
    /// lock to exclude all background activity for the duration of
    /// their pass.
    compaction_lock: Arc<RwLock<()>>,
    /// Tracks the sequence numbers of every live snapshot so compaction
    /// can drop versions that no snapshot and no current reader can
    /// see. A snapshot registers itself on creation and releases on
    /// drop; compaction queries `oldest_live_seq()` to compute its GC
    /// horizon.
    snapshot_registry: Arc<SnapshotRegistry>,
    options: EngineOptions,
    /// Bounded ring of writers waiting for a commit group. Writers push a
    /// ticket here and park; they never block on `pipeline`, which is what
    /// takes the WAL fsync off every writer's critical path.
    commit_ring: ArrayQueue<Arc<WriteSlot>>,
    /// Exclusion for the whole write pipeline, and the leader-owned
    /// staging buffers. Acquiring it *is* becoming the commit leader.
    /// Administrative operations that rotate the memtable or the WAL
    /// (`compact_range`, `ingest_external_files`, `checkpoint_capture`,
    /// `drop_all`, `close`) take it blockingly; no follower ever does.
    pipeline: Mutex<Pipeline>,
    /// Latched write-path failure. Set only when a failed commit group
    /// could not be rolled back out of the WAL, which leaves the log with
    /// a tail no later write may extend. Once set, every write fails loud
    /// with the original reason instead of appending after unknown bytes.
    wal_failure: Mutex<Option<(std::io::ErrorKind, String)>>,
    /// Cheap gate on `wal_failure`, checked on every write.
    wal_failed: AtomicBool,
    /// Signal used by foreground writers to wait out a "stop writes"
    /// condition (too many L0 files, too many unflushed memtables).
    /// The background compaction thread holds a clone of this `Arc`
    /// and calls [`StallSignal::notify_all`] after each compaction
    /// pass so blocked writers can re-check their thresholds.
    stall_signal: Arc<StallSignal>,
    /// Cached stall level: 0 = none, 1 = slowdown, 2 = stop.
    /// Updated by `rotate_memtable` (after changing L0/memtable
    /// counts) and by the compaction thread (after reducing them).
    /// Writers check this atomic first - the full `stall_state()`
    /// with its lock acquisitions is only called when the cached
    /// level is nonzero, saving 2 lock round-trips per write in
    /// the common no-stall case.
    cached_stall_level: AtomicU8,
    _db_lock: DbDirectoryLock,
}

impl LarkEngine {
    /// Open or create the database at the given path.
    pub(crate) fn open(db_dir: &Path, mut options: EngineOptions) -> std::io::Result<Arc<Self>> {
        options.read_only = false;
        let db_lock = DbDirectoryLock::acquire_exclusive(db_dir)?;
        let sst_dir = db_dir.join("sst");
        let wal_dir = db_dir.join("wal");

        std::fs::create_dir_all(&sst_dir)?;
        std::fs::create_dir_all(&wal_dir)?;

        let version_set =
            VersionSet::open_with_policy(db_dir, &sst_dir, options.metadata_policy())?;
        let version = version_set.current();
        let mut latest_seq = version.last_seq;

        // Replay WAL files to recover memtable state
        let memtable_config = MemTableConfig::new(
            options.arena_profile,
            options.write_buffer_size,
            options.max_write_buffer_number,
        );
        let memtable = Arc::new(MemTable::new(&memtable_config)?);
        let mut wal_files = list_wal_files(&wal_dir)?;
        wal_files.sort();
        wal_files.retain(|path| should_replay_wal(path, version.min_wal_id));

        let mut discarded_tails = Vec::new();
        let mut entries_per_file = Vec::with_capacity(wal_files.len());
        for wal_path in &wal_files {
            tracing::info!(path = %wal_path.display(), "Replaying WAL");
            let mut replay = WalReplayIter::open(wal_path)?;
            let mut entries = 0usize;
            while let Some(entry) = replay.next_entry()? {
                entries += 1;
                latest_seq = latest_seq.max(apply_replayed_wal_entry(&memtable, entry));
            }
            if let Some(tail) = replay.discarded_tail() {
                discarded_tails.push((wal_path.clone(), tail.offset, tail.discarded_bytes));
            }
            entries_per_file.push(entries);
        }
        reject_tail_discard_before_live_wal(&wal_files, &entries_per_file, &discarded_tails)?;
        if let Some(stats) = options.statistics.as_ref() {
            stats.add(
                crate::statistics::Ticker::WalTailDiscarded,
                discarded_tails.len() as u64,
            );
        }

        let wal_id = next_wal_id(version.next_file_id, &wal_files);
        let wal_path = wal_dir.join(wal_filename(wal_id));
        let mut wal = Wal::create(&wal_path)?;

        rewrite_recovered_memtable_to_wal(&memtable, &mut wal)?;

        for replayed_wal_path in &wal_files {
            if replayed_wal_path != &wal_path {
                match Wal::remove(replayed_wal_path) {
                    Ok(()) => {}
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                    Err(e) => return Err(e),
                }
            }
        }

        let versions = Arc::new(VersionStore::new(version_set));
        let view = Arc::new(ReadViewCell::new(ReadView {
            active: Arc::clone(&memtable),
            frozen: Vec::new(),
            version: versions.lock().current(),
        }));
        versions.attach_view(Arc::clone(&view));

        versions
            .lock()
            .apply(&[VersionEdit::SetNextFileId(wal_id + 1)])?;

        let cache = Arc::new(
            BlockCache::with_config(
                options.block_cache_size,
                options.block_cache_num_shard_bits,
                options.strict_capacity_limit,
            )
            .with_stats(options.statistics.clone()),
        );
        let compaction_opts = CompactionOptions {
            l0_compaction_trigger: options.l0_compaction_trigger,
            level_base_bytes: options.level_base_bytes,
            level_size_multiplier: options.level_size_multiplier,
            target_file_size: options.target_file_size,
            block_size: options.block_size,
            bloom_bits_per_key: options.bloom_bits_per_key,
            compression: options.compression,
            compression_per_level: options.compression_per_level.clone(),
            compaction_filter: options.compaction_filter.clone(),
            prefix_extractor: options.prefix_extractor.clone(),
            merge_operator: options.merge_operator.clone(),
            listeners: options.listeners.clone(),
            statistics: options.statistics.clone(),
            rate_limiter: options.rate_limiter.clone(),
            compaction_style: options.compaction_style,
            fifo_compaction_options: options.fifo_compaction_options,
            universal_compaction_options: options.universal_compaction_options,
            evict_compaction_data_from_page_cache: options.evict_compaction_data_from_page_cache,
            max_background_compactions: options.max_background_compactions,
            partitioned_index: options.partitioned_index,
            metadata_block_size: options.metadata_block_size,
            cache_index_and_filter_blocks: options.cache_index_and_filter_blocks,
        };

        let compaction_lock = Arc::new(RwLock::new(()));
        let snapshot_registry = Arc::new(SnapshotRegistry::new());
        let stall_signal = Arc::new(StallSignal::new());
        // A platform that cannot spawn the worker (a single-threaded
        // target) fails the open here instead of aborting; the
        // directory lock and the fresh WAL drop with this return.
        let compaction = CompactionScheduler::start(
            Arc::clone(&compaction_lock),
            Arc::clone(&snapshot_registry),
            Arc::clone(&versions),
            Arc::from(sst_dir.as_path()),
            Arc::clone(&cache),
            compaction_opts,
            Arc::clone(&stall_signal),
        )?;

        let engine = Arc::new(Self {
            view,
            memtable_config,
            versions,
            cache,
            latest_seq: AtomicU64::new(latest_seq),
            visible_seq: ReadHorizon::new(latest_seq),
            close_state: AtomicU8::new(CLOSE_STATE_OPEN),
            close_lock: Mutex::new(()),
            active_wal: Mutex::new(Some(wal)),
            wal_id: AtomicU64::new(wal_id),
            sst_dir,
            wal_dir,
            compaction: Mutex::new(compaction),
            compaction_lock,
            snapshot_registry,
            options,
            commit_ring: Self::new_commit_ring(),
            pipeline: Mutex::new(Pipeline::new()),
            wal_failure: Mutex::new(None),
            wal_failed: AtomicBool::new(false),
            stall_signal,
            cached_stall_level: AtomicU8::new(0),
            _db_lock: db_lock,
        });

        Ok(engine)
    }

    /// Open an existing database without mutating files or starting
    /// background writers.
    pub(crate) fn open_read_only(
        db_dir: &Path,
        mut options: EngineOptions,
    ) -> std::io::Result<Arc<Self>> {
        let db_lock = DbDirectoryLock::acquire_shared(db_dir)?;
        let sst_dir = db_dir.join("sst");
        let wal_dir = db_dir.join("wal");

        if !sst_dir.is_dir() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!(
                    "missing SST directory for read-only open: {}",
                    sst_dir.display()
                ),
            ));
        }
        if !wal_dir.is_dir() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!(
                    "missing WAL directory for read-only open: {}",
                    wal_dir.display()
                ),
            ));
        }

        options.read_only = true;
        let version_set = VersionSet::open_read_only(db_dir, &sst_dir, options.metadata_policy())?;
        let version = version_set.current();
        let mut latest_seq = version.last_seq;

        let memtable_config = MemTableConfig::new(
            options.arena_profile,
            options.write_buffer_size,
            options.max_write_buffer_number,
        );
        let memtable = Arc::new(MemTable::new(&memtable_config)?);
        let mut wal_files = list_wal_files(&wal_dir)?;
        wal_files.sort();
        wal_files.retain(|path| should_replay_wal(path, version.min_wal_id));

        let mut discarded_tails = Vec::new();
        let mut entries_per_file = Vec::with_capacity(wal_files.len());
        for wal_path in &wal_files {
            tracing::info!(path = %wal_path.display(), "Replaying WAL for read-only open");
            let mut replay = WalReplayIter::open(wal_path)?;
            let mut entries = 0usize;
            while let Some(entry) = replay.next_entry()? {
                entries += 1;
                latest_seq = latest_seq.max(apply_replayed_wal_entry(&memtable, entry));
            }
            if let Some(tail) = replay.discarded_tail() {
                discarded_tails.push((wal_path.clone(), tail.offset, tail.discarded_bytes));
            }
            entries_per_file.push(entries);
        }
        reject_tail_discard_before_live_wal(&wal_files, &entries_per_file, &discarded_tails)?;
        if let Some(stats) = options.statistics.as_ref() {
            stats.add(
                crate::statistics::Ticker::WalTailDiscarded,
                discarded_tails.len() as u64,
            );
        }

        let cache = Arc::new(
            BlockCache::with_config(
                options.block_cache_size,
                options.block_cache_num_shard_bits,
                options.strict_capacity_limit,
            )
            .with_stats(options.statistics.clone()),
        );
        let versions = Arc::new(VersionStore::new(version_set));
        let view = Arc::new(ReadViewCell::new(ReadView {
            active: Arc::clone(&memtable),
            frozen: Vec::new(),
            version: versions.lock().current(),
        }));
        versions.attach_view(Arc::clone(&view));
        let compaction_lock = Arc::new(RwLock::new(()));
        let snapshot_registry = Arc::new(SnapshotRegistry::new());
        let stall_signal = Arc::new(StallSignal::new());
        let wal_id = next_wal_id(version.next_file_id, &wal_files);

        Ok(Arc::new(Self {
            view,
            memtable_config,
            versions,
            cache,
            latest_seq: AtomicU64::new(latest_seq),
            visible_seq: ReadHorizon::new(latest_seq),
            close_state: AtomicU8::new(CLOSE_STATE_OPEN),
            close_lock: Mutex::new(()),
            active_wal: Mutex::new(None),
            wal_id: AtomicU64::new(wal_id),
            sst_dir,
            wal_dir,
            compaction: Mutex::new(CompactionScheduler::disabled()),
            compaction_lock,
            snapshot_registry,
            options,
            commit_ring: Self::new_commit_ring(),
            pipeline: Mutex::new(Pipeline::new()),
            wal_failure: Mutex::new(None),
            wal_failed: AtomicBool::new(false),
            stall_signal,
            cached_stall_level: AtomicU8::new(0),
            _db_lock: db_lock,
        }))
    }

    pub(crate) fn snapshot_seq(&self) -> u64 {
        self.visible_seq.visible()
    }

    pub(crate) fn is_read_only(&self) -> bool {
        self.options.read_only
    }

    pub(crate) fn is_closed(&self) -> bool {
        self.close_state.load(Ordering::Acquire) != CLOSE_STATE_OPEN
    }

    fn closed_error() -> std::io::Error {
        std::io::Error::new(std::io::ErrorKind::NotConnected, "database is closed")
    }

    fn ensure_open(&self) -> std::io::Result<()> {
        if self.is_closed() {
            Err(Self::closed_error())
        } else {
            Ok(())
        }
    }

    fn read_only_error() -> std::io::Error {
        std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "database was opened read-only",
        )
    }

    fn ensure_writable(&self) -> std::io::Result<()> {
        self.ensure_open()?;
        if self.wal_failed.load(Ordering::Acquire) {
            return Err(self.wal_failure_error());
        }
        if self.is_read_only() {
            Err(Self::read_only_error())
        } else {
            Ok(())
        }
    }

    /// Latch a write-ahead-log failure the engine cannot recover from
    /// on its own, so every later write fails loud with the reason
    /// rather than appending after a tail nobody can account for.
    pub(crate) fn latch_wal_failure(&self, err: &std::io::Error) {
        *self.wal_failure.lock() = Some((err.kind(), err.to_string()));
        self.wal_failed.store(true, Ordering::Release);
    }

    fn wal_failure_error(&self) -> std::io::Error {
        match self.wal_failure.lock().as_ref() {
            Some((kind, message)) => std::io::Error::new(
                *kind,
                format!("write-ahead log left in an unknown state: {message}"),
            ),
            None => std::io::Error::other("write-ahead log left in an unknown state"),
        }
    }

    fn validate_prefixed_key_size(&self, key: &[u8]) -> std::io::Result<()> {
        let user_key_len = key.len().saturating_sub(4);
        if user_key_len <= self.options.max_key_size {
            return Ok(());
        }

        Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "key length {} exceeds configured max_key_size {}",
                user_key_len, self.options.max_key_size
            ),
        ))
    }

    fn validate_value_size(&self, value: &[u8]) -> std::io::Result<()> {
        if value.len() <= self.options.max_value_size {
            return Ok(());
        }

        Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "value length {} exceeds configured max_value_size {}",
                value.len(),
                self.options.max_value_size
            ),
        ))
    }

    fn validate_ops_sizes(&self, ops: &[WriteBatchOp]) -> std::io::Result<()> {
        for op in ops {
            match op {
                WriteBatchOp::Put { key, value } => {
                    self.validate_prefixed_key_size(key)?;
                    self.validate_value_size(value)?;
                }
                WriteBatchOp::Delete { key } => self.validate_prefixed_key_size(key)?,
                WriteBatchOp::DeleteRange { start, end } => {
                    self.validate_prefixed_key_size(start)?;
                    self.validate_prefixed_key_size(end)?;
                }
                WriteBatchOp::Merge { key, operand } => {
                    self.validate_prefixed_key_size(key)?;
                    self.validate_value_size(operand)?;
                }
            }
        }
        Ok(())
    }

    /// Borrow the engine's `Statistics` sink if one is configured.
    /// Returning `Option<&Statistics>` lets instrumented call
    /// sites branch on a single `is_some()` check rather than
    /// cloning an `Arc` per operation.
    pub(crate) fn statistics(&self) -> Option<&crate::statistics::Statistics> {
        self.options.statistics.as_deref()
    }

    /// Clone the statistics `Arc` for consumers (like the public
    /// `Iter`) that need to carry the handle across a lifetime
    /// boundary where a borrowed reference wouldn't reach.
    pub(crate) fn statistics_arc(&self) -> Option<Arc<crate::statistics::Statistics>> {
        self.options.statistics.clone()
    }

    /// Register a new live snapshot at `seq` so compaction keeps
    /// every version it might need to see. Balanced by
    /// [`Self::release_snapshot`] when the snapshot drops.
    pub(crate) fn register_snapshot(&self, seq: u64) {
        if self.is_closed() {
            return;
        }
        self.snapshot_registry.register(seq);
        if let Some(s) = self.statistics() {
            s.add(crate::statistics::Ticker::SnapshotsRegistered, 1);
        }
    }

    /// Pin a snapshot at the current read horizon, sampling the
    /// horizon and registering the pin as one step so a concurrent
    /// compaction cannot compute its GC bound from a registry that
    /// does not yet contain this pin. Returns the pinned sequence.
    pub(crate) fn register_snapshot_at_horizon(&self) -> u64 {
        if self.is_closed() {
            return self.visible_seq.visible();
        }
        let seq = self
            .snapshot_registry
            .register_at(|| self.visible_seq.visible());
        if let Some(s) = self.statistics() {
            s.add(crate::statistics::Ticker::SnapshotsRegistered, 1);
        }
        seq
    }

    /// Release a snapshot pin previously taken via
    /// [`Self::register_snapshot`].
    pub(crate) fn release_snapshot(&self, seq: u64) {
        self.snapshot_registry.release(seq);
        if let Some(s) = self.statistics() {
            s.add(crate::statistics::Ticker::SnapshotsReleased, 1);
        }
    }

    /// Current GC horizon for compaction - the smallest live snapshot
    /// seq, or `u64::MAX` if no snapshot is currently pinned.
    pub(crate) fn oldest_live_seq(&self) -> u64 {
        self.snapshot_registry.oldest_live_seq()
    }

    /// Construct a streaming iterator over the latest published read
    /// horizon. The view is loaded before the horizon is sampled, for
    /// the reason spelled out on [`Self::get_latest`].
    pub(crate) fn new_iter_latest(&self) -> iterator::LarkIterator {
        let closed = self.is_closed();
        let view = self.view.load();
        let snapshot_seq = self.visible_seq.visible();
        self.iter_in_view(&view, snapshot_seq, closed)
    }

    /// Construct a streaming iterator rooted at a caller-pinned
    /// `snapshot_seq`. Captures one published view of the memtables and
    /// the version; no filesystem access happens here - file handles
    /// are already open in the pinned `Arc<LiveSst>`s carried by the
    /// version.
    pub(crate) fn new_iter_at(&self, snapshot_seq: u64) -> iterator::LarkIterator {
        let closed = self.is_closed();
        let view = self.view.load();
        self.iter_in_view(&view, snapshot_seq, closed)
    }

    fn iter_in_view(
        &self,
        view: &ReadView,
        snapshot_seq: u64,
        closed: bool,
    ) -> iterator::LarkIterator {
        let mut iter = iterator::LarkIterator::new(
            Arc::clone(&view.active),
            view.frozen.clone(),
            Arc::clone(&view.version),
            Arc::clone(&self.cache),
            snapshot_seq,
            self.options.prefix_extractor.clone(),
            self.options.merge_operator.clone(),
        );
        if closed {
            iter.set_error(Self::closed_error());
        }
        iter
    }

    /// Point lookup at the latest published read horizon.
    ///
    /// The order of the first two statements is load-bearing: the view
    /// is loaded FIRST and the horizon sampled SECOND. Sampling the
    /// horizon first lets a compaction garbage-collect the newest
    /// version at or below it - compaction's GC bound is
    /// `oldest_live_seq()`, and a read without a `Snapshot` registers
    /// nothing there - after which the key reads back as absent. A
    /// version that compaction dropped was shadowed by a newer one
    /// that had already been flushed into the version being published,
    /// so a horizon sampled after the view is always at least that
    /// newer version's sequence and the read finds it.
    ///
    /// The read linearizes at the moment it loads the view: a write
    /// that lands between the load and the horizon sample is simply
    /// not part of this read. Read-your-writes still holds, because a
    /// write applies into the active memtable of the view current at
    /// that moment and every later view still exposes that memtable's
    /// data, as `active`, as `frozen`, or folded into the version.
    pub(crate) fn get_latest(&self, key: &[u8]) -> std::io::Result<Option<Vec<u8>>> {
        self.ensure_open()?;
        let view = self.view.load();
        let lk = LookupKey::from_prefixed(key, self.visible_seq.visible());
        if self.options.merge_operator.is_some() {
            return Ok(self.get_with_merge(&lk)?.map(DbSlice::into_vec));
        }
        Ok(self
            .lookup_in_view(
                lk.prefixed_user_key(),
                lk.snapshot_seq(),
                &lk,
                Materialize::Value,
                &view,
            )?
            .and_then(|v| match v {
                PointValue::Value(value) => Some(value.into_vec()),
                PointValue::Length(_) => None,
            }))
    }

    /// Point lookup at a caller-pinned snapshot sequence. The
    /// snapshot's registration in [`SnapshotRegistry`] is what keeps
    /// compaction from dropping the versions it needs, so the load
    /// order does not matter here; the view is still loaded once so
    /// every source the read walks agrees with every other.
    pub(crate) fn get_at(&self, key: &[u8], snapshot_seq: u64) -> std::io::Result<Option<Vec<u8>>> {
        self.get(key, snapshot_seq)
    }

    /// Point lookup resolved against one already-loaded view.
    ///
    /// Walks sources newest→oldest (active memtable, frozen memtables
    /// newest first, L0 newest first, L1..Ln). At each source we check
    /// both the newest visible point entry and the newest visible
    /// covering range tombstone, carrying the largest RT seq forward so
    /// a range delete in a newer source can override a point entry in
    /// an older source. The first source yielding a decisive answer
    /// wins - a point entry with `seq > max_rt_so_far` gives its value;
    /// otherwise the range tombstone hides it.
    ///
    /// When a [`crate::MergeOperator`] is configured, the walk also
    /// collects any merge operands that sit on top of the terminator
    /// and calls the operator to collapse the chain into a final
    /// value at visibility time.
    pub(crate) fn get(
        &self,
        prefixed_key: &[u8],
        snapshot_seq: u64,
    ) -> std::io::Result<Option<Vec<u8>>> {
        let lk = LookupKey::from_prefixed(prefixed_key, snapshot_seq);
        Ok(self.get_slice(&lk)?.map(DbSlice::into_vec))
    }

    /// [`LarkEngine::get`] without copying the value. The returned
    /// [`DbSlice`] borrows the block or heap buffer the value already
    /// lives in and keeps that owner alive.
    /// The sequence a "read the latest" caller must use, sampled with
    /// the view it will read through already loaded.
    ///
    /// The order is load-bearing and it is the whole reason this is not
    /// `visible_seq.visible()` at the call site. Sampling the horizon
    /// first and loading the view afterwards leaves a window in
    /// between: `snapshot_seq()` registers nothing in the
    /// [`SnapshotRegistry`], so a compaction is free to drop the newest
    /// version at or below the sampled sequence, and the read then
    /// walks a view that no longer holds it and reports the key absent
    /// or an older value. That is a read travelling backwards.
    ///
    /// Loading the view first pins the sources, so every version the
    /// sampled horizon admits is still reachable through it.
    ///
    /// A caller reading at a *pinned* snapshot does not need this: the
    /// registration is what holds the versions, so the order does not
    /// matter there. See [`LarkEngine::get_at`].
    /// Newest visible value for `key`, without copying it.
    pub(crate) fn get_slice_latest(
        &self,
        cf_id: u32,
        key: &[u8],
    ) -> std::io::Result<Option<DbSlice>> {
        match self.lookup_latest_cf(cf_id, key, Materialize::Value)? {
            Some(PointValue::Value(value)) => Ok(Some(value)),
            Some(PointValue::Length(_)) => Err(std::io::Error::other(
                "point lookup produced a length where a value was requested",
            )),
            None => Ok(None),
        }
    }

    /// Length of the newest visible value for `key`, or `None`.
    pub(crate) fn get_size_latest(&self, cf_id: u32, key: &[u8]) -> std::io::Result<Option<usize>> {
        Ok(self
            .lookup_latest_cf(cf_id, key, Materialize::LengthOnly)?
            .map(|v| v.len()))
    }

    pub(crate) fn get_slice(&self, lk: &LookupKey) -> std::io::Result<Option<DbSlice>> {
        match self.lookup(lk, Materialize::Value)? {
            Some(PointValue::Value(value)) => Ok(Some(value)),
            Some(PointValue::Length(_)) => Err(std::io::Error::other(
                "point lookup produced a length where a value was requested",
            )),
            None => Ok(None),
        }
    }

    /// Length of the live value for `lk`, or `None` when there is none.
    ///
    /// Reads the same sources [`LarkEngine::get_slice`] does and pays
    /// the same block reads, but never takes a reference on the block
    /// or buffer the value lives in.
    pub(crate) fn get_size(&self, lk: &LookupKey) -> std::io::Result<Option<usize>> {
        Ok(self.lookup(lk, Materialize::LengthOnly)?.map(|v| v.len()))
    }

    /// Look one SSTable up, projecting the hit into whichever form the
    /// caller asked for. The two arms run the same bloom check, index
    /// search, block read and block scan; they differ only in whether
    /// the winning value is handed back as bytes or as a length.
    fn probe_file(
        &self,
        reader: &SsTableReader,
        lk: &LookupKey,
        materialize: Materialize,
    ) -> std::io::Result<LookupResult<PointValue>> {
        with_key_scratch(|buf| match materialize {
            Materialize::Value => Ok(reader
                .get(lk, buf, &self.cache)?
                .map_value(PointValue::Value)),
            Materialize::LengthOnly => Ok(reader
                .get_size(lk, buf, &self.cache)?
                .map_value(PointValue::Length)),
        })
    }

    /// The single point-read source walk. `materialize` decides only
    /// what the winning entry is projected into, never which sources
    /// are consulted or how MVCC and range-tombstone precedence are
    /// resolved, so every `get`-shaped entry point agrees by
    /// construction.
    fn lookup(
        &self,
        lk: &LookupKey,
        materialize: Materialize,
    ) -> std::io::Result<Option<PointValue>> {
        self.ensure_open()?;
        if self.options.merge_operator.is_some() {
            // A merge operator decides inside `full_merge` whether a
            // value exists at all, so a length-only request has to
            // collapse the chain exactly like a full read does.
            return Ok(self.get_with_merge(lk)?.map(PointValue::Value));
        }
        let view = self.view.load();
        self.lookup_in_view(
            lk.prefixed_user_key(),
            lk.snapshot_seq(),
            lk,
            materialize,
            &view,
        )
    }

    /// Read the newest visible value for `key`, sampling the horizon
    /// against the very view the read then walks.
    ///
    /// One view load, one sample, in that order. Sampling first and
    /// loading afterwards leaves a window where a compaction can drop
    /// the newest version at or below the sampled sequence, and the
    /// read reports the key absent or an older value: a read that
    /// travels backwards. Sampling under a *different* view load than
    /// the one the read walks has the same hole, only narrower, which
    /// is why the two happen here together rather than in a helper the
    /// caller composes.
    fn lookup_latest_cf(
        &self,
        cf_id: u32,
        key: &[u8],
        materialize: Materialize,
    ) -> std::io::Result<Option<PointValue>> {
        self.ensure_open()?;
        let view = self.view.load();
        // `LookupKey::new` writes the prefix into its own inline buffer,
        // so the read path stays allocation-free: prefixing into a
        // `Vec` here would put a malloc on every point read.
        let lk = LookupKey::new(cf_id, key, self.visible_seq.visible());
        let snapshot_seq = lk.snapshot_seq();
        if self.options.merge_operator.is_some() {
            return Ok(self.get_with_merge(&lk)?.map(PointValue::Value));
        }
        self.lookup_in_view(
            lk.prefixed_user_key(),
            snapshot_seq,
            &lk,
            materialize,
            &view,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn lookup_in_view(
        &self,
        key: &[u8],
        snapshot_seq: u64,
        lk: &LookupKey,
        materialize: Materialize,
        view: &ReadView,
    ) -> std::io::Result<Option<PointValue>> {
        let mut max_rt_seq: u64 = 0;

        // Memtable phase - timed via `PerfContext` at
        // `PerfLevel::EnableTime` so per-op breakdowns can
        // attribute time to "memtable vs SSTable".
        {
            let _t = crate::perf_context::PerfTimer::new(
                crate::perf_context::PerfTimerField::GetFromMemtable,
            );
            {
                let active = &view.active;
                let rt = active.covering_range_tombstone_seq(key, snapshot_seq);
                if rt > max_rt_seq {
                    max_rt_seq = rt;
                }
                if let Some((pseq, popt)) = active.get(lk) {
                    return Ok(if pseq > max_rt_seq {
                        popt.map(|v| PointValue::of(v, materialize))
                    } else {
                        None
                    });
                }
            }
            {
                let frozen = &view.frozen;
                for mt in frozen.iter().rev() {
                    let rt = mt.covering_range_tombstone_seq(key, snapshot_seq);
                    if rt > max_rt_seq {
                        max_rt_seq = rt;
                    }
                    if let Some((pseq, popt)) = mt.get(lk) {
                        return Ok(if pseq > max_rt_seq {
                            popt.map(|v| PointValue::of(v, materialize))
                        } else {
                            None
                        });
                    }
                }
            }
        }

        // SSTable phase - likewise timed. Everything below this
        // line is the "get_from_output_files" time.
        let _t_ssts = crate::perf_context::PerfTimer::new(
            crate::perf_context::PerfTimerField::GetFromOutputFiles,
        );
        let version = &view.version;

        // L0: check all files (may overlap), newest first. Readers are
        // already open in the pinned `Version`, so no filesystem access
        // happens here - concurrent compaction unlinking paths cannot
        // break us.
        for file in version.levels[0].iter().rev() {
            let rt = file.reader.covering_range_tombstone_seq(key, snapshot_seq);
            if rt > max_rt_seq {
                max_rt_seq = rt;
            }
            match self.probe_file(&file.reader, lk, materialize)? {
                LookupResult::Found { seq, value } => {
                    return Ok((seq > max_rt_seq).then_some(value));
                }
                LookupResult::FoundTombstone { .. } => return Ok(None),
                LookupResult::NotInTable => {}
            }
        }

        // L1+: point files are non-overlapping, but RT-only files can
        // share a boundary with neighboring point files. Scan metadata
        // ranges at the level so those empty files cannot hide the
        // actual point-containing SSTable.
        for level in 1..version.levels.len() {
            let files = &version.levels[level];
            if files.is_empty() {
                continue;
            }

            // Range tombstones can be stored in any file at this level whose
            // user-key range covers `key`, even if the point entry for `key`
            // lives in a different file (e.g. an RT-only SSTable). Scan each
            // overlapping file for RT coverage before the point lookup.
            for file in files {
                if file.meta.smallest_key.as_slice() <= key
                    && key <= file.meta.largest_key.as_slice()
                {
                    let rt = file.reader.covering_range_tombstone_seq(key, snapshot_seq);
                    if rt > max_rt_seq {
                        max_rt_seq = rt;
                    }
                }
            }

            for file in files {
                if file.meta.num_entries == 0 {
                    continue;
                }
                if file.meta.smallest_key.as_slice() > key || key > file.meta.largest_key.as_slice()
                {
                    continue;
                }
                match self.probe_file(&file.reader, lk, materialize)? {
                    LookupResult::Found { seq, value } => {
                        return Ok((seq > max_rt_seq).then_some(value));
                    }
                    LookupResult::FoundTombstone { .. } => return Ok(None),
                    LookupResult::NotInTable => {}
                }
            }
        }

        Ok(None)
    }

    /// Merge-aware point lookup. Walks every source newest→oldest
    /// collecting merge operands and any base value / deletion into
    /// a single chain, honors range-tombstone coverage, and calls
    /// [`crate::MergeOperator::full_merge`] at the end to materialize
    /// the final value.
    ///
    /// Callers are responsible for having checked that
    /// `self.options.merge_operator.is_some()`; this helper asserts
    /// internally.
    fn get_with_merge(&self, lk: &LookupKey) -> std::io::Result<Option<DbSlice>> {
        use internal_key::{VALUE_TYPE_DELETION, VALUE_TYPE_MERGE, VALUE_TYPE_VALUE};

        let key = lk.prefixed_user_key();
        let snapshot_seq = lk.snapshot_seq();

        let merge_op = self
            .options
            .merge_operator
            .as_ref()
            .expect("get_with_merge called without a merge operator");

        let view = self.view.load();
        // `chain` records visible entries for `key` in newest-seq-
        // first order, stopping at (and including) the first
        // terminator (`VALUE` or `DELETION`). Range tombstones that
        // cover the key are treated as virtual deletion terminators.
        let mut chain: Vec<(u64, u8, DbSlice)> = Vec::new();
        let mut max_rt_seq: u64 = 0;
        let mut terminated;

        // `consume_partial` appends entries from one source into the
        // running chain, short-circuiting if a terminator (real or
        // RT-synthesized) is reached.
        let consume_partial = |partial: Vec<(u64, u8, DbSlice)>,
                               max_rt_seq: u64,
                               chain: &mut Vec<(u64, u8, DbSlice)>|
         -> bool {
            for (seq, vt, value) in partial {
                if seq <= max_rt_seq {
                    // Range tombstone hides this and every older
                    // entry for the same key.
                    chain.push((max_rt_seq, VALUE_TYPE_DELETION, DbSlice::empty()));
                    return true;
                }
                chain.push((seq, vt, value));
                if vt != VALUE_TYPE_MERGE {
                    return true;
                }
            }
            false
        };

        // Walk sources newest → oldest.
        {
            let active = &view.active;
            let rt = active.covering_range_tombstone_seq(key, snapshot_seq);
            if rt > max_rt_seq {
                max_rt_seq = rt;
            }
            let mut partial = Vec::new();
            let _ = active.collect_merge_chain(lk, &mut partial);
            terminated = consume_partial(partial, max_rt_seq, &mut chain);
        }

        if !terminated {
            let frozen = &view.frozen;
            for mt in frozen.iter().rev() {
                let rt = mt.covering_range_tombstone_seq(key, snapshot_seq);
                if rt > max_rt_seq {
                    max_rt_seq = rt;
                }
                let mut partial = Vec::new();
                let _ = mt.collect_merge_chain(lk, &mut partial);
                terminated = consume_partial(partial, max_rt_seq, &mut chain);
                if terminated {
                    break;
                }
            }
        }

        if !terminated {
            let version = &view.version;

            // L0: newest-first.
            for file in version.levels[0].iter().rev() {
                let rt = file.reader.covering_range_tombstone_seq(key, snapshot_seq);
                if rt > max_rt_seq {
                    max_rt_seq = rt;
                }
                let mut partial = Vec::new();
                with_key_scratch(|buf| {
                    file.reader
                        .collect_merge_chain(lk, buf, &self.cache, &mut partial)
                })?;
                terminated = consume_partial(partial, max_rt_seq, &mut chain);
                if terminated {
                    break;
                }
            }

            // L1+: point files are non-overlapping, while RT coverage
            // may sit in sibling RT-only files.
            if !terminated {
                for level in 1..version.levels.len() {
                    let files = &version.levels[level];
                    if files.is_empty() {
                        continue;
                    }
                    for file in files {
                        if file.meta.smallest_key.as_slice() <= key
                            && key <= file.meta.largest_key.as_slice()
                        {
                            let rt = file.reader.covering_range_tombstone_seq(key, snapshot_seq);
                            if rt > max_rt_seq {
                                max_rt_seq = rt;
                            }
                        }
                    }
                    for file in files {
                        if file.meta.num_entries == 0 {
                            continue;
                        }
                        if file.meta.smallest_key.as_slice() > key
                            || key > file.meta.largest_key.as_slice()
                        {
                            continue;
                        }
                        let mut partial = Vec::new();
                        with_key_scratch(|buf| {
                            file.reader
                                .collect_merge_chain(lk, buf, &self.cache, &mut partial)
                        })?;
                        terminated = consume_partial(partial, max_rt_seq, &mut chain);
                        if terminated {
                            break;
                        }
                    }
                    if terminated {
                        break;
                    }
                }
            }
        }

        // Materialize the chain. `chain` is newest-first; the last
        // entry (if any) is either a real VALUE / DELETION terminator
        // or (if !terminated) the oldest visible merge operand.
        let (base_slice, has_terminator) = match chain.last() {
            Some((_, VALUE_TYPE_VALUE, value)) => (Some(value.as_slice()), true),
            Some((_, VALUE_TYPE_DELETION, _)) => (None, true),
            _ => (None, false),
        };

        // Build operands in oldest-first order: walk the merge part
        // of the chain (everything except the terminator slot, if
        // one is present) in reverse.
        let merge_end = if has_terminator {
            chain.len() - 1
        } else {
            chain.len()
        };
        let mut operands_owned: Vec<&[u8]> = Vec::with_capacity(merge_end);
        for entry in chain[..merge_end].iter().rev() {
            debug_assert_eq!(entry.1, VALUE_TYPE_MERGE);
            operands_owned.push(entry.2.as_slice());
        }

        if operands_owned.is_empty() {
            // No merges at all - the chain is just a plain
            // Value / Deletion / nothing. Return the base directly.
            drop(operands_owned);
            return Ok(match chain.pop() {
                Some((_, vt, value)) if vt == VALUE_TYPE_VALUE => Some(value),
                _ => None,
            });
        }

        match merge_op.full_merge(key, base_slice, &operands_owned) {
            Some(v) => Ok(Some(DbSlice::from_vec(v))),
            None => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("merge operator {} failed for key", merge_op.name()),
            )),
        }
    }

    /// Batched point lookup at a given snapshot. Returns one `Option<Vec<u8>>`
    /// per input key in the same order as `keys`. Duplicate keys in the
    /// input produce duplicate results.
    ///
    /// The batch amortizes per-call overhead - a single version snapshot,
    /// a single memtable lock acquisition per level, one logical walk of
    /// the source hierarchy - and short-circuits once every key has been
    /// resolved. All keys see the **same** consistent view, regardless of
    /// concurrent writers.
    pub(crate) fn multi_get_latest(&self, keys: &[&[u8]]) -> std::io::Result<Vec<Option<Vec<u8>>>> {
        self.ensure_open()?;
        let view = self.view.load();
        let snapshot_seq = self.visible_seq.visible();
        self.multi_get_in_view(keys, snapshot_seq, &view)
    }

    /// Batched point lookup at a caller-pinned snapshot sequence. See
    /// [`Self::get_at`] for why the load order is free here.
    pub(crate) fn multi_get_at(
        &self,
        keys: &[&[u8]],
        snapshot_seq: u64,
    ) -> std::io::Result<Vec<Option<Vec<u8>>>> {
        self.ensure_open()?;
        let view = self.view.load();
        self.multi_get_in_view(keys, snapshot_seq, &view)
    }

    /// Batched point lookup resolved against one already-loaded view.
    fn multi_get_in_view(
        &self,
        keys: &[&[u8]],
        snapshot_seq: u64,
        view: &ReadView,
    ) -> std::io::Result<Vec<Option<Vec<u8>>>> {
        // When a merge operator is configured, fall back to per-key
        // resolution - the batched walk's short-circuiting logic
        // doesn't compose cleanly with merge-chain collection, and
        // merges are rare enough that the cost difference isn't
        // worth a specialized batched path.
        if self.options.merge_operator.is_some() {
            let mut out = Vec::with_capacity(keys.len());
            let mut lk = LookupKey::from_prefixed(&[], snapshot_seq);
            for key in keys {
                lk.reset_prefixed(key, snapshot_seq);
                out.push(self.get_with_merge(&lk)?.map(DbSlice::into_vec));
            }
            return Ok(out);
        }

        let mut results: Vec<Option<Vec<u8>>> = vec![None; keys.len()];
        if keys.is_empty() {
            return Ok(results);
        }
        let mut entries = grouped_multi_get_entries(keys);
        let mut unresolved = entries.len();
        // One encoder, re-pointed per key, instead of one per key per
        // source.
        let mut lk = LookupKey::from_prefixed(&[], snapshot_seq);

        // 1. Active memtable.
        {
            let mt = &view.active;
            for entry in &mut entries {
                if entry.resolved {
                    continue;
                }
                let rt = mt.covering_range_tombstone_seq(&entry.key, snapshot_seq);
                if rt > entry.max_rt {
                    entry.max_rt = rt;
                }
                lk.reset_prefixed(&entry.key, snapshot_seq);
                if let Some((pseq, popt)) = mt.get(&lk) {
                    let value = resolve_multi_get_value(pseq, popt, entry.max_rt);
                    set_multi_get_result(entry, &mut results, value);
                    unresolved -= 1;
                }
            }
        }
        if unresolved == 0 {
            return Ok(results);
        }

        // 2. Frozen memtables, newest first.
        {
            let frozen = &view.frozen;
            for mt in frozen.iter().rev() {
                for entry in &mut entries {
                    if entry.resolved {
                        continue;
                    }
                    let rt = mt.covering_range_tombstone_seq(&entry.key, snapshot_seq);
                    if rt > entry.max_rt {
                        entry.max_rt = rt;
                    }
                    lk.reset_prefixed(&entry.key, snapshot_seq);
                    if let Some((pseq, popt)) = mt.get(&lk) {
                        let value = resolve_multi_get_value(pseq, popt, entry.max_rt);
                        set_multi_get_result(entry, &mut results, value);
                        unresolved -= 1;
                    }
                }
                if unresolved == 0 {
                    return Ok(results);
                }
            }
        }

        let version = &view.version;

        // 3. L0 SSTables, newest first.
        for file in version.levels[0].iter().rev() {
            for entry in &mut entries {
                if entry.resolved || !file_covers_key(file, &entry.key) {
                    continue;
                }
                let rt = file
                    .reader
                    .covering_range_tombstone_seq(&entry.key, snapshot_seq);
                if rt > entry.max_rt {
                    entry.max_rt = rt;
                }
                lk.reset_prefixed(&entry.key, snapshot_seq);
                match with_key_scratch(|buf| file.reader.get(&lk, buf, &self.cache))? {
                    LookupResult::Found { seq, value } => {
                        let value = resolve_multi_get_value(seq, Some(value), entry.max_rt);
                        set_multi_get_result(entry, &mut results, value);
                        unresolved -= 1;
                    }
                    LookupResult::FoundTombstone { .. } => {
                        set_multi_get_result(entry, &mut results, None);
                        unresolved -= 1;
                    }
                    LookupResult::NotInTable => {}
                }
            }
            if unresolved == 0 {
                return Ok(results);
            }
        }

        // 4. L1..Ln: point files are non-overlapping, while RT-only
        //    files can cover gaps or boundaries. Entries are sorted
        //    and deduplicated, so each file only examines the key
        //    subrange overlapped by its metadata instead of every
        //    unresolved input key.
        for level in 1..version.levels.len() {
            let files = &version.levels[level];
            if files.is_empty() {
                continue;
            }

            for file in files {
                for idx in key_range_for_file(&entries, file) {
                    let entry = &mut entries[idx];
                    if entry.resolved {
                        continue;
                    }
                    let rt = file
                        .reader
                        .covering_range_tombstone_seq(&entry.key, snapshot_seq);
                    if rt > entry.max_rt {
                        entry.max_rt = rt;
                    }
                }
            }

            for file in files {
                if file.meta.num_entries == 0 {
                    continue;
                }
                for idx in key_range_for_file(&entries, file) {
                    let entry = &mut entries[idx];
                    if entry.resolved {
                        continue;
                    }
                    lk.reset_prefixed(&entry.key, snapshot_seq);
                    match with_key_scratch(|buf| file.reader.get(&lk, buf, &self.cache))? {
                        LookupResult::Found { seq, value } => {
                            let value = resolve_multi_get_value(seq, Some(value), entry.max_rt);
                            set_multi_get_result(entry, &mut results, value);
                            unresolved -= 1;
                        }
                        LookupResult::FoundTombstone { .. } => {
                            set_multi_get_result(entry, &mut results, None);
                            unresolved -= 1;
                        }
                        LookupResult::NotInTable => {}
                    }
                    if unresolved == 0 {
                        return Ok(results);
                    }
                }
            }
            if unresolved == 0 {
                return Ok(results);
            }
        }

        Ok(results)
    }

    /// The version the last applied edit published.
    ///
    /// Cheaper than `versions.lock().current()` - one shared lock
    /// acquisition instead of the version set's exclusive one - and it
    /// is the same version, because every [`VersionStore`] guard
    /// publishes what its critical section installed before it
    /// releases the mutex.
    fn published_version(&self) -> Arc<manifest::Version> {
        Arc::clone(&self.view.load().version)
    }

    /// Retire the oldest frozen memtable in one publication. Called
    /// only once its contents are durable in an SSTable the published
    /// version already references, or once they proved to be empty.
    fn retire_oldest_frozen(&self) {
        self.view.update_memtables(|active, frozen| {
            let next = frozen.get(1..).map(<[_]>::to_vec).unwrap_or_default();
            (Arc::clone(active), next, ())
        });
    }

    /// Snapshot the current write-stall inputs: L0 file count,
    /// in-memory memtable count (active + frozen), and total bytes
    /// across all L0 files (lark's approximation of pending
    /// compaction bytes).
    fn stall_snapshot(&self) -> (usize, usize, u64) {
        let view = self.view.load();
        let l0 = view.version.levels[0].len();
        let pending_bytes: u64 = view.version.levels[0]
            .iter()
            .map(|f| f.meta.file_size)
            .sum();
        // The active memtable always counts as 1; frozen memtables
        // are whatever is still waiting for the flush path.
        let memtable_count = 1 + view.frozen.len();
        (l0, memtable_count, pending_bytes)
    }

    /// Classify the current state against the configured stall
    /// thresholds. Returns:
    ///
    /// * `None` - writes may proceed freely.
    /// * `Some(("...", true))` - hard stop: block writers until
    ///   compaction relieves the condition.
    /// * `Some(("...", false))` - slowdown: add a small delay per
    ///   write so the foreground write rate tracks compaction.
    fn stall_state(&self) -> Option<(&'static str, bool)> {
        let (l0, memtables, pending_bytes) = self.stall_snapshot();
        let opts = &self.options;
        // Stop conditions dominate over slowdown. An unconfigured
        // threshold (`0`) disables that particular trigger.
        if opts.level0_stop_writes_trigger > 0 && l0 >= opts.level0_stop_writes_trigger {
            return Some(("stop: too many L0 files", true));
        }
        if opts.max_write_buffer_number > 0
            && memtables >= opts.max_write_buffer_number.saturating_mul(2)
        {
            return Some(("stop: too many memtables", true));
        }
        if opts.hard_pending_compaction_bytes_limit > 0
            && pending_bytes >= opts.hard_pending_compaction_bytes_limit
        {
            return Some(("stop: pending compaction bytes over hard limit", true));
        }
        if opts.level0_slowdown_writes_trigger > 0 && l0 >= opts.level0_slowdown_writes_trigger {
            return Some(("slowdown: L0 files over trigger", false));
        }
        if opts.max_write_buffer_number > 0 && memtables > opts.max_write_buffer_number {
            return Some(("slowdown: memtables over trigger", false));
        }
        if opts.soft_pending_compaction_bytes_limit > 0
            && pending_bytes >= opts.soft_pending_compaction_bytes_limit
        {
            return Some(("slowdown: pending compaction bytes over soft limit", false));
        }
        None
    }

    /// Fixed per-write slowdown delay. Keeping this small (1 ms)
    /// gives foreground writers a steady back-pressure signal
    /// without freezing progress entirely; compaction gets cycles
    /// to catch up and the writer learns that the engine is under
    /// pressure.
    const SLOWDOWN_DELAY: std::time::Duration = std::time::Duration::from_millis(1);

    /// Block the current writer until the engine is ready to
    /// accept another write, or (when `no_slowdown` is set) return
    /// [`crate::Error::Busy`] immediately if any stall condition is
    /// active. Returns the number of microseconds the caller spent
    /// stalled, which is also published to the
    /// [`crate::statistics::Ticker::WriteStallMicros`] counter.
    /// Refresh the cached stall level from the current L0 /
    /// memtable / pending-bytes state. Called after any event that
    /// changes those counters (memtable rotation, compaction pass).
    pub(crate) fn refresh_stall_level(&self) {
        let level = match self.stall_state() {
            None => 0,
            Some((_, false)) => 1,
            Some((_, true)) => 2,
        };
        self.cached_stall_level.store(level, Ordering::Release);
    }

    pub(crate) fn wait_for_write_capacity(&self, no_slowdown: bool) -> Result<u64, crate::Error> {
        if self.is_closed() {
            return Err(crate::Error::Closed);
        }
        // Fast path: if the cached stall level is 0, skip the
        // stall_state() call that loads the read view and walks L0.
        // This saves a lock round-trip and a level scan per write in
        // the common no-stall scenario.
        if self.cached_stall_level.load(Ordering::Acquire) == 0 {
            return Ok(0);
        }

        let start = std::time::Instant::now();
        let mut any_stall = false;

        loop {
            if self.is_closed() {
                return Err(crate::Error::Closed);
            }
            match self.stall_state() {
                None => {
                    // Stall cleared - update the cache so
                    // subsequent writers take the fast path.
                    self.cached_stall_level.store(0, Ordering::Release);
                    break;
                }
                Some((reason, true)) => {
                    if no_slowdown {
                        return Err(crate::Error::Busy(reason));
                    }
                    any_stall = true;
                    // Bounded wait so a missed notification can't
                    // wedge a writer forever.
                    self.stall_signal
                        .wait(std::time::Duration::from_millis(100));
                }
                Some((reason, false)) => {
                    if no_slowdown {
                        return Err(crate::Error::Busy(reason));
                    }
                    any_stall = true;
                    std::thread::sleep(Self::SLOWDOWN_DELAY);
                    // One slowdown delay per call - don't loop, or
                    // a writer that just crossed the trigger would
                    // stall indefinitely at low rates.
                    break;
                }
            }
        }

        let micros = start.elapsed().as_micros() as u64;
        if any_stall && let Some(s) = self.statistics() {
            s.add(crate::statistics::Ticker::WriteStallMicros, micros);
        }
        Ok(micros)
    }

    pub(crate) fn commit_with_conflict_check(
        &self,
        conflict_keys: &[(Vec<u8>, u64)],
        point_ops: BTreeMap<Vec<u8>, Option<Vec<u8>>>,
        range_deletes: Vec<(Vec<u8>, Vec<u8>)>,
        merges: Vec<(Vec<u8>, Vec<u8>)>,
        durability: DurabilityMode,
    ) -> std::io::Result<CommitOutcome> {
        self.commit_optimistic(conflict_keys, point_ops, range_deletes, merges, durability)
    }

    /// Return the sequence number of the newest write that touched
    /// `key` across every source, or `None` if nothing ever wrote it.
    /// Used by transaction commit to detect conflicts, so it counts
    /// every kind of write: point entries, point tombstones, merge
    /// operands, and range tombstones that cover the key. The caller
    /// only needs to know "was this key written to again?".
    ///
    /// Sources are visited newest-first and range-tombstone coverage
    /// is accumulated on the way down, mirroring the read path: a
    /// tombstone in a newer source outranks a point entry found in an
    /// older one.
    fn latest_version_seq_in_view(
        &self,
        key: &[u8],
        view: &ReadView,
    ) -> std::io::Result<Option<u64>> {
        let snap = u64::MAX;
        let lk = LookupKey::from_prefixed(key, snap);
        let mut max_rt_seq: u64 = 0;
        let newest = |point_seq: u64, max_rt_seq: u64| Some(point_seq.max(max_rt_seq));
        {
            let active = &view.active;
            max_rt_seq = max_rt_seq.max(active.covering_range_tombstone_seq(key, snap));
            if let Some((seq, _)) = active.get(&lk) {
                return Ok(newest(seq, max_rt_seq));
            }
        }
        {
            let frozen = &view.frozen;
            for mt in frozen.iter().rev() {
                max_rt_seq = max_rt_seq.max(mt.covering_range_tombstone_seq(key, snap));
                if let Some((seq, _)) = mt.get(&lk) {
                    return Ok(newest(seq, max_rt_seq));
                }
            }
        }
        let version = &view.version;
        for file in version.levels[0].iter().rev() {
            max_rt_seq = max_rt_seq.max(file.reader.covering_range_tombstone_seq(key, snap));
            match with_key_scratch(|buf| file.reader.get(&lk, buf, &self.cache))? {
                LookupResult::Found { seq, .. } | LookupResult::FoundTombstone { seq } => {
                    return Ok(newest(seq, max_rt_seq));
                }
                LookupResult::NotInTable => {}
            }
        }
        for level in 1..version.levels.len() {
            let files = &version.levels[level];
            if files.is_empty() {
                continue;
            }
            for file in files {
                if file.meta.smallest_key.as_slice() <= key
                    && key <= file.meta.largest_key.as_slice()
                {
                    max_rt_seq =
                        max_rt_seq.max(file.reader.covering_range_tombstone_seq(key, snap));
                }
            }
            for file in files {
                if file.meta.num_entries == 0 {
                    continue;
                }
                if file.meta.smallest_key.as_slice() > key || key > file.meta.largest_key.as_slice()
                {
                    continue;
                }
                match with_key_scratch(|buf| file.reader.get(&lk, buf, &self.cache))? {
                    LookupResult::Found { seq, .. } | LookupResult::FoundTombstone { seq } => {
                        return Ok(newest(seq, max_rt_seq));
                    }
                    LookupResult::NotInTable => {}
                }
            }
        }
        if max_rt_seq > 0 {
            return Ok(Some(max_rt_seq));
        }
        Ok(None)
    }

    /// Rotate the active memtable when it has reached the write-buffer
    /// size. Called at the *start* of a write path so that a rotation
    /// failure is surfaced before the write is assigned a sequence or
    /// applied - keeping write errors determinate: a returned error means
    /// the write did not land, never that it landed but a later step
    /// failed. Caller must hold the pipeline mutex.
    fn rotate_if_full(&self) -> std::io::Result<()> {
        if self.view.load().active.approximate_size() >= self.options.write_buffer_size {
            self.rotate_memtable()?;
        }
        Ok(())
    }

    fn rotate_memtable(&self) -> std::io::Result<()> {
        self.ensure_writable()?;
        // One publication: the sealed memtable joins `frozen` in the
        // same view that hands writers the fresh active one, so no
        // reader can catch it in neither. The fresh memtable is built
        // before the publication so the fallible allocation happens
        // once, outside it.
        let fresh = Arc::new(MemTable::new(&self.memtable_config)?);
        self.view.update_memtables(|active, frozen| {
            let mut next_frozen = frozen.to_vec();
            next_frozen.push(Arc::clone(active));
            (fresh, next_frozen, ())
        });

        let new_wal_id = {
            let mut versions = self.versions.lock();
            let version = versions.current();
            let id = version.next_file_id;
            versions.apply(&[VersionEdit::SetNextFileId(id + 1)])?;
            id
        };

        let wal_path = self.wal_dir.join(wal_filename(new_wal_id));
        let new_wal = Wal::create(&wal_path)?;

        let old_wal = {
            let mut wal = self.active_wal.lock();
            wal.replace(new_wal).ok_or_else(Self::read_only_error)?
        };

        self.wal_id.store(new_wal_id, Ordering::Release);
        self.flush_frozen_memtable(old_wal)?;
        self.refresh_stall_level();

        Ok(())
    }

    fn flush_frozen_memtable(&self, old_wal: Wal) -> std::io::Result<()> {
        let flush_start = std::time::Instant::now();
        let memtable = match self.view.load().frozen.first() {
            Some(mt) => Arc::clone(mt),
            None => return Ok(()),
        };

        let range_tombstones = memtable.clone_range_tombstones();

        if memtable.is_empty() && range_tombstones.is_empty() {
            self.retire_oldest_frozen();
            let _ = Wal::remove(old_wal.path());
            return Ok(());
        }

        let file_id = {
            let mut versions = self.versions.lock();
            let version = versions.current();
            let id = version.next_file_id;
            versions.apply(&[VersionEdit::SetNextFileId(id + 1)])?;
            id
        };

        let sst_path = self.sst_dir.join(sst_filename(file_id));

        // Memtable flushes always land at L0 - pick L0's codec.
        let mut writer = SsTableWriter::new(
            &sst_path,
            self.options.block_size,
            self.options.bloom_bits_per_key,
            self.options.compression_for_level(0),
            self.options.prefix_extractor.clone(),
            self.options.partitioned_index,
            self.options.metadata_block_size,
        )?;

        // Walk the memtable in internal-key order and copy every version
        // and tombstone into the SSTable unchanged, preserving MVCC.
        // The walk streams straight out of the arena: a flush holds one
        // entry plus the block builder, never a second copy of the
        // whole memtable.
        memtable.try_for_each_entry(|internal_key, value| writer.add(internal_key, value))?;

        // Persist range tombstones alongside the point entries.
        for rt in &range_tombstones {
            writer.add_range_tombstone(&rt.start, &rt.end, rt.seq);
        }

        let summary = match writer.finish()? {
            Some(s) => s,
            None => {
                self.retire_oldest_frozen();
                let _ = Wal::remove(old_wal.path());
                let _ = std::fs::remove_file(&sst_path);
                return Ok(());
            }
        };

        let file_size = std::fs::metadata(&sst_path)?.len();
        let num_entries = summary.num_entries;

        // Throttle background I/O so bursts of flush writes don't
        // starve foreground traffic. Rate-limiting is opt-in via
        // `Options::rate_limiter`; a `None` limiter is a no-op.
        if let Some(limiter) = &self.options.rate_limiter {
            limiter.request(file_size, crate::rate_limiter::Priority::Low);
        }

        let reader = Arc::new(SsTableReader::open_with(
            &sst_path,
            file_id,
            self.options.metadata_policy(),
        )?);
        let file = LiveSst::new(
            SsTableMeta {
                file_id,
                smallest_key: summary.smallest_user_key,
                largest_key: summary.largest_user_key,
                file_size,
                num_entries,
            },
            reader,
        );

        let seq = self.latest_seq.load(Ordering::Acquire);
        let edits = vec![
            VersionEdit::AddFile { level: 0, file },
            VersionEdit::SetLastSeq(seq),
        ];
        self.versions.lock().apply(&edits)?;

        // Retired only now: until the `AddFile` above is published, the
        // flushed data lives in this memtable alone.
        self.retire_oldest_frozen();
        let _ = Wal::remove(old_wal.path());
        self.compaction.lock().notify();

        // Publish flush statistics before the listener dispatch
        // so callers that react to `on_flush_completed` can
        // already see the updated tickers.
        if let Some(s) = self.statistics() {
            s.add(crate::statistics::Ticker::FlushCount, 1);
            s.add(crate::statistics::Ticker::FlushBytesWritten, file_size);
            s.record(
                crate::statistics::Histogram::FlushTime,
                flush_start.elapsed().as_micros() as u64,
            );
        }

        // Dispatch lifecycle events to any registered listeners.
        // Two callbacks fire per flush: `on_table_file_created`
        // for the new SSTable and `on_flush_completed` with
        // memtable-level aggregates.
        if !self.options.listeners.is_empty() {
            let (smallest, largest) = {
                // Version was just applied; pull the newly-added
                // file's metadata back out so listeners see the
                // exact bounds the engine committed.
                let ver = self.versions.lock().current();
                if let Some(added) = ver.levels[0].iter().find(|f| f.meta.file_id == file_id) {
                    (
                        added.meta.smallest_key.clone(),
                        added.meta.largest_key.clone(),
                    )
                } else {
                    (Vec::new(), Vec::new())
                }
            };
            let duration = flush_start.elapsed();
            let create_info = event_listener::TableFileCreationInfo {
                file_id,
                file_path: sst_path.clone(),
                level: 0,
                reason: event_listener::TableFileCreationReason::Flush,
                file_size,
                num_entries,
            };
            let flush_info = event_listener::FlushJobInfo {
                file_id,
                file_path: sst_path.clone(),
                file_size,
                num_entries,
                smallest_key: smallest,
                largest_key: largest,
                duration,
            };
            event_listener::dispatch(&self.options.listeners, |l| {
                l.on_table_file_created(&create_info)
            });
            event_listener::dispatch(&self.options.listeners, |l| {
                l.on_flush_completed(&flush_info)
            });
        }

        tracing::info!(
            file_id,
            entries = num_entries,
            size = file_size,
            "Flushed memtable to L0 SSTable"
        );

        Ok(())
    }

    /// Synchronously compact SSTables overlapping the user-key range
    /// `[start, end)` down to the bottommost non-empty level.
    ///
    /// - Flushes the active memtable first so any matching in-memory
    ///   data reaches L0 before compaction picks inputs.
    /// - Acquires the engine-wide compaction lock so the background
    ///   scheduler can't pick an overlapping input set concurrently.
    /// - Walks levels 0..MAX_LEVELS-1, picking range-overlapping files
    ///   and merging them into the next level.
    pub(crate) fn compact_range(
        &self,
        start: Option<&[u8]>,
        end: Option<&[u8]>,
    ) -> std::io::Result<()> {
        self.ensure_writable()?;
        // 1. Flush the active memtable so any in-memory data that
        //    overlaps the range is materialized in L0. We only touch
        //    the write lock if there's actually data to flush. "Data"
        //    here includes range tombstones, not just point entries.
        let needs_flush = |mt: &MemTable| !mt.is_empty() || !mt.clone_range_tombstones().is_empty();
        if needs_flush(&self.view.load().active) {
            let _write_guard = self.pipeline.lock();
            if needs_flush(&self.view.load().active) {
                self.rotate_memtable()?;
            }
        }

        // 2. Exclude all background workers for the duration of the
        //    range walk. Write lock blocks until every in-flight
        //    background pass releases its read lock.
        let _compact_guard = self.compaction_lock.write();
        self.ensure_writable()?;

        // 3. Compute the snapshot-pinning GC horizon so compaction can
        //    drop versions that no live snapshot needs.
        let pin_seq = self.oldest_live_seq();

        // 4. Run the level-by-level push-down.
        let compaction_opts = compaction::CompactionOptions {
            l0_compaction_trigger: self.options.l0_compaction_trigger,
            level_base_bytes: self.options.level_base_bytes,
            level_size_multiplier: self.options.level_size_multiplier,
            target_file_size: self.options.target_file_size,
            block_size: self.options.block_size,
            bloom_bits_per_key: self.options.bloom_bits_per_key,
            compression: self.options.compression,
            compression_per_level: self.options.compression_per_level.clone(),
            compaction_filter: self.options.compaction_filter.clone(),
            prefix_extractor: self.options.prefix_extractor.clone(),
            merge_operator: self.options.merge_operator.clone(),
            listeners: self.options.listeners.clone(),
            statistics: self.options.statistics.clone(),
            rate_limiter: self.options.rate_limiter.clone(),
            compaction_style: self.options.compaction_style,
            fifo_compaction_options: self.options.fifo_compaction_options,
            universal_compaction_options: self.options.universal_compaction_options,
            evict_compaction_data_from_page_cache: self
                .options
                .evict_compaction_data_from_page_cache,
            max_background_compactions: self.options.max_background_compactions,
            partitioned_index: self.options.partitioned_index,
            metadata_block_size: self.options.metadata_block_size,
            cache_index_and_filter_blocks: self.options.cache_index_and_filter_blocks,
        };
        // Under FIFO compaction there is no level push-down; a
        // synchronous compact_range just flushes the memtable and
        // runs the FIFO picker so any pending files over the cap
        // get dropped deterministically.
        if matches!(
            self.options.compaction_style,
            crate::options::CompactionStyle::Fifo
        ) {
            let _ = compaction::run_fifo_pass(&self.versions, &self.sst_dir, &compaction_opts)?;
            return Ok(());
        }

        // Under Universal compaction a manual compact_range folds
        // every L0 file into one run, matching the "force full
        // compaction" semantics a caller expects.
        if matches!(
            self.options.compaction_style,
            crate::options::CompactionStyle::Universal
        ) {
            compaction::run_universal_full_compaction(
                &self.versions,
                &self.sst_dir,
                &self.cache,
                &compaction_opts,
                pin_seq,
            )?;
            return Ok(());
        }

        compaction::run_compact_range(
            &self.versions,
            &self.sst_dir,
            &self.cache,
            &compaction_opts,
            start,
            end,
            pin_seq,
        )
    }

    /// Bulk-ingest a set of externally-built SSTable files. Each file
    /// is re-emitted into `sst_dir` so its entries are rewritten with
    /// a freshly allocated sequence number (one per file), then the
    /// file is added to the version at an appropriate level:
    ///
    /// - L0 if the file's user-key range overlaps any file at any
    ///   existing level;
    /// - otherwise the deepest level whose files are all strictly
    ///   disjoint from the input range.
    ///
    /// If `ingest_behind` is set the file is forced to the bottommost
    /// level - any overlap is an error. If `snapshot_consistency` is
    /// set the call is rejected while any snapshot is pinned (ingest
    /// would otherwise inject a new seq that older snapshots cannot
    /// consistently observe).
    pub(crate) fn ingest_external_files<F>(
        &self,
        files: &[PathBuf],
        ingest_opts: &crate::sst_file_writer::IngestOptions,
        mut validate_user_key: F,
    ) -> std::io::Result<()>
    where
        F: FnMut(&[u8]) -> std::io::Result<()>,
    {
        use crate::engine::internal_key::decode_internal_key;

        self.ensure_writable()?;

        if files.is_empty() {
            return Ok(());
        }

        if ingest_opts.snapshot_consistency && self.oldest_live_seq() != u64::MAX {
            return Err(std::io::Error::other(
                "ingest_external_files: snapshot isolation would be violated \
                 because a live snapshot is pinned (use snapshot_consistency=false \
                 to override)",
            ));
        }

        // Pre-open every source file and compute its key range, so we
        // can reject bad inputs before we start mutating anything.
        let mut sources: Vec<IngestSource> = Vec::with_capacity(files.len());
        let _cache_guard = IngestCacheGuard::new(&self.cache, files.len());
        for (source_idx, path) in files.iter().enumerate() {
            // Source files are read through the engine's shared block
            // cache, which is keyed `(file_id, offset)`. They are not in
            // the version, so they have no allocated id; giving every
            // source the same one (or one a live file already owns) makes
            // the second source read the first source's cached blocks and
            // silently ingest the wrong bytes. Ids are handed out from the
            // top of the space, which `next_file_id` counts up from 1 and
            // never reaches.
            let cache_id = ingest_probe_file_id(source_idx);
            let reader = SsTableReader::open(path, cache_id).map_err(|e| {
                std::io::Error::new(e.kind(), format!("ingest: open {}: {e}", path.display()))
            })?;
            // Stream the source instead of materialising it: the
            // validation pass holds one entry and one data block, and
            // tracks the key range as it goes.
            let mut first_user_key: Option<Vec<u8>> = None;
            let mut last_user_key: Vec<u8> = Vec::new();
            {
                let mut entries = reader.iter_internal_stream(&self.cache)?;
                while let Some((ik, value)) = entries.next_entry()? {
                    let (user_key, _, _) = decode_internal_key(&ik);
                    self.validate_prefixed_key_size(user_key).map_err(|e| {
                        std::io::Error::new(
                            e.kind(),
                            format!(
                                "ingest: source file {} contains an over-sized key: {e}",
                                path.display()
                            ),
                        )
                    })?;
                    self.validate_value_size(&value).map_err(|e| {
                        std::io::Error::new(
                            e.kind(),
                            format!(
                                "ingest: source file {} contains an over-sized value: {e}",
                                path.display()
                            ),
                        )
                    })?;
                    validate_user_key(user_key).map_err(|e| {
                    std::io::Error::new(
                        e.kind(),
                        format!(
                            "ingest: source file {} contains a key outside live column families: {e}",
                            path.display()
                        ),
                    )
                })?;
                    if first_user_key.is_none() {
                        first_user_key = Some(user_key.to_vec());
                    }
                    last_user_key.clear();
                    last_user_key.extend_from_slice(user_key);
                }
            }
            let rts = reader.range_tombstones();
            for rt in rts {
                self.validate_prefixed_key_size(&rt.start).map_err(|e| {
                    std::io::Error::new(
                        e.kind(),
                        format!(
                            "ingest: source file {} contains an over-sized range tombstone start: {e}",
                            path.display()
                        ),
                    )
                })?;
                self.validate_prefixed_key_size(&rt.end).map_err(|e| {
                    std::io::Error::new(
                        e.kind(),
                        format!(
                            "ingest: source file {} contains an over-sized range tombstone end: {e}",
                            path.display()
                        ),
                    )
                })?;
                validate_user_key(&rt.start).map_err(|e| {
                    std::io::Error::new(
                        e.kind(),
                        format!(
                            "ingest: source file {} contains a range tombstone start outside live column families: {e}",
                            path.display()
                        ),
                    )
                })?;
                validate_user_key(&rt.end).map_err(|e| {
                    std::io::Error::new(
                        e.kind(),
                        format!(
                            "ingest: source file {} contains a range tombstone end outside live column families: {e}",
                            path.display()
                        ),
                    )
                })?;
            }
            let (smallest, largest) = if let Some(first) = first_user_key {
                (first, last_user_key)
            } else if !rts.is_empty() {
                let mut lo = rts[0].start.clone();
                let mut hi = rts[0].end.clone();
                for rt in rts.iter().skip(1) {
                    if rt.start < lo {
                        lo = rt.start.clone();
                    }
                    if rt.end > hi {
                        hi = rt.end.clone();
                    }
                }
                (lo, hi)
            } else {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("ingest: source file {} is empty", path.display()),
                ));
            };
            sources.push(IngestSource {
                path: path.clone(),
                reader,
                smallest,
                largest,
            });
        }

        // Exclude all background workers for the duration of the
        // ingest, the same pattern as `compact_range`.
        let _compact_guard = self.compaction_lock.write();
        self.ensure_writable()?;

        for source in &sources {
            self.ingest_one(source, ingest_opts)?;
        }

        self.compaction.lock().notify();
        Ok(())
    }

    fn ingest_one(
        &self,
        source: &IngestSource,
        ingest_opts: &crate::sst_file_writer::IngestOptions,
    ) -> std::io::Result<()> {
        use crate::engine::internal_key::{decode_internal_key, encode_internal_key};

        // Flush the active memtable if it overlaps the ingest range.
        // The cheapest correct thing is to flush unconditionally when
        // the memtable is non-empty and we might be landing at L0 -
        // which is the only level where a concurrent memtable could
        // shadow the ingested keys.
        let needs_flush = {
            let view = self.view.load();
            !view.active.is_empty() || !view.active.clone_range_tombstones().is_empty()
        };
        if needs_flush {
            let _write_guard = self.pipeline.lock();
            let view = self.view.load();
            if !view.active.is_empty() || !view.active.clone_range_tombstones().is_empty() {
                drop(view);
                self.rotate_memtable()?;
            }
        }

        // Compute the target level from the current version.
        let version = self.published_version();
        let target_level = compute_target_level(
            &version,
            &source.smallest,
            &source.largest,
            ingest_opts.ingest_behind,
        )?;

        // Allocate a new file_id and a new global seq. All entries in
        // the emitted file share the allocated seq.
        let file_id = {
            let mut guard = self.versions.lock();
            let current = guard.current();
            let id = current.next_file_id;
            guard.apply(&[VersionEdit::SetNextFileId(id + 1)])?;
            id
        };
        let ingest_seq = self.latest_seq.fetch_add(1, Ordering::AcqRel) + 1;

        let dest_path = self.sst_dir.join(sst_filename(file_id));
        let mut writer = SsTableWriter::new(
            &dest_path,
            self.options.block_size,
            self.options.bloom_bits_per_key,
            self.options.compression_for_level(target_level),
            self.options.prefix_extractor.clone(),
            self.options.partitioned_index,
            self.options.metadata_block_size,
        )?;

        // Re-encode every point entry with the ingest seq. The second
        // pass streams too, so an ingest never holds more than one
        // source data block regardless of how large the file is.
        let mut entries = source.reader.iter_internal_stream(&self.cache)?;
        while let Some((ik, value)) = entries.next_entry()? {
            let (uk, _old_seq, vt) = decode_internal_key(&ik);
            let new_ik = encode_internal_key(uk, ingest_seq, vt);
            writer.add(&new_ik, &value)?;
        }
        drop(entries);
        // Carry range tombstones across with the same seq rewrite.
        for rt in source.reader.range_tombstones() {
            writer.add_range_tombstone(&rt.start, &rt.end, ingest_seq);
        }

        let summary = match writer.finish()? {
            Some(s) => s,
            None => {
                let _ = std::fs::remove_file(&dest_path);
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!(
                        "ingest: source file {} produced empty output",
                        source.path.display()
                    ),
                ));
            }
        };

        let file_size = std::fs::metadata(&dest_path)?.len();
        let reader = Arc::new(SsTableReader::open_with(
            &dest_path,
            file_id,
            self.options.metadata_policy(),
        )?);
        let live = LiveSst::new(
            SsTableMeta {
                file_id,
                smallest_key: summary.smallest_user_key,
                largest_key: summary.largest_user_key,
                file_size,
                num_entries: summary.num_entries,
            },
            reader,
        );

        let edits = vec![
            VersionEdit::AddFile {
                level: target_level,
                file: live,
            },
            VersionEdit::SetLastSeq(ingest_seq),
        ];
        self.versions.lock().apply(&edits)?;

        // The ingested file is now installed in the version, so publish its
        // sequence. `fetch_max` guards against a concurrent write (ingest
        // holds the compaction lock, not the write lock) having already
        // published a higher horizon.
        self.visible_seq.publish(ingest_seq);

        if !self.options.listeners.is_empty() {
            // Fire the table-created event first (file-level
            // observation) and then the ingest-specific event
            // (caller-level observation carrying the original
            // external path).
            let create_info = event_listener::TableFileCreationInfo {
                file_id,
                file_path: dest_path.clone(),
                level: target_level,
                reason: event_listener::TableFileCreationReason::Recovery,
                file_size,
                num_entries: summary.num_entries,
            };
            event_listener::dispatch(&self.options.listeners, |l| {
                l.on_table_file_created(&create_info)
            });
            let ingest_info = event_listener::ExternalFileIngestionInfo {
                external_file_path: source.path.clone(),
                internal_file_id: file_id,
                level: target_level,
                num_entries: summary.num_entries,
                file_size,
            };
            event_listener::dispatch(&self.options.listeners, |l| {
                l.on_external_file_ingested(&ingest_info)
            });
        }

        tracing::info!(
            file_id,
            target_level,
            ingest_seq,
            entries = summary.num_entries,
            size = file_size,
            source = %source.path.display(),
            "Ingested external SSTable"
        );
        Ok(())
    }

    /// Atomically capture a consistent snapshot of the on-disk state
    /// for checkpoint / backup purposes.
    ///
    /// Flushes the active memtable (if non-empty) under the write
    /// lock so every live byte is in an SSTable, compacts the
    /// manifest so the on-disk form is a single rewrite of the
    /// current version, and returns the captured `Arc<Version>` plus
    /// the paths of the files a caller needs to copy or hard-link.
    ///
    /// The returned [`CheckpointSnapshot`] holds the engine's
    /// compaction lock, pinning every referenced file against
    /// concurrent unlink. Callers MUST drop the snapshot as soon as
    /// their filesystem work is done - a snapshot whose lifetime
    /// outlives the enclosing function scope can deadlock a
    /// concurrent `db.close()` / `drop(db)` that joins the background
    /// compaction thread (the thread will block on the same lock the
    /// snapshot holds).
    pub(crate) fn checkpoint_capture(&self) -> std::io::Result<CheckpointSnapshot> {
        self.ensure_writable()?;
        // Drain, do not merely seal. A checkpoint captures the SSTables
        // the current version names and copies no WAL, so a memtable that
        // is still waiting on a background flush would be missing from
        // the result entirely. Done before the compaction lock is taken,
        // because flushing needs it.
        self.drain_memtables(ActiveFlush::Always)?;

        let compaction_guard = self.compaction_lock.write_arc();
        self.ensure_writable()?;

        let version;
        let manifest_path;
        let manifest_len;
        {
            let mut versions = self.versions.lock();
            versions.compact_manifest()?;
            version = versions.current();
            manifest_path = versions.manifest_path().to_path_buf();
            // Record the manifest's on-disk size immediately after
            // compaction so a later copy can truncate at the exact
            // captured boundary. Concurrent flushes still append
            // `AddFile` records (they take `versions.lock()` but not
            // the compaction lock we're holding), and those records
            // reference files that are *not* in the captured
            // version. Copying them into a checkpoint manifest would
            // make recovery fail on the missing files.
            manifest_len = std::fs::metadata(&manifest_path)?.len();
        }

        Ok(CheckpointSnapshot {
            version,
            manifest_path,
            manifest_len,
            sst_dir: self.sst_dir.clone(),
            _compaction_guard: compaction_guard,
        })
    }

    /// Approximate on-disk bytes whose user key falls in
    /// `[start, end)`. Fans out over every SSTable whose own
    /// user-key range overlaps the query range and sums their
    /// per-range estimates. Index-only - no data-block decompression
    /// happens, so the cost scales with `num_files * log(num_blocks)`.
    pub(crate) fn approximate_size_in_range(&self, start: &[u8], end: &[u8]) -> u64 {
        if start >= end {
            return 0;
        }
        let version = self.published_version();
        let mut total: u64 = 0;
        for level in &version.levels {
            for file in level {
                if file.meta.largest_key.as_slice() < start {
                    continue;
                }
                if file.meta.smallest_key.as_slice() >= end {
                    continue;
                }
                total += file
                    .reader
                    .approximate_size_in_range(start, end, &self.cache);
            }
        }
        total
    }

    /// Exact `(count, size)` for every entry in the active memtable
    /// whose user key falls in `[start, end)`. Frozen memtables are
    /// *not* included - a caller that wants "everything in memory"
    /// should call this and also walk the frozen memtables
    /// separately.
    pub(crate) fn approximate_memtable_stats(&self, start: &[u8], end: &[u8]) -> (u64, u64) {
        self.view
            .load()
            .active
            .approximate_stats_for_range(start, end)
    }

    // ── property helpers ───────────────────────────────────────────────
    //
    // These return raw values consumed by `Db::get_property` /
    // `Db::get_int_property`. Every method is cheap - no block
    // reads, no locks held beyond a short `versions.lock()` or
    // memtable read.

    /// Number of SSTable files at a specific level. Returns 0 for
    /// out-of-range levels rather than panicking, so
    /// `lark.num-files-at-level<N>` for unknown levels reads
    /// cleanly as `Some(0)`.
    pub(crate) fn num_files_at_level(&self, level: usize) -> u64 {
        let version = self.published_version();
        version
            .levels
            .get(level)
            .map(|files| files.len() as u64)
            .unwrap_or(0)
    }

    /// Total size in bytes across every level of the current
    /// version - sum of every `LiveSst::meta.file_size`.
    pub(crate) fn total_sst_size(&self) -> u64 {
        let version = self.published_version();
        version
            .levels
            .iter()
            .flat_map(|level| level.iter())
            .map(|f| f.meta.file_size)
            .sum()
    }

    /// Total `num_entries` across every current SSTable. The
    /// manifest tracks this per file at ingest / flush /
    /// compaction time, so the sum is free to compute.
    pub(crate) fn total_sst_num_entries(&self) -> u64 {
        let version = self.published_version();
        version
            .levels
            .iter()
            .flat_map(|level| level.iter())
            .map(|f| f.meta.num_entries)
            .sum()
    }

    /// Approximate size of the active memtable (sum of every
    /// inserted internal key + value length seen so far; tracked
    /// by the memtable itself).
    pub(crate) fn active_memtable_size(&self) -> u64 {
        self.view.load().active.approximate_size() as u64
    }

    /// Bytes every in-memory memtable actually reserved from the global
    /// allocator: the sum of their arena chunk sizes plus the heap their
    /// range tombstones own.
    ///
    /// [`LarkEngine::active_memtable_size`] is the payload figure that
    /// `write_buffer_size` bounds; this is what the process is really
    /// holding, so the gap between them is the arena's rounding waste.
    pub(crate) fn memtables_reserved_size(&self) -> u64 {
        let active = self.view.load().active.reserved_size() as u64;
        let frozen: u64 = self
            .view
            .load()
            .frozen
            .iter()
            .map(|mt| mt.reserved_size() as u64)
            .sum();
        active + frozen
    }

    /// Bytes parked in the memtable arena's recycling pool, waiting to
    /// back the next memtable instead of being returned to the global
    /// allocator. Bounded by
    /// `write_buffer_size * max_write_buffer_number`.
    pub(crate) fn arena_pool_size(&self) -> u64 {
        self.memtable_config.pool_bytes().0 as u64
    }

    /// Total approximate size of every frozen memtable.
    pub(crate) fn frozen_memtables_size(&self) -> u64 {
        self.view
            .load()
            .frozen
            .iter()
            .map(|mt| mt.approximate_size() as u64)
            .sum()
    }

    /// Number of live snapshots. Counts pins, not distinct seqs -
    /// two snapshots taken at the same seq contribute two.
    pub(crate) fn live_snapshot_count(&self) -> u64 {
        self.snapshot_registry.live_count()
    }

    /// Total bytes currently held by the block cache across all
    /// shards. Used by the `lark.block-cache-usage` property.
    pub(crate) fn block_cache_usage(&self) -> usize {
        self.cache.usage()
    }

    /// Block cache capacity in bytes (the sum of every shard's
    /// budget). Used by the `lark.block-cache-capacity`
    /// property.
    pub(crate) fn block_cache_capacity(&self) -> usize {
        self.cache.capacity()
    }

    /// Bytes the currently-live SSTable readers hold *outside* the
    /// block cache budget: pinned indexes, pinned filter regions, any
    /// metadata block the cache refused, and range tombstones.
    ///
    /// This is the honest counterpart to `lark.block-cache-usage`:
    /// together they account for every byte of SSTable metadata the
    /// engine is holding. With
    /// [`Options::cache_index_and_filter_blocks`] off this number grows
    /// with the number of open files; with it on, only the pinned
    /// top-level index of each partitioned file and its range
    /// tombstones remain here.
    ///
    /// [`Options::cache_index_and_filter_blocks`]: crate::Options::cache_index_and_filter_blocks
    pub(crate) fn pinned_metadata_bytes(&self) -> usize {
        let version = self.versions.lock().current();
        version
            .levels
            .iter()
            .flat_map(|level| level.iter())
            .map(|file| file.reader.pinned_metadata_bytes())
            .sum()
    }

    /// Unix-seconds timestamp of the oldest live snapshot, or
    /// `None` when no snapshot is alive.
    pub(crate) fn oldest_snapshot_time_unix(&self) -> Option<u64> {
        self.snapshot_registry.oldest_snapshot_time_unix()
    }

    /// Borrow the current version so a caller can walk every
    /// SSTable's metadata - used by `lark.sstables` formatter.
    pub(crate) fn current_version(&self) -> Arc<manifest::Version> {
        self.published_version()
    }

    /// Drop all data in the engine.
    pub(crate) fn drop_all(&self) -> std::io::Result<()> {
        self.ensure_writable()?;
        let _write_guard = self.pipeline.lock();
        self.ensure_writable()?;

        // Published before the version `Reset` below, so no reader can
        // see the pre-drop memtables against the post-drop version.
        // `drop_all` holds `write_lock`, which excludes writers but not
        // readers: a concurrent reader may still briefly observe
        // pre-drop SSTable data, exactly as before this view existed.
        let fresh = Arc::new(MemTable::new(&self.memtable_config)?);
        self.view.update_memtables(|_, _| (fresh, Vec::new(), ()));

        let (old_version, wal_id, wal_path, new_wal) = {
            let mut versions = self.versions.lock();
            let old_version = versions.current();
            let id = old_version.next_file_id;
            let wal_path = self.wal_dir.join(wal_filename(id));
            let new_wal = Wal::create(&wal_path)?;
            versions.apply(&[VersionEdit::Reset {
                next_file_id: id + 1,
                min_wal_id: id,
            }])?;
            (old_version, id, wal_path, new_wal)
        };

        self.cache.clear();
        let _old_wal = self.active_wal.lock().replace(new_wal);
        self.wal_id.store(wal_id, Ordering::Release);
        self.latest_seq.store(0, Ordering::Release);
        self.visible_seq.reset();

        self.versions.lock().compact_manifest()?;

        remove_obsolete_sst_files(&self.sst_dir, &old_version)?;
        remove_obsolete_wal_files(&self.wal_dir, &wal_path)?;

        Ok(())
    }

    /// Test-only: whether the active memtable currently holds any
    /// entries. Used by `compact_range` tests to verify that the
    /// memtable was flushed to L0 as part of the range walk.
    #[cfg(test)]
    pub(crate) fn active_memtable_is_empty(&self) -> bool {
        let view = self.view.load();
        view.active.is_empty() && view.active.clone_range_tombstones().is_empty()
    }

    /// Test-only: number of SSTable files at `level` in the current
    /// version.
    #[cfg(test)]
    pub(crate) fn level_file_count(&self, level: usize) -> usize {
        self.published_version().levels[level].len()
    }

    /// Test-only: total number of SSTable files across every level.
    #[cfg(test)]
    pub(crate) fn total_file_count(&self) -> usize {
        let v = self.published_version();
        v.levels.iter().map(|level| level.len()).sum()
    }

    /// Test-only: collect every raw `(seq, value_type)` version of
    /// `user_key` currently persisted across all SSTables. Used by the
    /// snapshot-pinning GC tests to check the post-compaction on-disk
    /// state - not just what reads see.
    #[cfg(test)]
    pub(crate) fn all_persisted_versions_of(
        &self,
        user_key: &[u8],
    ) -> std::io::Result<Vec<(u64, u8)>> {
        use internal_key::decode_internal_key;

        let version = self.published_version();
        let mut out = Vec::new();
        for level in &version.levels {
            for file in level {
                let mut entries = file.reader.iter_internal_stream(&self.cache)?;
                while let Some((ik, _v)) = entries.next_entry()? {
                    let (uk, seq, vt) = decode_internal_key(&ik);
                    if uk == user_key {
                        out.push((seq, vt));
                    }
                }
            }
        }
        out.sort();
        Ok(out)
    }

    /// Flush all data to disk and shut down background threads.
    pub(crate) fn close(&self) -> std::io::Result<()> {
        let _close_guard = self.close_lock.lock();
        match self.close_state.load(Ordering::Acquire) {
            CLOSE_STATE_CLOSED => return Ok(()),
            CLOSE_STATE_CLOSING => return Err(Self::closed_error()),
            _ => {}
        }

        self.close_state
            .store(CLOSE_STATE_CLOSING, Ordering::Release);
        self.stall_signal.notify_all();

        match self.close_inner() {
            Ok(()) => {
                self.close_state
                    .store(CLOSE_STATE_CLOSED, Ordering::Release);
                Ok(())
            }
            Err(err) => {
                self.close_state.store(CLOSE_STATE_OPEN, Ordering::Release);
                self.stall_signal.notify_all();
                Err(err)
            }
        }
    }

    fn close_inner(&self) -> std::io::Result<()> {
        if self.is_read_only() {
            self.compaction.lock().shutdown();
            return Ok(());
        }

        self.flush_memtables_for_close()?;

        self.active_wal
            .lock()
            .as_mut()
            .ok_or_else(Self::read_only_error)?
            .sync_data()?;
        self.compaction.lock().shutdown();

        Ok(())
    }

    fn flush_memtables_for_close(&self) -> std::io::Result<()> {
        self.drain_memtables(ActiveFlush::WhenFull)
    }

    /// Flush frozen memtables until none remain, and the active one
    /// according to `active`.
    ///
    /// [`ActiveFlush::Always`] is what a checkpoint needs: it captures
    /// the SSTables the current version names and copies no WAL, so any
    /// write still sitting in a memtable would be absent from the
    /// result. Sealing without waiting is not enough either, because a
    /// sealed memtable is flushed by a background worker and the capture
    /// would race it.
    fn drain_memtables(&self, active: ActiveFlush) -> std::io::Result<()> {
        let active_pending = |view: &ReadView| match active {
            ActiveFlush::WhenFull => memtable_needs_flush(&view.active),
            ActiveFlush::Always => {
                !view.active.is_empty() || !view.active.clone_range_tombstones().is_empty()
            }
        };
        loop {
            let should_flush = {
                let view = self.view.load();
                active_pending(&view) || !view.frozen.is_empty()
            };
            if !should_flush {
                return Ok(());
            }

            let old_wal = {
                let _write_guard = self.pipeline.lock();
                let view = self.view.load();
                let active_needs_flush = active_pending(&view);
                let has_frozen = !view.frozen.is_empty();
                drop(view);
                if !active_needs_flush && !has_frozen {
                    continue;
                }

                let new_wal_id = {
                    let mut versions = self.versions.lock();
                    let version = versions.current();
                    let id = version.next_file_id;
                    versions.apply(&[VersionEdit::SetNextFileId(id + 1)])?;
                    id
                };
                let wal_for_flush = Wal::create(&self.wal_dir.join(wal_filename(new_wal_id)))?;

                if active_needs_flush {
                    // One publication for the seal and the enqueue: two
                    // would leave a window where the sealed memtable is
                    // in neither the active slot nor the frozen list.
                    let fresh = Arc::new(MemTable::new(&self.memtable_config)?);
                    self.view.update_memtables(|active, frozen| {
                        let sealed = Arc::clone(active);
                        let mut next_frozen = frozen.to_vec();
                        next_frozen.push(sealed);
                        (fresh, next_frozen, ())
                    });
                }

                let old_wal = self
                    .active_wal
                    .lock()
                    .replace(wal_for_flush)
                    .ok_or_else(Self::read_only_error)?;
                self.wal_id.store(new_wal_id, Ordering::Release);
                old_wal
            };

            self.flush_frozen_memtable(old_wal)?;
        }
    }
}

/// Block-cache file id for the `n`-th source of one ingest call.
///
/// Live file ids are allocated upward from 1 by `next_file_id`, so ids
/// taken from the top of the space cannot collide with a real file, and
/// distinct sources cannot collide with each other.
/// Whether a drain must flush the active memtable unconditionally, or
/// only once it has grown past the write buffer.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ActiveFlush {
    WhenFull,
    Always,
}

fn ingest_probe_file_id(source_idx: usize) -> u64 {
    u64::MAX - source_idx as u64
}

/// Drops the ingest sources' blocks from the shared block cache when the
/// ingest call returns, by any path.
///
/// Source blocks are read twice (validate, then rewrite) and are useless
/// afterwards, so leaving them resident would hold block-cache budget for
/// bytes no reader can ask for again.
struct IngestCacheGuard<'a> {
    cache: &'a BlockCache,
    sources: usize,
}

impl<'a> IngestCacheGuard<'a> {
    fn new(cache: &'a BlockCache, sources: usize) -> Self {
        Self { cache, sources }
    }
}

impl Drop for IngestCacheGuard<'_> {
    fn drop(&mut self) {
        for source_idx in 0..self.sources {
            self.cache.evict_file(ingest_probe_file_id(source_idx));
        }
    }
}

/// A validated ingest source: an open reader plus its user-key range.
struct IngestSource {
    path: PathBuf,
    reader: SsTableReader,
    smallest: Vec<u8>,
    largest: Vec<u8>,
}

/// Choose the target level for an ingest file covering the user-key
/// range `[smallest, largest]`. Returns L0 when the range overlaps any
/// existing file at any level; otherwise walks upward from the
/// bottommost non-empty level and picks the deepest level whose files
/// are all strictly disjoint from the input range. Level 0 is the
/// fallback when every other level also overlaps or when every level
/// is empty.
///
/// When `ingest_behind` is true the target is forced to the
/// bottommost level (MAX_LEVELS-1) and the call errors if any file at
/// any level overlaps the input range.
fn compute_target_level(
    version: &manifest::Version,
    smallest: &[u8],
    largest: &[u8],
    ingest_behind: bool,
) -> std::io::Result<usize> {
    let overlaps_level = |level: usize| -> bool {
        version.levels[level].iter().any(|f| {
            f.meta.smallest_key.as_slice() <= largest && f.meta.largest_key.as_slice() >= smallest
        })
    };

    if ingest_behind {
        for level in 0..manifest::MAX_LEVELS {
            if overlaps_level(level) {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "ingest_behind: input range overlaps existing SSTable",
                ));
            }
        }
        return Ok(manifest::MAX_LEVELS - 1);
    }

    // Any overlap at any level → land at L0. L0 is the only level
    // that tolerates overlapping files, so this is the only safe
    // destination when the ingest range is not disjoint from the
    // existing tree.
    for level in 0..manifest::MAX_LEVELS {
        if overlaps_level(level) {
            return Ok(0);
        }
    }
    // Every level is disjoint (or empty): land at the deepest level.
    Ok(manifest::MAX_LEVELS - 1)
}

/// A consistent snapshot of on-disk state captured by
/// [`LarkEngine::checkpoint_capture`]. Holds the engine's compaction
/// lock for its entire lifetime, so no background or foreground
/// compaction can unlink files referenced by `version` while the
/// snapshot is alive. Callers MUST drop this snapshot as soon as
/// they are done with the filesystem work - a snapshot whose
/// lifetime outlives the enclosing function scope can deadlock a
/// concurrent `db.close()` / `drop(db)` that joins the background
/// compaction thread (the thread will block waiting for the same
/// lock the snapshot holds).
pub(crate) struct CheckpointSnapshot {
    pub(crate) version: Arc<manifest::Version>,
    pub(crate) manifest_path: PathBuf,
    /// Number of bytes at the start of `manifest_path` that belong
    /// to this captured version. Copiers must truncate at this
    /// length to avoid picking up `AddFile` records written by
    /// concurrent flushes after the capture.
    pub(crate) manifest_len: u64,
    pub(crate) sst_dir: PathBuf,
    /// Compaction lock guard, scoped to the snapshot's lifetime.
    _compaction_guard: parking_lot::ArcRwLockWriteGuard<parking_lot::RawRwLock, ()>,
}

impl CheckpointSnapshot {
    /// Format an SSTable filename for the given file id - exposes
    /// the engine's naming scheme to callers outside the `engine`
    /// module so they can stage files into checkpoint / backup
    /// directories.
    pub(crate) fn sst_filename(id: u64) -> String {
        sst_filename(id)
    }
}

fn list_wal_files(dir: &Path) -> std::io::Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    if dir.exists() {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path
                .extension()
                .is_some_and(|ext| ext == "log" || ext == "wal")
            {
                files.push(path);
            }
        }
    }
    Ok(files)
}

fn remove_obsolete_sst_files(sst_dir: &Path, version: &manifest::Version) -> std::io::Result<()> {
    let mut removed_any = false;
    for level in &version.levels {
        for file in level {
            let path = sst_dir.join(sst_filename(file.meta.file_id));
            removed_any |= remove_file_if_exists(&path)?;
        }
    }
    if removed_any {
        durability::sync_dir(sst_dir)?;
    }
    Ok(())
}

fn remove_obsolete_wal_files(wal_dir: &Path, keep_path: &Path) -> std::io::Result<()> {
    let mut removed_any = false;
    for path in list_wal_files(wal_dir)? {
        if path == keep_path {
            continue;
        }
        removed_any |= remove_file_if_exists(&path)?;
    }
    if removed_any {
        durability::sync_dir(wal_dir)?;
    }
    Ok(())
}

fn remove_file_if_exists(path: &Path) -> std::io::Result<bool> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(true),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(err) => Err(err),
    }
}

fn wal_file_id(path: &Path) -> Option<u64> {
    let stem = path.file_stem()?.to_str()?;
    stem.strip_prefix("wal_")?.parse().ok()
}

fn should_replay_wal(path: &Path, min_wal_id: u64) -> bool {
    match wal_file_id(path) {
        Some(id) => id >= min_wal_id,
        // Legacy or temporary WAL names predate the reset marker. Once a
        // reset has committed, they must not be allowed to resurrect data.
        None => min_wal_id == 0,
    }
}

fn next_wal_id(manifest_next_file_id: u64, wal_files: &[PathBuf]) -> u64 {
    wal_files
        .iter()
        .filter_map(|path| wal_file_id(path))
        .map(|id| id.saturating_add(1))
        .fold(manifest_next_file_id, u64::max)
}

/// Refuse an open where a WAL file dropped a tail while a *later* WAL
/// file still yielded records.
///
/// `Wal::replay` judges one file's bytes and reports the discard rather
/// than deciding, because the torn-tail rule is only sound for the newest
/// file that contributes to replay. A torn write leaves nothing after it
/// anywhere, so records in a later file are proof that the earlier file's
/// missing tail is damage in the middle of the history. Opening on it
/// would serve a state that never existed: later writes present, earlier
/// acknowledged ones gone.
fn reject_tail_discard_before_live_wal(
    wal_files: &[PathBuf],
    entries_per_file: &[usize],
    discarded_tails: &[(PathBuf, u64, u64)],
) -> std::io::Result<()> {
    for (path, offset, discarded_bytes) in discarded_tails {
        let Some(index) = wal_files.iter().position(|p| p == path) else {
            continue;
        };
        let later = entries_per_file
            .iter()
            .enumerate()
            .skip(index + 1)
            .find(|(_, count)| **count > 0);
        let Some((later_index, later_count)) = later else {
            continue;
        };
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "{} is corrupt: the WAL record at offset {offset} is incomplete, but {later_count} \
                 later WAL record(s) follow it in {}, so the {discarded_bytes} discarded byte(s) \
                 are damage in the middle of the history rather than a torn write. Refusing to \
                 open rather than serve a state that never existed.",
                path.display(),
                wal_files[later_index].display(),
            ),
        ));
    }
    Ok(())
}

fn apply_replayed_wal_entry(memtable: &MemTable, entry: WalEntry) -> u64 {
    match entry {
        WalEntry::Put { key, value, seq } => {
            memtable.put(&key, &value, seq);
            seq
        }
        WalEntry::Delete { key, seq } => {
            memtable.delete(&key, seq);
            seq
        }
        WalEntry::DeleteRange { start, end, seq } => {
            memtable.delete_range(&start, &end, seq);
            seq
        }
        WalEntry::Merge { key, operand, seq } => {
            memtable.merge(&key, &operand, seq);
            seq
        }
    }
}

fn rewrite_recovered_memtable_to_wal(memtable: &MemTable, wal: &mut Wal) -> std::io::Result<()> {
    let mut wrote_record = false;

    memtable.try_for_each_entry(|internal_key, value| {
        let (user_key, seq, value_type) = internal_key::decode_internal_key(internal_key);
        match value_type {
            internal_key::VALUE_TYPE_VALUE => wal.append_put(user_key, value, seq)?,
            internal_key::VALUE_TYPE_DELETION => wal.append_delete(user_key, seq)?,
            internal_key::VALUE_TYPE_MERGE => wal.append_merge(user_key, value, seq)?,
            other => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("unknown value type {other} in recovered memtable"),
                ));
            }
        }
        wrote_record = true;
        Ok(())
    })?;

    for tombstone in memtable.clone_range_tombstones() {
        wal.append_delete_range(&tombstone.start, &tombstone.end, tombstone.seq)?;
        wrote_record = true;
    }

    if wrote_record {
        wal.sync_data()?;
    }

    Ok(())
}
