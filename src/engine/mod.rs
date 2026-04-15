pub(crate) mod block;
pub(crate) mod block_cache;
pub(crate) mod bloom;
pub(crate) mod compaction;
pub(crate) mod internal_key;
pub(crate) mod iterator;
pub(crate) mod manifest;
pub(crate) mod memtable;
pub(crate) mod range_tombstone;
pub(crate) mod snapshot_registry;
pub(crate) mod sstable;
pub(crate) mod wal;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::{Mutex, RwLock};

use block_cache::BlockCache;
use compaction::{CompactionOptions, CompactionScheduler};
use manifest::{VersionEdit, VersionSet};
use memtable::MemTable;
use snapshot_registry::SnapshotRegistry;

use crate::event_listener;
use sstable::{sst_filename, LiveSst, LookupResult, SsTableMeta, SsTableReader, SsTableWriter};
use wal::{wal_filename, Wal, WalEntry};

/// Controls when data is flushed to disk after a commit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DurabilityMode {
    Immediate,
    Eventual,
}

/// Outcome of [`LarkEngine::commit_optimistic`]. `Conflict`
/// indicates that another writer changed one of the tracked keys
/// after the transaction's snapshot seq; the caller typically
/// surfaces this as a retry-able error.
#[derive(Debug)]
pub(crate) enum CommitOutcome {
    Ok,
    Conflict {
        key: Vec<u8>,
        observed_seq: u64,
        latest_seq: u64,
    },
}

/// Configuration for the Lark engine.
#[derive(Clone)]
pub(crate) struct EngineOptions {
    pub(crate) write_buffer_size: usize,
    pub(crate) block_size: usize,
    pub(crate) block_cache_size: usize,
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
}

impl EngineOptions {
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
            block_size: 16 * 1024,
            block_cache_size: 512 * 1024 * 1024,
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
        }
    }
}

/// The core LSM-tree engine.
pub(crate) struct LarkEngine {
    active_memtable: RwLock<Arc<MemTable>>,
    frozen_memtables: RwLock<Vec<Arc<MemTable>>>,
    versions: Arc<Mutex<VersionSet>>,
    cache: Arc<BlockCache>,
    latest_seq: AtomicU64,
    active_wal: Mutex<Wal>,
    wal_id: AtomicU64,
    sst_dir: PathBuf,
    wal_dir: PathBuf,
    compaction: Mutex<CompactionScheduler>,
    /// Engine-wide mutex that serializes every compaction pass —
    /// background and foreground — so two compactions can't pick
    /// overlapping input sets and double-delete files out from under
    /// each other. Held around each `pick_and_run_compaction` call in
    /// the background scheduler, and around the whole
    /// [`LarkEngine::compact_range`] walk in the foreground path.
    compaction_lock: Arc<Mutex<()>>,
    /// Tracks the sequence numbers of every live snapshot so compaction
    /// can drop versions that no snapshot and no current reader can
    /// see. A snapshot registers itself on creation and releases on
    /// drop; compaction queries `oldest_live_seq()` to compute its GC
    /// horizon.
    snapshot_registry: Arc<SnapshotRegistry>,
    options: EngineOptions,
    write_lock: Mutex<()>,
}

impl LarkEngine {
    /// Open or create the database at the given path.
    pub(crate) fn open(db_dir: &Path, options: EngineOptions) -> std::io::Result<Arc<Self>> {
        let sst_dir = db_dir.join("sst");
        let wal_dir = db_dir.join("wal");

        std::fs::create_dir_all(&sst_dir)?;
        std::fs::create_dir_all(&wal_dir)?;

        let mut version_set = VersionSet::open(db_dir, &sst_dir)?;
        let version = version_set.current();
        let mut latest_seq = version.last_seq;

        // Replay WAL files to recover memtable state
        let memtable = Arc::new(MemTable::new());
        let mut wal_files = list_wal_files(&wal_dir)?;
        wal_files.sort();

        for wal_path in &wal_files {
            tracing::info!(path = %wal_path.display(), "Replaying WAL");
            match Wal::replay(wal_path) {
                Ok(entries) => {
                    for entry in entries {
                        match entry {
                            WalEntry::Put { key, value, seq } => {
                                memtable.put(&key, &value, seq);
                                latest_seq = latest_seq.max(seq);
                            }
                            WalEntry::Delete { key, seq } => {
                                memtable.delete(&key, seq);
                                latest_seq = latest_seq.max(seq);
                            }
                            WalEntry::DeleteRange { start, end, seq } => {
                                memtable.delete_range(&start, &end, seq);
                                latest_seq = latest_seq.max(seq);
                            }
                            WalEntry::Merge { key, operand, seq } => {
                                memtable.merge(&key, &operand, seq);
                                latest_seq = latest_seq.max(seq);
                            }
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!(path = %wal_path.display(), error = %e, "Failed to replay WAL");
                }
            }
        }

        for wal_path in &wal_files {
            let _ = std::fs::remove_file(wal_path);
        }

        let wal_id = version.next_file_id;
        let wal_path = wal_dir.join(wal_filename(wal_id));
        let wal = Wal::create(&wal_path)?;

        version_set.apply(&[VersionEdit::SetNextFileId(wal_id + 1)])?;

        let cache = Arc::new(
            BlockCache::new(options.block_cache_size).with_stats(options.statistics.clone()),
        );
        let versions = Arc::new(Mutex::new(version_set));

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
        };

        let compaction_lock = Arc::new(Mutex::new(()));
        let snapshot_registry = Arc::new(SnapshotRegistry::new());
        let compaction = CompactionScheduler::start(
            Arc::clone(&compaction_lock),
            Arc::clone(&snapshot_registry),
            Arc::clone(&versions),
            Arc::from(sst_dir.as_path()),
            Arc::clone(&cache),
            compaction_opts,
        );

        let engine = Arc::new(Self {
            active_memtable: RwLock::new(memtable),
            frozen_memtables: RwLock::new(Vec::new()),
            versions,
            cache,
            latest_seq: AtomicU64::new(latest_seq),
            active_wal: Mutex::new(wal),
            wal_id: AtomicU64::new(wal_id),
            sst_dir,
            wal_dir,
            compaction: Mutex::new(compaction),
            compaction_lock,
            snapshot_registry,
            options,
            write_lock: Mutex::new(()),
        });

        Ok(engine)
    }

    pub(crate) fn snapshot_seq(&self) -> u64 {
        self.latest_seq.load(Ordering::Acquire)
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
        self.snapshot_registry.register(seq);
        if let Some(s) = self.statistics() {
            s.add(crate::statistics::Ticker::SnapshotsRegistered, 1);
        }
    }

    /// Release a snapshot pin previously taken via
    /// [`Self::register_snapshot`].
    pub(crate) fn release_snapshot(&self, seq: u64) {
        self.snapshot_registry.release(seq);
        if let Some(s) = self.statistics() {
            s.add(crate::statistics::Ticker::SnapshotsReleased, 1);
        }
    }

    /// Current GC horizon for compaction — the smallest live snapshot
    /// seq, or `u64::MAX` if no snapshot is currently pinned.
    pub(crate) fn oldest_live_seq(&self) -> u64 {
        self.snapshot_registry.oldest_live_seq()
    }

    /// Construct a streaming iterator rooted at `snapshot_seq`. Captures
    /// the current memtable state and the current version; no filesystem
    /// access happens here — file handles are already open in the
    /// pinned `Arc<LiveSst>`s carried by the version.
    pub(crate) fn new_iter(&self, snapshot_seq: u64) -> iterator::LarkIterator {
        let active = Arc::clone(&self.active_memtable.read());
        let frozen: Vec<Arc<MemTable>> = self
            .frozen_memtables
            .read()
            .iter()
            .map(Arc::clone)
            .collect();
        let version = self.versions.lock().current();
        iterator::LarkIterator::new(
            active,
            frozen,
            version,
            Arc::clone(&self.cache),
            snapshot_seq,
            self.options.prefix_extractor.clone(),
            self.options.merge_operator.clone(),
        )
    }

    /// Point lookup at a given snapshot. Returns `Ok(Some(value))` or `Ok(None)`.
    ///
    /// Walks sources newest→oldest (active memtable, frozen memtables
    /// newest first, L0 newest first, L1..Ln). At each source we check
    /// both the newest visible point entry and the newest visible
    /// covering range tombstone, carrying the largest RT seq forward so
    /// a range delete in a newer source can override a point entry in
    /// an older source. The first source yielding a decisive answer
    /// wins — a point entry with `seq > max_rt_so_far` gives its value;
    /// otherwise the range tombstone hides it.
    ///
    /// When a [`crate::MergeOperator`] is configured, the walk also
    /// collects any merge operands that sit on top of the terminator
    /// and calls the operator to collapse the chain into a final
    /// value at visibility time.
    pub(crate) fn get(&self, key: &[u8], snapshot_seq: u64) -> std::io::Result<Option<Vec<u8>>> {
        if self.options.merge_operator.is_some() {
            return self.get_with_merge(key, snapshot_seq);
        }
        let mut max_rt_seq: u64 = 0;

        {
            let active = self.active_memtable.read();
            let rt = active.covering_range_tombstone_seq(key, snapshot_seq);
            if rt > max_rt_seq {
                max_rt_seq = rt;
            }
            if let Some((pseq, popt)) = active.get(key, snapshot_seq) {
                return Ok(if pseq > max_rt_seq { popt } else { None });
            }
        }

        {
            let frozen = self.frozen_memtables.read();
            for mt in frozen.iter().rev() {
                let rt = mt.covering_range_tombstone_seq(key, snapshot_seq);
                if rt > max_rt_seq {
                    max_rt_seq = rt;
                }
                if let Some((pseq, popt)) = mt.get(key, snapshot_seq) {
                    return Ok(if pseq > max_rt_seq { popt } else { None });
                }
            }
        }

        let version = self.versions.lock().current();

        // L0: check all files (may overlap), newest first. Readers are
        // already open in the pinned `Version`, so no filesystem access
        // happens here — concurrent compaction unlinking paths cannot
        // break us.
        for file in version.levels[0].iter().rev() {
            let rt = file.reader.covering_range_tombstone_seq(key, snapshot_seq);
            if rt > max_rt_seq {
                max_rt_seq = rt;
            }
            match file.reader.get(key, snapshot_seq, &self.cache)? {
                LookupResult::Found { seq, value } => {
                    return Ok(if seq > max_rt_seq { Some(value) } else { None });
                }
                LookupResult::FoundTombstone { .. } => return Ok(None),
                LookupResult::NotInTable => {}
            }
        }

        // L1+: binary search for the right SSTable per level.
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

            let idx = files.partition_point(|f| f.meta.largest_key.as_slice() < key);
            if idx < files.len() && files[idx].meta.smallest_key.as_slice() <= key {
                let file = &files[idx];
                match file.reader.get(key, snapshot_seq, &self.cache)? {
                    LookupResult::Found { seq, value } => {
                        return Ok(if seq > max_rt_seq { Some(value) } else { None });
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
    fn get_with_merge(&self, key: &[u8], snapshot_seq: u64) -> std::io::Result<Option<Vec<u8>>> {
        use internal_key::{VALUE_TYPE_DELETION, VALUE_TYPE_MERGE, VALUE_TYPE_VALUE};

        let merge_op = self
            .options
            .merge_operator
            .as_ref()
            .expect("get_with_merge called without a merge operator");

        // `chain` records visible entries for `key` in newest-seq-
        // first order, stopping at (and including) the first
        // terminator (`VALUE` or `DELETION`). Range tombstones that
        // cover the key are treated as virtual deletion terminators.
        let mut chain: Vec<(u64, u8, Vec<u8>)> = Vec::new();
        let mut max_rt_seq: u64 = 0;
        let mut terminated;

        // `consume_partial` appends entries from one source into the
        // running chain, short-circuiting if a terminator (real or
        // RT-synthesized) is reached.
        let consume_partial = |partial: Vec<(u64, u8, Vec<u8>)>,
                               max_rt_seq: u64,
                               chain: &mut Vec<(u64, u8, Vec<u8>)>|
         -> bool {
            for (seq, vt, value) in partial {
                if seq <= max_rt_seq {
                    // Range tombstone hides this and every older
                    // entry for the same key.
                    chain.push((max_rt_seq, VALUE_TYPE_DELETION, Vec::new()));
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
            let active = self.active_memtable.read();
            let rt = active.covering_range_tombstone_seq(key, snapshot_seq);
            if rt > max_rt_seq {
                max_rt_seq = rt;
            }
            let mut partial = Vec::new();
            let _ = active.collect_merge_chain(key, snapshot_seq, &mut partial);
            terminated = consume_partial(partial, max_rt_seq, &mut chain);
        }

        if !terminated {
            let frozen = self.frozen_memtables.read();
            for mt in frozen.iter().rev() {
                let rt = mt.covering_range_tombstone_seq(key, snapshot_seq);
                if rt > max_rt_seq {
                    max_rt_seq = rt;
                }
                let mut partial = Vec::new();
                let _ = mt.collect_merge_chain(key, snapshot_seq, &mut partial);
                terminated = consume_partial(partial, max_rt_seq, &mut chain);
                if terminated {
                    break;
                }
            }
        }

        if !terminated {
            let version = self.versions.lock().current();

            // L0: newest-first.
            for file in version.levels[0].iter().rev() {
                let rt = file.reader.covering_range_tombstone_seq(key, snapshot_seq);
                if rt > max_rt_seq {
                    max_rt_seq = rt;
                }
                let mut partial = Vec::new();
                file.reader
                    .collect_merge_chain(key, snapshot_seq, &self.cache, &mut partial)?;
                terminated = consume_partial(partial, max_rt_seq, &mut chain);
                if terminated {
                    break;
                }
            }

            // L1+: one file per level could contain the key. Also
            // scan every file at the level for RT coverage (RTs may
            // sit in a sibling file).
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
                    let idx = files.partition_point(|f| f.meta.largest_key.as_slice() < key);
                    if idx >= files.len() || files[idx].meta.smallest_key.as_slice() > key {
                        continue;
                    }
                    let file = &files[idx];
                    let mut partial = Vec::new();
                    file.reader.collect_merge_chain(
                        key,
                        snapshot_seq,
                        &self.cache,
                        &mut partial,
                    )?;
                    terminated = consume_partial(partial, max_rt_seq, &mut chain);
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
            Some((_, vt, _)) if *vt == VALUE_TYPE_VALUE || *vt == VALUE_TYPE_DELETION => {
                let base = if *vt == VALUE_TYPE_VALUE {
                    Some(chain.last().unwrap().2.as_slice())
                } else {
                    None
                };
                (base, true)
            }
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
            // No merges at all — the chain is just a plain
            // Value / Deletion / nothing. Return the base directly.
            return Ok(base_slice.map(|s| s.to_vec()));
        }

        match merge_op.full_merge(key, base_slice, &operands_owned) {
            Some(v) => Ok(Some(v)),
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
    /// The batch amortizes per-call overhead — a single version snapshot,
    /// a single memtable lock acquisition per level, one logical walk of
    /// the source hierarchy — and short-circuits once every key has been
    /// resolved. All keys see the **same** consistent view, regardless of
    /// concurrent writers.
    pub(crate) fn multi_get(
        &self,
        keys: &[&[u8]],
        snapshot_seq: u64,
    ) -> std::io::Result<Vec<Option<Vec<u8>>>> {
        // When a merge operator is configured, fall back to per-key
        // resolution — the batched walk's short-circuiting logic
        // doesn't compose cleanly with merge-chain collection, and
        // merges are rare enough that the cost difference isn't
        // worth a specialized batched path.
        if self.options.merge_operator.is_some() {
            let mut out = Vec::with_capacity(keys.len());
            for key in keys {
                out.push(self.get_with_merge(key, snapshot_seq)?);
            }
            return Ok(out);
        }

        let mut results: Vec<Option<Vec<u8>>> = vec![None; keys.len()];
        let mut resolved: Vec<bool> = vec![false; keys.len()];
        // Running max of range-tombstone seq covering `keys[i]` across
        // all sources walked so far (newest → oldest).
        let mut max_rt: Vec<u64> = vec![0; keys.len()];
        let mut unresolved = keys.len();
        if unresolved == 0 {
            return Ok(results);
        }

        // Helper that, given a point lookup result and the accumulated
        // max RT seq, decides the final per-key outcome.
        let resolve = |pseq: u64, popt: Option<Vec<u8>>, rt_seq: u64| -> Option<Vec<u8>> {
            if pseq > rt_seq {
                popt
            } else {
                None
            }
        };

        // 1. Active memtable.
        {
            let mt = self.active_memtable.read();
            for (i, k) in keys.iter().enumerate() {
                if resolved[i] {
                    continue;
                }
                let rt = mt.covering_range_tombstone_seq(k, snapshot_seq);
                if rt > max_rt[i] {
                    max_rt[i] = rt;
                }
                if let Some((pseq, popt)) = mt.get(k, snapshot_seq) {
                    results[i] = resolve(pseq, popt, max_rt[i]);
                    resolved[i] = true;
                    unresolved -= 1;
                }
            }
        }
        if unresolved == 0 {
            return Ok(results);
        }

        // 2. Frozen memtables, newest first.
        {
            let frozen = self.frozen_memtables.read();
            for mt in frozen.iter().rev() {
                for (i, k) in keys.iter().enumerate() {
                    if resolved[i] {
                        continue;
                    }
                    let rt = mt.covering_range_tombstone_seq(k, snapshot_seq);
                    if rt > max_rt[i] {
                        max_rt[i] = rt;
                    }
                    if let Some((pseq, popt)) = mt.get(k, snapshot_seq) {
                        results[i] = resolve(pseq, popt, max_rt[i]);
                        resolved[i] = true;
                        unresolved -= 1;
                    }
                }
                if unresolved == 0 {
                    return Ok(results);
                }
            }
        }

        let version = self.versions.lock().current();

        // 3. L0 SSTables, newest first.
        for file in version.levels[0].iter().rev() {
            for (i, k) in keys.iter().enumerate() {
                if resolved[i] {
                    continue;
                }
                let rt = file.reader.covering_range_tombstone_seq(k, snapshot_seq);
                if rt > max_rt[i] {
                    max_rt[i] = rt;
                }
                match file.reader.get(k, snapshot_seq, &self.cache)? {
                    LookupResult::Found { seq, value } => {
                        results[i] = resolve(seq, Some(value), max_rt[i]);
                        resolved[i] = true;
                        unresolved -= 1;
                    }
                    LookupResult::FoundTombstone { .. } => {
                        results[i] = None;
                        resolved[i] = true;
                        unresolved -= 1;
                    }
                    LookupResult::NotInTable => {}
                }
            }
            if unresolved == 0 {
                return Ok(results);
            }
        }

        // 4. L1..Ln: within each level files are non-overlapping, so a
        //    single partition_point locates the at-most-one file that
        //    could contain a given key's point entry. Range tombstones
        //    may live in a sibling file, though, so we scan all files
        //    whose user-key range covers `k` for RT coverage first.
        for level in 1..version.levels.len() {
            let files = &version.levels[level];
            if files.is_empty() {
                continue;
            }
            for (i, k) in keys.iter().enumerate() {
                if resolved[i] {
                    continue;
                }
                for file in files {
                    if file.meta.smallest_key.as_slice() <= *k
                        && *k <= file.meta.largest_key.as_slice()
                    {
                        let rt = file.reader.covering_range_tombstone_seq(k, snapshot_seq);
                        if rt > max_rt[i] {
                            max_rt[i] = rt;
                        }
                    }
                }
                let idx = files.partition_point(|f| f.meta.largest_key.as_slice() < k);
                if idx >= files.len() || files[idx].meta.smallest_key.as_slice() > *k {
                    continue;
                }
                let file = &files[idx];
                match file.reader.get(k, snapshot_seq, &self.cache)? {
                    LookupResult::Found { seq, value } => {
                        results[i] = resolve(seq, Some(value), max_rt[i]);
                        resolved[i] = true;
                        unresolved -= 1;
                    }
                    LookupResult::FoundTombstone { .. } => {
                        results[i] = None;
                        resolved[i] = true;
                        unresolved -= 1;
                    }
                    LookupResult::NotInTable => {}
                }
            }
            if unresolved == 0 {
                return Ok(results);
            }
        }

        Ok(results)
    }

    /// Apply a batch of point writes, range deletes, and merge
    /// operands atomically.
    ///
    /// `point_ops` carries put/delete operations keyed by user key;
    /// `range_deletes` is a list of `(start, end)` half-open
    /// intervals; `merges` is a list of `(key, operand)` pairs that
    /// each layer a merge operand on top of whatever base value the
    /// key currently has.
    ///
    /// All operations in a single call are assigned consecutive
    /// sequence numbers in this order: point ops, then range
    /// deletes, then merges.
    ///
    /// `durability` controls WAL fsync semantics (`Immediate` = fsync
    /// per call, `Eventual` = buffered flush). `disable_wal` skips
    /// the WAL append entirely — the caller accepts that a crash
    /// before the next memtable flush loses the write.
    pub(crate) fn apply_batch(
        &self,
        point_ops: BTreeMap<Vec<u8>, Option<Vec<u8>>>,
        range_deletes: Vec<(Vec<u8>, Vec<u8>)>,
        merges: Vec<(Vec<u8>, Vec<u8>)>,
        durability: DurabilityMode,
        disable_wal: bool,
    ) -> std::io::Result<()> {
        if point_ops.is_empty() && range_deletes.is_empty() && merges.is_empty() {
            return Ok(());
        }

        let _write_guard = self.write_lock.lock();
        self.apply_batch_locked(point_ops, range_deletes, merges, durability, disable_wal)
    }

    /// Apply a batch assuming the caller already holds
    /// `self.write_lock`. Used by [`Self::apply_batch`] and by the
    /// transaction commit path, which needs to interleave a
    /// conflict-detection step with the write while holding the
    /// write lock the whole time.
    fn apply_batch_locked(
        &self,
        point_ops: BTreeMap<Vec<u8>, Option<Vec<u8>>>,
        range_deletes: Vec<(Vec<u8>, Vec<u8>)>,
        merges: Vec<(Vec<u8>, Vec<u8>)>,
        durability: DurabilityMode,
        disable_wal: bool,
    ) -> std::io::Result<()> {
        if point_ops.is_empty() && range_deletes.is_empty() && merges.is_empty() {
            return Ok(());
        }

        let total_ops = point_ops.len() + range_deletes.len() + merges.len();
        let base_seq = self
            .latest_seq
            .fetch_add(total_ops as u64, Ordering::AcqRel)
            + 1;
        let range_delete_base = base_seq + point_ops.len() as u64;
        let merge_base = range_delete_base + range_deletes.len() as u64;

        if !disable_wal {
            let wal_start = std::time::Instant::now();
            let mut wal = self.active_wal.lock();
            let mut wal_bytes: u64 = 0;
            for (i, (key, value)) in point_ops.iter().enumerate() {
                let seq = base_seq + i as u64;
                match value {
                    Some(v) => {
                        wal.append_put(key, v, seq)?;
                        wal_bytes += (key.len() + v.len() + 8) as u64;
                    }
                    None => {
                        wal.append_delete(key, seq)?;
                        wal_bytes += (key.len() + 8) as u64;
                    }
                }
            }
            for (j, (start, end)) in range_deletes.iter().enumerate() {
                let seq = range_delete_base + j as u64;
                wal.append_delete_range(start, end, seq)?;
                wal_bytes += (start.len() + end.len() + 8) as u64;
            }
            for (k, (key, operand)) in merges.iter().enumerate() {
                let seq = merge_base + k as u64;
                wal.append_merge(key, operand, seq)?;
                wal_bytes += (key.len() + operand.len() + 8) as u64;
            }
            match durability {
                DurabilityMode::Immediate => wal.sync()?,
                DurabilityMode::Eventual => wal.flush()?,
            }
            if let Some(s) = self.statistics() {
                s.add(crate::statistics::Ticker::WalBytesWritten, wal_bytes);
                if matches!(durability, DurabilityMode::Immediate) {
                    s.add(crate::statistics::Ticker::WalSyncCount, 1);
                }
                s.record(
                    crate::statistics::Histogram::WalWriteTime,
                    wal_start.elapsed().as_micros() as u64,
                );
            }
        }

        {
            let memtable = self.active_memtable.read();
            for (i, (key, value)) in point_ops.iter().enumerate() {
                let seq = base_seq + i as u64;
                match value {
                    Some(v) => memtable.put(key, v, seq),
                    None => memtable.delete(key, seq),
                }
            }
            for (j, (start, end)) in range_deletes.iter().enumerate() {
                let seq = range_delete_base + j as u64;
                memtable.delete_range(start, end, seq);
            }
            for (k, (key, operand)) in merges.iter().enumerate() {
                let seq = merge_base + k as u64;
                memtable.merge(key, operand, seq);
            }
        }

        if self.active_memtable.read().approximate_size() >= self.options.write_buffer_size {
            self.rotate_memtable()?;
        }

        Ok(())
    }

    /// Attempt to commit an optimistic transaction's buffered
    /// writes. Performs the write-write conflict check under the
    /// engine write-lock: for every key the transaction wrote (or
    /// explicitly tracked for conflict detection), verify that no
    /// version newer than `snapshot_seq` has landed since the
    /// transaction started. On conflict returns
    /// `Ok(CommitOutcome::Conflict { key })`. Otherwise applies the
    /// buffered writes atomically and returns `Ok(CommitOutcome::Ok)`.
    ///
    /// Range-delete operations from a transaction are applied
    /// unconditionally — their conflict semantics require tracking
    /// every key the range could shadow, which the initial
    /// transaction impl does not support. See the transaction
    /// module for the caveat.
    pub(crate) fn commit_optimistic(
        &self,
        conflict_keys: &[Vec<u8>],
        snapshot_seq: u64,
        point_ops: BTreeMap<Vec<u8>, Option<Vec<u8>>>,
        range_deletes: Vec<(Vec<u8>, Vec<u8>)>,
        merges: Vec<(Vec<u8>, Vec<u8>)>,
        durability: DurabilityMode,
    ) -> std::io::Result<CommitOutcome> {
        let _write_guard = self.write_lock.lock();

        // Conflict check: for each tracked key, peek at the latest
        // visible version without a snapshot bound. If its seq is
        // newer than `snapshot_seq`, someone wrote to the key after
        // the transaction began.
        for key in conflict_keys {
            if let Some(latest_seq) = self.latest_version_seq(key)? {
                if latest_seq > snapshot_seq {
                    return Ok(CommitOutcome::Conflict {
                        key: key.clone(),
                        observed_seq: snapshot_seq,
                        latest_seq,
                    });
                }
            }
        }

        self.apply_batch_locked(point_ops, range_deletes, merges, durability, false)?;
        Ok(CommitOutcome::Ok)
    }

    /// Return the sequence number of the newest visible point
    /// entry for `key` across every source, or `None` if no live
    /// version exists. Used by optimistic transaction commit to
    /// detect write-write conflicts. Ignores range tombstones and
    /// merge operators — the caller only needs to know "was this
    /// key written to again?".
    fn latest_version_seq(&self, key: &[u8]) -> std::io::Result<Option<u64>> {
        let snap = u64::MAX;
        {
            let active = self.active_memtable.read();
            if let Some((seq, _)) = active.get(key, snap) {
                return Ok(Some(seq));
            }
        }
        {
            let frozen = self.frozen_memtables.read();
            for mt in frozen.iter().rev() {
                if let Some((seq, _)) = mt.get(key, snap) {
                    return Ok(Some(seq));
                }
            }
        }
        let version = self.versions.lock().current();
        for file in version.levels[0].iter().rev() {
            match file.reader.get(key, snap, &self.cache)? {
                LookupResult::Found { seq, .. } | LookupResult::FoundTombstone { seq } => {
                    return Ok(Some(seq));
                }
                LookupResult::NotInTable => {}
            }
        }
        for level in 1..version.levels.len() {
            let files = &version.levels[level];
            if files.is_empty() {
                continue;
            }
            let idx = files.partition_point(|f| f.meta.largest_key.as_slice() < key);
            if idx < files.len() && files[idx].meta.smallest_key.as_slice() <= key {
                match files[idx].reader.get(key, snap, &self.cache)? {
                    LookupResult::Found { seq, .. } | LookupResult::FoundTombstone { seq } => {
                        return Ok(Some(seq));
                    }
                    LookupResult::NotInTable => {}
                }
            }
        }
        Ok(None)
    }

    fn rotate_memtable(&self) -> std::io::Result<()> {
        let old_memtable = {
            let mut active = self.active_memtable.write();
            let old = Arc::clone(&active);
            *active = Arc::new(MemTable::new());
            old
        };

        self.frozen_memtables.write().push(old_memtable);

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
            std::mem::replace(&mut *wal, new_wal)
        };

        self.wal_id.store(new_wal_id, Ordering::Release);
        self.flush_frozen_memtable(old_wal)?;

        Ok(())
    }

    fn flush_frozen_memtable(&self, old_wal: Wal) -> std::io::Result<()> {
        let flush_start = std::time::Instant::now();
        let memtable = {
            let frozen = self.frozen_memtables.read();
            if frozen.is_empty() {
                return Ok(());
            }
            Arc::clone(&frozen[0])
        };

        let range_tombstones = memtable.clone_range_tombstones();

        if memtable.is_empty() && range_tombstones.is_empty() {
            self.frozen_memtables.write().remove(0);
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

        // Memtable flushes always land at L0 — pick L0's codec.
        let mut writer = SsTableWriter::new(
            &sst_path,
            self.options.block_size,
            self.options.bloom_bits_per_key,
            self.options.compression_for_level(0),
            self.options.prefix_extractor.clone(),
        )?;

        // Walk the memtable in internal-key order and copy every version
        // and tombstone into the SSTable unchanged, preserving MVCC.
        let entries = memtable.iter_internal();
        for (internal_key, value) in &entries {
            writer.add(internal_key, value)?;
        }

        // Persist range tombstones alongside the point entries.
        for rt in &range_tombstones {
            writer.add_range_tombstone(&rt.start, &rt.end, rt.seq);
        }

        let summary = match writer.finish()? {
            Some(s) => s,
            None => {
                self.frozen_memtables.write().remove(0);
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

        let reader = Arc::new(SsTableReader::open(&sst_path, file_id)?);
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

        self.frozen_memtables.write().remove(0);
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
        // 1. Flush the active memtable so any in-memory data that
        //    overlaps the range is materialized in L0. We only touch
        //    the write lock if there's actually data to flush. "Data"
        //    here includes range tombstones, not just point entries.
        let needs_flush = |mt: &MemTable| !mt.is_empty() || !mt.clone_range_tombstones().is_empty();
        if needs_flush(&self.active_memtable.read()) {
            let _write_guard = self.write_lock.lock();
            if needs_flush(&self.active_memtable.read()) {
                self.rotate_memtable()?;
            }
        }

        // 2. Serialize with background compaction for the duration of
        //    the range walk.
        let _compact_guard = self.compaction_lock.lock();

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
        };
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
    /// level — any overlap is an error. If `snapshot_consistency` is
    /// set the call is rejected while any snapshot is pinned (ingest
    /// would otherwise inject a new seq that older snapshots cannot
    /// consistently observe).
    pub(crate) fn ingest_external_files(
        &self,
        files: &[PathBuf],
        ingest_opts: &crate::sst_file_writer::IngestOptions,
    ) -> std::io::Result<()> {
        use crate::engine::internal_key::decode_internal_key;

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
        for path in files {
            let reader = SsTableReader::open(path, 0).map_err(|e| {
                std::io::Error::new(e.kind(), format!("ingest: open {}: {e}", path.display()))
            })?;
            let entries = reader.iter_internal(&self.cache)?;
            let rts = reader.range_tombstones();
            let (smallest, largest) =
                if let (Some(first), Some(last)) = (entries.first(), entries.last()) {
                    let (uk_lo, _, _) = decode_internal_key(&first.0);
                    let (uk_hi, _, _) = decode_internal_key(&last.0);
                    (uk_lo.to_vec(), uk_hi.to_vec())
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

        // Serialize with background compaction for the duration of the
        // ingest — same pattern as `compact_range`.
        let _compact_guard = self.compaction_lock.lock();

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
        // the memtable is non-empty and we might be landing at L0 —
        // which is the only level where a concurrent memtable could
        // shadow the ingested keys.
        let needs_flush = {
            let mt = self.active_memtable.read();
            !mt.is_empty() || !mt.clone_range_tombstones().is_empty()
        };
        if needs_flush {
            let _write_guard = self.write_lock.lock();
            let mt = self.active_memtable.read();
            if !mt.is_empty() || !mt.clone_range_tombstones().is_empty() {
                drop(mt);
                self.rotate_memtable()?;
            }
        }

        // Compute the target level from the current version.
        let version = self.versions.lock().current();
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
        )?;

        // Re-encode every point entry with the ingest seq.
        let entries = source.reader.iter_internal(&self.cache)?;
        for (ik, value) in entries {
            let (uk, _old_seq, vt) = decode_internal_key(&ik);
            let new_ik = encode_internal_key(uk, ingest_seq, vt);
            writer.add(&new_ik, &value)?;
        }
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
        let reader = Arc::new(SsTableReader::open(&dest_path, file_id)?);
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
    /// their filesystem work is done — a snapshot whose lifetime
    /// outlives the enclosing function scope can deadlock a
    /// concurrent `db.close()` / `drop(db)` that joins the background
    /// compaction thread (the thread will block on the same lock the
    /// snapshot holds).
    pub(crate) fn checkpoint_capture(&self) -> std::io::Result<CheckpointSnapshot> {
        {
            let has_any = {
                let mt = self.active_memtable.read();
                !mt.is_empty() || !mt.clone_range_tombstones().is_empty()
            };
            if has_any {
                let _write_guard = self.write_lock.lock();
                let still_has_any = {
                    let mt = self.active_memtable.read();
                    !mt.is_empty() || !mt.clone_range_tombstones().is_empty()
                };
                if still_has_any {
                    self.rotate_memtable()?;
                }
            }
        }

        let compaction_guard = self.compaction_lock.lock_arc();

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
            // reference files that are *not* in the captured version
            // — copying them into a checkpoint manifest would cause
            // recovery to fail on the missing files.
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
    /// per-range estimates. Index-only — no data-block decompression
    /// happens, so the cost scales with `num_files * log(num_blocks)`.
    pub(crate) fn approximate_size_in_range(&self, start: &[u8], end: &[u8]) -> u64 {
        if start >= end {
            return 0;
        }
        let version = self.versions.lock().current();
        let mut total: u64 = 0;
        for level in &version.levels {
            for file in level {
                if file.meta.largest_key.as_slice() < start {
                    continue;
                }
                if file.meta.smallest_key.as_slice() >= end {
                    continue;
                }
                total += file.reader.approximate_size_in_range(start, end);
            }
        }
        total
    }

    /// Exact `(count, size)` for every entry in the active memtable
    /// whose user key falls in `[start, end)`. Frozen memtables are
    /// *not* included — a caller that wants "everything in memory"
    /// should call this and also walk the frozen memtables
    /// separately.
    pub(crate) fn approximate_memtable_stats(&self, start: &[u8], end: &[u8]) -> (u64, u64) {
        self.active_memtable
            .read()
            .approximate_stats_for_range(start, end)
    }

    // ── property helpers ───────────────────────────────────────────────
    //
    // These return raw values consumed by `Db::get_property` /
    // `Db::get_int_property`. Every method is cheap — no block
    // reads, no locks held beyond a short `versions.lock()` or
    // memtable read.

    /// Number of SSTable files at a specific level. Returns 0 for
    /// out-of-range levels rather than panicking, so
    /// `rocksdb.num-files-at-level<N>` for unknown levels reads
    /// cleanly as `Some(0)`.
    pub(crate) fn num_files_at_level(&self, level: usize) -> u64 {
        let version = self.versions.lock().current();
        version
            .levels
            .get(level)
            .map(|files| files.len() as u64)
            .unwrap_or(0)
    }

    /// Total size in bytes across every level of the current
    /// version — sum of every `LiveSst::meta.file_size`.
    pub(crate) fn total_sst_size(&self) -> u64 {
        let version = self.versions.lock().current();
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
        let version = self.versions.lock().current();
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
        self.active_memtable.read().approximate_size() as u64
    }

    /// Total approximate size of every frozen memtable.
    pub(crate) fn frozen_memtables_size(&self) -> u64 {
        self.frozen_memtables
            .read()
            .iter()
            .map(|mt| mt.approximate_size() as u64)
            .sum()
    }

    /// Number of live snapshots. Counts pins, not distinct seqs —
    /// two snapshots taken at the same seq contribute two.
    pub(crate) fn live_snapshot_count(&self) -> u64 {
        self.snapshot_registry.live_count()
    }

    /// Unix-seconds timestamp of the oldest live snapshot, or
    /// `None` when no snapshot is alive.
    pub(crate) fn oldest_snapshot_time_unix(&self) -> Option<u64> {
        self.snapshot_registry.oldest_snapshot_time_unix()
    }

    /// Borrow the current version so a caller can walk every
    /// SSTable's metadata — used by `rocksdb.sstables` formatter.
    pub(crate) fn current_version(&self) -> Arc<manifest::Version> {
        self.versions.lock().current()
    }

    /// Drop all data in the engine.
    pub(crate) fn drop_all(&self) -> std::io::Result<()> {
        let _write_guard = self.write_lock.lock();

        *self.active_memtable.write() = Arc::new(MemTable::new());
        self.frozen_memtables.write().clear();

        let version = self.versions.lock().current();
        for level in &version.levels {
            for file in level {
                let path = self.sst_dir.join(sst_filename(file.meta.file_id));
                let _ = std::fs::remove_file(&path);
            }
        }

        self.cache.clear();

        {
            let mut versions = self.versions.lock();
            let mut edits = Vec::new();
            let ver = versions.current();
            for (level_idx, level) in ver.levels.iter().enumerate() {
                for file in level {
                    edits.push(VersionEdit::RemoveFile {
                        level: level_idx,
                        file_id: file.meta.file_id,
                    });
                }
            }
            if !edits.is_empty() {
                versions.apply(&edits)?;
            }
            versions.compact_manifest()?;
        }

        let wal_files = list_wal_files(&self.wal_dir)?;
        for path in wal_files {
            let _ = std::fs::remove_file(&path);
        }

        let wal_id = {
            let mut versions = self.versions.lock();
            let version = versions.current();
            let id = version.next_file_id;
            versions.apply(&[VersionEdit::SetNextFileId(id + 1)])?;
            id
        };
        let wal_path = self.wal_dir.join(wal_filename(wal_id));
        let new_wal = Wal::create(&wal_path)?;
        *self.active_wal.lock() = new_wal;
        self.wal_id.store(wal_id, Ordering::Release);
        self.latest_seq.store(0, Ordering::Release);

        Ok(())
    }

    /// Test-only: whether the active memtable currently holds any
    /// entries. Used by `compact_range` tests to verify that the
    /// memtable was flushed to L0 as part of the range walk.
    #[cfg(test)]
    pub(crate) fn active_memtable_is_empty(&self) -> bool {
        self.active_memtable.read().is_empty()
            && self
                .active_memtable
                .read()
                .clone_range_tombstones()
                .is_empty()
    }

    /// Test-only: number of SSTable files at `level` in the current
    /// version.
    #[cfg(test)]
    pub(crate) fn level_file_count(&self, level: usize) -> usize {
        self.versions.lock().current().levels[level].len()
    }

    /// Test-only: total number of SSTable files across every level.
    #[cfg(test)]
    pub(crate) fn total_file_count(&self) -> usize {
        let v = self.versions.lock().current();
        v.levels.iter().map(|level| level.len()).sum()
    }

    /// Test-only: collect every raw `(seq, value_type)` version of
    /// `user_key` currently persisted across all SSTables. Used by the
    /// snapshot-pinning GC tests to check the post-compaction on-disk
    /// state — not just what reads see.
    #[cfg(test)]
    pub(crate) fn all_persisted_versions_of(
        &self,
        user_key: &[u8],
    ) -> std::io::Result<Vec<(u64, u8)>> {
        use internal_key::decode_internal_key;

        let version = self.versions.lock().current();
        let mut out = Vec::new();
        for level in &version.levels {
            for file in level {
                for (ik, _v) in file.reader.iter_internal(&self.cache)? {
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
        let has_any = {
            let mt = self.active_memtable.read();
            !mt.is_empty() || !mt.clone_range_tombstones().is_empty()
        };
        if has_any {
            let _write_guard = self.write_lock.lock();

            let old_memtable = {
                let mut active = self.active_memtable.write();
                let old = Arc::clone(&active);
                *active = Arc::new(MemTable::new());
                old
            };

            if !old_memtable.is_empty() {
                self.frozen_memtables.write().push(old_memtable);

                let wal_for_flush = Wal::create(&self.wal_dir.join("flush_tmp.wal"))?;
                let old_wal = std::mem::replace(&mut *self.active_wal.lock(), wal_for_flush);

                drop(_write_guard);
                self.flush_frozen_memtable(old_wal)?;
            }
        }

        self.active_wal.lock().sync()?;
        self.compaction.lock().shutdown();

        Ok(())
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
/// they are done with the filesystem work — a snapshot whose
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
    _compaction_guard: parking_lot::ArcMutexGuard<parking_lot::RawMutex, ()>,
}

impl CheckpointSnapshot {
    /// Format an SSTable filename for the given file id — exposes
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
