use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

use super::block_cache::BlockCache;
use super::internal_key::{compare_internal_keys, decode_internal_key, user_key_of};
use super::manifest::{VersionEdit, VersionSet, MAX_LEVELS};
use super::range_tombstone::{max_covering_seq, RangeTombstone};
use super::snapshot_registry::SnapshotRegistry;
use super::sstable::{
    remove_sst, sst_filename, LiveSst, SsTableMeta, SsTableReader, SsTableWriter,
};

/// Default compaction trigger: flush L0 → L1 when L0 has this many SSTables.
pub(crate) const L0_COMPACTION_TRIGGER: usize = 4;

/// Default level size multiplier between levels.
pub(crate) const LEVEL_SIZE_MULTIPLIER: u64 = 10;

/// Default max bytes for level 1 (256 MB).
pub(crate) const DEFAULT_LEVEL_BASE_BYTES: u64 = 256 * 1024 * 1024;

/// Default target SSTable file size (64 MB).
pub(crate) const DEFAULT_TARGET_FILE_SIZE: u64 = 64 * 1024 * 1024;

/// Manages background compaction on one or more dedicated OS threads.
pub(crate) struct CompactionScheduler {
    shutdown: Arc<AtomicBool>,
    trigger: Arc<(Mutex<bool>, Condvar)>,
    handles: Vec<thread::JoinHandle<()>>,
}

impl CompactionScheduler {
    /// Start one or more background compaction threads.
    ///
    /// `compaction_lock` is the engine-wide RwLock that serializes
    /// foreground callers (write lock) with background workers (read
    /// lock). Multiple workers can run concurrently; the in-progress
    /// set inside `compaction_loop` ensures they don't pick overlapping
    /// input sets.
    ///
    /// `snapshot_registry` lets each compaction pass query the current
    /// pin seq so it can drop versions that no live snapshot needs.
    pub(crate) fn start(
        compaction_lock: Arc<parking_lot::RwLock<()>>,
        snapshot_registry: Arc<SnapshotRegistry>,
        versions: Arc<parking_lot::Mutex<VersionSet>>,
        sst_dir: Arc<Path>,
        cache: Arc<BlockCache>,
        opts: CompactionOptions,
        stall_signal: Arc<crate::engine::StallSignal>,
    ) -> Self {
        let shutdown = Arc::new(AtomicBool::new(false));
        let trigger = Arc::new((Mutex::new(false), Condvar::new()));
        let in_progress: Arc<parking_lot::Mutex<HashSet<u64>>> =
            Arc::new(parking_lot::Mutex::new(HashSet::new()));

        let worker_count = opts.max_background_compactions.max(1);
        let mut handles = Vec::with_capacity(worker_count);

        for i in 0..worker_count {
            let shutdown_clone = Arc::clone(&shutdown);
            let trigger_clone = Arc::clone(&trigger);
            let lock_clone = Arc::clone(&compaction_lock);
            let registry_clone = Arc::clone(&snapshot_registry);
            let versions_clone = Arc::clone(&versions);
            let sst_dir_clone = Arc::clone(&sst_dir);
            let cache_clone = Arc::clone(&cache);
            let opts_clone = opts.clone();
            let stall_clone = Arc::clone(&stall_signal);
            let in_progress_clone = Arc::clone(&in_progress);

            let handle = thread::Builder::new()
                .name(format!("lark-compaction-{i}"))
                .spawn(move || {
                    compaction_loop(
                        shutdown_clone,
                        trigger_clone,
                        lock_clone,
                        registry_clone,
                        versions_clone,
                        sst_dir_clone,
                        cache_clone,
                        opts_clone,
                        stall_clone,
                        in_progress_clone,
                    );
                })
                .expect("failed to spawn compaction thread");
            handles.push(handle);
        }

        Self {
            shutdown,
            trigger,
            handles,
        }
    }

    /// Notify the compaction thread that work may be available.
    pub(crate) fn notify(&self) {
        let (lock, cvar) = &*self.trigger;
        let mut triggered = lock.lock().unwrap();
        *triggered = true;
        cvar.notify_one();
    }

    /// Shut down all compaction threads.
    pub(crate) fn shutdown(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        self.notify();
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
    }
}

impl Drop for CompactionScheduler {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[derive(Clone)]
pub(crate) struct CompactionOptions {
    pub(crate) l0_compaction_trigger: usize,
    pub(crate) level_base_bytes: u64,
    pub(crate) level_size_multiplier: u64,
    pub(crate) target_file_size: u64,
    pub(crate) block_size: usize,
    pub(crate) bloom_bits_per_key: usize,
    pub(crate) compression: crate::options::CompressionType,
    pub(crate) compression_per_level: Option<Vec<crate::options::CompressionType>>,
    pub(crate) compaction_filter: Option<Arc<dyn crate::options::CompactionFilter>>,
    pub(crate) prefix_extractor: Option<Arc<dyn crate::options::PrefixExtractor>>,
    pub(crate) merge_operator: Option<Arc<dyn crate::options::MergeOperator>>,
    pub(crate) listeners: Vec<Arc<dyn crate::event_listener::EventListener>>,
    pub(crate) statistics: Option<Arc<crate::statistics::Statistics>>,
    pub(crate) rate_limiter: Option<Arc<dyn crate::rate_limiter::RateLimiter>>,
    pub(crate) compaction_style: crate::options::CompactionStyle,
    pub(crate) fifo_compaction_options: crate::options::FifoCompactionOptions,
    pub(crate) universal_compaction_options: crate::options::UniversalCompactionOptions,
    pub(crate) use_direct_io_for_compaction: bool,
    pub(crate) max_subcompactions: usize,
    pub(crate) max_background_compactions: usize,
    pub(crate) partitioned_index: bool,
    pub(crate) metadata_block_size: usize,
}

impl CompactionOptions {
    /// Resolve the codec used when writing an output SSTable destined
    /// for `level`. Mirrors `EngineOptions::compression_for_level`.
    pub(crate) fn compression_for_level(&self, level: usize) -> crate::options::CompressionType {
        match &self.compression_per_level {
            Some(per_level) if level < per_level.len() => per_level[level],
            _ => self.compression,
        }
    }
}

impl Default for CompactionOptions {
    fn default() -> Self {
        Self {
            l0_compaction_trigger: L0_COMPACTION_TRIGGER,
            level_base_bytes: DEFAULT_LEVEL_BASE_BYTES,
            level_size_multiplier: LEVEL_SIZE_MULTIPLIER,
            target_file_size: DEFAULT_TARGET_FILE_SIZE,
            block_size: 16 * 1024,
            bloom_bits_per_key: 10,
            compression: crate::options::CompressionType::Lz4,
            compression_per_level: None,
            compaction_filter: None,
            prefix_extractor: None,
            merge_operator: None,
            listeners: Vec::new(),
            statistics: None,
            rate_limiter: None,
            compaction_style: crate::options::CompactionStyle::Level,
            fifo_compaction_options: crate::options::FifoCompactionOptions::default(),
            universal_compaction_options: crate::options::UniversalCompactionOptions::default(),
            use_direct_io_for_compaction: false,
            max_subcompactions: 1,
            max_background_compactions: 1,
            partitioned_index: false,
            metadata_block_size: 4096,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn compaction_loop(
    shutdown: Arc<AtomicBool>,
    trigger: Arc<(Mutex<bool>, Condvar)>,
    compaction_lock: Arc<parking_lot::RwLock<()>>,
    snapshot_registry: Arc<SnapshotRegistry>,
    versions: Arc<parking_lot::Mutex<VersionSet>>,
    sst_dir: Arc<Path>,
    cache: Arc<BlockCache>,
    opts: CompactionOptions,
    stall_signal: Arc<crate::engine::StallSignal>,
    in_progress: Arc<parking_lot::Mutex<HashSet<u64>>>,
) {
    loop {
        // Wait for trigger or periodic check
        {
            let (lock, cvar) = &*trigger;
            let mut triggered = lock.lock().unwrap();
            if !*triggered {
                let _ = cvar.wait_timeout(triggered, std::time::Duration::from_secs(1));
                triggered = lock.lock().unwrap();
            }
            *triggered = false;
        }

        if shutdown.load(Ordering::Acquire) {
            break;
        }

        // Drive compactions until there's nothing to do. Each pass
        // takes a read lock so multiple workers can run concurrently.
        // Foreground callers (`compact_range`, `ingest_external_files`,
        // `checkpoint_capture`) take the write lock, which blocks until
        // all workers finish their current pass.
        loop {
            let did_work = {
                let _guard = compaction_lock.read();
                // Recompute the GC horizon on every pass: a snapshot
                // may have dropped since the previous pass, unpinning
                // more versions.
                let pin_seq = snapshot_registry.oldest_live_seq();
                match pick_and_run_compaction(
                    &versions,
                    &sst_dir,
                    &cache,
                    &opts,
                    pin_seq,
                    &in_progress,
                ) {
                    Ok(did_work) => did_work,
                    Err(e) => {
                        tracing::error!(error = %e, "Compaction failed");
                        // Surface the failure to any registered
                        // listeners so metrics pipelines and
                        // debuggers notice it — the scheduler
                        // itself keeps running.
                        if !opts.listeners.is_empty() {
                            let err =
                                crate::Error::Io(std::io::Error::new(e.kind(), e.to_string()));
                            crate::event_listener::dispatch(&opts.listeners, |l| {
                                l.on_background_error(
                                    crate::event_listener::BackgroundErrorReason::Compaction,
                                    &err,
                                )
                            });
                        }
                        false
                    }
                }
            };

            // After each pass, wake any foreground writer that was
            // blocked by a "stop writes" condition so it can re-check
            // thresholds against the freshly-updated version.
            stall_signal.notify_all();

            if !did_work || shutdown.load(Ordering::Acquire) {
                break;
            }
        }
    }
}

fn pick_and_run_compaction(
    versions: &Arc<parking_lot::Mutex<VersionSet>>,
    sst_dir: &Path,
    cache: &BlockCache,
    opts: &CompactionOptions,
    pin_seq: u64,
    in_progress: &parking_lot::Mutex<HashSet<u64>>,
) -> std::io::Result<bool> {
    match opts.compaction_style {
        crate::options::CompactionStyle::Level => {
            pick_and_run_level_compaction(versions, sst_dir, cache, opts, pin_seq, in_progress)
        }
        crate::options::CompactionStyle::Fifo => {
            // Skip if any file is already being compacted — with a
            // single L0 pool there's nothing safe to pick in parallel.
            if !in_progress.lock().is_empty() {
                return Ok(false);
            }
            run_fifo_pass(versions, sst_dir, opts)
        }
        crate::options::CompactionStyle::Universal => {
            // Same: universal merges all L0 files, so parallel picks
            // would conflict. Skip until the in-progress set clears.
            if !in_progress.lock().is_empty() {
                return Ok(false);
            }
            pick_and_run_universal(versions, sst_dir, cache, opts, pin_seq)
        }
    }
}

/// Universal (size-tiered) picker. Every L0 file is a "run"; the
/// picker either merges the newest handful of files (size-ratio
/// rule) or folds every file into one run (size-amplification
/// rule). See the [`crate::UniversalCompactionOptions`] doc for
/// the meaning of each trigger.
///
/// The picker returns after at most one merge per call; the
/// background compaction loop re-invokes it until there's nothing
/// left to do.
fn pick_and_run_universal(
    versions: &Arc<parking_lot::Mutex<VersionSet>>,
    sst_dir: &Path,
    cache: &BlockCache,
    opts: &CompactionOptions,
    pin_seq: u64,
) -> std::io::Result<bool> {
    let universal = opts.universal_compaction_options;

    // Snapshot L0 under the version lock, then make decisions
    // without holding it.
    let l0_files: Vec<Arc<LiveSst>> = {
        let version = versions.lock().current();
        version.levels[0].clone()
    };
    if l0_files.len() < 2 {
        return Ok(false);
    }

    // Sort newest-first. Age is approximated by `file_id`: every
    // fresh flush / merge output gets a strictly larger id than
    // any previous one (handed out monotonically by the version
    // set), so largest = newest.
    let mut by_age: Vec<Arc<LiveSst>> = l0_files;
    by_age.sort_by_key(|f| std::cmp::Reverse(f.meta.file_id));

    // Rule 1 — size ratio merge. Accumulate newest files into a
    // candidate group; keep adding until the group's total size
    // grows past `size_ratio` percent of the next file's size, at
    // which point the next file is too much larger to be worth
    // folding in. If we end up with `min_merge_width` or more
    // files in the group, merge them.
    let max_width = universal.max_merge_width.max(1) as usize;
    let min_width = universal.min_merge_width.max(2) as usize;
    let ratio = universal.size_ratio as u128;
    let mut group: Vec<Arc<LiveSst>> = Vec::new();
    let mut group_size: u128 = 0;
    for (i, file) in by_age.iter().enumerate() {
        if group.len() >= max_width {
            break;
        }
        if group.is_empty() {
            group.push(Arc::clone(file));
            group_size = file.meta.file_size as u128;
            continue;
        }
        // Does adding this file keep the "size-tier" invariant?
        // The rule: the running total must be within
        // `size_ratio` percent of the new candidate — i.e. the
        // candidate should not dwarf the accumulator.
        let candidate_size = file.meta.file_size as u128;
        if group_size * (100 + ratio) / 100 >= candidate_size {
            group.push(Arc::clone(file));
            group_size += candidate_size;
        } else {
            break;
        }
        // Safety: never include the final file unless we also
        // trigger the size-amp rule below.
        let _ = i;
    }
    if group.len() >= min_width {
        return perform_universal_merge(versions, sst_dir, cache, opts, group, pin_seq)
            .map(|_| true);
    }

    // Rule 2 — size amplification. Compute the ratio of "all
    // files except the oldest" to "the oldest file". When it
    // exceeds the configured percent, fold everything into one
    // run so the database stops accumulating redundancy.
    let mut oldest_first = by_age.clone();
    oldest_first.reverse();
    let oldest_size = oldest_first[0].meta.file_size as u128;
    if oldest_size == 0 {
        return Ok(false);
    }
    let younger_total: u128 = oldest_first
        .iter()
        .skip(1)
        .map(|f| f.meta.file_size as u128)
        .sum();
    let amp_percent = younger_total * 100 / oldest_size;
    if amp_percent >= universal.max_size_amplification_percent as u128 {
        return perform_universal_merge(versions, sst_dir, cache, opts, by_age, pin_seq)
            .map(|_| true);
    }

    Ok(false)
}

/// Merge every file in `inputs` into a single new L0 file. Input
/// files are removed from the version; the new file gets a fresh
/// id (so it sorts as the newest L0 file under the picker's
/// age-by-file_id ordering).
fn perform_universal_merge(
    versions: &Arc<parking_lot::Mutex<VersionSet>>,
    sst_dir: &Path,
    cache: &BlockCache,
    opts: &CompactionOptions,
    inputs: Vec<Arc<LiveSst>>,
    pin_seq: u64,
) -> std::io::Result<()> {
    // Universal merges are L0 → L0. The shared
    // `perform_compaction_to` helper handles the merge-and-write
    // path; we just tell it to target L0 and pass no overlap
    // files (there is no "next level" to drag in).
    perform_compaction_to(
        versions,
        sst_dir,
        cache,
        opts,
        0,
        0,
        inputs,
        Vec::new(),
        pin_seq,
    )
}

/// Public wrapper for the FIFO picker, used by the synchronous
/// `compact_range` path so callers can trigger a deterministic
/// drop of over-cap files.
pub(crate) fn run_fifo_pass(
    versions: &Arc<parking_lot::Mutex<VersionSet>>,
    sst_dir: &Path,
    opts: &CompactionOptions,
) -> std::io::Result<bool> {
    pick_and_run_fifo(versions, sst_dir, opts)
}

/// Force-merge every L0 file into a single run. Called by the
/// synchronous `Db::compact_range` path when the configured
/// compaction style is [`crate::CompactionStyle::Universal`], so a
/// caller asking for a manual compaction gets full deduplication
/// across every live file regardless of the picker's ratio rules.
pub(crate) fn run_universal_full_compaction(
    versions: &Arc<parking_lot::Mutex<VersionSet>>,
    sst_dir: &Path,
    cache: &BlockCache,
    opts: &CompactionOptions,
    pin_seq: u64,
) -> std::io::Result<()> {
    let l0_files: Vec<Arc<LiveSst>> = {
        let version = versions.lock().current();
        version.levels[0].clone()
    };
    if l0_files.len() < 2 {
        return Ok(());
    }
    perform_universal_merge(versions, sst_dir, cache, opts, l0_files, pin_seq)
}

fn pick_and_run_level_compaction(
    versions: &Arc<parking_lot::Mutex<VersionSet>>,
    sst_dir: &Path,
    cache: &BlockCache,
    opts: &CompactionOptions,
    pin_seq: u64,
    in_progress: &parking_lot::Mutex<HashSet<u64>>,
) -> std::io::Result<bool> {
    let version = versions.lock().current();

    // Check L0 first
    if version.l0_count() >= opts.l0_compaction_trigger {
        return compact_l0(versions, sst_dir, cache, opts, pin_seq, in_progress);
    }

    // Check other levels
    for level in 1..MAX_LEVELS - 1 {
        let target = level_target_size(level, opts);
        if version.level_size(level) > target {
            return compact_level(versions, sst_dir, cache, opts, level, pin_seq, in_progress);
        }
    }

    Ok(false)
}

/// FIFO picker: every flush lands a new L0 file, and lark never
/// merges in this style. After each pass, if the total bytes held
/// across L0 exceed `max_table_files_size`, we unlink the oldest
/// file (smallest `file_id`) and emit a `RemoveFile` edit. We stop
/// when the cap is satisfied or only one file remains — a single
/// oversized file is not worth deleting because that would lose
/// data with no successor on disk.
fn pick_and_run_fifo(
    versions: &Arc<parking_lot::Mutex<VersionSet>>,
    sst_dir: &Path,
    opts: &CompactionOptions,
) -> std::io::Result<bool> {
    let limit = opts.fifo_compaction_options.max_table_files_size;
    if limit == 0 {
        return Ok(false);
    }

    // Snapshot the current L0 contents under the version lock, then
    // make decisions outside it so we don't hold the lock across
    // the unlink syscall.
    let l0_files: Vec<Arc<crate::engine::sstable::LiveSst>> = {
        let version = versions.lock().current();
        version.levels[0].clone()
    };

    let total: u64 = l0_files.iter().map(|f| f.meta.file_size).sum();
    if total <= limit {
        return Ok(false);
    }

    // Sort by file_id ascending — smaller id was allocated earlier,
    // so it is the oldest file by construction.
    let mut by_age: Vec<Arc<crate::engine::sstable::LiveSst>> = l0_files;
    by_age.sort_by_key(|f| f.meta.file_id);

    let mut running_total = total;
    let mut edits: Vec<VersionEdit> = Vec::new();
    let mut removed_paths: Vec<std::path::PathBuf> = Vec::new();
    for file in &by_age {
        // Always keep at least one file alive — dropping the only
        // remaining file would delete every byte of user data the
        // database currently holds.
        if by_age.len() - edits.len() <= 1 {
            break;
        }
        if running_total <= limit {
            break;
        }
        edits.push(VersionEdit::RemoveFile {
            level: 0,
            file_id: file.meta.file_id,
        });
        removed_paths.push(sst_dir.join(sst_filename(file.meta.file_id)));
        running_total = running_total.saturating_sub(file.meta.file_size);
    }

    if edits.is_empty() {
        return Ok(false);
    }

    versions.lock().apply(&edits)?;

    // Best-effort unlink. The `LiveSst` reader Arcs in older
    // versions (held by long-running snapshots / iterators) keep
    // file descriptors open, so the kernel preserves the inode
    // until those readers drop. A failure here doesn't corrupt the
    // database — the manifest already reflects the removal.
    for path in &removed_paths {
        let _ = std::fs::remove_file(path);
    }

    if let Some(s) = &opts.statistics {
        s.add(
            crate::statistics::Ticker::CompactionCount,
            edits.len() as u64,
        );
    }

    Ok(true)
}

fn level_target_size(level: usize, opts: &CompactionOptions) -> u64 {
    let mut size = opts.level_base_bytes;
    for _ in 1..level {
        size = size.saturating_mul(opts.level_size_multiplier);
    }
    size
}

/// Compact all L0 SSTables into L1.
fn compact_l0(
    versions: &Arc<parking_lot::Mutex<VersionSet>>,
    sst_dir: &Path,
    cache: &BlockCache,
    opts: &CompactionOptions,
    pin_seq: u64,
    in_progress: &parking_lot::Mutex<HashSet<u64>>,
) -> std::io::Result<bool> {
    // If any L0 file is already being compacted by another worker,
    // skip — L0 files may overlap so concurrent picks would produce
    // conflicting output sets.
    {
        let ip = in_progress.lock();
        let version = versions.lock().current();
        if version.levels[0]
            .iter()
            .any(|f| ip.contains(&f.meta.file_id))
        {
            return Ok(false);
        }
    }
    compact_level(versions, sst_dir, cache, opts, 0, pin_seq, in_progress)
}

/// Compact a level into the next level using the standard size-based
/// heuristic (all L0 files, or the first file of L1+). Used by the
/// background scheduler.
fn compact_level(
    versions: &Arc<parking_lot::Mutex<VersionSet>>,
    sst_dir: &Path,
    cache: &BlockCache,
    opts: &CompactionOptions,
    level: usize,
    pin_seq: u64,
    in_progress: &parking_lot::Mutex<HashSet<u64>>,
) -> std::io::Result<bool> {
    let target_level = level + 1;
    if target_level >= MAX_LEVELS {
        return Ok(false);
    }

    let version = versions.lock().current();

    let input_files: Vec<Arc<LiveSst>> = version.levels[level].clone();
    if input_files.is_empty() {
        return Ok(false);
    }

    // For L0, all files may overlap. For other levels, pick the first
    // file that is not already being compacted by another worker.
    let (input_files, overlap_files) = if level == 0 {
        let l0_files = input_files;
        let (min_key, max_key) = key_range(&l0_files);
        let overlapping = find_overlapping(&version.levels[target_level], &min_key, &max_key);
        (l0_files, overlapping)
    } else {
        let ip = in_progress.lock();
        let candidate = input_files
            .iter()
            .find(|f| !ip.contains(&f.meta.file_id))
            .map(Arc::clone);
        drop(ip);
        let picked = match candidate {
            Some(f) => vec![f],
            None => return Ok(false),
        };
        let (min_key, max_key) = key_range(&picked);
        // Also skip if any overlap file is already in-progress.
        let overlapping = find_overlapping(&version.levels[target_level], &min_key, &max_key);
        {
            let ip = in_progress.lock();
            if overlapping.iter().any(|f| ip.contains(&f.meta.file_id)) {
                return Ok(false);
            }
        }
        (picked, overlapping)
    };

    // Register all input and overlap file ids before releasing the
    // version snapshot so concurrent workers see them immediately.
    let all_ids: Vec<u64> = input_files
        .iter()
        .chain(overlap_files.iter())
        .map(|f| f.meta.file_id)
        .collect();
    {
        let mut ip = in_progress.lock();
        for &id in &all_ids {
            ip.insert(id);
        }
    }

    let result = perform_compaction(
        versions,
        sst_dir,
        cache,
        opts,
        level,
        input_files,
        overlap_files,
        pin_seq,
    );

    // Always deregister, even on error, so workers don't stall
    // forever waiting for a slot that will never clear.
    {
        let mut ip = in_progress.lock();
        for id in &all_ids {
            ip.remove(id);
        }
    }

    result?;
    Ok(true)
}

/// Run a manual `compact_range` pass synchronously: for every level from
/// 0 down to MAX_LEVELS-2, pick the files overlapping `[start, end)` and
/// push them down to the next level. Returns once no level has any more
/// files in the range that can be pushed down.
///
/// Callers must hold the engine-wide compaction lock so the background
/// scheduler can't pick an overlapping input set concurrently.
pub(crate) fn run_compact_range(
    versions: &Arc<parking_lot::Mutex<VersionSet>>,
    sst_dir: &Path,
    cache: &BlockCache,
    opts: &CompactionOptions,
    start: Option<&[u8]>,
    end: Option<&[u8]>,
    pin_seq: u64,
) -> std::io::Result<()> {
    for level in 0..MAX_LEVELS - 1 {
        loop {
            let version = versions.lock().current();
            let files = &version.levels[level];
            if files.is_empty() {
                break;
            }

            // At L0 files overlap each other, so picking any file that
            // intersects the range drags in every other L0 file it
            // overlaps — simplest correct move: take every L0 file
            // that intersects the range in one shot.
            //
            // At L1+ files are non-overlapping, so we can pick them
            // one at a time.
            let inputs: Vec<Arc<LiveSst>> = if level == 0 {
                files
                    .iter()
                    .filter(|f| file_overlaps_range(f, start, end))
                    .map(Arc::clone)
                    .collect()
            } else {
                files
                    .iter()
                    .find(|f| file_overlaps_range(f, start, end))
                    .map(Arc::clone)
                    .into_iter()
                    .collect()
            };

            if inputs.is_empty() {
                break;
            }

            let (min_key, max_key) = key_range(&inputs);
            let target_level = level + 1;
            let overlap_files = find_overlapping(&version.levels[target_level], &min_key, &max_key);

            perform_compaction(
                versions,
                sst_dir,
                cache,
                opts,
                level,
                inputs,
                overlap_files,
                pin_seq,
            )?;

            // At L0 we handled every range-overlapping file in one
            // shot, so we're done with this level.
            if level == 0 {
                break;
            }
            // At L1+, loop again to pick the next file in the range,
            // if any.
        }
    }
    Ok(())
}

/// Inner body of a compaction: read the merged entries from
/// `input_files` + `overlap_files`, write new output SSTables at
/// `target_level`, and atomically apply the version edit (Remove old
/// files, Add new ones). File descriptors of old files stay alive via
/// any `Arc<LiveSst>` still referenced by older versions / iterators.
///
/// `pin_seq` is the snapshot-pinning GC horizon: every version with a
/// seq older than the version visible to `pin_seq` can be dropped.
/// When no snapshot is live, callers pass `u64::MAX` and only the
/// newest version of each user key is retained.
#[allow(clippy::too_many_arguments)]
fn perform_compaction(
    versions: &Arc<parking_lot::Mutex<VersionSet>>,
    sst_dir: &Path,
    cache: &BlockCache,
    opts: &CompactionOptions,
    level: usize,
    input_files: Vec<Arc<LiveSst>>,
    overlap_files: Vec<Arc<LiveSst>>,
    pin_seq: u64,
) -> std::io::Result<()> {
    // Leveled callers always push down one level.
    perform_compaction_to(
        versions,
        sst_dir,
        cache,
        opts,
        level,
        level + 1,
        input_files,
        overlap_files,
        pin_seq,
    )
}

/// Core compaction body. Accepts an explicit `output_level` so the
/// caller can target either the next level (leveled push-down) or
/// the same level (universal in-place merge at L0). `overlap_files`
/// are read as additional inputs and their version entries at
/// `output_level` are removed alongside the `input_files` at
/// `input_level`.
#[allow(clippy::too_many_arguments)]
fn perform_compaction_to(
    versions: &Arc<parking_lot::Mutex<VersionSet>>,
    sst_dir: &Path,
    cache: &BlockCache,
    opts: &CompactionOptions,
    level: usize,
    target_level: usize,
    input_files: Vec<Arc<LiveSst>>,
    overlap_files: Vec<Arc<LiveSst>>,
    pin_seq: u64,
) -> std::io::Result<()> {
    let compaction_start = std::time::Instant::now();

    // Snapshot input file ids up-front so the on_compaction_begin
    // callback has them even if the job errors out later.
    let (begin_inputs_l, begin_inputs_l1): (Vec<u64>, Vec<u64>) = (
        input_files.iter().map(|f| f.meta.file_id).collect(),
        overlap_files.iter().map(|f| f.meta.file_id).collect(),
    );
    if !opts.listeners.is_empty() {
        let info = crate::event_listener::CompactionJobInfo {
            input_level: level,
            output_level: target_level,
            input_files_input_level: begin_inputs_l.clone(),
            input_files_output_level: begin_inputs_l1.clone(),
            output_files: Vec::new(),
            duration: std::time::Duration::ZERO,
        };
        crate::event_listener::dispatch(&opts.listeners, |l| l.on_compaction_begin(&info));
    }

    // Read all input entries as raw internal-key / value pairs so every
    // version and tombstone is preserved through the merge. Readers are
    // already open in the pinned version — no fresh `File::open`.
    let mut all_entries: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
    let mut merged_range_tombstones: Vec<RangeTombstone> = Vec::new();
    for file in input_files.iter().chain(overlap_files.iter()) {
        all_entries.extend(file.reader.iter_internal(cache)?);
        for rt in file.reader.range_tombstones() {
            merged_range_tombstones.push(rt.clone());
        }
        // Tell the OS it can drop pages we just read from the
        // page cache. Opt-in via `use_direct_io_for_compaction`;
        // on non-Linux targets the hint is a no-op.
        if opts.use_direct_io_for_compaction {
            let path = sst_dir.join(sst_filename(file.meta.file_id));
            crate::os_hint::drop_page_cache_by_path(&path);
        }
    }

    // Sort by internal key (which orders newer seqs first within each user
    // key) and drop any exact duplicates (same user_key + seq + type).
    all_entries.sort_by(|a, b| compare_internal_keys(&a.0, &b.0));
    all_entries.dedup_by(|a, b| a.0 == b.0);

    // User compaction filter for range tombstones. Run this **before**
    // the point-entry RT shadow pass so a filter that drops a range
    // tombstone doesn't leave orphaned point entries already wiped
    // out by that same RT. Only runs when no snapshot is pinned.
    if pin_seq == u64::MAX {
        if let Some(filter) = opts.compaction_filter.as_ref() {
            merged_range_tombstones.retain(|rt| {
                !matches!(
                    filter.filter_range_delete(target_level, &rt.start, &rt.end),
                    crate::options::CompactionDecision::Remove
                )
            });
        }
    }

    // Drop point entries that a range tombstone from the merged input
    // set shadows — i.e. any `(user_key, seq)` where some RT covering
    // `user_key` has `rt_seq > seq`. This shrinks the live set before
    // the snapshot-pin GC so we don't rewrite bytes that no reader can
    // ever see. We still keep range tombstones themselves in the output
    // SSTable (see below), since a future compaction with lower levels
    // may still need them to shadow older point entries.
    all_entries.retain(|(ik, _v)| {
        let (uk, seq, _vt) = decode_internal_key(ik);
        let rt_seq = max_covering_seq(&merged_range_tombstones, uk, u64::MAX);
        rt_seq <= seq
    });

    // Snapshot-pinning GC: drop versions that no live snapshot and no
    // current reader can see. See `gc_old_versions` for the rule.
    let all_entries = gc_old_versions(all_entries, pin_seq);

    // Point-entry compaction filter. Snapshot-gated like the RT
    // filter above.
    let all_entries = if pin_seq == u64::MAX {
        if let Some(filter) = opts.compaction_filter.as_ref() {
            apply_compaction_filter(all_entries, filter.as_ref(), target_level)
        } else {
            all_entries
        }
    } else {
        all_entries
    };

    // Merge-operator chain collapse. For each user-key group, if a
    // terminator (`Value` / `Deletion`) is present in the merged
    // input, call `full_merge` and replace the entire group with a
    // single `Value` entry at the terminator's seq. Otherwise, if
    // `partial_merge` is available, fold the operand chain pairwise
    // into a single operand. Snapshot-gated for the same reason as
    // the compaction filter: with a live snapshot we cannot safely
    // drop intermediate versions the snapshot might still read.
    let all_entries = if pin_seq == u64::MAX {
        if let Some(op) = opts.merge_operator.as_ref() {
            collapse_merge_chains(all_entries, op.as_ref())
        } else {
            all_entries
        }
    } else {
        all_entries
    };

    // Dedup merged range tombstones by (start, end, seq) — a single
    // logical RT may appear in multiple input files after previous
    // compactions carried it forward.
    merged_range_tombstones.sort_by(|a, b| {
        (a.start.as_slice(), a.end.as_slice(), a.seq).cmp(&(
            b.start.as_slice(),
            b.end.as_slice(),
            b.seq,
        ))
    });
    merged_range_tombstones.dedup_by(|a, b| a.start == b.start && a.end == b.end && a.seq == b.seq);

    let mut edits = Vec::new();

    for file in &input_files {
        edits.push(VersionEdit::RemoveFile {
            level,
            file_id: file.meta.file_id,
        });
    }
    for file in &overlap_files {
        edits.push(VersionEdit::RemoveFile {
            level: target_level,
            file_id: file.meta.file_id,
        });
    }

    if all_entries.is_empty() && merged_range_tombstones.is_empty() {
        versions.lock().apply(&edits)?;
        delete_old_files(sst_dir, &input_files, &overlap_files, cache);
        return Ok(());
    }

    // Split output across multiple SSTables at user-key boundaries so
    // that all versions of a given user key live in exactly one file
    // (required for the non-overlap invariant at L1+).
    //
    // Range tombstones are replicated into every output file produced by
    // this compaction. A future compaction of any of these files will
    // merge and dedup the RTs again, so there's no runaway duplication.
    // Replicating keeps each file self-sufficient for reads: a scan that
    // only picks up one of the files still sees the right RT coverage.
    //
    // When `max_subcompactions > 1`, the sorted entry list is split at
    // user-key boundaries into that many chunks and each chunk is
    // written on its own scoped OS thread; output file ids are still
    // allocated atomically via the version lock. An empty-entries
    // "RT-only" output is always single-threaded because it produces
    // exactly one file.
    let (new_file_edits, new_output_infos) = write_compaction_outputs(
        versions,
        sst_dir,
        opts,
        target_level,
        &all_entries,
        &merged_range_tombstones,
    )?;
    let output_file_infos: Vec<OutputFileInfo> = new_output_infos;
    edits.extend(new_file_edits);

    // Atomically apply the remove / add edits. `SetNextFileId` is not
    // needed here — each output file already advanced it when it was
    // allocated above.
    versions.lock().apply(&edits)?;

    // Unlink the old SSTable paths. Their file descriptors stay alive
    // through any `Arc<LiveSst>` still held by older versions or by
    // iterators, so the data remains readable until those Arcs drop.
    delete_old_files(sst_dir, &input_files, &overlap_files, cache);

    // Publish compaction statistics. Bytes-in is the sum of
    // every input file's on-disk size; bytes-out is the sum of
    // the freshly-emitted output files. Both the foreground
    // `compact_range` path and the background scheduler route
    // through this function, so the tickers cover both.
    if let Some(s) = opts.statistics.as_deref() {
        let bytes_in: u64 = input_files
            .iter()
            .chain(overlap_files.iter())
            .map(|f| f.meta.file_size)
            .sum();
        let bytes_out: u64 = output_file_infos.iter().map(|f| f.file_size).sum();
        s.add(crate::statistics::Ticker::CompactionCount, 1);
        s.add(crate::statistics::Ticker::CompactionBytesRead, bytes_in);
        s.add(crate::statistics::Ticker::CompactionBytesWritten, bytes_out);
        s.record(
            crate::statistics::Histogram::CompactionTime,
            compaction_start.elapsed().as_micros() as u64,
        );
    }

    // Dispatch compaction completion + per-file creation +
    // per-file deletion events to registered listeners. The
    // caller may fire many file-level callbacks plus one
    // compaction-level aggregate.
    if !opts.listeners.is_empty() {
        for out in &output_file_infos {
            let info = crate::event_listener::TableFileCreationInfo {
                file_id: out.file_id,
                file_path: out.path.clone(),
                level: target_level,
                reason: crate::event_listener::TableFileCreationReason::Compaction,
                file_size: out.file_size,
                num_entries: out.num_entries,
            };
            crate::event_listener::dispatch(&opts.listeners, |l| l.on_table_file_created(&info));
        }
        for f in input_files.iter().chain(overlap_files.iter()) {
            let info = crate::event_listener::TableFileDeletionInfo {
                file_id: f.meta.file_id,
                file_path: sst_dir.join(sst_filename(f.meta.file_id)),
            };
            crate::event_listener::dispatch(&opts.listeners, |l| l.on_table_file_deleted(&info));
        }
        let job_info = crate::event_listener::CompactionJobInfo {
            input_level: level,
            output_level: target_level,
            input_files_input_level: begin_inputs_l,
            input_files_output_level: begin_inputs_l1,
            output_files: output_file_infos.iter().map(|f| f.file_id).collect(),
            duration: compaction_start.elapsed(),
        };
        crate::event_listener::dispatch(&opts.listeners, |l| l.on_compaction_completed(&job_info));
    }

    tracing::info!(
        level,
        target_level,
        input_files = input_files.len() + overlap_files.len(),
        "Compaction completed"
    );

    Ok(())
}

/// Drop every version that no live snapshot and no current reader
/// can observe.
///
/// Input `entries` is in internal-key order, so within a user-key
/// group entries appear **newest-seq first** (because the internal
/// key encodes `!seq`). The rule is:
///
/// 1. Keep every entry with `seq > pin_seq`. These are visible to
///    newer snapshots or to current reads.
/// 2. For the stretch of entries with `seq <= pin_seq`, keep only
///    the *first* one we see — that's the largest seq not exceeding
///    `pin_seq`, i.e. the version the oldest live snapshot actually
///    reads. Drop everything strictly older than that.
///
/// When `pin_seq == u64::MAX` (no live snapshot), rule (1) vacuously
/// keeps nothing and rule (2) keeps only the newest version of each
/// user key — the aggressive GC case. When `pin_seq` is somewhere in
/// the middle, older versions still visible to some snapshot are
/// conservatively preserved.
///
/// Tombstones participate in the same rule: the newest tombstone in
/// a user-key group survives as long as `seq > pin_seq`, or as the
/// single pin entry if all versions fall at or below `pin_seq`.
/// Dropping tombstones at the bottommost level when no deeper data
/// references them is a future optimization (tracked as a follow-up).
fn gc_old_versions(entries: Vec<(Vec<u8>, Vec<u8>)>, pin_seq: u64) -> Vec<(Vec<u8>, Vec<u8>)> {
    use super::internal_key::VALUE_TYPE_MERGE;

    let mut out = Vec::with_capacity(entries.len());
    let mut current_user_key: Option<Vec<u8>> = None;
    // Once set to `true`, every subsequent entry for the current
    // user key that sits at or below `pin_seq` is shadowed by an
    // already-emitted terminator and can be dropped. Merge
    // operands do *not* set this flag — they form an open chain
    // that must be preserved until a non-merge terminator arrives.
    let mut chain_terminated = false;

    for (ik, value) in entries {
        let (uk, seq, vt) = decode_internal_key(&ik);

        if current_user_key.as_deref() != Some(uk) {
            current_user_key = Some(uk.to_vec());
            chain_terminated = false;
        }

        if seq > pin_seq {
            out.push((ik, value));
            continue;
        }

        // `seq <= pin_seq` and we've already reached a terminator
        // for this user key — the entry is strictly older and
        // shadowed. Drop it.
        if chain_terminated {
            continue;
        }

        out.push((ik, value));
        if vt != VALUE_TYPE_MERGE {
            chain_terminated = true;
        }
    }

    out
}

/// Run the user [`crate::options::CompactionFilter`] over every
/// `Value` entry in `entries` and apply its decision:
///
/// - [`CompactionDecision::Keep`] — pass the entry through unchanged.
/// - [`CompactionDecision::Change`] — pass through with the filter's
///   new value (key and seq preserved).
/// - [`CompactionDecision::Remove`] — replace the entry with a
///   deletion tombstone at the same seq. The tombstone prevents an
///   older version of the same user key (living deeper in the LSM)
///   from resurfacing after the filtered value disappears.
///
/// Deletion internal keys are passed through without consulting the
/// filter — the filter's contract is about the user's own values,
/// not about tombstones lark writes itself.
fn apply_compaction_filter(
    entries: Vec<(Vec<u8>, Vec<u8>)>,
    filter: &dyn crate::options::CompactionFilter,
    level: usize,
) -> Vec<(Vec<u8>, Vec<u8>)> {
    use super::internal_key::{encode_internal_key, VALUE_TYPE_DELETION, VALUE_TYPE_VALUE};

    let mut out = Vec::with_capacity(entries.len());
    for (ik, value) in entries {
        let (uk, seq, vt) = decode_internal_key(&ik);
        if vt != VALUE_TYPE_VALUE {
            out.push((ik, value));
            continue;
        }
        match filter.filter(level, uk, &value) {
            crate::options::CompactionDecision::Keep => out.push((ik, value)),
            crate::options::CompactionDecision::Change(new_value) => out.push((ik, new_value)),
            crate::options::CompactionDecision::Remove => {
                // Replace with a same-seq deletion so lower levels
                // can't resurrect the filtered value.
                let tombstone_key = encode_internal_key(uk, seq, VALUE_TYPE_DELETION);
                out.push((tombstone_key, Vec::new()));
            }
        }
    }
    out
}

/// Walk the entries in user-key groups and collapse merge chains
/// where possible, using the configured [`MergeOperator`]. Entries
/// arrive in internal-key order, so within each group the newest
/// seq appears first.
///
/// Two transformations happen:
///
/// 1. **Full collapse:** if a group contains a `Value` or
///    `Deletion` terminator AND one or more `Merge` operands
///    layered on top of it, call `full_merge(base, operands)` and
///    replace the whole group with a single `Value` entry at the
///    newest merge's seq (or with the original terminator if
///    `full_merge` fails — we conservatively keep the raw chain).
///
/// 2. **Partial fold:** if a group is pure merges (no terminator in
///    the compaction's input set) and the operator's
///    `partial_merge` is available, fold the operand chain pairwise
///    into a single operand. The result replaces the chain at the
///    newest merge's seq.
///
/// Anything the operator rejects (`None` return) is left intact so
/// a compaction-time merge failure never loses data — the raw
/// operands survive to be retried on the next compaction or
/// materialized by a reader.
fn collapse_merge_chains(
    entries: Vec<(Vec<u8>, Vec<u8>)>,
    op: &dyn crate::options::MergeOperator,
) -> Vec<(Vec<u8>, Vec<u8>)> {
    use super::internal_key::{
        encode_internal_key, VALUE_TYPE_DELETION, VALUE_TYPE_MERGE, VALUE_TYPE_VALUE,
    };

    let mut out: Vec<(Vec<u8>, Vec<u8>)> = Vec::with_capacity(entries.len());
    let mut i = 0;
    while i < entries.len() {
        // Find the group of entries sharing this user key.
        let (uk_head, _, _) = decode_internal_key(&entries[i].0);
        let uk = uk_head.to_vec();
        let start = i;
        let mut end = i + 1;
        while end < entries.len() {
            let (uk_next, _, _) = decode_internal_key(&entries[end].0);
            if uk_next != uk.as_slice() {
                break;
            }
            end += 1;
        }

        // Classify the group.
        //
        // Walk newest → oldest: collect merge operands until we hit
        // a terminator (Value / Deletion) or run off the end.
        let group = &entries[start..end];
        let mut operands_newest_first: Vec<(u64, Vec<u8>)> = Vec::new();
        let mut terminator: Option<(u64, u8, Vec<u8>)> = None;
        let mut chain_end_offset = 0usize;
        for (offset, entry) in group.iter().enumerate() {
            let (_, seq, vt) = decode_internal_key(&entry.0);
            match vt {
                VALUE_TYPE_MERGE => {
                    operands_newest_first.push((seq, entry.1.clone()));
                    chain_end_offset = offset + 1;
                }
                VALUE_TYPE_VALUE | VALUE_TYPE_DELETION => {
                    terminator = Some((seq, vt, entry.1.clone()));
                    chain_end_offset = offset + 1;
                    break;
                }
                _ => {
                    chain_end_offset = group.len();
                    break;
                }
            }
        }

        let everything_scanned = chain_end_offset == group.len();
        let has_operands = !operands_newest_first.is_empty();

        match (has_operands, &terminator) {
            (false, _) => {
                // No merges in this group — nothing to collapse.
                out.extend(group.iter().cloned());
            }
            (true, Some((_term_seq, term_vt, term_value))) => {
                // Full collapse. Build operands in oldest-first
                // order, pick base from terminator type, call
                // full_merge. Newest merge's seq becomes the single
                // output's seq so it shadows everything that was
                // already shadowed by the original terminator.
                let newest_seq = operands_newest_first[0].0;
                let base: Option<&[u8]> = if *term_vt == VALUE_TYPE_VALUE {
                    Some(term_value.as_slice())
                } else {
                    None
                };
                let operand_refs: Vec<&[u8]> = operands_newest_first
                    .iter()
                    .rev()
                    .map(|(_, v)| v.as_slice())
                    .collect();
                match op.full_merge(&uk, base, &operand_refs) {
                    Some(collapsed) => {
                        let new_key = encode_internal_key(&uk, newest_seq, VALUE_TYPE_VALUE);
                        out.push((new_key, collapsed));
                        // Older entries past the terminator are
                        // already shadowed by it; emit them as-is
                        // in case a later pass wants them.
                        out.extend(group[chain_end_offset..].iter().cloned());
                    }
                    None => {
                        // Conservative: keep the raw chain.
                        out.extend(group.iter().cloned());
                    }
                }
            }
            (true, None) => {
                // No terminator in this group. Try pairwise partial
                // merge to shrink the chain.
                if everything_scanned && operands_newest_first.len() > 1 {
                    // Fold oldest→newest, carrying an accumulator.
                    let mut iter = operands_newest_first.iter().rev();
                    let first = iter.next().unwrap();
                    let mut acc: Vec<u8> = first.1.clone();
                    let mut newest_seq = first.0;
                    let mut succeeded_any = false;
                    for (seq, val) in iter {
                        match op.partial_merge(&uk, &acc, val) {
                            Some(folded) => {
                                acc = folded;
                                newest_seq = *seq;
                                succeeded_any = true;
                            }
                            None => {
                                succeeded_any = false;
                                break;
                            }
                        }
                    }
                    if succeeded_any {
                        let new_key = encode_internal_key(&uk, newest_seq, VALUE_TYPE_MERGE);
                        out.push((new_key, acc));
                    } else {
                        out.extend(group.iter().cloned());
                    }
                } else {
                    out.extend(group.iter().cloned());
                }
            }
        }

        i = end;
    }

    out
}

/// Whether a file's user-key range intersects `[start, end)`. `None`
/// bounds are treated as unbounded.
fn file_overlaps_range(file: &Arc<LiveSst>, start: Option<&[u8]>, end: Option<&[u8]>) -> bool {
    if let Some(s) = start {
        if file.meta.largest_key.as_slice() < s {
            return false;
        }
    }
    if let Some(e) = end {
        if file.meta.smallest_key.as_slice() >= e {
            return false;
        }
    }
    true
}

/// Info a compaction run accumulates for each output file it
/// produces, so the listener-dispatch block at the end has
/// enough data to fire `on_table_file_created` /
/// `on_compaction_completed` without going back to the version.
struct OutputFileInfo {
    file_id: u64,
    path: PathBuf,
    file_size: u64,
    num_entries: u64,
}

/// Smallest subcompaction a worker will ever take on. Below this
/// many user-key groups per worker, the thread-spawn overhead
/// eats any parallel speedup and we fall back to single-threaded
/// writing.
const MIN_GROUPS_PER_SUBCOMPACTION: usize = 64;

/// Dispatch the output-writing phase of a compaction across
/// one or more worker threads depending on `opts.max_subcompactions`.
/// Returns the `AddFile` edits and `OutputFileInfo`s for every
/// file this compaction produced.
///
/// Subcompactions always split at user-key boundaries so the
/// non-overlap invariant at L1+ is preserved: every version of a
/// given user key lives in exactly one output file, just as the
/// single-threaded writer guaranteed.
fn write_compaction_outputs(
    versions: &Arc<parking_lot::Mutex<VersionSet>>,
    sst_dir: &Path,
    opts: &CompactionOptions,
    target_level: usize,
    all_entries: &[(Vec<u8>, Vec<u8>)],
    merged_range_tombstones: &[RangeTombstone],
) -> std::io::Result<(Vec<VersionEdit>, Vec<OutputFileInfo>)> {
    // Empty point entries + non-empty RTs → one RT-only output.
    // Always single-threaded.
    if all_entries.is_empty() && !merged_range_tombstones.is_empty() {
        return write_chunk_outputs(
            versions,
            sst_dir,
            opts,
            target_level,
            &[],
            merged_range_tombstones,
            /* rt_only */ true,
        );
    }

    // Decide how many workers we actually want. Capped by the
    // number of user-key groups in the input so chunks never end
    // up empty.
    let group_count = count_user_key_groups(all_entries);
    let requested = opts.max_subcompactions.max(1);
    let worker_count = requested.min(group_count.max(1));
    let worker_count =
        if worker_count > 1 && group_count / worker_count >= MIN_GROUPS_PER_SUBCOMPACTION {
            worker_count
        } else {
            1
        };

    if worker_count <= 1 {
        return write_chunk_outputs(
            versions,
            sst_dir,
            opts,
            target_level,
            all_entries,
            merged_range_tombstones,
            /* rt_only */ false,
        );
    }

    // Split indices at user-key boundaries roughly equidistant
    // by group count (not by bytes — good enough in practice and
    // avoids a second full pass over the entry list).
    let split_indices = compute_subcompaction_splits(all_entries, group_count, worker_count);

    // One sub-slice per worker. Each thread runs `write_chunk_outputs`
    // independently against its slice; they do not share any
    // mutable state beyond the `versions` lock (used only inside
    // atomic file-id allocation) and the optional rate limiter.
    let chunks: Vec<&[(Vec<u8>, Vec<u8>)]> = split_indices
        .windows(2)
        .map(|w| &all_entries[w[0]..w[1]])
        .collect();

    let mut aggregated_edits: Vec<VersionEdit> = Vec::new();
    let mut aggregated_infos: Vec<OutputFileInfo> = Vec::new();

    std::thread::scope(|scope| -> std::io::Result<()> {
        let handles: Vec<_> = chunks
            .into_iter()
            .map(|chunk| {
                scope.spawn(move || {
                    write_chunk_outputs(
                        versions,
                        sst_dir,
                        opts,
                        target_level,
                        chunk,
                        merged_range_tombstones,
                        /* rt_only */ false,
                    )
                })
            })
            .collect();
        for h in handles {
            let (edits, infos) = h
                .join()
                .map_err(|_| std::io::Error::other("subcompaction worker panicked"))??;
            aggregated_edits.extend(edits);
            aggregated_infos.extend(infos);
        }
        Ok(())
    })?;

    Ok((aggregated_edits, aggregated_infos))
}

/// Count the number of distinct user keys in a sorted internal-
/// key entry list. Used by the subcompaction planner to decide
/// how many workers to spin up and where to split.
fn count_user_key_groups(entries: &[(Vec<u8>, Vec<u8>)]) -> usize {
    let mut count = 0usize;
    let mut prev: Option<&[u8]> = None;
    for (ik, _) in entries {
        let uk = user_key_of(ik);
        if prev != Some(uk) {
            count += 1;
            prev = Some(uk);
        }
    }
    count
}

/// Compute split indices into `entries` so every subcompaction
/// chunk boundary falls on a user-key boundary. Returns a vector
/// with `workers + 1` entries where element 0 is always 0 and
/// element `workers` is always `entries.len()`.
fn compute_subcompaction_splits(
    entries: &[(Vec<u8>, Vec<u8>)],
    group_count: usize,
    workers: usize,
) -> Vec<usize> {
    debug_assert!(workers > 1);
    debug_assert!(group_count >= workers);
    let groups_per_worker = group_count / workers;

    let mut result: Vec<usize> = Vec::with_capacity(workers + 1);
    result.push(0);
    let mut groups_seen = 0usize;
    let mut prev_uk: Option<&[u8]> = None;
    let mut target_boundary = groups_per_worker;
    for (idx, (ik, _)) in entries.iter().enumerate() {
        let uk = user_key_of(ik);
        if prev_uk != Some(uk) {
            groups_seen += 1;
            prev_uk = Some(uk);
            if groups_seen > target_boundary && result.len() < workers {
                result.push(idx);
                target_boundary += groups_per_worker;
            }
        }
    }
    // Pad out any missing split points (happens when the last
    // couple of groups fall past the final target) and cap at
    // `entries.len()` exactly.
    while result.len() < workers {
        result.push(entries.len());
    }
    result.push(entries.len());
    result
}

/// Inner body that writes one chunk's worth of entries as one
/// or more output SSTables. Factored out of the old single-
/// threaded writer loop so both the single- and multi-threaded
/// paths share exactly the same file-building logic.
#[allow(clippy::too_many_arguments)]
fn write_chunk_outputs(
    versions: &Arc<parking_lot::Mutex<VersionSet>>,
    sst_dir: &Path,
    opts: &CompactionOptions,
    target_level: usize,
    entries: &[(Vec<u8>, Vec<u8>)],
    merged_range_tombstones: &[RangeTombstone],
    rt_only: bool,
) -> std::io::Result<(Vec<VersionEdit>, Vec<OutputFileInfo>)> {
    let mut edits: Vec<VersionEdit> = Vec::new();
    let mut infos: Vec<OutputFileInfo> = Vec::new();
    let mut chunk_start = 0usize;
    let mut wrote_rt_only = false;

    loop {
        if !rt_only && chunk_start >= entries.len() {
            break;
        }
        if rt_only && wrote_rt_only {
            break;
        }

        // Allocate an output file id under the version lock. This
        // is a short critical section (one atomic counter bump),
        // so many subcompaction workers contending for it is
        // fine in practice — writes are the expensive part and
        // they happen without holding the lock.
        let file_id = {
            let mut guard = versions.lock();
            let current = guard.current();
            let id = current.next_file_id;
            guard.apply(&[VersionEdit::SetNextFileId(id + 1)])?;
            id
        };

        let path = sst_dir.join(sst_filename(file_id));
        let mut writer = SsTableWriter::new(
            &path,
            opts.block_size,
            opts.bloom_bits_per_key,
            opts.compression_for_level(target_level),
            opts.prefix_extractor.clone(),
            opts.partitioned_index,
            opts.metadata_block_size,
        )?;

        let mut estimated_size: u64 = 0;
        let mut current_user_key: Option<Vec<u8>> = None;

        while chunk_start < entries.len() {
            let (ik, value) = &entries[chunk_start];
            let uk = user_key_of(ik);
            let at_boundary = current_user_key.as_deref() != Some(uk);

            if estimated_size >= opts.target_file_size && at_boundary {
                break;
            }

            writer.add(ik, value)?;
            estimated_size += (ik.len() + value.len()) as u64;
            if at_boundary {
                current_user_key = Some(uk.to_vec());
            }
            chunk_start += 1;
        }

        for rt in merged_range_tombstones {
            writer.add_range_tombstone(&rt.start, &rt.end, rt.seq);
        }
        if rt_only {
            wrote_rt_only = true;
        }

        let summary = match writer.finish()? {
            Some(s) => s,
            None => {
                let _ = std::fs::remove_file(&path);
                continue;
            }
        };

        let file_size = std::fs::metadata(&path)?.len();

        if let Some(limiter) = &opts.rate_limiter {
            limiter.request(file_size, crate::rate_limiter::Priority::Low);
        }

        if opts.use_direct_io_for_compaction {
            crate::os_hint::drop_page_cache_by_path(&path);
        }

        let reader = Arc::new(SsTableReader::open(&path, file_id)?);
        let new_file = LiveSst::new(
            SsTableMeta {
                file_id,
                smallest_key: summary.smallest_user_key,
                largest_key: summary.largest_user_key,
                file_size,
                num_entries: summary.num_entries,
            },
            reader,
        );

        infos.push(OutputFileInfo {
            file_id,
            path: path.clone(),
            file_size,
            num_entries: summary.num_entries,
        });

        edits.push(VersionEdit::AddFile {
            level: target_level,
            file: new_file,
        });
    }

    Ok((edits, infos))
}

fn key_range(files: &[Arc<LiveSst>]) -> (Vec<u8>, Vec<u8>) {
    let mut min_key = files[0].meta.smallest_key.clone();
    let mut max_key = files[0].meta.largest_key.clone();

    for file in &files[1..] {
        if file.meta.smallest_key < min_key {
            min_key = file.meta.smallest_key.clone();
        }
        if file.meta.largest_key > max_key {
            max_key = file.meta.largest_key.clone();
        }
    }

    (min_key, max_key)
}

fn find_overlapping(files: &[Arc<LiveSst>], min_key: &[u8], max_key: &[u8]) -> Vec<Arc<LiveSst>> {
    files
        .iter()
        .filter(|f| {
            f.meta.smallest_key.as_slice() <= max_key && f.meta.largest_key.as_slice() >= min_key
        })
        .map(Arc::clone)
        .collect()
}

fn delete_old_files(
    sst_dir: &Path,
    input_files: &[Arc<LiveSst>],
    overlap_files: &[Arc<LiveSst>],
    cache: &BlockCache,
) {
    for file in input_files.iter().chain(overlap_files.iter()) {
        let path = sst_dir.join(sst_filename(file.meta.file_id));
        cache.evict_file(file.meta.file_id);
        if let Err(e) = remove_sst(&path) {
            tracing::warn!(
                file_id = file.meta.file_id,
                error = %e,
                "Failed to delete old SSTable"
            );
        }
    }
}
