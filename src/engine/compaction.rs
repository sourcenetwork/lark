use std::cmp::Ordering as CmpOrdering;
use std::collections::{BinaryHeap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
// The periodic worker poll is the only user of these, and the worker
// itself does not exist on a target without threads.
#[cfg(not(target_arch = "wasm32"))]
use std::time::{Duration, Instant};

#[cfg(not(target_arch = "wasm32"))]
use kovan_channel::RecvDeadline;
#[cfg(not(target_arch = "wasm32"))]
use kovan_channel::unbounded::Receiver;
use kovan_channel::unbounded::Sender;

// Through the portability shim so a target without 64-bit atomics still
// builds; see `src/portability.rs`.
use crate::portability::{AtomicBool, Ordering};

use super::block_cache::BlockCache;
use super::internal_key::{compare_internal_keys, decode_internal_key, user_key_of};
use super::manifest::{MAX_LEVELS, VersionEdit};
use super::range_tombstone::{
    RangeTombstone, RangeTombstoneSet, exclusive_successor, sort_dedup_tombstones,
};
use super::read_view::VersionStore;
use super::snapshot_registry::SnapshotRegistry;
use super::sstable::{
    LiveSst, MetadataPolicy, SsTableInternalIter, SsTableMeta, SsTableReader, SsTableWriter,
    remove_sst_in, sst_filename,
};
use crate::env::{Env, JoinHandle};

/// Default compaction trigger: flush L0 → L1 when L0 has this many SSTables.
pub(crate) const L0_COMPACTION_TRIGGER: usize = 4;

/// Default level size multiplier between levels.
pub(crate) const LEVEL_SIZE_MULTIPLIER: u64 = 10;

/// Default max bytes for level 1 (256 MB).
pub(crate) const DEFAULT_LEVEL_BASE_BYTES: u64 = 256 * 1024 * 1024;

/// Default target SSTable file size (64 MB).
pub(crate) const DEFAULT_TARGET_FILE_SIZE: u64 = 64 * 1024 * 1024;

/// How long a worker sleeps between periodic checks when no wake-up
/// arrives. The backstop that keeps compaction live if a coalesced
/// notification is ever dropped.
#[cfg(not(target_arch = "wasm32"))]
const WORKER_POLL_INTERVAL: Duration = Duration::from_secs(1);

/// Manages background compaction on one or more dedicated OS threads.
pub(crate) struct CompactionScheduler {
    shutdown: Arc<AtomicBool>,
    /// Wake-up channel. One token wakes one idle worker.
    trigger: Sender<()>,
    /// Coalescing gate: at most one unconsumed token is ever in flight,
    /// so a flush storm cannot queue thousands of wake-ups for work a
    /// single pass already covers.
    pending: Arc<AtomicBool>,
    /// Joined through the env, so a target that cannot spawn threads
    /// runs compaction in the foreground instead of failing to build.
    handles: Vec<Box<dyn JoinHandle>>,
}

impl CompactionScheduler {
    /// Construct a scheduler with no background workers.
    pub(crate) fn disabled() -> Self {
        let (trigger, _rx) = kovan_channel::unbounded();
        Self {
            shutdown: Arc::new(AtomicBool::new(true)),
            trigger,
            pending: Arc::new(AtomicBool::new(false)),
            handles: Vec::new(),
        }
    }

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
    ///
    /// `in_progress` is the engine-wide set of file ids currently being
    /// compacted. The engine owns it and shares it with the foreground
    /// inline compaction path, so a worker and a foreground pass can
    /// never pick overlapping inputs.
    ///
    /// `max_background_compactions == 0` starts no worker at all and
    /// returns a scheduler equivalent to [`CompactionScheduler::disabled`];
    /// compaction then runs on whichever thread asks for it. That is the
    /// only mode a single-threaded host such as `wasm32-wasip1` can open
    /// in, because `std::thread::spawn` reports
    /// [`std::io::ErrorKind::Unsupported`] there.
    ///
    /// Returns the spawn error if a worker thread cannot be created,
    /// after shutting down and joining any worker already started, so
    /// a failed open leaves no detached thread behind.
    // Every parameter is a distinct shared handle the workers need;
    // `compaction_loop` below carries the same fan-out for the same
    // reason.
    #[allow(clippy::too_many_arguments)]
    // On wasm32 the spawn half below is compiled out, which leaves the
    // shared handles the workers would have taken unread.
    #[cfg_attr(target_arch = "wasm32", allow(unused_variables))]
    pub(crate) fn start(
        compaction_lock: Arc<crate::sync::Gate>,
        snapshot_registry: Arc<SnapshotRegistry>,
        versions: Arc<VersionStore>,
        sst_dir: Arc<Path>,
        cache: Arc<BlockCache>,
        opts: CompactionOptions,
        stall_signal: Arc<crate::engine::StallSignal>,
        in_progress: Arc<crate::sync::Mutex<HashSet<u64>>>,
    ) -> std::io::Result<Self> {
        let shutdown = Arc::new(AtomicBool::new(false));
        let (trigger, receiver) = kovan_channel::unbounded::<()>();
        let pending = Arc::new(AtomicBool::new(false));

        // Zero means no background worker: compaction runs on the
        // calling thread, which is what a target without threads needs.
        let worker_count = opts.max_background_compactions;
        if worker_count == 0 {
            return Ok(Self::disabled());
        }

        // wasm32 has no threads at all, so `Env::spawn` reports
        // `Unsupported` for every worker. Report it up front and leave
        // the whole worker loop out of the build rather than carrying
        // machinery the target can never reach.
        #[cfg(target_arch = "wasm32")]
        return Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "failed to spawn compaction thread 0: this target has no threads; set \
             Options::max_background_compactions = 0 to run compaction on the calling thread",
        ));

        #[cfg(not(target_arch = "wasm32"))]
        {
            let mut scheduler = Self {
                shutdown: Arc::clone(&shutdown),
                trigger,
                pending: Arc::clone(&pending),
                handles: Vec::with_capacity(worker_count),
            };

            for i in 0..worker_count {
                let shutdown_clone = Arc::clone(&shutdown);
                let receiver_clone = receiver.clone();
                let pending_clone = Arc::clone(&pending);
                let lock_clone = Arc::clone(&compaction_lock);
                let registry_clone = Arc::clone(&snapshot_registry);
                let versions_clone = Arc::clone(&versions);
                let sst_dir_clone = Arc::clone(&sst_dir);
                let cache_clone = Arc::clone(&cache);
                let opts_clone = opts.clone();
                let stall_clone = Arc::clone(&stall_signal);
                let in_progress_clone = Arc::clone(&in_progress);

                let spawned = spawn_worker(&*opts.env, i, move || {
                    compaction_loop(
                        shutdown_clone,
                        receiver_clone,
                        pending_clone,
                        lock_clone,
                        registry_clone,
                        versions_clone,
                        sst_dir_clone,
                        cache_clone,
                        opts_clone,
                        stall_clone,
                        in_progress_clone,
                    );
                });

                match spawned {
                    Ok(handle) => scheduler.handles.push(handle),
                    Err(e) => {
                        // Dropping `scheduler` on the way out signals
                        // shutdown and joins the workers already started,
                        // which is the same path `shutdown` takes.
                        return Err(std::io::Error::new(
                            e.kind(),
                            format!(
                                "failed to spawn compaction thread {i}: {e}; set \
                             Options::max_background_compactions = 0 to run compaction \
                             on the calling thread"
                            ),
                        ));
                    }
                }
            }

            Ok(scheduler)
        }
    }

    /// Notify the compaction workers that work may be available.
    ///
    /// Coalesced: a notification arriving while one is still unconsumed
    /// is dropped, because the worker that picks the outstanding token up
    /// re-checks the whole version before it decides there is nothing to
    /// do. Send is non-blocking, so this never stalls a flush.
    pub(crate) fn notify(&self) {
        if !self.pending.swap(true, Ordering::AcqRel) {
            self.trigger.send(());
        }
    }

    /// Shut down all compaction threads.
    ///
    /// Wakes every worker, not just one: `join` below waits for all of
    /// them, so a `notify_one` here would leave the rest sleeping until
    /// their condvar timeout and make close latency scale with
    /// `max_background_compactions`.
    pub(crate) fn shutdown(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        // One token per worker, bypassing the coalescing gate, so every
        // thread wakes now instead of waiting out its periodic poll.
        for _ in 0..self.handles.len() {
            self.trigger.send(());
        }
        for handle in self.handles.drain(..) {
            handle.join();
        }
    }
}

impl Drop for CompactionScheduler {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Spawn one named compaction worker.
///
/// Split out of [`CompactionScheduler::start`] so tests can force the
/// spawn failure that a single-threaded target produces natively.
#[cfg(not(target_arch = "wasm32"))]
fn spawn_worker<F>(env: &dyn Env, index: usize, body: F) -> std::io::Result<Box<dyn JoinHandle>>
where
    F: FnOnce() + Send + 'static,
{
    #[cfg(test)]
    {
        if SPAWN_FAILURE_AFTER.with(|limit| limit.get().is_some_and(|after| index >= after)) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "test seam: compaction worker spawn disabled",
            ));
        }
    }
    env.spawn(&format!("lark-compaction-{index}"), Box::new(body))
}

#[cfg(test)]
thread_local! {
    /// Test seam: index of the first compaction worker whose spawn
    /// reports `Unsupported`, mirroring what a single-threaded target
    /// does natively. `None` disables the seam. Thread-local because
    /// `start` runs entirely on its caller's thread, so a parallel
    /// test never observes another test's setting.
    static SPAWN_FAILURE_AFTER: std::cell::Cell<Option<usize>> =
        const { std::cell::Cell::new(None) };
}

/// Test-only guard that makes compaction-worker spawning fail on the
/// current thread for the guard's lifetime.
#[cfg(test)]
pub(crate) struct SpawnFailureGuard;

#[cfg(test)]
impl SpawnFailureGuard {
    /// Let the first `after` workers spawn and fail every one past
    /// them until the guard is dropped.
    pub(crate) fn allowing(after: usize) -> Self {
        SPAWN_FAILURE_AFTER.with(|limit| limit.set(Some(after)));
        Self
    }
}

#[cfg(test)]
impl Drop for SpawnFailureGuard {
    fn drop(&mut self) {
        SPAWN_FAILURE_AFTER.with(|limit| limit.set(None));
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
    pub(crate) evict_compaction_data_from_page_cache: bool,
    pub(crate) max_background_compactions: usize,
    pub(crate) partitioned_index: bool,
    pub(crate) metadata_block_size: usize,
    pub(crate) cache_index_and_filter_blocks: bool,
    /// The host platform. Compaction reads, writes, and unlinks
    /// SSTables through it, and starts its workers on it.
    pub(crate) env: Arc<dyn Env>,
}

impl CompactionOptions {
    /// How readers opened by compaction should hold their index and
    /// filter blocks. Mirrors `EngineOptions::metadata_policy`.
    pub(crate) fn metadata_policy(&self) -> MetadataPolicy {
        if self.cache_index_and_filter_blocks {
            MetadataPolicy::Cached
        } else {
            MetadataPolicy::Pinned
        }
    }

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
            evict_compaction_data_from_page_cache: false,
            max_background_compactions: 1,
            partitioned_index: false,
            metadata_block_size: 4096,
            cache_index_and_filter_blocks: false,
            env: crate::env::std_env(),
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[allow(clippy::too_many_arguments)]
fn compaction_loop(
    shutdown: Arc<AtomicBool>,
    trigger: Receiver<()>,
    pending: Arc<AtomicBool>,
    compaction_lock: Arc<crate::sync::Gate>,
    snapshot_registry: Arc<SnapshotRegistry>,
    versions: Arc<VersionStore>,
    sst_dir: Arc<Path>,
    cache: Arc<BlockCache>,
    opts: CompactionOptions,
    stall_signal: Arc<crate::engine::StallSignal>,
    in_progress: Arc<crate::sync::Mutex<HashSet<u64>>>,
) {
    loop {
        // Hand back this thread's share of the block cache's deferred
        // reclamation before parking. A thread that has taken a
        // reclamation guard and then idles pins every batch holding a
        // node born before its last published epoch, so a worker that
        // read one block and then slept would hold the cache's evicted
        // entries for as long as it stayed asleep - which, between
        // compactions, is most of the time. Measured on the block-cache
        // overhead probe, skipping this left live heap at 1.79x the byte
        // budget against 1.04x with it.
        kovan::flush();

        // Wait for a trigger, or fall through on the periodic poll.
        // The gate is reopened before the pass runs, so a notification
        // that lands mid-pass queues a token instead of being swallowed.
        match trigger.recv_deadline(Instant::now() + WORKER_POLL_INTERVAL) {
            RecvDeadline::Msg(()) => pending.store(false, Ordering::Release),
            RecvDeadline::Timeout => {}
            RecvDeadline::Disconnected => break,
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
                    Ok(outcome) => outcome == CompactionOutcome::DidWork,
                    Err(e) => {
                        tracing::error!(error = %e, "Compaction failed");
                        // Surface the failure to any registered
                        // listeners so metrics pipelines and
                        // debuggers notice it - the scheduler
                        // itself keeps running.
                        if !opts.listeners.is_empty() {
                            let err =
                                crate::Error::from(std::io::Error::new(e.kind(), e.to_string()));
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

/// What one call to [`pick_and_run_compaction`] achieved.
///
/// The distinction between [`CompactionOutcome::Idle`] and
/// [`CompactionOutcome::Contended`] is load-bearing, not cosmetic. A
/// writer stalling with no background worker performs the compaction
/// itself; if it treated "another thread already holds these inputs"
/// as "there is nothing to compact", it would fail the write with
/// [`crate::Error::Busy`] the moment a second thread was mid-pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CompactionOutcome {
    /// A compaction job ran to completion.
    DidWork,
    /// No level was over its trigger. Nothing to do.
    Idle,
    /// Work is pending, but another thread holds the input files. The
    /// caller should wait for that thread rather than conclude the
    /// engine is idle.
    Contended,
}

/// Pick the highest-priority pending compaction job for the configured
/// style and run it to completion on the calling thread. Returns
/// whether any work was done.
///
/// Shared by the background worker loop and by the engine's foreground
/// pass ([`crate::engine::LarkEngine::run_one_compaction_pass`]) so the
/// two can never diverge on what "one job" means. Callers hold either
/// side of the engine-wide compaction lock and pass the engine-wide
/// `in_progress` set.
pub(crate) fn pick_and_run_compaction(
    versions: &Arc<VersionStore>,
    sst_dir: &Path,
    cache: &BlockCache,
    opts: &CompactionOptions,
    pin_seq: u64,
    in_progress: &crate::sync::Mutex<HashSet<u64>>,
) -> std::io::Result<CompactionOutcome> {
    match opts.compaction_style {
        crate::options::CompactionStyle::Level => {
            pick_and_run_level_compaction(versions, sst_dir, cache, opts, pin_seq, in_progress)
        }
        crate::options::CompactionStyle::Fifo => {
            // Skip if any file is already being compacted - with a
            // single L0 pool there's nothing safe to pick in parallel.
            if !in_progress.lock().is_empty() {
                return Ok(CompactionOutcome::Contended);
            }
            Ok(outcome(run_fifo_pass(versions, sst_dir, opts)?))
        }
        crate::options::CompactionStyle::Universal => {
            // Same: universal merges all L0 files, so parallel picks
            // would conflict. Skip until the in-progress set clears.
            if !in_progress.lock().is_empty() {
                return Ok(CompactionOutcome::Contended);
            }
            Ok(outcome(pick_and_run_universal(
                versions, sst_dir, cache, opts, pin_seq,
            )?))
        }
    }
}

/// Map a picker that has no notion of contention onto an outcome.
fn outcome(did_work: bool) -> CompactionOutcome {
    if did_work {
        CompactionOutcome::DidWork
    } else {
        CompactionOutcome::Idle
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
    versions: &Arc<VersionStore>,
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

    // Rule 1 - size ratio merge. Accumulate newest files into a
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
        // `size_ratio` percent of the new candidate - i.e. the
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

    // Rule 2 - size amplification. Compute the ratio of "all
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
///
/// The output is deliberately **one** file regardless of
/// [`CompactionOptions::target_file_size`]. Under this style each L0
/// file is one sorted run and the picker's size-ratio rule reasons
/// about runs, so splitting a merge across several files re-creates
/// the very group that was just merged: the picker re-selects it, the
/// writer re-splits it, and the pass never terminates while burning a
/// full merge each time. Emitting one run makes every merge reduce the
/// L0 file count by at least one, which is what bounds the loop in
/// [`crate::Db::compact_step`] and in the inline stall path.
fn perform_universal_merge(
    versions: &Arc<VersionStore>,
    sst_dir: &Path,
    cache: &BlockCache,
    opts: &CompactionOptions,
    inputs: Vec<Arc<LiveSst>>,
    pin_seq: u64,
) -> std::io::Result<()> {
    let mut run_opts = opts.clone();
    run_opts.target_file_size = u64::MAX;
    // Universal merges are L0 -> L0. The shared
    // `perform_compaction_to` helper handles the merge-and-write
    // path; we just tell it to target L0 and pass no overlap
    // files (there is no "next level" to drag in).
    perform_compaction_to(
        versions,
        sst_dir,
        cache,
        &run_opts,
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
    versions: &Arc<VersionStore>,
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
    versions: &Arc<VersionStore>,
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
    versions: &Arc<VersionStore>,
    sst_dir: &Path,
    cache: &BlockCache,
    opts: &CompactionOptions,
    pin_seq: u64,
    in_progress: &crate::sync::Mutex<HashSet<u64>>,
) -> std::io::Result<CompactionOutcome> {
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

    Ok(CompactionOutcome::Idle)
}

/// FIFO picker: every flush lands a new L0 file, and lark never
/// merges in this style. After each pass, if the total bytes held
/// across L0 exceed `max_table_files_size`, we unlink the oldest
/// file (smallest `file_id`) and emit a `RemoveFile` edit. We stop
/// when the cap is satisfied or only one file remains - a single
/// oversized file is not worth deleting because that would lose
/// data with no successor on disk.
fn pick_and_run_fifo(
    versions: &Arc<VersionStore>,
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

    // Sort by file_id ascending - smaller id was allocated earlier,
    // so it is the oldest file by construction.
    let mut by_age: Vec<Arc<crate::engine::sstable::LiveSst>> = l0_files;
    by_age.sort_by_key(|f| f.meta.file_id);

    let mut running_total = total;
    let mut edits: Vec<VersionEdit> = Vec::new();
    let mut removed_paths: Vec<std::path::PathBuf> = Vec::new();
    for file in &by_age {
        // Always keep at least one file alive - dropping the only
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
    // database - the manifest already reflects the removal.
    for path in &removed_paths {
        let _ = opts.env.remove_file(path);
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
    versions: &Arc<VersionStore>,
    sst_dir: &Path,
    cache: &BlockCache,
    opts: &CompactionOptions,
    pin_seq: u64,
    in_progress: &crate::sync::Mutex<HashSet<u64>>,
) -> std::io::Result<CompactionOutcome> {
    // If any L0 file is already being compacted by another worker,
    // skip - L0 files may overlap so concurrent picks would produce
    // conflicting output sets.
    {
        let ip = in_progress.lock();
        let version = versions.lock().current();
        if version.levels[0]
            .iter()
            .any(|f| ip.contains(&f.meta.file_id))
        {
            return Ok(CompactionOutcome::Contended);
        }
    }
    compact_level(versions, sst_dir, cache, opts, 0, pin_seq, in_progress)
}

/// Compact a level into the next level using the standard size-based
/// heuristic (all L0 files, or the first file of L1+). Used by the
/// background scheduler.
fn compact_level(
    versions: &Arc<VersionStore>,
    sst_dir: &Path,
    cache: &BlockCache,
    opts: &CompactionOptions,
    level: usize,
    pin_seq: u64,
    in_progress: &crate::sync::Mutex<HashSet<u64>>,
) -> std::io::Result<CompactionOutcome> {
    let target_level = level + 1;
    if target_level >= MAX_LEVELS {
        return Ok(CompactionOutcome::Idle);
    }

    let version = versions.lock().current();

    let input_files: Vec<Arc<LiveSst>> = version.levels[level].clone();
    if input_files.is_empty() {
        return Ok(CompactionOutcome::Idle);
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
            None => return Ok(CompactionOutcome::Contended),
        };
        let (min_key, max_key) = key_range(&picked);
        // Also skip if any overlap file is already in-progress.
        let overlapping = find_overlapping(&version.levels[target_level], &min_key, &max_key);
        {
            let ip = in_progress.lock();
            if overlapping.iter().any(|f| ip.contains(&f.meta.file_id)) {
                return Ok(CompactionOutcome::Contended);
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
    Ok(CompactionOutcome::DidWork)
}

/// Run a manual `compact_range` pass synchronously: for every level from
/// 0 down to MAX_LEVELS-2, pick the files overlapping `[start, end)` and
/// push them down to the next level. Returns once no level has any more
/// files in the range that can be pushed down.
///
/// Callers must hold the engine-wide compaction lock so the background
/// scheduler can't pick an overlapping input set concurrently.
pub(crate) fn run_compact_range(
    versions: &Arc<VersionStore>,
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
            // overlaps - simplest correct move: take every L0 file
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
    versions: &Arc<VersionStore>,
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
    versions: &Arc<VersionStore>,
    sst_dir: &Path,
    cache: &BlockCache,
    opts: &CompactionOptions,
    level: usize,
    target_level: usize,
    input_files: Vec<Arc<LiveSst>>,
    overlap_files: Vec<Arc<LiveSst>>,
    pin_seq: u64,
) -> std::io::Result<()> {
    let compaction_start = opts.env.now_micros();

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

    // Range tombstones are small metadata relative to the point-entry
    // stream, and readers need the complete covering set to decide
    // whether a point version is shadowed. Point entries themselves are
    // merged below by a k-way stream so compaction memory no longer
    // scales with total input data bytes.
    let mut merged_range_tombstones: Vec<RangeTombstone> = Vec::new();
    for file in input_files.iter().chain(overlap_files.iter()) {
        for rt in file.reader.range_tombstones() {
            merged_range_tombstones.push(rt.clone());
        }
    }

    // User compaction filter for range tombstones. Run this **before**
    // the point-entry RT shadow pass so a filter that drops a range
    // tombstone doesn't leave orphaned point entries already wiped
    // out by that same RT. Only runs when no snapshot is pinned.
    if pin_seq == u64::MAX
        && let Some(filter) = opts.compaction_filter.as_ref()
    {
        merged_range_tombstones.retain(|rt| {
            !matches!(
                filter.filter_range_delete(target_level, &rt.start, &rt.end),
                crate::options::CompactionDecision::Remove
            )
        });
    }

    // Dedup merged range tombstones by (start, end, seq) - a single
    // logical RT may appear in multiple input files after previous
    // compactions carried it forward.
    let merged_range_tombstones = RangeTombstoneSet::from_vec(merged_range_tombstones);

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

    let (new_file_edits, output_file_infos) = stream_compaction_outputs(
        versions,
        sst_dir,
        cache,
        opts,
        target_level,
        &input_files,
        &overlap_files,
        pin_seq,
        &merged_range_tombstones,
    )?;
    edits.extend(new_file_edits);

    // Tell the OS it can drop pages we just read from the page cache.
    // Opt-in via `evict_compaction_data_from_page_cache`; on non-Linux targets
    // the hint is a no-op.
    if opts.evict_compaction_data_from_page_cache {
        for file in input_files.iter().chain(overlap_files.iter()) {
            let path = sst_dir.join(sst_filename(file.meta.file_id));
            opts.env.drop_page_cache(&path);
        }
    }

    // Atomically apply the remove / add edits. `SetNextFileId` is not
    // needed here - each output file already advanced it when it was
    // allocated above.
    versions.lock().apply(&edits)?;

    // Unlink the old SSTable paths. Their file descriptors stay alive
    // through any `Arc<LiveSst>` still held by older versions or by
    // iterators, so the data remains readable until those Arcs drop.
    delete_old_files(&*opts.env, sst_dir, &input_files, &overlap_files, cache);

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
        if let Some(micros) = crate::env::elapsed_micros(&*opts.env, compaction_start) {
            s.record(crate::statistics::Histogram::CompactionTime, micros);
        }
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
            duration: std::time::Duration::from_micros(
                crate::env::elapsed_micros(&*opts.env, compaction_start).unwrap_or(0),
            ),
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
///    the *first* one we see - that's the largest seq not exceeding
///    `pin_seq`, i.e. the version the oldest live snapshot actually
///    reads. Drop everything strictly older than that.
///
/// When `pin_seq == u64::MAX` (no live snapshot), rule (1) vacuously
/// keeps nothing and rule (2) keeps only the newest version of each
/// user key - the aggressive GC case. When `pin_seq` is somewhere in
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
    // operands do *not* set this flag - they form an open chain
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
        // for this user key - the entry is strictly older and
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
/// - [`CompactionDecision::Keep`] - pass the entry through unchanged.
/// - [`CompactionDecision::Change`] - pass through with the filter's
///   new value (key and seq preserved).
/// - [`CompactionDecision::Remove`] - replace the entry with a
///   deletion tombstone at the same seq. The tombstone prevents an
///   older version of the same user key (living deeper in the LSM)
///   from resurfacing after the filtered value disappears.
///
/// Deletion internal keys are passed through without consulting the
/// filter - the filter's contract is about the user's own values,
/// not about tombstones lark writes itself.
fn apply_compaction_filter(
    entries: Vec<(Vec<u8>, Vec<u8>)>,
    filter: &dyn crate::options::CompactionFilter,
    level: usize,
) -> Vec<(Vec<u8>, Vec<u8>)> {
    use super::internal_key::{VALUE_TYPE_DELETION, VALUE_TYPE_VALUE, encode_internal_key};

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
///    `full_merge` fails - we conservatively keep the raw chain).
///
/// 2. **Partial fold:** if a group is pure merges (no terminator in
///    the compaction's input set) and the operator's
///    `partial_merge` is available, fold the operand chain pairwise
///    into a single operand. The result replaces the chain at the
///    newest merge's seq.
///
/// Anything the operator rejects (`None` return) is left intact so
/// a compaction-time merge failure never loses data - the raw
/// operands survive to be retried on the next compaction or
/// materialized by a reader.
fn collapse_merge_chains(
    entries: Vec<(Vec<u8>, Vec<u8>)>,
    op: &dyn crate::options::MergeOperator,
) -> Vec<(Vec<u8>, Vec<u8>)> {
    use super::internal_key::{
        VALUE_TYPE_DELETION, VALUE_TYPE_MERGE, VALUE_TYPE_VALUE, encode_internal_key,
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
                // No merges in this group - nothing to collapse.
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
                    // The length check above guarantees a first item.
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
    if let Some(s) = start
        && file.meta.largest_key.as_slice() < s
    {
        return false;
    }
    if let Some(e) = end
        && file.meta.smallest_key.as_slice() >= e
    {
        return false;
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

struct MergeHeapEntry {
    key: Vec<u8>,
    value: Vec<u8>,
    stream_idx: usize,
}

impl PartialEq for MergeHeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.stream_idx == other.stream_idx && self.key == other.key
    }
}

impl Eq for MergeHeapEntry {}

impl PartialOrd for MergeHeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
        Some(self.cmp(other))
    }
}

impl Ord for MergeHeapEntry {
    fn cmp(&self, other: &Self) -> CmpOrdering {
        match compare_internal_keys(&other.key, &self.key) {
            CmpOrdering::Equal => other.stream_idx.cmp(&self.stream_idx),
            ordering => ordering,
        }
    }
}

struct CompactionInputStream<'a> {
    iter: SsTableInternalIter<'a>,
}

#[allow(clippy::too_many_arguments)]
fn stream_compaction_outputs(
    versions: &Arc<VersionStore>,
    sst_dir: &Path,
    cache: &BlockCache,
    opts: &CompactionOptions,
    target_level: usize,
    input_files: &[Arc<LiveSst>],
    overlap_files: &[Arc<LiveSst>],
    pin_seq: u64,
    merged_range_tombstones: &RangeTombstoneSet,
) -> std::io::Result<(Vec<VersionEdit>, Vec<OutputFileInfo>)> {
    let all_files: Vec<Arc<LiveSst>> = input_files
        .iter()
        .chain(overlap_files.iter())
        .map(Arc::clone)
        .collect();
    let mut streams = Vec::with_capacity(all_files.len());
    let mut heap = BinaryHeap::new();

    for file in &all_files {
        let mut iter = file.reader.iter_internal_stream(cache)?;
        let stream_idx = streams.len();
        if let Some((key, value)) = iter.next_entry()? {
            heap.push(MergeHeapEntry {
                key,
                value,
                stream_idx,
            });
        }
        streams.push(CompactionInputStream { iter });
    }

    let mut writer = StreamingCompactionWriter::new(
        versions,
        sst_dir,
        opts,
        target_level,
        merged_range_tombstones,
    );
    let mut last_internal_key: Option<Vec<u8>> = None;
    let mut current_user_key: Option<Vec<u8>> = None;
    let mut group: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();

    while let Some(entry) = heap.pop() {
        if let Some((key, value)) = streams[entry.stream_idx].iter.next_entry()? {
            heap.push(MergeHeapEntry {
                key,
                value,
                stream_idx: entry.stream_idx,
            });
        }

        if last_internal_key.as_deref() == Some(entry.key.as_slice()) {
            continue;
        }
        last_internal_key = Some(entry.key.clone());

        let user_key = user_key_of(&entry.key);
        if current_user_key
            .as_deref()
            .is_some_and(|current| current != user_key)
        {
            let transformed =
                transform_compaction_group(std::mem::take(&mut group), pin_seq, opts, target_level);
            writer.add_group(&transformed)?;
        }
        if current_user_key.as_deref() != Some(user_key) {
            current_user_key = Some(user_key.to_vec());
        }

        let (_, seq, _) = decode_internal_key(&entry.key);
        let rt_seq = merged_range_tombstones.max_covering_seq(user_key, pin_seq);
        if rt_seq <= seq {
            group.push((entry.key, entry.value));
        }
    }

    if current_user_key.is_some() {
        let transformed = transform_compaction_group(group, pin_seq, opts, target_level);
        writer.add_group(&transformed)?;
    }

    writer.finish()
}

fn transform_compaction_group(
    mut group: Vec<(Vec<u8>, Vec<u8>)>,
    pin_seq: u64,
    opts: &CompactionOptions,
    target_level: usize,
) -> Vec<(Vec<u8>, Vec<u8>)> {
    if group.is_empty() {
        return group;
    }

    group = gc_old_versions(group, pin_seq);
    if pin_seq == u64::MAX {
        if let Some(filter) = opts.compaction_filter.as_ref() {
            group = apply_compaction_filter(group, filter.as_ref(), target_level);
        }
        if let Some(op) = opts.merge_operator.as_ref() {
            group = collapse_merge_chains(group, op.as_ref());
        }
    }
    group
}

struct StreamingOutputBuilder {
    file_id: u64,
    path: PathBuf,
    writer: SsTableWriter,
    estimated_size: u64,
    smallest_user_key: Option<Vec<u8>>,
    largest_user_key: Option<Vec<u8>>,
}

struct StreamingCompactionWriter<'a> {
    versions: &'a Arc<VersionStore>,
    sst_dir: &'a Path,
    opts: &'a CompactionOptions,
    target_level: usize,
    range_tombstones: &'a RangeTombstoneSet,
    point_ranges: Vec<(Vec<u8>, Vec<u8>)>,
    current: Option<StreamingOutputBuilder>,
    edits: Vec<VersionEdit>,
    infos: Vec<OutputFileInfo>,
}

impl<'a> StreamingCompactionWriter<'a> {
    fn new(
        versions: &'a Arc<VersionStore>,
        sst_dir: &'a Path,
        opts: &'a CompactionOptions,
        target_level: usize,
        range_tombstones: &'a RangeTombstoneSet,
    ) -> Self {
        Self {
            versions,
            sst_dir,
            opts,
            target_level,
            range_tombstones,
            point_ranges: Vec::new(),
            current: None,
            edits: Vec::new(),
            infos: Vec::new(),
        }
    }

    fn add_group(&mut self, entries: &[(Vec<u8>, Vec<u8>)]) -> std::io::Result<()> {
        if entries.is_empty() {
            return Ok(());
        }

        if self
            .current
            .as_ref()
            .is_some_and(|current| current.estimated_size >= self.opts.target_file_size)
        {
            self.finish_current()?;
        }

        // `ensure_current` either installs an output or returns an
        // error, so `current` is `Some` on this line.
        self.ensure_current()?;
        let current = self.current.as_mut().expect("current output is open");
        let user_key = user_key_of(&entries[0].0);
        if current.smallest_user_key.is_none() {
            current.smallest_user_key = Some(user_key.to_vec());
        }
        current.largest_user_key = Some(user_key.to_vec());
        for (ik, value) in entries {
            current.writer.add(ik, value)?;
            current.estimated_size += (ik.len() + value.len()) as u64;
        }
        Ok(())
    }

    fn finish(mut self) -> std::io::Result<(Vec<VersionEdit>, Vec<OutputFileInfo>)> {
        self.finish_current()?;
        self.write_uncovered_range_tombstones()?;
        Ok((self.edits, self.infos))
    }

    fn ensure_current(&mut self) -> std::io::Result<()> {
        if self.current.is_some() {
            return Ok(());
        }

        let file_id = {
            let mut guard = self.versions.lock();
            let current = guard.current();
            let id = current.next_file_id;
            guard.apply(&[VersionEdit::SetNextFileId(id + 1)])?;
            id
        };

        let path = self.sst_dir.join(sst_filename(file_id));
        let writer = SsTableWriter::new_in(
            &self.opts.env,
            &path,
            self.opts.block_size,
            self.opts.bloom_bits_per_key,
            self.opts.compression_for_level(self.target_level),
            self.opts.prefix_extractor.clone(),
            self.opts.partitioned_index,
            self.opts.metadata_block_size,
        )?;

        self.current = Some(StreamingOutputBuilder {
            file_id,
            path,
            writer,
            estimated_size: 0,
            smallest_user_key: None,
            largest_user_key: None,
        });
        Ok(())
    }

    fn finish_current(&mut self) -> std::io::Result<()> {
        let Some(mut current) = self.current.take() else {
            return Ok(());
        };

        let point_range = current
            .smallest_user_key
            .as_ref()
            .zip(current.largest_user_key.as_ref())
            .map(|(smallest, largest)| (smallest.clone(), exclusive_successor(largest)));

        if let Some((start, end)) = &point_range {
            for rt in self.range_tombstones.clipped_overlaps(start, end) {
                current
                    .writer
                    .add_range_tombstone(&rt.start, &rt.end, rt.seq);
            }
        }

        let summary = match current.writer.finish()? {
            Some(summary) => summary,
            None => {
                let _ = self.opts.env.remove_file(&current.path);
                return Ok(());
            }
        };

        let num_entries = summary.num_entries;
        let file_size = self.opts.env.metadata(&current.path)?.len;

        if let Some(limiter) = &self.opts.rate_limiter {
            limiter.request(file_size, crate::rate_limiter::Priority::Low);
        }

        if self.opts.evict_compaction_data_from_page_cache {
            self.opts.env.drop_page_cache(&current.path);
        }

        let reader = Arc::new(SsTableReader::open_with(
            &self.opts.env,
            &current.path,
            current.file_id,
            self.opts.metadata_policy(),
        )?);
        let (smallest_key, largest_key) = match (
            current.smallest_user_key.take(),
            current.largest_user_key.take(),
        ) {
            (Some(smallest), Some(largest)) => {
                if let Some((start, end)) = point_range {
                    self.point_ranges.push((start, end));
                }
                (smallest, largest)
            }
            _ => (summary.smallest_user_key, summary.largest_user_key),
        };
        let new_file = LiveSst::new(
            SsTableMeta {
                file_id: current.file_id,
                smallest_key,
                largest_key,
                file_size,
                num_entries,
            },
            reader,
        );

        self.infos.push(OutputFileInfo {
            file_id: current.file_id,
            path: current.path.clone(),
            file_size,
            num_entries,
        });

        self.edits.push(VersionEdit::AddFile {
            level: self.target_level,
            file: new_file,
        });

        Ok(())
    }

    fn write_uncovered_range_tombstones(&mut self) -> std::io::Result<()> {
        if self.range_tombstones.is_empty() {
            return Ok(());
        }

        let fragments = uncovered_range_tombstone_fragments(
            self.range_tombstones.as_slice(),
            &self.point_ranges,
        );
        if fragments.is_empty() {
            return Ok(());
        }

        for chunk in group_range_tombstone_fragments(fragments) {
            self.write_range_tombstone_only_file(&chunk)?;
        }

        Ok(())
    }

    fn write_range_tombstone_only_file(
        &mut self,
        tombstones: &[RangeTombstone],
    ) -> std::io::Result<()> {
        if tombstones.is_empty() {
            return Ok(());
        }

        let file_id = {
            let mut guard = self.versions.lock();
            let current = guard.current();
            let id = current.next_file_id;
            guard.apply(&[VersionEdit::SetNextFileId(id + 1)])?;
            id
        };

        let path = self.sst_dir.join(sst_filename(file_id));
        let mut writer = SsTableWriter::new_in(
            &self.opts.env,
            &path,
            self.opts.block_size,
            self.opts.bloom_bits_per_key,
            self.opts.compression_for_level(self.target_level),
            self.opts.prefix_extractor.clone(),
            self.opts.partitioned_index,
            self.opts.metadata_block_size,
        )?;
        for rt in tombstones {
            writer.add_range_tombstone(&rt.start, &rt.end, rt.seq);
        }

        let Some(summary) = writer.finish()? else {
            let _ = self.opts.env.remove_file(&path);
            return Ok(());
        };

        let file_size = self.opts.env.metadata(&path)?.len;

        if let Some(limiter) = &self.opts.rate_limiter {
            limiter.request(file_size, crate::rate_limiter::Priority::Low);
        }

        if self.opts.evict_compaction_data_from_page_cache {
            self.opts.env.drop_page_cache(&path);
        }

        let reader = Arc::new(SsTableReader::open_with(
            &self.opts.env,
            &path,
            file_id,
            self.opts.metadata_policy(),
        )?);
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

        self.infos.push(OutputFileInfo {
            file_id,
            path: path.clone(),
            file_size,
            num_entries: summary.num_entries,
        });

        self.edits.push(VersionEdit::AddFile {
            level: self.target_level,
            file: new_file,
        });

        Ok(())
    }
}

fn uncovered_range_tombstone_fragments(
    tombstones: &[RangeTombstone],
    covered_ranges: &[(Vec<u8>, Vec<u8>)],
) -> Vec<RangeTombstone> {
    let covered_ranges = merge_key_ranges(covered_ranges);
    let mut fragments = Vec::new();

    for rt in tombstones {
        let mut cursor = rt.start.clone();
        for (covered_start, covered_end) in &covered_ranges {
            if covered_end.as_slice() <= cursor.as_slice() {
                continue;
            }
            if covered_start.as_slice() >= rt.end.as_slice() {
                break;
            }

            if cursor.as_slice() < covered_start.as_slice() {
                let fragment_end = min_key(covered_start, &rt.end);
                if cursor.as_slice() < fragment_end.as_slice() {
                    fragments.push(RangeTombstone::new(cursor.clone(), fragment_end, rt.seq));
                }
            }

            if cursor.as_slice() < covered_end.as_slice() {
                cursor = max_key(covered_end, &cursor);
            }
            if cursor.as_slice() >= rt.end.as_slice() {
                break;
            }
        }

        if cursor.as_slice() < rt.end.as_slice() {
            fragments.push(RangeTombstone::new(cursor, rt.end.clone(), rt.seq));
        }
    }

    sort_dedup_tombstones(&mut fragments);
    fragments
}

fn group_range_tombstone_fragments(mut fragments: Vec<RangeTombstone>) -> Vec<Vec<RangeTombstone>> {
    sort_dedup_tombstones(&mut fragments);
    let mut groups: Vec<Vec<RangeTombstone>> = Vec::new();
    let mut current: Vec<RangeTombstone> = Vec::new();
    let mut current_end: Option<Vec<u8>> = None;

    for rt in fragments {
        if let Some(end) = &current_end
            && rt.start.as_slice() > end.as_slice()
        {
            groups.push(std::mem::take(&mut current));
            current_end = None;
        }

        if current_end
            .as_ref()
            .is_none_or(|end| end.as_slice() < rt.end.as_slice())
        {
            current_end = Some(rt.end.clone());
        }
        current.push(rt);
    }

    if !current.is_empty() {
        groups.push(current);
    }

    groups
}

fn merge_key_ranges(ranges: &[(Vec<u8>, Vec<u8>)]) -> Vec<(Vec<u8>, Vec<u8>)> {
    let mut ranges: Vec<(Vec<u8>, Vec<u8>)> = ranges
        .iter()
        .filter(|(start, end)| start < end)
        .cloned()
        .collect();
    ranges.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

    let mut merged: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
    for (start, end) in ranges {
        match merged.last_mut() {
            Some((_, merged_end)) if start.as_slice() <= merged_end.as_slice() => {
                if merged_end.as_slice() < end.as_slice() {
                    *merged_end = end;
                }
            }
            _ => merged.push((start, end)),
        }
    }
    merged
}

fn min_key(a: &[u8], b: &[u8]) -> Vec<u8> {
    if a <= b { a.to_vec() } else { b.to_vec() }
}

fn max_key(a: &[u8], b: &[u8]) -> Vec<u8> {
    if a >= b { a.to_vec() } else { b.to_vec() }
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
    env: &dyn Env,
    sst_dir: &Path,
    input_files: &[Arc<LiveSst>],
    overlap_files: &[Arc<LiveSst>],
    cache: &BlockCache,
) {
    for file in input_files.iter().chain(overlap_files.iter()) {
        let path = sst_dir.join(sst_filename(file.meta.file_id));
        cache.evict_file(file.meta.file_id);
        if let Err(e) = remove_sst_in(env, &path) {
            tracing::warn!(
                file_id = file.meta.file_id,
                error = %e,
                "Failed to delete old SSTable"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rt(start: &[u8], end: &[u8], seq: u64) -> RangeTombstone {
        RangeTombstone::new(start.to_vec(), end.to_vec(), seq)
    }

    #[test]
    fn db_open_returns_an_error_when_a_compaction_worker_cannot_spawn() {
        let dir = tempfile::tempdir().unwrap();

        let err = {
            let _guard = SpawnFailureGuard::allowing(0);
            crate::Db::open(dir.path(), crate::Options::default()).unwrap_err()
        };
        match err {
            crate::Error::Io(e) => assert_eq!(e.kind(), std::io::ErrorKind::Unsupported),
            other => panic!("expected an I/O error, got {other:?}"),
        }

        // The failed open released the directory lock and left no
        // half-built database behind.
        let db = crate::Db::open(dir.path(), crate::Options::default()).unwrap();
        db.put(b"k", b"v").unwrap();
        assert_eq!(db.get(b"k").unwrap().as_deref(), Some(&b"v"[..]));
    }

    #[test]
    fn failed_start_joins_the_workers_it_already_spawned() {
        let dir = tempfile::tempdir().unwrap();
        let sst_dir = dir.path().join("sst");
        std::fs::create_dir_all(&sst_dir).unwrap();

        let versions = Arc::new(VersionStore::new(
            super::super::manifest::VersionSet::open(dir.path(), &sst_dir).unwrap(),
        ));
        let opts = CompactionOptions {
            max_background_compactions: 3,
            ..CompactionOptions::default()
        };

        let started = {
            let _guard = SpawnFailureGuard::allowing(2);
            CompactionScheduler::start(
                Arc::new(crate::sync::Gate::new()),
                Arc::new(SnapshotRegistry::new()),
                Arc::clone(&versions),
                Arc::from(sst_dir.as_path()),
                Arc::new(BlockCache::new(4096)),
                opts,
                Arc::new(crate::engine::StallSignal::new()),
                Arc::new(crate::sync::Mutex::new(HashSet::new())),
            )
        };

        match started {
            Ok(_) => panic!("expected the third worker spawn to fail"),
            Err(e) => assert_eq!(e.kind(), std::io::ErrorKind::Unsupported),
        }
        // Every worker that did start has exited and been joined, so
        // its clone of the version set is gone.
        assert_eq!(Arc::strong_count(&versions), 1);
    }

    #[test]
    fn uncovered_fragments_split_around_point_ranges() {
        let fragments = uncovered_range_tombstone_fragments(
            &[rt(b"a", b"z", 7)],
            &[(b"m".to_vec(), b"m\0".to_vec())],
        );

        assert_eq!(fragments.len(), 2);
        assert_eq!(fragments[0].start, b"a");
        assert_eq!(fragments[0].end, b"m");
        assert_eq!(fragments[0].seq, 7);
        assert_eq!(fragments[1].start, b"m\0");
        assert_eq!(fragments[1].end, b"z");
        assert_eq!(fragments[1].seq, 7);
    }

    #[test]
    fn uncovered_fragments_drop_fully_covered_ranges() {
        let fragments = uncovered_range_tombstone_fragments(
            &[rt(b"b", b"d", 3)],
            &[(b"a".to_vec(), b"z".to_vec())],
        );
        assert!(fragments.is_empty());
    }

    #[test]
    fn tombstone_fragments_group_overlapping_gaps() {
        let groups = group_range_tombstone_fragments(vec![
            rt(b"a", b"c", 1),
            rt(b"b", b"d", 2),
            rt(b"f", b"g", 3),
        ]);

        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].len(), 2);
        assert_eq!(groups[1].len(), 1);
    }

    #[test]
    fn failed_open_preserves_data_already_written() {
        let dir = tempfile::tempdir().unwrap();
        {
            let db = crate::Db::open(dir.path(), crate::Options::default()).unwrap();
            for i in 0..300 {
                db.put(format!("k{i:04}").as_bytes(), format!("v{i:04}").as_bytes())
                    .unwrap();
            }
        }

        for _ in 0..3 {
            let _guard = SpawnFailureGuard::allowing(0);
            assert!(crate::Db::open(dir.path(), crate::Options::default()).is_err());
        }

        let db = crate::Db::open(dir.path(), crate::Options::default()).unwrap();
        for i in 0..300 {
            assert_eq!(
                db.get(format!("k{i:04}").as_bytes()).unwrap().as_deref(),
                Some(format!("v{i:04}").as_bytes()),
                "key {i} lost across failed opens"
            );
        }
    }

    #[test]
    fn failed_start_with_many_workers_returns_promptly() {
        let dir = tempfile::tempdir().unwrap();
        let sst_dir = dir.path().join("sst");
        std::fs::create_dir_all(&sst_dir).unwrap();
        let versions = Arc::new(VersionStore::new(
            super::super::manifest::VersionSet::open(dir.path(), &sst_dir).unwrap(),
        ));
        let opts = CompactionOptions {
            max_background_compactions: 8,
            ..CompactionOptions::default()
        };
        let start = std::time::Instant::now();
        let started = {
            let _guard = SpawnFailureGuard::allowing(7);
            CompactionScheduler::start(
                Arc::new(crate::sync::Gate::new()),
                Arc::new(SnapshotRegistry::new()),
                Arc::clone(&versions),
                Arc::from(sst_dir.as_path()),
                Arc::new(BlockCache::new(4096)),
                opts,
                Arc::new(crate::engine::StallSignal::new()),
                Arc::new(crate::sync::Mutex::new(HashSet::new())),
            )
        };
        let elapsed = start.elapsed();
        assert!(started.is_err());
        println!("failed start with 8 workers took {elapsed:?}");
        assert_eq!(Arc::strong_count(&versions), 1);
    }
}
