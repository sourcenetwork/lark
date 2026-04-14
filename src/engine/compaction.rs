use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

use super::block_cache::BlockCache;
use super::internal_key::{decode_internal_key, user_key_of};
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

/// Manages background compaction on a dedicated OS thread.
pub(crate) struct CompactionScheduler {
    shutdown: Arc<AtomicBool>,
    trigger: Arc<(Mutex<bool>, Condvar)>,
    handle: Option<thread::JoinHandle<()>>,
}

impl CompactionScheduler {
    /// Start the background compaction thread.
    ///
    /// `compaction_lock` is the engine-wide mutex that serializes
    /// background compactions with any foreground caller of
    /// [`run_compact_range`] — acquiring it around every compaction pass
    /// ensures both paths don't try to pick overlapping input sets or
    /// double-delete the same file.
    ///
    /// `snapshot_registry` lets each compaction pass query the current
    /// pin seq so it can drop versions that no live snapshot needs.
    pub(crate) fn start(
        compaction_lock: Arc<parking_lot::Mutex<()>>,
        snapshot_registry: Arc<SnapshotRegistry>,
        versions: Arc<parking_lot::Mutex<VersionSet>>,
        sst_dir: Arc<Path>,
        cache: Arc<BlockCache>,
        opts: CompactionOptions,
    ) -> Self {
        let shutdown = Arc::new(AtomicBool::new(false));
        let trigger = Arc::new((Mutex::new(false), Condvar::new()));

        let shutdown_clone = Arc::clone(&shutdown);
        let trigger_clone = Arc::clone(&trigger);

        let handle = thread::Builder::new()
            .name("lark-compaction".into())
            .spawn(move || {
                compaction_loop(
                    shutdown_clone,
                    trigger_clone,
                    compaction_lock,
                    snapshot_registry,
                    versions,
                    sst_dir,
                    cache,
                    opts,
                );
            })
            .expect("failed to spawn compaction thread");

        Self {
            shutdown,
            trigger,
            handle: Some(handle),
        }
    }

    /// Notify the compaction thread that work may be available.
    pub(crate) fn notify(&self) {
        let (lock, cvar) = &*self.trigger;
        let mut triggered = lock.lock().unwrap();
        *triggered = true;
        cvar.notify_one();
    }

    /// Shut down the compaction thread.
    pub(crate) fn shutdown(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        self.notify();
        if let Some(handle) = self.handle.take() {
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
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn compaction_loop(
    shutdown: Arc<AtomicBool>,
    trigger: Arc<(Mutex<bool>, Condvar)>,
    compaction_lock: Arc<parking_lot::Mutex<()>>,
    snapshot_registry: Arc<SnapshotRegistry>,
    versions: Arc<parking_lot::Mutex<VersionSet>>,
    sst_dir: Arc<Path>,
    cache: Arc<BlockCache>,
    opts: CompactionOptions,
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
        // acquires `compaction_lock` so a foreground `compact_range`
        // caller can interleave cleanly — foreground grabs the lock
        // for the duration of its walk and we block until it's done.
        loop {
            let did_work = {
                let _guard = compaction_lock.lock();
                // Recompute the GC horizon on every pass: a snapshot
                // may have dropped since the previous pass, unpinning
                // more versions.
                let pin_seq = snapshot_registry.oldest_live_seq();
                match pick_and_run_compaction(&versions, &sst_dir, &cache, &opts, pin_seq) {
                    Ok(did_work) => did_work,
                    Err(e) => {
                        tracing::error!(error = %e, "Compaction failed");
                        false
                    }
                }
            };

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
) -> std::io::Result<bool> {
    let version = versions.lock().current();

    // Check L0 first
    if version.l0_count() >= opts.l0_compaction_trigger {
        return compact_l0(versions, sst_dir, cache, opts, pin_seq);
    }

    // Check other levels
    for level in 1..MAX_LEVELS - 1 {
        let target = level_target_size(level, opts);
        if version.level_size(level) > target {
            return compact_level(versions, sst_dir, cache, opts, level, pin_seq);
        }
    }

    Ok(false)
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
) -> std::io::Result<bool> {
    compact_level(versions, sst_dir, cache, opts, 0, pin_seq)
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

    // For L0, all files may overlap. For other levels, pick the first file.
    let (input_files, overlap_files) = if level == 0 {
        let l0_files = input_files;
        let (min_key, max_key) = key_range(&l0_files);
        let overlapping = find_overlapping(&version.levels[target_level], &min_key, &max_key);
        (l0_files, overlapping)
    } else {
        let picked = vec![Arc::clone(&input_files[0])];
        let (min_key, max_key) = key_range(&picked);
        let overlapping = find_overlapping(&version.levels[target_level], &min_key, &max_key);
        (picked, overlapping)
    };

    perform_compaction(
        versions,
        sst_dir,
        cache,
        opts,
        level,
        input_files,
        overlap_files,
        pin_seq,
    )?;
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
    let target_level = level + 1;

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
    }

    // Sort by internal key (which orders newer seqs first within each user
    // key) and drop any exact duplicates (same user_key + seq + type).
    all_entries.sort_by(|a, b| a.0.cmp(&b.0));
    all_entries.dedup_by(|a, b| a.0 == b.0);

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

    // Split output across multiple SSTables at user-key boundaries so that
    // all versions of a given user key live in exactly one file (required
    // for the non-overlap invariant at L1+).
    //
    // Range tombstones are replicated into every output file produced by
    // this compaction. A future compaction of any of these files will
    // merge and dedup the RTs again, so there's no runaway duplication.
    // Replicating keeps each file self-sufficient for reads: a scan that
    // only picks up one of the files still sees the right RT coverage.
    let mut chunk_start = 0;
    let rt_only_output = all_entries.is_empty() && !merged_range_tombstones.is_empty();
    let mut wrote_rt_only = false;
    loop {
        if !rt_only_output && chunk_start >= all_entries.len() {
            break;
        }
        if rt_only_output && wrote_rt_only {
            break;
        }
        // Allocate the output file_id atomically from the *current*
        // version inside `versions.lock()` — same pattern
        // `flush_frozen_memtable` uses. Using a captured
        // `version.next_file_id` would race with a concurrent flush
        // that advances the counter on the current version; both
        // paths would then pick the same id and the second
        // `File::create(path)` would truncate the first path's newly
        // written file.
        let file_id = {
            let mut guard = versions.lock();
            let current = guard.current();
            let id = current.next_file_id;
            guard.apply(&[VersionEdit::SetNextFileId(id + 1)])?;
            id
        };

        let path = sst_dir.join(sst_filename(file_id));
        // Output files land at `target_level`, so pick that level's
        // codec — this is what makes `compression_per_level` actually
        // shape the on-disk layout as files migrate down through the
        // tree.
        let mut writer = SsTableWriter::new(
            &path,
            opts.block_size,
            opts.bloom_bits_per_key,
            opts.compression_for_level(target_level),
        )?;

        let mut estimated_size: u64 = 0;
        let mut current_user_key: Option<Vec<u8>> = None;

        while chunk_start < all_entries.len() {
            let (ik, value) = &all_entries[chunk_start];
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

        for rt in &merged_range_tombstones {
            writer.add_range_tombstone(&rt.start, &rt.end, rt.seq);
        }
        if rt_only_output {
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

        edits.push(VersionEdit::AddFile {
            level: target_level,
            file: new_file,
        });
    }

    // Atomically apply the remove / add edits. `SetNextFileId` is not
    // needed here — each output file already advanced it when it was
    // allocated above.
    versions.lock().apply(&edits)?;

    // Unlink the old SSTable paths. Their file descriptors stay alive
    // through any `Arc<LiveSst>` still held by older versions or by
    // iterators, so the data remains readable until those Arcs drop.
    delete_old_files(sst_dir, &input_files, &overlap_files, cache);

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
    let mut out = Vec::with_capacity(entries.len());
    let mut current_user_key: Option<Vec<u8>> = None;
    let mut pin_emitted = false;

    for (ik, value) in entries {
        let (uk, seq, _) = decode_internal_key(&ik);

        if current_user_key.as_deref() != Some(uk) {
            current_user_key = Some(uk.to_vec());
            pin_emitted = false;
        }

        if seq > pin_seq {
            out.push((ik, value));
            continue;
        }

        // `seq <= pin_seq` — first one is the pin entry for this
        // group, every later one is strictly older and shadowed.
        if !pin_emitted {
            out.push((ik, value));
            pin_emitted = true;
        }
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
