use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

use super::block_cache::BlockCache;
use super::internal_key::user_key_of;
use super::manifest::{VersionEdit, VersionSet, MAX_LEVELS};
use super::sstable::{remove_sst, sst_filename, SsTableMeta, SsTableReader, SsTableWriter};

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
    pub(crate) fn start(
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
    pub(crate) compression: bool,
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
            compression: true,
        }
    }
}

fn compaction_loop(
    shutdown: Arc<AtomicBool>,
    trigger: Arc<(Mutex<bool>, Condvar)>,
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

        // Check if compaction is needed
        loop {
            let did_work = match pick_and_run_compaction(&versions, &sst_dir, &cache, &opts) {
                Ok(did_work) => did_work,
                Err(e) => {
                    tracing::error!(error = %e, "Compaction failed");
                    false
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
) -> std::io::Result<bool> {
    let version = versions.lock().current();

    // Check L0 first
    if version.l0_count() >= opts.l0_compaction_trigger {
        return compact_l0(versions, sst_dir, cache, opts);
    }

    // Check other levels
    for level in 1..MAX_LEVELS - 1 {
        let target = level_target_size(level, opts);
        if version.level_size(level) > target {
            return compact_level(versions, sst_dir, cache, opts, level);
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
) -> std::io::Result<bool> {
    compact_level(versions, sst_dir, cache, opts, 0)
}

/// Compact a level into the next level.
fn compact_level(
    versions: &Arc<parking_lot::Mutex<VersionSet>>,
    sst_dir: &Path,
    cache: &BlockCache,
    opts: &CompactionOptions,
    level: usize,
) -> std::io::Result<bool> {
    let target_level = level + 1;
    if target_level >= MAX_LEVELS {
        return Ok(false);
    }

    let version = versions.lock().current();

    let input_files: Vec<SsTableMeta> = version.levels[level].clone();
    if input_files.is_empty() {
        return Ok(false);
    }

    // For L0, all files may overlap. For other levels, pick the first file.
    let (input_metas, overlap_metas) = if level == 0 {
        // All L0 files
        let l0_files = input_files;
        // Find overlapping L1 files
        let (min_key, max_key) = key_range(&l0_files);
        let overlapping = find_overlapping(&version.levels[target_level], &min_key, &max_key);
        (l0_files, overlapping)
    } else {
        // Pick first file from this level
        let picked = vec![input_files[0].clone()];
        let (min_key, max_key) = key_range(&picked);
        let overlapping = find_overlapping(&version.levels[target_level], &min_key, &max_key);
        (picked, overlapping)
    };

    // Read all input entries as raw internal-key / value pairs so every
    // version and tombstone is preserved through the merge.
    let mut all_entries: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();

    for meta in input_metas.iter().chain(overlap_metas.iter()) {
        let path = sst_dir.join(sst_filename(meta.file_id));
        let reader = SsTableReader::open(&path, meta.file_id)?;
        all_entries.extend(reader.iter_internal(cache)?);
    }

    // Sort by internal key (which orders newer seqs first within each user
    // key) and drop any exact duplicates (same user_key + seq + type).
    all_entries.sort_by(|a, b| a.0.cmp(&b.0));
    all_entries.dedup_by(|a, b| a.0 == b.0);

    let mut edits = Vec::new();

    for meta in &input_metas {
        edits.push(VersionEdit::RemoveFile {
            level,
            file_id: meta.file_id,
        });
    }
    for meta in &overlap_metas {
        edits.push(VersionEdit::RemoveFile {
            level: target_level,
            file_id: meta.file_id,
        });
    }

    let mut next_file_id = version.next_file_id;

    if all_entries.is_empty() {
        edits.push(VersionEdit::SetNextFileId(next_file_id));
        versions.lock().apply(&edits)?;
        delete_old_files(sst_dir, &input_metas, &overlap_metas, cache);
        return Ok(true);
    }

    // Split output across multiple SSTables at user-key boundaries so that
    // all versions of a given user key live in exactly one file (required
    // for the non-overlap invariant at L1+).
    let mut chunk_start = 0;
    while chunk_start < all_entries.len() {
        let file_id = next_file_id;
        next_file_id += 1;

        let path = sst_dir.join(sst_filename(file_id));
        let mut writer = SsTableWriter::new(
            &path,
            opts.block_size,
            opts.bloom_bits_per_key,
            opts.compression,
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

        let summary = match writer.finish()? {
            Some(s) => s,
            None => {
                let _ = std::fs::remove_file(&path);
                continue;
            }
        };

        let file_size = std::fs::metadata(&path)?.len();

        edits.push(VersionEdit::AddFile {
            level: target_level,
            meta: SsTableMeta {
                file_id,
                smallest_key: summary.smallest_user_key,
                largest_key: summary.largest_user_key,
                file_size,
                num_entries: summary.num_entries,
            },
        });
    }

    edits.push(VersionEdit::SetNextFileId(next_file_id));

    // Atomically apply version edits
    versions.lock().apply(&edits)?;

    // Delete old SSTable files (safe because version no longer references them)
    delete_old_files(sst_dir, &input_metas, &overlap_metas, cache);

    tracing::info!(
        level,
        target_level,
        input_files = input_metas.len() + overlap_metas.len(),
        "Compaction completed"
    );

    Ok(true)
}

fn key_range(files: &[SsTableMeta]) -> (Vec<u8>, Vec<u8>) {
    let mut min_key = files[0].smallest_key.clone();
    let mut max_key = files[0].largest_key.clone();

    for meta in &files[1..] {
        if meta.smallest_key < min_key {
            min_key = meta.smallest_key.clone();
        }
        if meta.largest_key > max_key {
            max_key = meta.largest_key.clone();
        }
    }

    (min_key, max_key)
}

fn find_overlapping(files: &[SsTableMeta], min_key: &[u8], max_key: &[u8]) -> Vec<SsTableMeta> {
    files
        .iter()
        .filter(|f| {
            // Overlaps if: file.smallest <= max_key AND file.largest >= min_key
            f.smallest_key.as_slice() <= max_key && f.largest_key.as_slice() >= min_key
        })
        .cloned()
        .collect()
}

fn delete_old_files(
    sst_dir: &Path,
    input_metas: &[SsTableMeta],
    overlap_metas: &[SsTableMeta],
    cache: &BlockCache,
) {
    for meta in input_metas.iter().chain(overlap_metas.iter()) {
        let path = sst_dir.join(sst_filename(meta.file_id));
        cache.evict_file(meta.file_id);
        if let Err(e) = remove_sst(&path) {
            tracing::warn!(
                file_id = meta.file_id,
                error = %e,
                "Failed to delete old SSTable"
            );
        }
    }
}
