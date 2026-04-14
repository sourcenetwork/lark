pub(crate) mod block;
pub(crate) mod block_cache;
pub(crate) mod bloom;
pub(crate) mod compaction;
pub(crate) mod internal_key;
pub(crate) mod iterator;
pub(crate) mod manifest;
pub(crate) mod memtable;
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
use sstable::{sst_filename, LiveSst, LookupResult, SsTableMeta, SsTableReader, SsTableWriter};
use wal::{wal_filename, Wal, WalEntry};

/// Controls when data is flushed to disk after a commit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DurabilityMode {
    Immediate,
    Eventual,
}

/// Configuration for the Lark engine.
#[derive(Clone)]
pub(crate) struct EngineOptions {
    pub(crate) write_buffer_size: usize,
    pub(crate) block_size: usize,
    pub(crate) block_cache_size: usize,
    pub(crate) bloom_bits_per_key: usize,
    pub(crate) compression: bool,
    pub(crate) l0_compaction_trigger: usize,
    pub(crate) level_base_bytes: u64,
    pub(crate) level_size_multiplier: u64,
    pub(crate) target_file_size: u64,
}

impl Default for EngineOptions {
    fn default() -> Self {
        Self {
            write_buffer_size: 64 * 1024 * 1024,
            block_size: 16 * 1024,
            block_cache_size: 512 * 1024 * 1024,
            bloom_bits_per_key: 10,
            compression: true,
            l0_compaction_trigger: compaction::L0_COMPACTION_TRIGGER,
            level_base_bytes: compaction::DEFAULT_LEVEL_BASE_BYTES,
            level_size_multiplier: compaction::LEVEL_SIZE_MULTIPLIER,
            target_file_size: compaction::DEFAULT_TARGET_FILE_SIZE,
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

        let cache = Arc::new(BlockCache::new(options.block_cache_size));
        let versions = Arc::new(Mutex::new(version_set));

        let compaction_opts = CompactionOptions {
            l0_compaction_trigger: options.l0_compaction_trigger,
            level_base_bytes: options.level_base_bytes,
            level_size_multiplier: options.level_size_multiplier,
            target_file_size: options.target_file_size,
            block_size: options.block_size,
            bloom_bits_per_key: options.bloom_bits_per_key,
            compression: options.compression,
        };

        let compaction = CompactionScheduler::start(
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
            options,
            write_lock: Mutex::new(()),
        });

        Ok(engine)
    }

    pub(crate) fn snapshot_seq(&self) -> u64 {
        self.latest_seq.load(Ordering::Acquire)
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
        )
    }

    /// Point lookup at a given snapshot. Returns `Ok(Some(value))` or `Ok(None)`.
    pub(crate) fn get(&self, key: &[u8], snapshot_seq: u64) -> std::io::Result<Option<Vec<u8>>> {
        if let Some(result) = self.active_memtable.read().get(key, snapshot_seq) {
            return Ok(result);
        }

        {
            let frozen = self.frozen_memtables.read();
            for mt in frozen.iter().rev() {
                if let Some(result) = mt.get(key, snapshot_seq) {
                    return Ok(result);
                }
            }
        }

        let version = self.versions.lock().current();

        // L0: check all files (may overlap), newest first. Readers are
        // already open in the pinned `Version`, so no filesystem access
        // happens here — concurrent compaction unlinking paths cannot
        // break us.
        for file in version.levels[0].iter().rev() {
            match file.reader.get(key, snapshot_seq, &self.cache)? {
                LookupResult::Found(value) => return Ok(Some(value)),
                LookupResult::FoundTombstone => return Ok(None),
                LookupResult::NotInTable => {}
            }
        }

        // L1+: binary search for the right SSTable per level.
        for level in 1..version.levels.len() {
            let files = &version.levels[level];
            if files.is_empty() {
                continue;
            }

            let idx = files.partition_point(|f| f.meta.largest_key.as_slice() < key);
            if idx < files.len() && files[idx].meta.smallest_key.as_slice() <= key {
                let file = &files[idx];
                match file.reader.get(key, snapshot_seq, &self.cache)? {
                    LookupResult::Found(value) => return Ok(Some(value)),
                    LookupResult::FoundTombstone => return Ok(None),
                    LookupResult::NotInTable => {}
                }
            }
        }

        Ok(None)
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
        let mut results: Vec<Option<Vec<u8>>> = vec![None; keys.len()];
        // `None` = still pending, `Some(_)` = resolved (the result is
        // already stored in `results[i]`).
        let mut resolved: Vec<bool> = vec![false; keys.len()];
        let mut unresolved = keys.len();
        if unresolved == 0 {
            return Ok(results);
        }

        // 1. Active memtable.
        {
            let mt = self.active_memtable.read();
            for (i, k) in keys.iter().enumerate() {
                if resolved[i] {
                    continue;
                }
                if let Some(result) = mt.get(k, snapshot_seq) {
                    results[i] = result;
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
                    if let Some(result) = mt.get(k, snapshot_seq) {
                        results[i] = result;
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
                match file.reader.get(k, snapshot_seq, &self.cache)? {
                    LookupResult::Found(v) => {
                        results[i] = Some(v);
                        resolved[i] = true;
                        unresolved -= 1;
                    }
                    LookupResult::FoundTombstone => {
                        // tombstone hides any older versions.
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
        //    could contain a given key.
        for level in 1..version.levels.len() {
            let files = &version.levels[level];
            if files.is_empty() {
                continue;
            }
            for (i, k) in keys.iter().enumerate() {
                if resolved[i] {
                    continue;
                }
                let idx = files.partition_point(|f| f.meta.largest_key.as_slice() < k);
                if idx >= files.len() || files[idx].meta.smallest_key.as_slice() > *k {
                    continue;
                }
                let file = &files[idx];
                match file.reader.get(k, snapshot_seq, &self.cache)? {
                    LookupResult::Found(v) => {
                        results[i] = Some(v);
                        resolved[i] = true;
                        unresolved -= 1;
                    }
                    LookupResult::FoundTombstone => {
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

    /// Apply a batch of writes atomically.
    pub(crate) fn apply_batch(
        &self,
        writes: BTreeMap<Vec<u8>, Option<Vec<u8>>>,
        durability: DurabilityMode,
    ) -> std::io::Result<()> {
        if writes.is_empty() {
            return Ok(());
        }

        let _write_guard = self.write_lock.lock();

        let base_seq = self
            .latest_seq
            .fetch_add(writes.len() as u64, Ordering::AcqRel)
            + 1;

        {
            let mut wal = self.active_wal.lock();
            for (i, (key, value)) in writes.iter().enumerate() {
                let seq = base_seq + i as u64;
                match value {
                    Some(v) => wal.append_put(key, v, seq)?,
                    None => wal.append_delete(key, seq)?,
                }
            }
            match durability {
                DurabilityMode::Immediate => wal.sync()?,
                DurabilityMode::Eventual => wal.flush()?,
            }
        }

        {
            let memtable = self.active_memtable.read();
            for (i, (key, value)) in writes.iter().enumerate() {
                let seq = base_seq + i as u64;
                match value {
                    Some(v) => memtable.put(key, v, seq),
                    None => memtable.delete(key, seq),
                }
            }
        }

        if self.active_memtable.read().approximate_size() >= self.options.write_buffer_size {
            self.rotate_memtable()?;
        }

        Ok(())
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
        let memtable = {
            let frozen = self.frozen_memtables.read();
            if frozen.is_empty() {
                return Ok(());
            }
            Arc::clone(&frozen[0])
        };

        if memtable.is_empty() {
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

        let mut writer = SsTableWriter::new(
            &sst_path,
            self.options.block_size,
            self.options.bloom_bits_per_key,
            self.options.compression,
        )?;

        // Walk the memtable in internal-key order and copy every version
        // and tombstone into the SSTable unchanged, preserving MVCC.
        let entries = memtable.iter_internal();
        for (internal_key, value) in &entries {
            writer.add(internal_key, value)?;
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

        tracing::info!(
            file_id,
            entries = num_entries,
            size = file_size,
            "Flushed memtable to L0 SSTable"
        );

        Ok(())
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

    /// Flush all data to disk and shut down background threads.
    pub(crate) fn close(&self) -> std::io::Result<()> {
        if !self.active_memtable.read().is_empty() {
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
