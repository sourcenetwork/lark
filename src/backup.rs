//! Content-addressed incremental backups.
//!
//! [`BackupEngine`] keeps a directory of backups that deduplicate
//! SSTable files across generations. Files are keyed in a `shared/`
//! subdirectory by an xxh3-based content id, and each backup's
//! manifest lists the `(level, file_id, content id, meta)` tuples that
//! reconstruct its version. Backing up an unchanged database a
//! second time adds only the per-backup manifest, not the data.
//! The content id and manifest checksum are fast accidental-corruption
//! guards, not adversarial tamper protection.
//!
//! On-disk layout:
//!
//! ```text
//! backup_dir/
//!   meta/
//!     000001.backup       # per-backup manifest
//!     000002.backup
//!   shared/
//!     <32-hex>.sst        # content-hashed SST files
//! ```
//!
//! The backup directory may live on a different filesystem from
//! the source database - files are byte-copied, not hard-linked.
//! Restores stream bytes back out of `shared/` into a fresh
//! target directory and write a new MANIFEST reflecting the
//! captured version.

use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::engine::{CheckpointSnapshot, checksum, durability};
use crate::{Db, Error, Result};

/// Opaque identifier for a single backup generation. Monotonically
/// increasing within a backup directory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BackupId(pub u64);

impl std::fmt::Display for BackupId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// High-level summary of a backup returned from [`BackupEngine::list_backups`].
#[derive(Debug, Clone)]
pub struct BackupInfo {
    /// Backup identifier.
    pub id: BackupId,
    /// Unix timestamp (seconds) at which the backup was created.
    pub created_at_unix: u64,
    /// Number of SSTable files captured in the backup.
    pub file_count: usize,
    /// Total logical size (sum of SSTable file sizes) in bytes.
    pub bytes: u64,
}

/// Content-addressed backup repository for one or more databases.
///
/// Multiple [`BackupEngine`] instances should not share a backup
/// directory - there is no cross-process locking. A single process
/// may reuse one instance for many backups.
pub struct BackupEngine {
    root: PathBuf,
    meta_dir: PathBuf,
    shared_dir: PathBuf,
}

#[derive(Debug, Clone)]
struct BackupFileEntry {
    level: u32,
    file_id: u64,
    file_size: u64,
    hash: u128,
    smallest_key: Vec<u8>,
    largest_key: Vec<u8>,
    num_entries: u64,
}

#[derive(Debug, Clone)]
struct BackupManifest {
    created_at_unix: u64,
    files: Vec<BackupFileEntry>,
    next_file_id: u64,
    last_seq: u64,
}

/// Identifier at the head of a backup manifest: `REGOMBKP`.
const BACKUP_MANIFEST_MAGIC: [u8; 8] = *b"REGOMBKP";

/// The 4-byte identifier earlier builds wrote (`LKBP`). Read, never
/// written, so an existing backup repository still restores.
const BACKUP_MANIFEST_MAGIC_LEGACY: u32 = 0x4C4B_4250;

/// Bumped with the identifier: a manifest carrying `REGOMBKP` is v2.
const BACKUP_MANIFEST_VERSION: u32 = 2;
const BACKUP_MANIFEST_VERSION_LEGACY: u32 = 1;

impl BackupEngine {
    /// Open or create a backup repository at `backup_dir`.
    pub fn open<P: AsRef<Path>>(backup_dir: P) -> Result<Self> {
        let root = backup_dir.as_ref().to_path_buf();
        let meta_dir = root.join("meta");
        let shared_dir = root.join("shared");
        fs::create_dir_all(&meta_dir).map_err(Error::from)?;
        fs::create_dir_all(&shared_dir).map_err(Error::from)?;
        durability::sync_parent_dir(&root).map_err(Error::from)?;
        durability::sync_dir(&root).map_err(Error::from)?;
        durability::sync_dir(&meta_dir).map_err(Error::from)?;
        durability::sync_dir(&shared_dir).map_err(Error::from)?;
        Ok(Self {
            root,
            meta_dir,
            shared_dir,
        })
    }

    /// Root directory of this backup repository.
    pub fn path(&self) -> &Path {
        &self.root
    }

    /// Create a new backup of `db`, deduping against any files
    /// already present in the shared pool.
    pub fn create_backup(&mut self, db: &Db) -> Result<BackupId> {
        let id = BackupId(self.next_backup_id()?);
        let created_at_unix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        // Take the snapshot in a limited scope so its held
        // compaction lock is released before we touch the backup
        // metadata directory. `checkpoint_capture` holds the
        // engine's compaction lock, which pins the captured file
        // set against concurrent unlink while we hash + copy.
        let (files, next_file_id, last_seq) = {
            let snapshot = db.engine().checkpoint_capture().map_err(Error::from)?;

            let mut files = Vec::new();
            for (level_idx, level) in snapshot.version.levels.iter().enumerate() {
                for file in level {
                    let src = snapshot
                        .sst_dir
                        .join(CheckpointSnapshot::sst_filename(file.meta.file_id));
                    let hash = hash_file(&src).map_err(Error::from)?;
                    let shared_name = shared_filename(hash);
                    let shared_path = self.shared_dir.join(&shared_name);
                    ensure_shared_file(&src, &shared_path, hash, file.meta.file_size)
                        .map_err(Error::from)?;
                    files.push(BackupFileEntry {
                        level: level_idx as u32,
                        file_id: file.meta.file_id,
                        file_size: file.meta.file_size,
                        hash,
                        smallest_key: file.meta.smallest_key.clone(),
                        largest_key: file.meta.largest_key.clone(),
                        num_entries: file.meta.num_entries,
                    });
                }
            }
            (
                files,
                snapshot.version.next_file_id,
                snapshot.version.last_seq,
            )
            // snapshot drops here, releasing the compaction lock.
        };

        let manifest = BackupManifest {
            created_at_unix,
            files,
            next_file_id,
            last_seq,
        };
        let bytes = encode_manifest(&manifest);
        let manifest_path = self.meta_dir.join(backup_filename(id.0));
        atomic_write(&manifest_path, &bytes).map_err(Error::from)?;
        Ok(id)
    }

    /// Return a summary of every backup currently stored. Ordered
    /// by backup id (creation order).
    pub fn list_backups(&self) -> Vec<BackupInfo> {
        let mut out = Vec::new();
        let Ok(entries) = fs::read_dir(&self.meta_dir) else {
            return out;
        };
        let mut ids: Vec<u64> = entries
            .filter_map(|e| e.ok())
            .filter_map(|e| parse_backup_id(&e.file_name().to_string_lossy()))
            .collect();
        ids.sort_unstable();
        for id in ids {
            let path = self.meta_dir.join(backup_filename(id));
            let Ok(bytes) = fs::read(&path) else { continue };
            let Ok(manifest) = decode_manifest(&bytes) else {
                continue;
            };
            let bytes_sum: u64 = manifest.files.iter().map(|f| f.file_size).sum();
            out.push(BackupInfo {
                id: BackupId(id),
                created_at_unix: manifest.created_at_unix,
                file_count: manifest.files.len(),
                bytes: bytes_sum,
            });
        }
        out
    }

    /// Restore `backup_id` into `target_dir`. The target directory
    /// is created if it does not exist and must be empty (or
    /// contain only empty `sst/`/`wal/` subdirectories). The
    /// resulting directory opens cleanly as a new [`Db`].
    pub fn restore<P: AsRef<Path>>(&self, backup_id: BackupId, target_dir: P) -> Result<()> {
        let manifest = self.read_manifest(backup_id)?;
        let target_dir = target_dir.as_ref();
        let target_sst = target_dir.join("sst");
        let target_wal = target_dir.join("wal");
        fs::create_dir_all(&target_sst).map_err(Error::from)?;
        fs::create_dir_all(&target_wal).map_err(Error::from)?;
        durability::sync_parent_dir(target_dir).map_err(Error::from)?;
        durability::sync_dir(target_dir).map_err(Error::from)?;
        durability::sync_dir(&target_sst).map_err(Error::from)?;
        durability::sync_dir(&target_wal).map_err(Error::from)?;

        for f in &manifest.files {
            let src = self.shared_dir.join(shared_filename(f.hash));
            verify_shared_file(&src, f.hash, f.file_size).map_err(Error::from)?;
            let dst = target_sst.join(CheckpointSnapshot::sst_filename(f.file_id));
            copy_file_atomic(&src, &dst).map_err(Error::from)?;
        }

        let manifest_bytes = encode_engine_manifest(&manifest);
        let manifest_path = target_dir.join("MANIFEST");
        atomic_write(&manifest_path, &manifest_bytes).map_err(Error::from)?;
        Ok(())
    }

    /// Delete `backup_id`. Shared files whose reference count drops
    /// to zero are removed from disk.
    pub fn delete_backup(&mut self, backup_id: BackupId) -> Result<()> {
        let path = self.meta_dir.join(backup_filename(backup_id.0));
        if !path.exists() {
            return Ok(());
        }
        let manifest = self.read_manifest(backup_id)?;
        durability::remove_file_and_sync_parent(&path).map_err(Error::from)?;
        self.gc_shared(&manifest)?;
        Ok(())
    }

    /// Delete every backup except the `keep` most recent.
    pub fn purge_old_backups(&mut self, keep: usize) -> Result<()> {
        let infos = self.list_backups();
        if infos.len() <= keep {
            return Ok(());
        }
        let to_remove = infos.len() - keep;
        for info in infos.into_iter().take(to_remove) {
            self.delete_backup(info.id)?;
        }
        Ok(())
    }

    fn gc_shared(&self, removed: &BackupManifest) -> Result<()> {
        let mut still_referenced = std::collections::HashSet::new();
        let Ok(entries) = fs::read_dir(&self.meta_dir) else {
            return Ok(());
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            if parse_backup_id(&name.to_string_lossy()).is_none() {
                continue;
            }
            if let Ok(bytes) = fs::read(entry.path())
                && let Ok(m) = decode_manifest(&bytes)
            {
                for f in m.files {
                    still_referenced.insert(f.hash);
                }
            }
        }
        for f in &removed.files {
            if !still_referenced.contains(&f.hash) {
                let p = self.shared_dir.join(shared_filename(f.hash));
                match durability::remove_file_and_sync_parent(&p) {
                    Ok(()) => {}
                    Err(e) if e.kind() == io::ErrorKind::NotFound => {}
                    Err(e) => return Err(Error::from(e)),
                }
            }
        }
        Ok(())
    }

    fn read_manifest(&self, id: BackupId) -> Result<BackupManifest> {
        let path = self.meta_dir.join(backup_filename(id.0));
        let bytes = fs::read(&path).map_err(Error::from)?;
        decode_manifest(&bytes).map_err(Error::from)
    }

    fn next_backup_id(&self) -> Result<u64> {
        let mut max_id = 0u64;
        for entry in fs::read_dir(&self.meta_dir).map_err(Error::from)? {
            let entry = entry.map_err(Error::from)?;
            if let Some(id) = parse_backup_id(&entry.file_name().to_string_lossy())
                && id > max_id
            {
                max_id = id;
            }
        }
        Ok(max_id + 1)
    }
}

fn hash_file(path: &Path) -> io::Result<u128> {
    let mut f = File::open(path)?;
    checksum::backup_shared_file(&mut f)
}

fn ensure_shared_file(
    src: &Path,
    dst: &Path,
    expected_hash: u128,
    expected_size: u64,
) -> io::Result<()> {
    match verify_shared_file(dst, expected_hash, expected_size) {
        Ok(()) => return Ok(()),
        Err(e) if e.kind() == io::ErrorKind::NotFound => {}
        Err(e) if e.kind() == io::ErrorKind::InvalidData => {
            if dst.is_dir() {
                return Err(e);
            }
        }
        Err(e) => return Err(e),
    }

    copy_file_atomic(src, dst)?;
    verify_shared_file(dst, expected_hash, expected_size)
}

fn verify_shared_file(path: &Path, expected_hash: u128, expected_size: u64) -> io::Result<()> {
    let meta = fs::metadata(path)?;
    if !meta.is_file() {
        return Err(invalid_data(format!(
            "backup shared object {} is not a regular file",
            path.display()
        )));
    }
    if meta.len() != expected_size {
        return Err(invalid_data(format!(
            "backup shared object {} has size {}, expected {expected_size}",
            path.display(),
            meta.len()
        )));
    }
    let actual_hash = hash_file(path)?;
    if actual_hash != expected_hash {
        return Err(invalid_data(format!(
            "backup shared object {} content id mismatch",
            path.display()
        )));
    }
    Ok(())
}

fn copy_file_atomic(src: &Path, dst: &Path) -> io::Result<()> {
    let tmp = dst.with_extension("tmp");
    {
        let mut input = File::open(src)?;
        let mut output = File::create(&tmp)?;
        io::copy(&mut input, &mut output)?;
        output.sync_all()?;
    }
    fs::rename(&tmp, dst)?;
    durability::sync_parent_dir(dst)?;
    Ok(())
}

fn atomic_write(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let tmp = path.with_extension("tmp");
    {
        let mut f = File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }
    fs::rename(&tmp, path)?;
    durability::sync_parent_dir(path)?;
    Ok(())
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn shared_filename(hash: u128) -> String {
    format!("{:032x}.sst", hash)
}

fn backup_filename(id: u64) -> String {
    format!("{:06}.backup", id)
}

fn parse_backup_id(name: &str) -> Option<u64> {
    let stem = name.strip_suffix(".backup")?;
    stem.parse::<u64>().ok()
}

fn encode_manifest(m: &BackupManifest) -> Vec<u8> {
    let mut body = Vec::new();
    body.extend_from_slice(&BACKUP_MANIFEST_MAGIC);
    body.extend_from_slice(&BACKUP_MANIFEST_VERSION.to_le_bytes());
    body.extend_from_slice(&m.created_at_unix.to_le_bytes());
    body.extend_from_slice(&m.next_file_id.to_le_bytes());
    body.extend_from_slice(&m.last_seq.to_le_bytes());
    body.extend_from_slice(&(m.files.len() as u32).to_le_bytes());
    for f in &m.files {
        body.extend_from_slice(&f.level.to_le_bytes());
        body.extend_from_slice(&f.file_id.to_le_bytes());
        body.extend_from_slice(&f.file_size.to_le_bytes());
        body.extend_from_slice(&f.num_entries.to_le_bytes());
        body.extend_from_slice(&f.hash.to_le_bytes());
        body.extend_from_slice(&(f.smallest_key.len() as u32).to_le_bytes());
        body.extend_from_slice(&f.smallest_key);
        body.extend_from_slice(&(f.largest_key.len() as u32).to_le_bytes());
        body.extend_from_slice(&f.largest_key);
    }
    let checksum = checksum::backup_manifest(&body);
    let mut out = Vec::with_capacity(body.len() + 8);
    out.extend_from_slice(&body);
    out.extend_from_slice(&checksum.to_le_bytes());
    out
}

fn decode_manifest(data: &[u8]) -> io::Result<BackupManifest> {
    if data.len() < 8 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "short backup"));
    }
    let body_len = data.len() - 8;
    let body = &data[..body_len];
    let stored_cksum = u64::from_le_bytes(data[body_len..].try_into().unwrap());
    if checksum::backup_manifest(body) != stored_cksum
        && checksum::legacy_payload_u64(body) != stored_cksum
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "backup manifest checksum mismatch",
        ));
    }
    let mut p = 0usize;
    // `REGOMBKP` is eight bytes; the identifier earlier builds wrote is
    // four. Reading eight and falling back keeps an existing repository
    // restorable instead of stranding it.
    let expected_version = if body.len() >= BACKUP_MANIFEST_MAGIC.len()
        && body[..BACKUP_MANIFEST_MAGIC.len()] == BACKUP_MANIFEST_MAGIC
    {
        p = BACKUP_MANIFEST_MAGIC.len();
        BACKUP_MANIFEST_VERSION
    } else if read_u32(body, &mut p)? == BACKUP_MANIFEST_MAGIC_LEGACY {
        BACKUP_MANIFEST_VERSION_LEGACY
    } else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "backup manifest bad magic",
        ));
    };
    let version = read_u32(body, &mut p)?;
    if version != expected_version {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported backup manifest version {version}"),
        ));
    }
    let created_at_unix = read_u64(body, &mut p)?;
    let next_file_id = read_u64(body, &mut p)?;
    let last_seq = read_u64(body, &mut p)?;
    let count = read_u32(body, &mut p)? as usize;
    let mut files = Vec::with_capacity(count);
    for _ in 0..count {
        let level = read_u32(body, &mut p)?;
        let file_id = read_u64(body, &mut p)?;
        let file_size = read_u64(body, &mut p)?;
        let num_entries = read_u64(body, &mut p)?;
        let hash = read_u128(body, &mut p)?;
        let smallest_key = read_var_bytes(body, &mut p)?;
        let largest_key = read_var_bytes(body, &mut p)?;
        files.push(BackupFileEntry {
            level,
            file_id,
            file_size,
            num_entries,
            hash,
            smallest_key,
            largest_key,
        });
    }
    Ok(BackupManifest {
        created_at_unix,
        files,
        next_file_id,
        last_seq,
    })
}

fn read_u32(data: &[u8], p: &mut usize) -> io::Result<u32> {
    if *p + 4 > data.len() {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "short"));
    }
    let v = u32::from_le_bytes(data[*p..*p + 4].try_into().unwrap());
    *p += 4;
    Ok(v)
}

fn read_u64(data: &[u8], p: &mut usize) -> io::Result<u64> {
    if *p + 8 > data.len() {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "short"));
    }
    let v = u64::from_le_bytes(data[*p..*p + 8].try_into().unwrap());
    *p += 8;
    Ok(v)
}

fn read_u128(data: &[u8], p: &mut usize) -> io::Result<u128> {
    if *p + 16 > data.len() {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "short"));
    }
    let v = u128::from_le_bytes(data[*p..*p + 16].try_into().unwrap());
    *p += 16;
    Ok(v)
}

fn read_var_bytes(data: &[u8], p: &mut usize) -> io::Result<Vec<u8>> {
    let len = read_u32(data, p)? as usize;
    if *p + len > data.len() {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "short"));
    }
    let v = data[*p..*p + len].to_vec();
    *p += len;
    Ok(v)
}

/// Encode a restored backup as an engine-compatible MANIFEST stream.
///
/// Mirrors the record layout produced by
/// `VersionSet::encode_records` / `ManifestRecord::encode` in
/// [`crate::engine::manifest`]. Each record is `[len u32][body][cksum u32]`
/// with the same accidental-corruption checksum used by the engine manifest.
fn encode_engine_manifest(m: &BackupManifest) -> Vec<u8> {
    const TAG_ADD_FILE: u8 = 1;
    const TAG_LAST_SEQ: u8 = 3;
    const TAG_NEXT_FILE_ID: u8 = 4;

    let mut records: Vec<Vec<u8>> = Vec::new();
    {
        let mut r = Vec::new();
        r.push(TAG_NEXT_FILE_ID);
        r.extend_from_slice(&m.next_file_id.to_le_bytes());
        records.push(r);
    }
    {
        let mut r = Vec::new();
        r.push(TAG_LAST_SEQ);
        r.extend_from_slice(&m.last_seq.to_le_bytes());
        records.push(r);
    }
    for f in &m.files {
        let mut r = Vec::new();
        r.push(TAG_ADD_FILE);
        r.extend_from_slice(&f.level.to_le_bytes());
        r.extend_from_slice(&f.file_id.to_le_bytes());
        r.extend_from_slice(&(f.smallest_key.len() as u32).to_le_bytes());
        r.extend_from_slice(&f.smallest_key);
        r.extend_from_slice(&(f.largest_key.len() as u32).to_le_bytes());
        r.extend_from_slice(&f.largest_key);
        r.extend_from_slice(&f.file_size.to_le_bytes());
        r.extend_from_slice(&f.num_entries.to_le_bytes());
        records.push(r);
    }

    let mut out = Vec::new();
    for record_buf in &records {
        let len = record_buf.len() as u32;
        let checksum = checksum::manifest_record(len, record_buf);
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(record_buf);
        out.extend_from_slice(&checksum.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Options;
    use tempfile::TempDir;

    fn tiny_flush_opts() -> Options {
        Options {
            write_buffer_size: 4 * 1024,
            ..Options::default()
        }
    }

    fn populate(db: &Db, prefix: &str, n: usize) {
        let filler = vec![0u8; 256];
        for i in 0..n {
            let k = format!("{}_{:05}", prefix, i);
            let mut v = filler.clone();
            v.extend_from_slice(k.as_bytes());
            db.put(k.as_bytes(), &v).unwrap();
        }
    }

    fn assert_has(db: &Db, prefix: &str, n: usize) {
        let filler = vec![0u8; 256];
        for i in 0..n {
            let k = format!("{}_{:05}", prefix, i);
            let mut expected = filler.clone();
            expected.extend_from_slice(k.as_bytes());
            assert_eq!(db.get(k.as_bytes()).unwrap(), Some(expected));
        }
    }

    fn shared_bytes(dir: &Path) -> u64 {
        let shared = dir.join("shared");
        let mut total = 0u64;
        if let Ok(entries) = fs::read_dir(&shared) {
            for entry in entries.flatten() {
                if let Ok(md) = entry.metadata() {
                    total += md.len();
                }
            }
        }
        total
    }

    fn shared_count(dir: &Path) -> usize {
        let shared = dir.join("shared");
        fs::read_dir(&shared)
            .map(|it| it.flatten().count())
            .unwrap_or(0)
    }

    fn shared_file_paths(dir: &Path) -> Vec<PathBuf> {
        let shared = dir.join("shared");
        let mut paths: Vec<_> = fs::read_dir(&shared)
            .unwrap()
            .flatten()
            .map(|entry| entry.path())
            .collect();
        paths.sort();
        paths
    }

    fn corrupt_file_same_size(path: &Path) {
        let mut bytes = fs::read(path).unwrap();
        assert!(!bytes.is_empty());
        bytes[0] ^= 0xFF;
        fs::write(path, bytes).unwrap();
    }

    #[test]
    fn backup_and_restore_roundtrip() {
        let src_dir = TempDir::new().unwrap();
        let bkp_dir = TempDir::new().unwrap();
        let tgt_dir = TempDir::new().unwrap();

        let db = Db::open(src_dir.path(), tiny_flush_opts()).unwrap();
        populate(&db, "k", 300);

        let mut engine = BackupEngine::open(bkp_dir.path()).unwrap();
        let id = engine.create_backup(&db).unwrap();
        let infos = engine.list_backups();
        assert_eq!(infos.len(), 1);
        assert_eq!(infos[0].id, id);
        assert!(infos[0].file_count >= 1);

        engine.restore(id, tgt_dir.path()).unwrap();
        drop(db);

        let reopened = Db::open(tgt_dir.path(), Options::default()).unwrap();
        assert_has(&reopened, "k", 300);
    }

    #[test]
    fn incremental_backup_dedupes() {
        let src_dir = TempDir::new().unwrap();
        let bkp_dir = TempDir::new().unwrap();
        let db = Db::open(src_dir.path(), tiny_flush_opts()).unwrap();
        populate(&db, "x", 400);
        db.compact_range(None, None).unwrap();

        let mut engine = BackupEngine::open(bkp_dir.path()).unwrap();
        let _id1 = engine.create_backup(&db).unwrap();
        let shared_bytes_1 = shared_bytes(bkp_dir.path());
        let shared_count_1 = shared_count(bkp_dir.path());

        // No writes between backups - second backup must add no
        // shared files.
        let _id2 = engine.create_backup(&db).unwrap();
        let shared_bytes_2 = shared_bytes(bkp_dir.path());
        let shared_count_2 = shared_count(bkp_dir.path());

        assert_eq!(shared_bytes_1, shared_bytes_2);
        assert_eq!(shared_count_1, shared_count_2);
        assert_eq!(engine.list_backups().len(), 2);
    }

    #[test]
    fn create_backup_replaces_corrupt_existing_shared_object() {
        let src_dir = TempDir::new().unwrap();
        let bkp_dir = TempDir::new().unwrap();
        let tgt_dir = TempDir::new().unwrap();
        let db = Db::open(src_dir.path(), tiny_flush_opts()).unwrap();
        populate(&db, "r", 400);
        db.compact_range(None, None).unwrap();

        let mut engine = BackupEngine::open(bkp_dir.path()).unwrap();
        let _id1 = engine.create_backup(&db).unwrap();
        let shared_files = shared_file_paths(bkp_dir.path());
        assert!(!shared_files.is_empty());
        let original_sizes: Vec<u64> = shared_files
            .iter()
            .map(|path| fs::metadata(path).unwrap().len())
            .collect();
        for path in &shared_files {
            corrupt_file_same_size(path);
        }

        let id2 = engine.create_backup(&db).unwrap();

        for (path, original_size) in shared_files.iter().zip(original_sizes) {
            assert_eq!(fs::metadata(path).unwrap().len(), original_size);
        }
        engine.restore(id2, tgt_dir.path()).unwrap();
        drop(db);
        let restored = Db::open(tgt_dir.path(), Options::default()).unwrap();
        assert_has(&restored, "r", 400);
    }

    #[test]
    fn restore_rejects_corrupt_shared_object() {
        let src_dir = TempDir::new().unwrap();
        let bkp_dir = TempDir::new().unwrap();
        let tgt_dir = TempDir::new().unwrap();
        let db = Db::open(src_dir.path(), tiny_flush_opts()).unwrap();
        populate(&db, "bad", 300);
        db.compact_range(None, None).unwrap();

        let mut engine = BackupEngine::open(bkp_dir.path()).unwrap();
        let id = engine.create_backup(&db).unwrap();
        let shared_files = shared_file_paths(bkp_dir.path());
        assert!(!shared_files.is_empty());
        corrupt_file_same_size(&shared_files[0]);

        let kind = match engine.restore(id, tgt_dir.path()) {
            Err(Error::Corruption(e)) => e.kind(),
            Err(e) => panic!("expected corruption error, got {e:?}"),
            Ok(()) => panic!("expected restore to reject corrupt shared object"),
        };
        assert_eq!(kind, io::ErrorKind::InvalidData);
    }

    #[test]
    fn delete_backup_gcs_unreferenced_files() {
        let src_dir = TempDir::new().unwrap();
        let bkp_dir = TempDir::new().unwrap();
        let db = Db::open(src_dir.path(), tiny_flush_opts()).unwrap();
        populate(&db, "a", 200);
        db.compact_range(None, None).unwrap();

        let mut engine = BackupEngine::open(bkp_dir.path()).unwrap();
        let id1 = engine.create_backup(&db).unwrap();
        let shared_after_1 = shared_count(bkp_dir.path());
        assert!(shared_after_1 >= 1);

        // New data that doesn't overlap prior SSTs.
        populate(&db, "z", 200);
        db.compact_range(None, None).unwrap();
        let id2 = engine.create_backup(&db).unwrap();
        let shared_after_2 = shared_count(bkp_dir.path());

        // Remove the first backup - any file it held that backup 2
        // does not also reference should be gone.
        engine.delete_backup(id1).unwrap();
        let shared_after_delete = shared_count(bkp_dir.path());
        assert!(shared_after_delete <= shared_after_2);
        assert_eq!(engine.list_backups().len(), 1);

        // backup 2 still restores cleanly.
        let tgt_dir = TempDir::new().unwrap();
        engine.restore(id2, tgt_dir.path()).unwrap();
        drop(db);
        let reopened = Db::open(tgt_dir.path(), Options::default()).unwrap();
        assert_has(&reopened, "z", 200);
    }

    #[test]
    fn delete_only_backup_removes_all_shared() {
        let src_dir = TempDir::new().unwrap();
        let bkp_dir = TempDir::new().unwrap();
        let db = Db::open(src_dir.path(), tiny_flush_opts()).unwrap();
        populate(&db, "q", 300);
        db.compact_range(None, None).unwrap();

        let mut engine = BackupEngine::open(bkp_dir.path()).unwrap();
        let id = engine.create_backup(&db).unwrap();
        assert!(shared_count(bkp_dir.path()) > 0);

        engine.delete_backup(id).unwrap();
        assert_eq!(shared_count(bkp_dir.path()), 0);
        assert_eq!(engine.list_backups().len(), 0);
    }

    #[test]
    fn purge_keeps_newest() {
        let src_dir = TempDir::new().unwrap();
        let bkp_dir = TempDir::new().unwrap();
        let db = Db::open(src_dir.path(), tiny_flush_opts()).unwrap();

        populate(&db, "g1", 100);
        db.compact_range(None, None).unwrap();
        let mut engine = BackupEngine::open(bkp_dir.path()).unwrap();
        let _b1 = engine.create_backup(&db).unwrap();

        populate(&db, "g2", 100);
        db.compact_range(None, None).unwrap();
        let _b2 = engine.create_backup(&db).unwrap();

        populate(&db, "g3", 100);
        db.compact_range(None, None).unwrap();
        let b3 = engine.create_backup(&db).unwrap();

        engine.purge_old_backups(1).unwrap();
        let infos = engine.list_backups();
        assert_eq!(infos.len(), 1);
        assert_eq!(infos[0].id, b3);
    }

    #[test]
    fn restore_independent_of_source() {
        let src_dir = TempDir::new().unwrap();
        let bkp_dir = TempDir::new().unwrap();
        let tgt_dir = TempDir::new().unwrap();

        let db = Db::open(src_dir.path(), tiny_flush_opts()).unwrap();
        populate(&db, "ind", 250);

        let mut engine = BackupEngine::open(bkp_dir.path()).unwrap();
        let id = engine.create_backup(&db).unwrap();

        db.close().unwrap();
        drop(db);
        fs::remove_dir_all(src_dir.path()).unwrap();

        engine.restore(id, tgt_dir.path()).unwrap();
        let reopened = Db::open(tgt_dir.path(), Options::default()).unwrap();
        assert_has(&reopened, "ind", 250);
    }
}
