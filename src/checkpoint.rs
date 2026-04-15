//! Hard-link checkpoints of a live [`Db`].
//!
//! A checkpoint is a directory that reflects the contents of the
//! source database at a single well-defined point in time. It can be
//! opened with [`Db::open`] as an independent database, copied
//! off-host, or used as the seed for a test harness. The checkpoint
//! is produced by hard-linking every live SSTable into a target
//! `sst/` directory and copying the manifest — the data itself is
//! never duplicated on disk, so creating a checkpoint is O(file
//! count) and costs a few inodes.
//!
//! # Same-filesystem requirement
//!
//! Hard links cannot cross filesystems. If the target directory is
//! on a different device, [`Checkpoint::create`] fails with the
//! underlying I/O error rather than silently falling back to a
//! byte-for-byte copy.
//!
//! # Atomicity
//!
//! The implementation briefly holds the engine write-lock to flush
//! the active memtable into an L0 SSTable and compact the live
//! manifest. The lock is released before any files are linked.
//! Because the captured version pins every referenced SSTable
//! against concurrent compaction reclamation, writers running in
//! parallel with [`Checkpoint::create`] cannot race a file unlink or
//! tear the checkpoint.

use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::Path;

use crate::engine::CheckpointSnapshot;
use crate::{Db, Error, Result};

/// A handle to a [`Db`] prepared for checkpointing. The handle itself
/// captures no engine state — the actual flush, version snapshot,
/// and filesystem work all happen inside [`Checkpoint::create`] under
/// a single held compaction lock, so dropping the `Checkpoint`
/// without calling `create` is free.
pub struct Checkpoint<'db> {
    db: &'db Db,
}

impl<'db> Checkpoint<'db> {
    /// Prepare to checkpoint `db`. Does no filesystem work itself.
    pub fn new(db: &'db Db) -> Result<Self> {
        Ok(Self { db })
    }

    /// Materialize the checkpoint in `target_dir`.
    ///
    /// Atomically:
    /// 1. Flushes the active memtable into an L0 SSTable.
    /// 2. Compacts the manifest so the on-disk form is a single
    ///    rewrite of the current version.
    /// 3. Captures an `Arc<Version>` that pins every referenced file.
    /// 4. Hard-links every captured SSTable into `target_dir/sst/`.
    /// 5. Copies the compacted manifest into `target_dir/MANIFEST`.
    ///
    /// The engine's compaction lock is held for steps 2–5 so no
    /// concurrent compaction can unlink a referenced file mid-copy.
    /// It is released as soon as `create` returns — so dropping the
    /// [`Checkpoint`] or the source [`Db`] after the call cannot
    /// deadlock.
    ///
    /// The target directory is created (recursively) if it does not
    /// exist. Fails if the target already contains a non-empty
    /// `sst/` directory, if a hard-link attempt crosses filesystems,
    /// or if the target path cannot be written.
    pub fn create<P: AsRef<Path>>(&self, target_dir: P) -> Result<()> {
        let target_dir = target_dir.as_ref();
        let target_sst = target_dir.join("sst");
        let target_wal = target_dir.join("wal");

        fs::create_dir_all(&target_sst).map_err(Error::Io)?;
        fs::create_dir_all(&target_wal).map_err(Error::Io)?;

        if target_sst.read_dir().map_err(Error::Io)?.next().is_some() {
            return Err(Error::Io(std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                "checkpoint target sst directory is not empty",
            )));
        }

        // `checkpoint_capture` holds the compaction lock for the
        // lifetime of the returned snapshot, so the captured file
        // set cannot be unlinked while we hard-link it. The
        // snapshot is a local — it drops before `create` returns,
        // releasing the lock without outlasting this function.
        // Keeping the lock scoped this way is what lets a caller
        // safely call `db.close()` or `drop(db)` after `create`.
        let snapshot = self.db.engine().checkpoint_capture().map_err(Error::Io)?;

        for level in &snapshot.version.levels {
            for file in level {
                let name = CheckpointSnapshot::sst_filename(file.meta.file_id);
                let src = snapshot.sst_dir.join(&name);
                let dst = target_sst.join(&name);
                fs::hard_link(&src, &dst).map_err(Error::Io)?;
            }
        }

        let target_manifest = target_dir.join("MANIFEST");
        copy_truncated(
            &snapshot.manifest_path,
            &target_manifest,
            snapshot.manifest_len,
        )
        .map_err(Error::Io)?;

        Ok(())
    }
}

/// Copy exactly `len` bytes from `src` to `dst`, creating `dst`
/// anew. Used to stage a point-in-time manifest snapshot without
/// picking up `AddFile` records appended by concurrent flushes.
fn copy_truncated(src: &Path, dst: &Path, len: u64) -> std::io::Result<()> {
    let mut reader = File::open(src)?;
    let mut writer = File::create(dst)?;
    let mut remaining = len;
    let mut buf = [0u8; 16 * 1024];
    while remaining > 0 {
        let want = remaining.min(buf.len() as u64) as usize;
        let n = reader.read(&mut buf[..want])?;
        if n == 0 {
            break;
        }
        writer.write_all(&buf[..n])?;
        remaining -= n as u64;
    }
    writer.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Options, WriteBatch};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::thread;
    use tempfile::TempDir;

    fn tiny_flush_opts() -> Options {
        Options {
            write_buffer_size: 4 * 1024,
            ..Options::default()
        }
    }

    fn force_flush(db: &Db, tag: &str) {
        let payload = vec![0u8; 512];
        for i in 0..32 {
            let key = format!("__flush_{}_{:04}", tag, i);
            db.put(key.as_bytes(), &payload).unwrap();
        }
    }

    #[test]
    fn checkpoint_empty_db() {
        let src_dir = TempDir::new().unwrap();
        let tgt_dir = TempDir::new().unwrap();
        let db = Db::open(src_dir.path(), Options::default()).unwrap();

        let cp = Checkpoint::new(&db).unwrap();
        cp.create(tgt_dir.path()).unwrap();
        drop(db);

        let reopened = Db::open(tgt_dir.path(), Options::default()).unwrap();
        assert_eq!(reopened.get(b"missing").unwrap(), None);
    }

    #[test]
    fn checkpoint_memtable_only() {
        let src_dir = TempDir::new().unwrap();
        let tgt_dir = TempDir::new().unwrap();
        let db = Db::open(src_dir.path(), Options::default()).unwrap();

        db.put(b"a", b"1").unwrap();
        db.put(b"b", b"2").unwrap();
        db.put(b"c", b"3").unwrap();

        let cp = Checkpoint::new(&db).unwrap();
        cp.create(tgt_dir.path()).unwrap();

        // Source DB stays alive through the reopen.
        let reopened = Db::open(tgt_dir.path(), Options::default()).unwrap();
        assert_eq!(reopened.get(b"a").unwrap(), Some(b"1".to_vec()));
        assert_eq!(reopened.get(b"b").unwrap(), Some(b"2".to_vec()));
        assert_eq!(reopened.get(b"c").unwrap(), Some(b"3".to_vec()));

        drop(reopened);
        drop(db);
    }

    #[test]
    fn checkpoint_flushed_and_compacted() {
        let src_dir = TempDir::new().unwrap();
        let tgt_dir = TempDir::new().unwrap();
        let db = Db::open(src_dir.path(), tiny_flush_opts()).unwrap();

        for i in 0..200 {
            let k = format!("key_{:05}", i);
            db.put(k.as_bytes(), k.as_bytes()).unwrap();
        }
        force_flush(&db, "a");
        db.compact_range(None, None).unwrap();

        // Add a few memtable-only writes after the compaction to
        // exercise the flush-before-capture path.
        db.put(b"post_1", b"pv1").unwrap();
        db.put(b"post_2", b"pv2").unwrap();

        let cp = Checkpoint::new(&db).unwrap();
        cp.create(tgt_dir.path()).unwrap();

        let reopened = Db::open(tgt_dir.path(), Options::default()).unwrap();
        for i in 0..200 {
            let k = format!("key_{:05}", i);
            assert_eq!(reopened.get(k.as_bytes()).unwrap(), Some(k.into_bytes()));
        }
        assert_eq!(reopened.get(b"post_1").unwrap(), Some(b"pv1".to_vec()));
        assert_eq!(reopened.get(b"post_2").unwrap(), Some(b"pv2".to_vec()));
    }

    #[test]
    fn checkpoint_with_concurrent_writer() {
        let src_dir = TempDir::new().unwrap();
        let tgt_dir = TempDir::new().unwrap();
        let db = Arc::new(Db::open(src_dir.path(), tiny_flush_opts()).unwrap());

        for i in 0..50 {
            let k = format!("seed_{:03}", i);
            db.put(k.as_bytes(), k.as_bytes()).unwrap();
        }

        let stop = Arc::new(AtomicBool::new(false));
        let writer_db = Arc::clone(&db);
        let writer_stop = Arc::clone(&stop);
        let writer = thread::spawn(move || {
            let mut i = 0u64;
            while !writer_stop.load(Ordering::Relaxed) {
                let mut batch = WriteBatch::new();
                let k = format!("live_{:06}", i);
                batch.put(k.as_bytes(), k.as_bytes());
                writer_db.write(batch).unwrap();
                i += 1;
            }
            i
        });

        // Let the writer make some progress, then take a checkpoint.
        for _ in 0..5 {
            let cp = Checkpoint::new(&db).unwrap();
            cp.create(tgt_dir.path()).unwrap();
            // Reset target for the next iteration.
            std::fs::remove_dir_all(tgt_dir.path()).unwrap();
            std::fs::create_dir_all(tgt_dir.path()).unwrap();
        }

        // Final checkpoint we'll actually open.
        let cp = Checkpoint::new(&db).unwrap();
        cp.create(tgt_dir.path()).unwrap();

        stop.store(true, Ordering::Relaxed);
        let _total_writes = writer.join().unwrap();

        // Source survives and is still usable.
        db.put(b"after_checkpoint", b"ok").unwrap();
        assert_eq!(db.get(b"after_checkpoint").unwrap(), Some(b"ok".to_vec()));

        // Checkpoint opens cleanly and contains the seed data it
        // definitely saw. It must NOT be corrupted.
        let reopened = Db::open(tgt_dir.path(), Options::default()).unwrap();
        for i in 0..50 {
            let k = format!("seed_{:03}", i);
            assert_eq!(reopened.get(k.as_bytes()).unwrap(), Some(k.into_bytes()));
        }
    }

    #[test]
    fn source_can_be_dropped_after_checkpoint() {
        let src_dir = TempDir::new().unwrap();
        let tgt_dir = TempDir::new().unwrap();
        let db = Db::open(src_dir.path(), tiny_flush_opts()).unwrap();

        for i in 0..100 {
            let k = format!("dur_{:04}", i);
            db.put(k.as_bytes(), k.as_bytes()).unwrap();
        }
        force_flush(&db, "x");

        let cp = Checkpoint::new(&db).unwrap();
        cp.create(tgt_dir.path()).unwrap();

        db.close().unwrap();
        drop(db);
        // Wipe the source to prove the checkpoint doesn't depend on
        // it post-creation — hard-linked inodes outlive the original
        // directory entry.
        std::fs::remove_dir_all(src_dir.path()).unwrap();

        let reopened = Db::open(tgt_dir.path(), Options::default()).unwrap();
        for i in 0..100 {
            let k = format!("dur_{:04}", i);
            assert_eq!(reopened.get(k.as_bytes()).unwrap(), Some(k.into_bytes()));
        }
    }
}
