//! Hard-link checkpoints of a live [`Db`].
//!
//! A checkpoint is a directory that reflects the contents of the
//! source database at a single well-defined point in time. It can be
//! opened with [`Db::open`] as an independent database, copied
//! off-host, or used as the seed for a test harness. The checkpoint
//! is produced by hard-linking every live SSTable into a target
//! `sst/` directory and copying the manifest - the data itself is
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

use std::path::Path;

use crate::engine::CheckpointSnapshot;
use crate::env::{Env, WriteMode};
use crate::{Db, Error, Result};

/// A handle to a [`Db`] prepared for checkpointing. The handle itself
/// captures no engine state - the actual flush, version snapshot,
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
    /// The engine's compaction lock is held for steps 2-5 so no
    /// concurrent compaction can unlink a referenced file mid-copy.
    /// It is released as soon as `create` returns - so dropping the
    /// [`Checkpoint`] or the source [`Db`] after the call cannot
    /// deadlock.
    ///
    /// The target directory is created (recursively) if it does not
    /// exist. Fails if the target already contains a non-empty
    /// `sst/` directory, if a hard-link attempt crosses filesystems,
    /// or if the target path cannot be written.
    pub fn create<P: AsRef<Path>>(&self, target_dir: P) -> Result<()> {
        self.create_inner(target_dir.as_ref(), |_| {})
    }

    /// [`Checkpoint::create`] with a hook that runs after the capture
    /// and before anything is copied.
    ///
    /// That window is where a concurrent flush or a manifest rewrite
    /// would land, and it is far too narrow to hit reliably by running
    /// writers alongside. The hook makes it deterministic.
    #[cfg(test)]
    pub(crate) fn create_between<P: AsRef<Path>>(
        &self,
        target_dir: P,
        after_capture: impl FnOnce(&Db),
    ) -> Result<()> {
        self.create_inner(target_dir.as_ref(), after_capture)
    }

    fn create_inner(&self, target_dir: &Path, after_capture: impl FnOnce(&Db)) -> Result<()> {
        let target_sst = target_dir.join("sst");
        let target_wal = target_dir.join("wal");

        let env = self.db.engine().env();
        env.create_dir_all(&target_sst).map_err(Error::from)?;
        env.create_dir_all(&target_wal).map_err(Error::from)?;

        if !env.read_dir(&target_sst).map_err(Error::from)?.is_empty() {
            return Err(Error::Io(std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                "checkpoint target sst directory is not empty",
            )));
        }

        // `checkpoint_capture` holds the compaction lock for the
        // lifetime of the returned snapshot, so the captured file
        // set cannot be unlinked while we hard-link it. The
        // snapshot is a local - it drops before `create` returns,
        // releasing the lock without outlasting this function.
        // Keeping the lock scoped this way is what lets a caller
        // safely call `db.close()` or `drop(db)` after `create`.
        let snapshot = self.db.engine().checkpoint_capture().map_err(Error::from)?;
        after_capture(self.db);

        for level in &snapshot.version.levels {
            for file in level {
                let name = CheckpointSnapshot::sst_filename(file.meta.file_id);
                let src = snapshot.sst_dir.join(&name);
                let dst = target_sst.join(&name);
                // An environment without hard links copies the bytes
                // instead, which is slower but produces the same
                // checkpoint. `Capabilities::hard_link` is what says
                // which one happened.
                if env.capabilities().hard_link {
                    env.hard_link(&src, &dst).map_err(Error::from)?;
                } else {
                    let len = env.metadata(&src).map_err(Error::from)?.len;
                    copy_truncated(&**env, &src, &dst, len).map_err(Error::from)?;
                }
            }
        }

        // Written from the bytes captured under the version lock, not
        // re-read from the source. Re-reading would race a concurrent
        // flush appending records for files this checkpoint did not
        // hardlink, and a manifest rewrite replacing the file outright.
        let target_manifest = target_dir.join("MANIFEST");
        let mut manifest = env
            .open_write(&target_manifest, WriteMode::Truncate)
            .map_err(Error::from)?;
        manifest
            .write_all(&snapshot.manifest_bytes)
            .map_err(Error::from)?;
        manifest.sync_all().map_err(Error::from)?;

        Ok(())
    }
}

/// Copy exactly `len` bytes from `src` to `dst`, creating `dst`
/// anew. Used to stage a point-in-time manifest snapshot without
/// picking up `AddFile` records appended by concurrent flushes.
fn copy_truncated(env: &dyn Env, src: &Path, dst: &Path, len: u64) -> std::io::Result<()> {
    let reader = env.open_read(src)?;
    let mut writer = env.open_write(dst, WriteMode::Truncate)?;
    // A fixed 16 KiB window, so copying a multi-gigabyte SSTable
    // costs one buffer rather than its own size in memory.
    let mut buf = [0u8; 16 * 1024];
    let available = reader.len()?.min(len);
    let mut offset = 0u64;
    while offset < available {
        let want = (available - offset).min(buf.len() as u64) as usize;
        reader.read_exact_at(offset, &mut buf[..want])?;
        writer.write_all(&buf[..want])?;
        offset += want as u64;
    }
    writer.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Options, WriteBatch};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
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

    /// Options with no background worker at all.
    ///
    /// This is what makes the two tests below deterministic. With a
    /// worker running, a memtable that a checkpoint merely *sealed*
    /// usually gets flushed before the capture reads the version, and
    /// the bug hides; the data is only lost when the flush happens to
    /// lag, which is why it read as a flake under load. With no worker,
    /// nothing else can flush it, so sealing without draining loses the
    /// data every time.
    fn no_background_flush() -> Options {
        Options {
            max_background_compactions: 0,
            ..Default::default()
        }
    }

    /// A checkpoint captures the SSTables the current version names and
    /// copies no WAL, so anything still in a memtable has to be flushed
    /// before the capture, not merely sealed.
    #[test]
    fn a_checkpoint_captures_data_that_never_reached_an_sstable() {
        let src_dir = TempDir::new().unwrap();
        let tgt_dir = TempDir::new().unwrap();
        let db = Db::open(src_dir.path(), no_background_flush()).unwrap();

        // Far below the write buffer, so nothing flushes on its own.
        for i in 0..200u64 {
            let k = format!("unflushed_{i:04}");
            db.put(k.as_bytes(), k.as_bytes()).unwrap();
        }

        let cp = Checkpoint::new(&db).unwrap();
        cp.create(tgt_dir.path()).unwrap();

        let reopened = Db::open(tgt_dir.path(), no_background_flush()).unwrap();
        for i in 0..200u64 {
            let k = format!("unflushed_{i:04}");
            assert_eq!(
                reopened.get(k.as_bytes()).unwrap(),
                Some(k.clone().into_bytes()),
                "{k} was acknowledged before the checkpoint but is not in it"
            );
        }
    }

    /// The same property for range tombstones, which live beside the
    /// entries rather than in them and are dropped by a different path.
    #[test]
    fn a_checkpoint_captures_unflushed_range_deletes() {
        let src_dir = TempDir::new().unwrap();
        let tgt_dir = TempDir::new().unwrap();
        let db = Db::open(src_dir.path(), no_background_flush()).unwrap();

        for i in 0..50u64 {
            let k = format!("k_{i:04}");
            db.put(k.as_bytes(), b"v").unwrap();
        }
        db.delete_range(b"k_0010", b"k_0020").unwrap();

        let cp = Checkpoint::new(&db).unwrap();
        cp.create(tgt_dir.path()).unwrap();

        let reopened = Db::open(tgt_dir.path(), no_background_flush()).unwrap();
        for i in 0..50u64 {
            let k = format!("k_{i:04}");
            let deleted = (10..20).contains(&i);
            assert_eq!(
                reopened.get(k.as_bytes()).unwrap().is_none(),
                deleted,
                "{k}: range delete did not survive the checkpoint"
            );
        }
    }

    /// The postcondition the capture relies on, asserted directly rather
    /// than through a reopen: once `checkpoint_capture` returns, no
    /// memtable still holds data the captured version does not name.
    #[test]
    fn capturing_a_checkpoint_leaves_no_memtable_holding_data() {
        let dir = TempDir::new().unwrap();
        let db = Db::open(dir.path(), no_background_flush()).unwrap();
        for i in 0..200u64 {
            db.put(format!("k{i:04}").as_bytes(), b"v").unwrap();
        }

        let snapshot = db.engine().checkpoint_capture().unwrap();
        drop(snapshot);

        assert!(
            db.engine().memtables_hold_no_data(),
            "a memtable still held data after the capture, so the checkpoint \
             names SSTables that do not contain it"
        );
    }

    /// The manifest a checkpoint writes must describe exactly the
    /// SSTables it hardlinked.
    ///
    /// The capture holds the compaction lock, but a concurrent flush
    /// takes only the version lock, so it can append records naming
    /// files the checkpoint will not copy. Worse, the manifest is
    /// rewritten once it grows past a multiple of its canonical size, so
    /// the file can be replaced wholesale between the capture and the
    /// copy. This drives both while a checkpoint is in flight and
    /// requires the result to still open and read back.
    #[test]
    fn a_checkpoint_manifest_survives_a_rewrite_racing_the_capture() {
        let src_dir = TempDir::new().unwrap();
        let db = Arc::new(Db::open(src_dir.path(), Options::default()).unwrap());

        for i in 0..300u64 {
            let k = format!("seed_{i:04}");
            db.put(k.as_bytes(), k.as_bytes()).unwrap();
        }

        let stop = Arc::new(AtomicBool::new(false));
        let writer = {
            let db = Arc::clone(&db);
            let stop = Arc::clone(&stop);
            thread::spawn(move || {
                let mut i = 0u64;
                while !stop.load(Ordering::Relaxed) {
                    // Enough edits to push the manifest past its rewrite
                    // threshold while the checkpoints below run.
                    let k = format!("churn_{i:08}");
                    let _ = db.put(k.as_bytes(), &vec![b'x'; 512]);
                    i += 1;
                }
                i
            })
        };

        for _ in 0..8 {
            let tgt = TempDir::new().unwrap();
            let cp = Checkpoint::new(&db).unwrap();
            cp.create(tgt.path()).unwrap();

            let reopened = Db::open(tgt.path(), Options::default()).unwrap();
            for i in 0..300u64 {
                let k = format!("seed_{i:04}");
                assert_eq!(
                    reopened.get(k.as_bytes()).unwrap(),
                    Some(k.clone().into_bytes()),
                    "{k} is missing from a checkpoint taken under a concurrent writer"
                );
            }
        }

        stop.store(true, Ordering::Relaxed);
        let _ = writer.join().unwrap();
    }

    /// A manifest rewrite landing between the capture and the copy must
    /// not change what the checkpoint writes.
    ///
    /// The capture holds the compaction lock, but a flush takes only the
    /// version lock, and a rewrite replaces the manifest file wholesale.
    /// Re-reading the path at copy time would then pick up a *different*
    /// manifest, and truncating it at a length measured against the old
    /// one can land mid-record. The hook forces exactly that, so this
    /// fails if the copy ever goes back to re-reading the source.
    #[test]
    fn a_manifest_rewritten_between_capture_and_copy_does_not_corrupt_the_checkpoint() {
        let src_dir = TempDir::new().unwrap();
        let tgt_dir = TempDir::new().unwrap();
        let db = Db::open(src_dir.path(), Options::default()).unwrap();

        for i in 0..400u64 {
            let k = format!("seed_{i:04}");
            db.put(k.as_bytes(), k.as_bytes()).unwrap();
        }
        db.flush().unwrap();

        let cp = Checkpoint::new(&db).unwrap();
        cp.create_between(tgt_dir.path(), |db| {
            // Grow the manifest, then rewrite it, so the file on disk is
            // both different and a different length from the captured
            // one.
            for i in 0..400u64 {
                let k = format!("after_{i:04}");
                db.put(k.as_bytes(), &vec![b'z'; 256]).unwrap();
            }
            db.flush().unwrap();
            db.engine().force_manifest_rewrite().unwrap();
        })
        .unwrap();

        let reopened = Db::open(tgt_dir.path(), Options::default()).unwrap();
        for i in 0..400u64 {
            let k = format!("seed_{i:04}");
            assert_eq!(
                reopened.get(k.as_bytes()).unwrap(),
                Some(k.clone().into_bytes()),
                "{k} was captured but is not in the checkpoint"
            );
        }
        // Nothing written after the capture belongs in it.
        assert_eq!(
            reopened.get(b"after_0000").unwrap(),
            None,
            "the checkpoint picked up a write that happened after its capture"
        );
    }

    /// A checkpoint's manifest must not name a file the checkpoint did
    /// not hardlink. Opening it is the check: recovery opens a reader
    /// for every SSTable the manifest references, so a dangling name
    /// fails the open rather than passing quietly.
    #[test]
    fn a_checkpoint_names_only_files_it_copied() {
        let src_dir = TempDir::new().unwrap();
        let tgt_dir = TempDir::new().unwrap();
        let db = Db::open(src_dir.path(), Options::default()).unwrap();

        for i in 0..500u64 {
            db.put(format!("k{i:05}").as_bytes(), &vec![b'v'; 256])
                .unwrap();
        }
        db.flush().unwrap();
        for i in 500..1000u64 {
            db.put(format!("k{i:05}").as_bytes(), &vec![b'v'; 256])
                .unwrap();
        }

        let cp = Checkpoint::new(&db).unwrap();
        cp.create(tgt_dir.path()).unwrap();

        let copied: std::collections::HashSet<_> = std::fs::read_dir(tgt_dir.path().join("sst"))
            .unwrap()
            .filter_map(|e| e.ok().map(|e| e.file_name()))
            .collect();

        // Opening replays the manifest and opens every file it names.
        let reopened = Db::open(tgt_dir.path(), Options::default()).unwrap();
        for i in 0..1000u64 {
            let k = format!("k{i:05}");
            assert!(
                reopened.get(k.as_bytes()).unwrap().is_some(),
                "{k} missing from the checkpoint"
            );
        }
        assert!(
            !copied.is_empty(),
            "the checkpoint hardlinked no SSTable at all"
        );
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
        // it post-creation - hard-linked inodes outlive the original
        // directory entry.
        std::fs::remove_dir_all(src_dir.path()).unwrap();

        let reopened = Db::open(tgt_dir.path(), Options::default()).unwrap();
        for i in 0..100 {
            let k = format!("dur_{:04}", i);
            assert_eq!(reopened.get(k.as_bytes()).unwrap(), Some(k.into_bytes()));
        }
    }
}
