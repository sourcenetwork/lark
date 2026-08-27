//! Cross-process locking of a database directory, for [`StdEnv`].
//!
//! [`super::StdEnv`] excludes other processes with `flock` on unix
//! and with an exclusive share mode on Windows. Both take a `LOCK`
//! file inside the database directory and both release it when the
//! guard drops.
//!
//! # Where there is no flock
//!
//! Every other target gets [`DirectoryRegistry`]: exclusion between
//! [`crate::Db`] handles recorded **in process memory**, with no file
//! on disk. That is a deliberate decision, not a cfg accident, and the
//! narrower guarantee is reported honestly through
//! [`super::Capabilities::file_lock`], which is `false` there and which
//! [`crate::Db::capabilities`] hands back to the caller.
//!
//! The registry is a complete guarantee exactly where it is used. wasm
//! is one process with one linear memory holding one copy of lark's
//! state, so a second writer can only come from a second [`crate::Db`]
//! in that same process, which the registry sees. A [`super::MemEnv`]
//! and an OPFS mount are likewise scoped to one process, so each keeps
//! its own registry rather than sharing the process-wide one: two
//! `MemEnv`s are two different filesystems and must not collide on a
//! shared path.
//!
//! The alternative lark used to ship was a `create_new(true)` proxy:
//! opening failed when a `LOCK` file already existed. That has the
//! severe failure mode a registry does not - a crash or an unclean
//! unload leaves the file behind, and every later open fails
//! permanently with a message blaming a second process that cannot
//! exist. Registry state dies with the process that holds it, so a
//! crash can never leave a stale lock.
//!
//! Two `Db` handles opened on one path inside a single process are
//! rejected on every target: `flock` rejects them because lark opens a
//! separate descriptor per handle, and the registry rejects them
//! because it records the path. What neither prevents is a second
//! *process* on a target with no file locking, and there lark says so
//! through `Capabilities::file_lock` instead of pretending.

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use super::FileLock;

/// In-process exclusion between database handles on one directory.
///
/// Used by every environment whose
/// [`super::Capabilities::file_lock`] is `false`. Holding a guard
/// marks the directory taken; dropping it releases the mark. One
/// writer excludes every other holder, and any number of readers
/// coexist.
#[derive(Debug, Default)]
pub(super) struct DirectoryRegistry {
    /// `None` means one writer holds the path; `Some(n)` means `n`
    /// readers do. A vacant entry means nobody does.
    held: Mutex<HashMap<PathBuf, Option<usize>>>,
}

impl DirectoryRegistry {
    /// Take `db_dir`, exclusively when `exclusive`.
    ///
    /// Reports [`std::io::ErrorKind::AlreadyExists`] when a
    /// conflicting handle is already open on the same path in this
    /// process, matching what `flock` reports on unix.
    pub(super) fn acquire(
        self: &Arc<Self>,
        db_dir: &Path,
        exclusive: bool,
    ) -> io::Result<Box<dyn FileLock>> {
        let path = db_dir.to_path_buf();
        let mut held = self
            .held
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match held.entry(path.clone()) {
            Entry::Occupied(mut slot) => match (slot.get_mut(), exclusive) {
                (Some(readers), false) => *readers += 1,
                _ => return Err(conflict(&path)),
            },
            Entry::Vacant(slot) => {
                slot.insert(if exclusive { None } else { Some(1) });
            }
        }
        drop(held);
        Ok(Box::new(RegistryLock {
            registry: Arc::clone(self),
            path,
            exclusive,
        }))
    }

    fn release(&self, path: &Path, exclusive: bool) {
        let mut held = self
            .held
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if exclusive {
            held.remove(path);
            return;
        }
        if let Entry::Occupied(mut slot) = held.entry(path.to_path_buf()) {
            match slot.get_mut() {
                Some(readers) if *readers > 1 => *readers -= 1,
                _ => {
                    slot.remove();
                }
            }
        }
    }
}

/// The process-wide registry, for [`super::StdEnv`] on a target with
/// no `flock`. One registry for the process is right there because the
/// paths it records are real filesystem paths that every `StdEnv` in
/// the process shares.
#[cfg(not(any(unix, windows)))]
pub(super) fn process_registry() -> &'static Arc<DirectoryRegistry> {
    static REGISTRY: std::sync::OnceLock<Arc<DirectoryRegistry>> = std::sync::OnceLock::new();
    REGISTRY.get_or_init(Arc::default)
}

fn conflict(path: &Path) -> io::Error {
    io::Error::new(
        io::ErrorKind::AlreadyExists,
        format!(
            "database directory is already open in this process: {}",
            path.display()
        ),
    )
}

/// A held entry in a [`DirectoryRegistry`]. Releases on drop.
#[derive(Debug)]
struct RegistryLock {
    registry: Arc<DirectoryRegistry>,
    path: PathBuf,
    exclusive: bool,
}

impl FileLock for RegistryLock {}

impl Drop for RegistryLock {
    fn drop(&mut self) {
        self.registry.release(&self.path, self.exclusive);
    }
}

/// Whether this target has real cross-process file locking.
pub(super) const SUPPORTS_FILE_LOCK: bool = cfg!(any(unix, windows));

/// Acquire the directory lock for `db_dir`.
///
/// `exclusive` takes a write lock (read-write open); otherwise a
/// shared lock (read-only open). Reports
/// [`std::io::ErrorKind::AlreadyExists`] when another process holds a
/// conflicting lock.
pub(super) fn acquire(db_dir: &Path, exclusive: bool) -> io::Result<Box<dyn FileLock>> {
    #[cfg(any(unix, windows))]
    {
        real::acquire(db_dir, exclusive)
    }
    #[cfg(not(any(unix, windows)))]
    {
        process_registry().acquire(db_dir, exclusive)
    }
}

#[cfg(any(unix, windows))]
mod real {
    use std::fs::{File, OpenOptions};
    use std::io;
    use std::path::Path;

    use super::FileLock;

    /// Name of the lock file inside the database directory.
    const LOCK_FILE: &str = "LOCK";

    /// Content stamped into the lock file.
    ///
    /// The lock itself is advisory and needs no bytes, but an unmarked
    /// zero-length `LOCK` is indistinguishable from any other tool's,
    /// and from a stray file in a directory someone pointed lark at by
    /// mistake. The stamp is checked on the exclusive path, so lark
    /// refuses to write a database into a directory that is not its own
    /// rather than taking it over.
    const LOCK_STAMP: &[u8; 8] = b"REGOLOCK";

    /// A held `flock` (unix) or exclusive share mode (Windows) on the
    /// database directory's `LOCK` file.
    #[derive(Debug)]
    pub(super) struct DbDirectoryLock {
        file: File,
    }

    impl FileLock for DbDirectoryLock {}

    impl Drop for DbDirectoryLock {
        fn drop(&mut self) {
            unlock(&self.file);
        }
    }

    pub(super) fn acquire(db_dir: &Path, exclusive: bool) -> io::Result<Box<dyn FileLock>> {
        let path = db_dir.join(LOCK_FILE);
        let file = if exclusive {
            std::fs::create_dir_all(db_dir)?;
            open_lock_file(&path).map_err(|e| lock_error(&path, e))?
        } else {
            OpenOptions::new()
                .read(true)
                .open(&path)
                .map_err(|e| lock_error(&path, e))?
        };

        if exclusive {
            lock_exclusive(&file).map_err(|e| lock_error(&path, e))?;
            // Stamped only once the lock is held, so two processes
            // cannot race to write it, and only after any existing
            // content has been vetted.
            stamp_lock_file(&file).map_err(|e| lock_error(&path, e))?;
        } else {
            lock_shared(&file).map_err(|e| lock_error(&path, e))?;
        }

        Ok(Box::new(DbDirectoryLock { file }))
    }

    /// Verify the stamp when the file already carries content, write it
    /// when it does not.
    ///
    /// An empty file is what a build older than the stamp left behind,
    /// so it is adopted rather than refused: the directory is lark's, it
    /// simply predates the stamp. Content that is neither empty nor the
    /// stamp belongs to something else.
    fn stamp_lock_file(file: &File) -> io::Result<()> {
        use std::io::{Read, Seek, SeekFrom, Write};

        let mut existing = Vec::new();
        (&*file).seek(SeekFrom::Start(0))?;
        (&*file).read_to_end(&mut existing)?;

        if !existing.is_empty() && existing.as_slice() != LOCK_STAMP {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "LOCK exists but was not written by lark: refusing to open a database here",
            ));
        }
        if existing.is_empty() {
            (&*file).seek(SeekFrom::Start(0))?;
            (&*file).write_all(LOCK_STAMP)?;
            file.sync_data()?;
        }
        Ok(())
    }

    #[cfg(unix)]
    fn open_lock_file(path: &Path) -> io::Result<File> {
        OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)
    }

    #[cfg(windows)]
    fn open_lock_file(path: &Path) -> io::Result<File> {
        use std::os::windows::fs::OpenOptionsExt;

        OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .share_mode(0)
            .open(path)
    }

    fn lock_error(path: &Path, err: io::Error) -> io::Error {
        if is_lock_conflict(&err) {
            io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!(
                    "database directory is already locked for read-write access: {}",
                    path.display()
                ),
            )
        } else {
            io::Error::new(
                err.kind(),
                format!(
                    "failed to lock database directory {}: {err}",
                    path.display()
                ),
            )
        }
    }

    fn is_lock_conflict(err: &io::Error) -> bool {
        err.kind() == io::ErrorKind::WouldBlock
            || err.kind() == io::ErrorKind::AlreadyExists
            || is_windows_share_violation(err)
    }

    #[cfg(windows)]
    fn is_windows_share_violation(err: &io::Error) -> bool {
        const ERROR_SHARING_VIOLATION: i32 = 32;
        err.raw_os_error() == Some(ERROR_SHARING_VIOLATION)
    }

    #[cfg(not(windows))]
    fn is_windows_share_violation(_err: &io::Error) -> bool {
        false
    }

    #[cfg(unix)]
    fn lock_exclusive(file: &File) -> io::Result<()> {
        rustix::fs::flock(file, rustix::fs::FlockOperation::NonBlockingLockExclusive)
            .map_err(|e| io::Error::from_raw_os_error(e.raw_os_error()))
    }

    #[cfg(unix)]
    fn lock_shared(file: &File) -> io::Result<()> {
        rustix::fs::flock(file, rustix::fs::FlockOperation::NonBlockingLockShared)
            .map_err(|e| io::Error::from_raw_os_error(e.raw_os_error()))
    }

    #[cfg(unix)]
    fn unlock(file: &File) {
        let _ = rustix::fs::flock(file, rustix::fs::FlockOperation::Unlock);
    }

    #[cfg(windows)]
    fn lock_exclusive(_file: &File) -> io::Result<()> {
        Ok(())
    }

    #[cfg(windows)]
    fn lock_shared(_file: &File) -> io::Result<()> {
        Ok(())
    }

    #[cfg(windows)]
    fn unlock(_file: &File) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `Box<dyn FileLock>` is not `Debug`, so `unwrap_err` cannot be
    /// used on an acquire result. Take the error kind directly.
    fn kind(result: io::Result<Box<dyn FileLock>>) -> Option<io::ErrorKind> {
        result.err().map(|e| e.kind())
    }

    #[test]
    fn a_second_exclusive_holder_is_rejected_on_every_target() {
        let dir = tempfile::TempDir::new().unwrap();
        let first = acquire(dir.path(), true).unwrap();
        assert_eq!(
            kind(acquire(dir.path(), true)),
            Some(io::ErrorKind::AlreadyExists),
            "a second exclusive lock must fail"
        );
        drop(first);
        assert!(acquire(dir.path(), true).is_ok());
    }

    #[test]
    fn the_registry_excludes_a_writer_and_admits_concurrent_readers() {
        let registry: Arc<DirectoryRegistry> = Arc::default();
        let dir = Path::new("/registry/db");

        let writer = registry.acquire(dir, true).unwrap();
        assert_eq!(
            kind(registry.acquire(dir, true)),
            Some(io::ErrorKind::AlreadyExists)
        );
        assert_eq!(
            kind(registry.acquire(dir, false)),
            Some(io::ErrorKind::AlreadyExists)
        );
        drop(writer);

        let first_reader = registry.acquire(dir, false).unwrap();
        let second_reader = registry.acquire(dir, false).unwrap();
        assert_eq!(
            kind(registry.acquire(dir, true)),
            Some(io::ErrorKind::AlreadyExists),
            "a writer must not join live readers"
        );
        drop(first_reader);
        assert!(
            registry.acquire(dir, true).is_err(),
            "one reader still holds the path"
        );
        drop(second_reader);
        assert!(registry.acquire(dir, true).is_ok());
    }

    #[test]
    fn separate_registries_do_not_collide_on_one_path() {
        let one: Arc<DirectoryRegistry> = Arc::default();
        let two: Arc<DirectoryRegistry> = Arc::default();
        let dir = Path::new("/db");
        let _held = one.acquire(dir, true).unwrap();
        assert!(
            two.acquire(dir, true).is_ok(),
            "two MemEnvs are two filesystems and must not share exclusion"
        );
    }
}
