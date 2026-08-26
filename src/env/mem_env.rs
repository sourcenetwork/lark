//! An entirely in-memory [`Env`].
//!
//! [`MemEnv`] exists to prove the seam. Every pre-existing test runs
//! through [`super::StdEnv`], so a leftover `std::fs` call inside the
//! engine would still pass; the same lifecycle run against `MemEnv`
//! fails loudly instead, because there is no filesystem behind it at
//! all.
//!
//! It is also the honest way to test the paths lark takes when the
//! host is missing a capability: `MemEnv` has no hard links, no
//! directory fsync, and no threads, and it says so through
//! [`super::Capabilities`].
//!
//! # Shape
//!
//! A flat `BTreeMap` from path to file, plus a set of directories.
//! Files are `Arc<Mutex<Vec<u8>>>` so a reader and a writer can hold
//! the same file without holding the directory lock, which is what
//! keeps a compaction reading one file while a flush writes another
//! from deadlocking.

use std::collections::{BTreeMap, BTreeSet};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;

use super::db_lock::DirectoryRegistry;
use super::{
    Capabilities, DirEntry, Env, FileLock, FileMeta, JoinHandle, ReadFile, WriteFile, WriteMode,
};

type SharedFile = Arc<Mutex<Vec<u8>>>;

#[derive(Default)]
struct MemFs {
    files: BTreeMap<PathBuf, SharedFile>,
    dirs: BTreeSet<PathBuf>,
}

/// An [`Env`] whose filesystem is a map in memory.
///
/// Clones share one filesystem, so a `MemEnv` handed to
/// [`crate::Options::env`] can be kept by the caller and inspected
/// after the database closes.
#[derive(Clone, Default)]
pub struct MemEnv {
    fs: Arc<Mutex<MemFs>>,
    clock: Arc<Mutex<MemClock>>,
    /// Exclusion between database handles on this filesystem. Scoped
    /// to the `MemEnv` rather than the process because two `MemEnv`s
    /// are two unrelated filesystems that may legitimately hold the
    /// same path.
    open_dirs: Arc<DirectoryRegistry>,
}

#[derive(Debug, Clone, Copy)]
struct MemClock {
    micros: Option<u64>,
    unix_secs: Option<u64>,
}

impl Default for MemClock {
    fn default() -> Self {
        Self {
            micros: Some(0),
            unix_secs: Some(0),
        }
    }
}

impl std::fmt::Debug for MemEnv {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let fs = self.fs.lock();
        f.debug_struct("MemEnv")
            .field("files", &fs.files.len())
            .field("dirs", &fs.dirs.len())
            .finish()
    }
}

impl MemEnv {
    /// An empty in-memory filesystem with a clock starting at zero.
    pub fn new() -> Self {
        Self::default()
    }

    /// Advance the monotonic clock by `micros` and the wall clock by
    /// the whole seconds that implies.
    ///
    /// Time in a `MemEnv` only moves when a test moves it, so a test
    /// that depends on elapsed time is deterministic instead of
    /// racing the machine it runs on.
    pub fn advance_micros(&self, micros: u64) {
        let mut clock = self.clock.lock();
        if let Some(now) = clock.micros {
            clock.micros = Some(now.saturating_add(micros));
        }
        if let Some(secs) = clock.unix_secs {
            clock.unix_secs = Some(secs.saturating_add(micros / 1_000_000));
        }
    }

    /// Remove the monotonic clock, the wall clock, or both, to
    /// exercise what lark does on a platform that has none.
    pub fn set_clocks(&self, micros: Option<u64>, unix_secs: Option<u64>) {
        let mut clock = self.clock.lock();
        clock.micros = micros;
        clock.unix_secs = unix_secs;
    }

    /// Total bytes currently held across every file.
    pub fn total_bytes(&self) -> u64 {
        self.fs
            .lock()
            .files
            .values()
            .map(|f| f.lock().len() as u64)
            .sum()
    }

    /// Number of files currently present.
    pub fn file_count(&self) -> usize {
        self.fs.lock().files.len()
    }

    fn lookup(&self, path: &Path) -> io::Result<SharedFile> {
        self.fs
            .lock()
            .files
            .get(path)
            .cloned()
            .ok_or_else(|| not_found(path))
    }
}

fn not_found(path: &Path) -> io::Error {
    io::Error::new(
        io::ErrorKind::NotFound,
        format!("no such file in MemEnv: {}", path.display()),
    )
}

impl Env for MemEnv {
    fn create_dir_all(&self, path: &Path) -> io::Result<()> {
        let mut fs = self.fs.lock();
        let mut cursor = Some(path);
        while let Some(dir) = cursor {
            if fs.files.contains_key(dir) {
                return Err(io::Error::new(
                    io::ErrorKind::AlreadyExists,
                    format!("{} exists and is not a directory", dir.display()),
                ));
            }
            fs.dirs.insert(dir.to_path_buf());
            cursor = dir.parent().filter(|p| !p.as_os_str().is_empty());
        }
        Ok(())
    }

    fn read_dir(&self, path: &Path) -> io::Result<Vec<DirEntry>> {
        let fs = self.fs.lock();
        if !fs.dirs.contains(path) {
            return Err(not_found(path));
        }
        let mut out = Vec::new();
        for file in fs.files.keys() {
            if file.parent() == Some(path) {
                out.push(DirEntry {
                    path: file.clone(),
                    is_dir: false,
                });
            }
        }
        for dir in &fs.dirs {
            if dir.parent() == Some(path) {
                out.push(DirEntry {
                    path: dir.clone(),
                    is_dir: true,
                });
            }
        }
        Ok(out)
    }

    fn open_read(&self, path: &Path) -> io::Result<Box<dyn ReadFile>> {
        Ok(Box::new(MemReadFile {
            data: self.lookup(path)?,
        }))
    }

    fn open_write(&self, path: &Path, mode: WriteMode) -> io::Result<Box<dyn WriteFile>> {
        let mut fs = self.fs.lock();
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
            && !fs.dirs.contains(parent)
        {
            return Err(not_found(parent));
        }
        let data = fs.files.entry(path.to_path_buf()).or_default().clone();
        drop(fs);
        if mode == WriteMode::Truncate {
            data.lock().clear();
        }
        Ok(Box::new(MemWriteFile { data }))
    }

    fn metadata(&self, path: &Path) -> io::Result<FileMeta> {
        let fs = self.fs.lock();
        if let Some(file) = fs.files.get(path) {
            return Ok(FileMeta {
                len: file.lock().len() as u64,
                is_dir: false,
            });
        }
        if fs.dirs.contains(path) {
            return Ok(FileMeta {
                len: 0,
                is_dir: true,
            });
        }
        Err(not_found(path))
    }

    fn remove_file(&self, path: &Path) -> io::Result<()> {
        self.fs
            .lock()
            .files
            .remove(path)
            .map(|_| ())
            .ok_or_else(|| not_found(path))
    }

    fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
        let mut fs = self.fs.lock();
        let file = fs.files.remove(from).ok_or_else(|| not_found(from))?;
        fs.files.insert(to.to_path_buf(), file);
        Ok(())
    }

    fn hard_link(&self, _src: &Path, _dst: &Path) -> io::Result<()> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "MemEnv has no hard links; see Capabilities::hard_link",
        ))
    }

    fn sync_dir(&self, _path: &Path) -> io::Result<()> {
        Ok(())
    }

    fn lock_file(&self, path: &Path, exclusive: bool) -> io::Result<Box<dyn FileLock>> {
        self.open_dirs.acquire(path, exclusive)
    }

    fn capabilities(&self) -> Capabilities {
        // Nothing here survives the process, so `durable_sync` is
        // false: a `sync_all` on a MemEnv file is a no-op and saying
        // otherwise would be a durability claim lark cannot keep.
        Capabilities::none().with_atomic_rename(true)
    }

    fn now_micros(&self) -> Option<u64> {
        self.clock.lock().micros
    }

    fn unix_secs(&self) -> Option<u64> {
        self.clock.lock().unix_secs
    }

    fn spawn(
        &self,
        _name: &str,
        _body: Box<dyn FnOnce() + Send + 'static>,
    ) -> io::Result<Box<dyn JoinHandle>> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "MemEnv does not start threads; see Capabilities::threads",
        ))
    }

    fn sleep(&self, _dur: Duration) {}
}

struct MemReadFile {
    data: SharedFile,
}

impl ReadFile for MemReadFile {
    fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        let data = self.data.lock();
        let start = usize::try_from(offset).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "offset is too large to address",
            )
        })?;
        let end = start
            .checked_add(buf.len())
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "read range overflows"))?;
        if end > data.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "failed to fill whole buffer",
            ));
        }
        buf.copy_from_slice(&data[start..end]);
        Ok(())
    }

    fn len(&self) -> io::Result<u64> {
        Ok(self.data.lock().len() as u64)
    }
}

struct MemWriteFile {
    data: SharedFile,
}

impl WriteFile for MemWriteFile {
    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        let mut data = self.data.lock();
        data.try_reserve(buf.len()).map_err(|_| {
            io::Error::new(
                io::ErrorKind::OutOfMemory,
                format!("cannot grow a MemEnv file by {} bytes", buf.len()),
            )
        })?;
        data.extend_from_slice(buf);
        Ok(())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }

    fn sync_all(&mut self) -> io::Result<()> {
        Ok(())
    }

    fn set_len(&mut self, len: u64) -> io::Result<()> {
        let len = usize::try_from(len).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "length is too large to address",
            )
        })?;
        let mut data = self.data.lock();
        let grow_by = len.saturating_sub(data.len());
        if grow_by > 0 {
            data.try_reserve(grow_by).map_err(|_| {
                io::Error::new(io::ErrorKind::OutOfMemory, "cannot extend a MemEnv file")
            })?;
        }
        data.resize(len, 0);
        Ok(())
    }

    fn len(&self) -> io::Result<u64> {
        Ok(self.data.lock().len() as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env() -> MemEnv {
        let env = MemEnv::new();
        env.create_dir_all(Path::new("/db")).unwrap();
        env
    }

    #[test]
    fn write_read_round_trips() {
        let env = env();
        let path = Path::new("/db/file");
        env.write(path, b"payload").unwrap();
        assert_eq!(env.read(path).unwrap(), b"payload");
        assert_eq!(env.metadata(path).unwrap().len, 7);
        assert_eq!(env.file_count(), 1);
        assert_eq!(env.total_bytes(), 7);
    }

    #[test]
    fn writing_into_a_missing_directory_fails() {
        let env = env();
        let err = env
            .open_write(Path::new("/nowhere/file"), WriteMode::Truncate)
            .map(|_| ())
            .unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::NotFound);
    }

    #[test]
    fn truncate_discards_and_append_keeps() {
        let env = env();
        let path = Path::new("/db/f");
        env.write(path, b"0123456789").unwrap();
        {
            let mut w = env.open_write(path, WriteMode::Append).unwrap();
            w.write_all(b"ab").unwrap();
        }
        assert_eq!(env.read(path).unwrap(), b"0123456789ab");
        env.write(path, b"z").unwrap();
        assert_eq!(env.read(path).unwrap(), b"z");
    }

    #[test]
    fn read_dir_lists_only_direct_children() {
        let env = env();
        env.create_dir_all(Path::new("/db/sst")).unwrap();
        env.write(Path::new("/db/MANIFEST"), b"m").unwrap();
        env.write(Path::new("/db/sst/1.sst"), b"s").unwrap();

        let mut names: Vec<String> = env
            .read_dir(Path::new("/db"))
            .unwrap()
            .iter()
            .map(|e| e.file_name())
            .collect();
        names.sort();
        assert_eq!(names, vec!["MANIFEST".to_string(), "sst".to_string()]);
    }

    #[test]
    fn rename_moves_content_and_clears_the_source() {
        let env = env();
        env.write(Path::new("/db/tmp"), b"new").unwrap();
        env.write(Path::new("/db/live"), b"old").unwrap();
        env.rename(Path::new("/db/tmp"), Path::new("/db/live"))
            .unwrap();
        assert_eq!(env.read(Path::new("/db/live")).unwrap(), b"new");
        assert!(!env.exists(Path::new("/db/tmp")));
    }

    #[test]
    fn positional_reads_are_independent() {
        let env = env();
        let path = Path::new("/db/pos");
        env.write(path, b"0123456789").unwrap();
        let f = env.open_read(path).unwrap();
        let mut a = [0u8; 2];
        let mut b = [0u8; 2];
        f.read_exact_at(8, &mut a).unwrap();
        f.read_exact_at(0, &mut b).unwrap();
        assert_eq!(&a, b"89");
        assert_eq!(&b, b"01");
        assert_eq!(
            f.read_exact_at(9, &mut a).unwrap_err().kind(),
            io::ErrorKind::UnexpectedEof
        );
    }

    #[test]
    fn missing_capabilities_are_declared_and_enforced() {
        let env = env();
        let caps = env.capabilities();
        assert!(!caps.hard_link);
        assert!(!caps.sync_dir);
        assert!(!caps.threads);
        assert!(!caps.file_lock);
        assert!(!caps.durable_sync);
        assert!(caps.atomic_rename);

        assert_eq!(
            env.hard_link(Path::new("/db/a"), Path::new("/db/b"))
                .unwrap_err()
                .kind(),
            io::ErrorKind::Unsupported
        );
        assert_eq!(
            env.spawn("x", Box::new(|| {}))
                .map(|_| ())
                .unwrap_err()
                .kind(),
            io::ErrorKind::Unsupported
        );
    }

    #[test]
    fn the_clock_only_moves_when_a_test_moves_it() {
        let env = MemEnv::new();
        assert_eq!(env.now_micros(), Some(0));
        assert_eq!(env.unix_secs(), Some(0));
        env.advance_micros(2_500_000);
        assert_eq!(env.now_micros(), Some(2_500_000));
        assert_eq!(env.unix_secs(), Some(2));
        env.set_clocks(None, None);
        assert!(env.now_micros().is_none());
        assert!(env.unix_secs().is_none());
    }

    #[test]
    fn set_len_truncates_and_extends_with_zeroes() {
        let env = env();
        let path = Path::new("/db/sized");
        env.write(path, b"abcdef").unwrap();
        let mut w = env.open_write(path, WriteMode::Append).unwrap();
        w.set_len(3).unwrap();
        assert_eq!(w.len().unwrap(), 3);
        w.set_len(5).unwrap();
        drop(w);
        assert_eq!(env.read(path).unwrap(), b"abc\0\0");
    }

    #[test]
    fn clones_share_one_filesystem() {
        let env = env();
        let twin = env.clone();
        env.write(Path::new("/db/shared"), b"x").unwrap();
        assert!(twin.exists(Path::new("/db/shared")));
    }

    #[test]
    fn a_lock_on_a_mem_env_excludes_a_second_holder_and_creates_no_file() {
        let env = env();
        let first = env.lock_file(Path::new("/db"), true).unwrap();
        assert!(
            env.lock_file(Path::new("/db"), true).is_err(),
            "a second read-write handle on one directory must be refused"
        );
        drop(first);
        let again = env.lock_file(Path::new("/db"), true);
        assert!(again.is_ok(), "the directory must be free again after drop");
        drop(again);
        assert_eq!(env.file_count(), 0, "no LOCK file is ever created");
    }

    #[test]
    fn two_mem_envs_do_not_exclude_each_other() {
        let one = env();
        let two = env();
        let _held = one.lock_file(Path::new("/db"), true).unwrap();
        assert!(
            two.lock_file(Path::new("/db"), true).is_ok(),
            "two MemEnvs are two filesystems and must not share exclusion"
        );
    }
}
