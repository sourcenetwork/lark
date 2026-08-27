//! The default [`Env`]: `std::fs`, `std::thread`, `std::time`.
//!
//! Every call here is the same call lark made before the [`Env`]
//! trait existed, in the same order, returning the same error kinds.
//! This is the environment [`crate::Options::default`] installs, so a
//! native user sees no behavior change at all.

use std::fs::{File, OpenOptions};
use std::io::{self, Seek, SeekFrom, Write};
use std::path::Path;
use std::time::Duration;
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
use std::time::{SystemTime, UNIX_EPOCH};

use super::db_lock;
use super::{
    Capabilities, DirEntry, Env, FileLock, FileMeta, JoinHandle, ReadFile, WriteFile, WriteMode,
};

/// Whether this target has a real directory fsync.
///
/// `cfg!(unix)` alone is not enough: it is `false` on
/// `wasm32-wasip1`, and opening a preopened directory there and
/// calling `sync_all` on it reports `EBADF`. Keeping the constant and
/// the behavior derived from the same expression is what stops
/// [`Capabilities::sync_dir`] from claiming a durability the platform
/// does not provide.
const SUPPORTS_DIR_SYNC: bool = cfg!(unix) && !cfg!(target_family = "wasm");

/// Whether this target can start a thread.
///
/// A wasm module built without the `atomics` target feature has
/// exactly one thread, and `std::thread::spawn` there reports
/// [`std::io::ErrorKind::Unsupported`]. Declaring that up front lets
/// a caller choose a single-threaded configuration before an open
/// fails rather than after.
const SUPPORTS_THREADS: bool = !cfg!(all(target_family = "wasm", not(target_feature = "atomics")));

/// The host platform as lark has always used it.
#[derive(Debug, Default, Clone, Copy)]
pub struct StdEnv;

impl StdEnv {
    /// Construct the standard environment.
    ///
    /// It holds no state, so [`crate::env::std_env`] shares one
    /// instance across the whole process rather than allocating per
    /// database.
    pub const fn new() -> Self {
        Self
    }
}

impl Env for StdEnv {
    fn create_dir_all(&self, path: &Path) -> io::Result<()> {
        std::fs::create_dir_all(path)
    }

    fn read_dir(&self, path: &Path) -> io::Result<Vec<DirEntry>> {
        let mut out = Vec::new();
        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            let is_dir = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);
            out.push(DirEntry {
                path: entry.path(),
                is_dir,
            });
        }
        Ok(out)
    }

    fn open_read(&self, path: &Path) -> io::Result<Box<dyn ReadFile>> {
        Ok(Box::new(StdReadFile {
            file: File::open(path)?,
            #[cfg(not(any(unix, windows)))]
            cursor: crate::sync::Mutex::new(()),
        }))
    }

    fn open_write(&self, path: &Path, mode: WriteMode) -> io::Result<Box<dyn WriteFile>> {
        let file = match mode {
            WriteMode::Truncate => OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(path)?,
            WriteMode::Append => OpenOptions::new().create(true).append(true).open(path)?,
            // `truncate(false)` is stated rather than left to default:
            // keeping the contents is the whole difference between this
            // mode and `Truncate`.
            WriteMode::Update => OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(false)
                .open(path)?,
        };
        Ok(Box::new(StdWriteFile { file }))
    }

    fn metadata(&self, path: &Path) -> io::Result<FileMeta> {
        let meta = std::fs::metadata(path)?;
        Ok(FileMeta {
            len: meta.len(),
            is_dir: meta.is_dir(),
        })
    }

    fn remove_file(&self, path: &Path) -> io::Result<()> {
        std::fs::remove_file(path)
    }

    fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
        std::fs::rename(from, to)
    }

    fn hard_link(&self, src: &Path, dst: &Path) -> io::Result<()> {
        std::fs::hard_link(src, dst)
    }

    fn sync_dir(&self, path: &Path) -> io::Result<()> {
        if SUPPORTS_DIR_SYNC {
            File::open(path)?.sync_all()
        } else {
            Ok(())
        }
    }

    fn lock_file(&self, path: &Path, exclusive: bool) -> io::Result<Box<dyn FileLock>> {
        db_lock::acquire(path, exclusive)
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities::posix()
            .with_sync_dir(SUPPORTS_DIR_SYNC)
            .with_file_lock(db_lock::SUPPORTS_FILE_LOCK)
            .with_threads(SUPPORTS_THREADS)
    }

    fn now_micros(&self) -> Option<u64> {
        super::platform_micros()
    }

    fn unix_secs(&self) -> Option<u64> {
        #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
        {
            // `SystemTime::now` panics on this target rather than
            // returning an error, so it is never called there.
            None
        }
        #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
        {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .ok()
                .map(|d| d.as_secs())
        }
    }

    fn spawn(
        &self,
        name: &str,
        body: Box<dyn FnOnce() + Send + 'static>,
    ) -> io::Result<Box<dyn JoinHandle>> {
        let handle = std::thread::Builder::new()
            .name(name.to_string())
            .spawn(body)?;
        Ok(Box::new(StdJoinHandle(handle)))
    }

    fn sleep(&self, dur: Duration) {
        std::thread::sleep(dur);
    }

    fn drop_page_cache(&self, path: &Path) {
        // Compaction reads and writes whole SSTables sequentially and
        // will not re-read them soon; without this hint the kernel
        // keeps those pages resident and evicts hot foreground data.
        // A failure only means the cache stayed warmer than we wanted,
        // so every error on this path is deliberately dropped.
        if let Ok(file) = OpenOptions::new().read(true).open(path) {
            drop_page_cache_for(&file);
        }
    }
}

#[cfg(target_os = "linux")]
fn drop_page_cache_for(file: &File) {
    // `rustix::fs::fadvise` is a safe wrapper around
    // `posix_fadvise`; offset 0 with no length means "the whole
    // file".
    let _ = rustix::fs::fadvise(file, 0, None, rustix::fs::Advice::DontNeed);
}

#[cfg(not(target_os = "linux"))]
fn drop_page_cache_for(_file: &File) {
    // macOS offers `F_NOCACHE`, but it must be set before any read
    // happens, which needs a flag threaded through the SSTable reader
    // open path. Nothing to do here until then.
}

/// A `std::fs::File` read positionally.
///
/// Unix and Windows have a positional read syscall, so the handle
/// carries no cursor and concurrent readers never serialize. Any
/// other target falls back to seek-then-read behind a lock, because
/// the cursor is then shared state; `wasm32-wasip1` lands there and
/// is single-threaded, so that lock is uncontended by construction.
#[derive(Debug)]
struct StdReadFile {
    file: File,
    #[cfg(not(any(unix, windows)))]
    cursor: crate::sync::Mutex<()>,
}

impl ReadFile for StdReadFile {
    #[cfg(unix)]
    fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        use std::os::unix::fs::FileExt;
        self.file.read_exact_at(buf, offset)
    }

    #[cfg(windows)]
    fn read_exact_at(&self, mut offset: u64, buf: &mut [u8]) -> io::Result<()> {
        use std::os::windows::fs::FileExt;
        let mut written = 0;
        while written < buf.len() {
            match self.file.seek_read(&mut buf[written..], offset) {
                Ok(0) => {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "failed to fill whole buffer",
                    ));
                }
                Ok(n) => {
                    written += n;
                    offset += n as u64;
                }
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }

    #[cfg(not(any(unix, windows)))]
    fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        use std::io::{Read, Seek, SeekFrom};
        let _guard = self.cursor.lock();
        let mut file = &self.file;
        file.seek(SeekFrom::Start(offset))?;
        file.read_exact(buf)
    }

    fn len(&self) -> io::Result<u64> {
        Ok(self.file.metadata()?.len())
    }
}

/// A `std::fs::File` opened for appending.
#[derive(Debug)]
struct StdWriteFile {
    file: File,
}

impl WriteFile for StdWriteFile {
    /// One `writev` for the whole record. `write_all_vectored` on a POSIX file
    /// is the syscall this exists for: without it, a WAL payload at or above
    /// the buffer capacity forces the 5-byte header out in a syscall of its
    /// own, so one record costs two writes and leaves a wider window between
    /// the header and the payload.
    fn write_all_vectored(&mut self, slices: &[&[u8]]) -> io::Result<()> {
        use std::io::IoSlice;
        let mut bufs: Vec<IoSlice<'_>> = slices
            .iter()
            .filter(|s| !s.is_empty())
            .map(|s| IoSlice::new(s))
            .collect();
        let mut rest = &mut bufs[..];
        while !rest.is_empty() {
            let n = self.file.write_vectored(rest)?;
            if n == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::WriteZero,
                    "vectored write made no progress",
                ));
            }
            // A short vectored write is normal: advance past the slices it
            // consumed and re-issue the remainder.
            IoSlice::advance_slices(&mut rest, n);
        }
        Ok(())
    }

    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        self.file.write_all(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.file.flush()
    }

    fn sync_all(&mut self) -> io::Result<()> {
        self.file.sync_all()
    }

    fn sync_data(&mut self) -> io::Result<()> {
        self.file.sync_data()
    }

    fn set_len(&mut self, len: u64) -> io::Result<()> {
        self.file.set_len(len)?;
        // `set_len` does not move the cursor, so without this the next
        // write lands past the new end and leaves a hole behind it.
        self.file.seek(SeekFrom::Start(len))?;
        Ok(())
    }

    fn len(&self) -> io::Result<u64> {
        Ok(self.file.metadata()?.len())
    }
}

struct StdJoinHandle(std::thread::JoinHandle<()>);

impl JoinHandle for StdJoinHandle {
    fn join(self: Box<Self>) {
        // A worker that panicked is joined like any other: the panic
        // has already been reported on its own thread, and shutdown
        // must not turn it into a second one here.
        let _ = self.0.join();
    }
}

#[cfg(test)]
mod update_mode_tests {
    use super::*;
    use crate::env::WriteMode;

    /// The regression this guards: manifest recovery trims a torn tail
    /// with `set_len`, and it used to do that through the same append
    /// handle it then wrote through. On Windows an append handle is
    /// opened without `FILE_WRITE_DATA`, which `SetEndOfFile` needs, so
    /// every reopen of a database whose MANIFEST had a torn tail failed
    /// with "Access is denied" - which is to say, every reopen after the
    /// ordinary kind of crash.
    #[test]
    fn a_file_opened_for_update_keeps_its_contents_and_can_be_shortened() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("log");
        let env = StdEnv::new();

        {
            let mut w = Env::open_write(&env, &path, WriteMode::Truncate).expect("create");
            w.write_all(b"0123456789").expect("write");
            w.sync_all().expect("sync");
        }

        {
            let mut w = Env::open_write(&env, &path, WriteMode::Update).expect("open update");
            w.set_len(4).expect(
                "shortening through an Update handle is what manifest recovery does on every \
                 reopen after a crash",
            );
            w.sync_all().expect("sync");
        }

        assert_eq!(
            Env::read(&env, &path).expect("read"),
            b"0123456789"[..4].to_vec(),
            "Update must keep what it did not remove",
        );

        // And the log is still appendable afterwards, which is the next
        // thing recovery does.
        {
            let mut w = Env::open_write(&env, &path, WriteMode::Append).expect("open append");
            w.write_all(b"xy").expect("append");
            w.sync_all().expect("sync");
        }
        assert_eq!(Env::read(&env, &path).expect("read"), b"0123xy".to_vec());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::std_env;
    use tempfile::TempDir;

    #[test]
    fn write_read_round_trips_through_the_trait() {
        let dir = TempDir::new().unwrap();
        let env = std_env();
        let path = dir.path().join("round.trip");
        env.write(&path, b"payload").unwrap();
        assert_eq!(env.read(&path).unwrap(), b"payload");
        assert_eq!(env.metadata(&path).unwrap().len, 7);
        assert!(env.exists(&path));
        env.remove_file(&path).unwrap();
        assert!(!env.exists(&path));
    }

    #[test]
    fn append_mode_keeps_prior_contents() {
        let dir = TempDir::new().unwrap();
        let env = std_env();
        let path = dir.path().join("appended");
        env.write(&path, b"first").unwrap();
        let mut w = env.open_write(&path, WriteMode::Append).unwrap();
        w.write_all(b"second").unwrap();
        w.flush().unwrap();
        drop(w);
        assert_eq!(env.read(&path).unwrap(), b"firstsecond");
    }

    #[test]
    fn truncate_mode_discards_prior_contents() {
        let dir = TempDir::new().unwrap();
        let env = std_env();
        let path = dir.path().join("truncated");
        env.write(&path, b"0123456789").unwrap();
        env.write(&path, b"ab").unwrap();
        assert_eq!(env.read(&path).unwrap(), b"ab");
    }

    #[test]
    fn positional_reads_do_not_disturb_each_other() {
        let dir = TempDir::new().unwrap();
        let env = std_env();
        let path = dir.path().join("positional");
        env.write(&path, b"0123456789").unwrap();
        let file = env.open_read(&path).unwrap();
        let mut a = [0u8; 3];
        let mut b = [0u8; 3];
        file.read_exact_at(6, &mut a).unwrap();
        file.read_exact_at(0, &mut b).unwrap();
        assert_eq!(&a, b"678");
        assert_eq!(&b, b"012");
        assert_eq!(file.len().unwrap(), 10);
    }

    #[test]
    fn read_past_the_end_is_an_unexpected_eof() {
        let dir = TempDir::new().unwrap();
        let env = std_env();
        let path = dir.path().join("short");
        env.write(&path, b"ab").unwrap();
        let file = env.open_read(&path).unwrap();
        let mut buf = [0u8; 8];
        let err = file.read_exact_at(0, &mut buf).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::UnexpectedEof);
    }

    #[test]
    fn read_dir_lists_files_and_directories() {
        let dir = TempDir::new().unwrap();
        let env = std_env();
        env.create_dir_all(&dir.path().join("sub")).unwrap();
        env.write(&dir.path().join("file"), b"x").unwrap();

        let mut entries = env.read_dir(dir.path()).unwrap();
        entries.sort_by(|a, b| a.path.cmp(&b.path));
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].file_name(), "file");
        assert!(!entries[0].is_dir);
        assert_eq!(entries[1].file_name(), "sub");
        assert!(entries[1].is_dir);
    }

    #[test]
    fn rename_replaces_the_destination() {
        let dir = TempDir::new().unwrap();
        let env = std_env();
        let from = dir.path().join("from");
        let to = dir.path().join("to");
        env.write(&from, b"new").unwrap();
        env.write(&to, b"old").unwrap();
        env.rename(&from, &to).unwrap();
        assert_eq!(env.read(&to).unwrap(), b"new");
        assert!(!env.exists(&from));
    }

    #[test]
    fn sync_dir_matches_the_declared_capability() {
        let dir = TempDir::new().unwrap();
        let env = std_env();
        env.sync_dir(dir.path()).unwrap();
        assert_eq!(env.capabilities().sync_dir, SUPPORTS_DIR_SYNC);
    }

    #[test]
    fn hard_link_shares_content() {
        let dir = TempDir::new().unwrap();
        let env = std_env();
        let src = dir.path().join("src");
        let dst = dir.path().join("dst");
        env.write(&src, b"linked").unwrap();
        env.hard_link(&src, &dst).unwrap();
        assert_eq!(env.read(&dst).unwrap(), b"linked");
    }

    #[test]
    fn spawn_runs_the_body_and_join_waits_for_it() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};

        let env = std_env();
        let done = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&done);
        let handle = env
            .spawn(
                "lark-env-test",
                Box::new(move || flag.store(true, Ordering::Release)),
            )
            .unwrap();
        handle.join();
        assert!(done.load(Ordering::Acquire));
    }

    #[test]
    fn the_clocks_report_something_on_a_hosted_target() {
        let env = std_env();
        assert!(env.now_micros().is_some());
        assert!(env.unix_secs().is_some());
    }

    #[test]
    fn set_len_truncates_the_file() {
        let dir = TempDir::new().unwrap();
        let env = std_env();
        let path = dir.path().join("trunc");
        env.write(&path, b"0123456789").unwrap();
        let mut w = env.open_write(&path, WriteMode::Append).unwrap();
        w.set_len(3).unwrap();
        w.sync_all().unwrap();
        assert_eq!(w.len().unwrap(), 3);
        drop(w);
        assert_eq!(env.read(&path).unwrap(), b"012");
    }

    #[test]
    fn drop_page_cache_on_a_real_file_does_not_panic() {
        let dir = TempDir::new().unwrap();
        let env = std_env();
        let path = dir.path().join("hint.bin");
        env.write(&path, &vec![0u8; 4096]).unwrap();
        env.drop_page_cache(&path);
    }

    #[test]
    fn drop_page_cache_on_a_missing_path_is_silent() {
        let dir = TempDir::new().unwrap();
        std_env().drop_page_cache(&dir.path().join("never_existed"));
    }
}
