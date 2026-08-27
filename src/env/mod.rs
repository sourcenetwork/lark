//! The host platform, behind one trait.
//!
//! The engine performs no `std::fs`, `std::thread`, or `std::time`
//! call of its own. Everything that touches the host goes through the
//! [`Env`] installed on [`crate::Options::env`], which defaults to
//! [`StdEnv`]: the same `std::fs` + `std::thread` + `std::time` calls
//! lark made before this trait existed, in the same order, with the
//! same error kinds.
//!
//! # Why a trait object
//!
//! `Options` already carries `Arc<dyn CompactionFilter>`,
//! `Arc<dyn PrefixExtractor>`, `Arc<dyn MergeOperator>` and friends,
//! so `Arc<dyn Env>` extends the pattern that is already here rather
//! than introducing a generic parameter that would infect `Db`,
//! `Snapshot`, `Iter`, and every signature that mentions them. The
//! indirect call costs a few nanoseconds against a syscall that costs
//! hundreds, and the cached read path never reaches an `Env` at all:
//! a block-cache hit returns before any file is touched.
//!
//! # What a backend must decide
//!
//! Not every host has hard links, directory fsync, cross-process file
//! locking, or threads. [`Capabilities`] is how a backend says so and
//! how lark reports it back through [`crate::Db::capabilities`],
//! rather than claiming a guarantee it is not providing.
//!
//! # Internal naming convention
//!
//! Engine constructors that touch the host take the `Env` as their
//! first argument and carry an `_in` suffix (`Wal::create_in`,
//! `SsTableReader::open_in`). The unsuffixed names remain as
//! `#[cfg(test)]` shims over [`std_env`] so the module-local unit
//! tests keep exercising the same code through the standard
//! environment.

mod buffered;
mod db_lock;
mod mem_env;
mod types;
// The browser's Origin Private File System. Only compiled for
// `wasm32-unknown-unknown`, the one target with no `std::fs` at all;
// it is also the only target that has `js-sys` as a dependency.
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
pub mod opfs;
mod std_env;
// WASI has `std::fs` but no threads. `WasiEnv` delegates every
// filesystem call to `StdEnv` and says so where it matters: a spawn
// that explains itself instead of a bare `Unsupported`, and no
// pointless open per compacted file for a page-cache hint the
// platform does not have. See `src/env/wasi.rs`.
#[cfg(target_os = "wasi")]
mod wasi;

pub use mem_env::MemEnv;
pub use std_env::StdEnv;
pub use types::{Capabilities, DirEntry, FileMeta, WriteMode};
#[cfg(target_os = "wasi")]
pub use wasi::WasiEnv;

pub(crate) use buffered::BufferedWriter;

use std::io;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

/// Filesystem, clock, and scheduling for one database.
///
/// An implementation is shared across threads (a compaction worker
/// holds the same `Arc` a foreground writer does), so `Send + Sync`
/// are required on every target. `Debug` is required so a bug report
/// can name which environment was installed.
///
/// Paths are whatever the host understands. lark only ever joins
/// names onto the database directory it was given, so a backend with
/// a flat namespace is free to treat the whole path as an opaque key.
pub trait Env: Send + Sync + std::fmt::Debug {
    /// Create `path` and every missing parent. Succeeds when the
    /// directory already exists.
    fn create_dir_all(&self, path: &Path) -> io::Result<()>;

    /// The entries directly under `path`, in unspecified order. lark
    /// sorts whatever it needs sorted.
    fn read_dir(&self, path: &Path) -> io::Result<Vec<DirEntry>>;

    /// Open `path` for positional reading.
    fn open_read(&self, path: &Path) -> io::Result<Box<dyn ReadFile>>;

    /// Open `path` for writing, creating it if it does not exist.
    fn open_write(&self, path: &Path, mode: WriteMode) -> io::Result<Box<dyn WriteFile>>;

    /// Size and kind of the entry at `path`.
    fn metadata(&self, path: &Path) -> io::Result<FileMeta>;

    /// Remove the file at `path`.
    fn remove_file(&self, path: &Path) -> io::Result<()>;

    /// Rename `from` to `to`, replacing `to` if it exists.
    ///
    /// lark uses this for its write-new-then-rename update of the
    /// MANIFEST and of backup metadata, so a backend whose rename is
    /// not crash-atomic must report that through
    /// [`Capabilities::atomic_rename`].
    fn rename(&self, from: &Path, to: &Path) -> io::Result<()>;

    /// Create a hard link at `dst` pointing at `src`.
    ///
    /// Only reached when [`Capabilities::hard_link`] is `true`.
    fn hard_link(&self, src: &Path, dst: &Path) -> io::Result<()>;

    /// Make a directory's entries durable, so a file created,
    /// renamed, or removed inside it survives a crash once its
    /// contents are synced.
    ///
    /// An environment whose [`Capabilities::sync_dir`] is `false`
    /// returns `Ok(())` without doing anything. That flag, not this
    /// return value, is what tells the truth about the guarantee.
    fn sync_dir(&self, path: &Path) -> io::Result<()>;

    /// Exclude other processes from the database directory `path`.
    ///
    /// The lock is released when the returned guard is dropped. An
    /// environment whose [`Capabilities::file_lock`] is `false`
    /// returns a guard that excludes nothing and creates no file.
    fn lock_file(&self, path: &Path, exclusive: bool) -> io::Result<Box<dyn FileLock>>;

    /// What this environment can actually do.
    fn capabilities(&self) -> Capabilities;

    /// Monotonic microseconds from an arbitrary origin, for measuring
    /// durations.
    ///
    /// `None` on a platform with no monotonic clock. lark then records
    /// no timing at all rather than recording a zero that reads like a
    /// measurement.
    fn now_micros(&self) -> Option<u64>;

    /// Seconds since the Unix epoch.
    ///
    /// `None` on a platform with no wall clock. lark then reports the
    /// affected timestamps as absent rather than as the epoch.
    fn unix_secs(&self) -> Option<u64>;

    /// Run `body` on another thread.
    ///
    /// A single-threaded host reports
    /// [`std::io::ErrorKind::Unsupported`] here, and lark's caller
    /// turns that into an open error rather than a panic.
    fn spawn(
        &self,
        name: &str,
        body: Box<dyn FnOnce() + Send + 'static>,
    ) -> io::Result<Box<dyn JoinHandle>>;

    /// Block the calling thread for `dur`.
    ///
    /// A no-op on a platform that cannot block. The only caller is the
    /// per-write slowdown delay, which is back-pressure rather than
    /// correctness, so skipping it costs throughput smoothing and
    /// nothing else.
    fn sleep(&self, dur: Duration);

    /// Whether anything exists at `path`.
    fn exists(&self, path: &Path) -> bool {
        self.metadata(path).is_ok()
    }

    /// Whether `path` is a directory. `false` when it does not exist.
    fn is_dir(&self, path: &Path) -> bool {
        self.metadata(path).map(|m| m.is_dir).unwrap_or(false)
    }

    /// Read a whole file into memory.
    ///
    /// The length comes from the host, so it is bounded with
    /// `try_reserve_exact`: a file larger than the available heap
    /// reports [`std::io::ErrorKind::OutOfMemory`] instead of
    /// aborting the process.
    fn read(&self, path: &Path) -> io::Result<Vec<u8>> {
        let file = self.open_read(path)?;
        let len = usize::try_from(file.len()?).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{} is too large to address", path.display()),
            )
        })?;
        let mut buf = Vec::new();
        buf.try_reserve_exact(len).map_err(|_| {
            io::Error::new(
                io::ErrorKind::OutOfMemory,
                format!("cannot allocate {len} bytes to read {}", path.display()),
            )
        })?;
        buf.resize(len, 0);
        file.read_exact_at(0, &mut buf)?;
        Ok(buf)
    }

    /// Replace the whole contents of `path`, creating it if needed.
    ///
    /// Buffered like `std::fs::write`: the bytes have reached the
    /// host when this returns, but durability still needs a
    /// [`WriteFile::sync_all`] on a handle the caller keeps.
    fn write(&self, path: &Path, data: &[u8]) -> io::Result<()> {
        let mut file = self.open_write(path, WriteMode::Truncate)?;
        file.write_all(data)?;
        file.flush()
    }

    /// Hint that the host may evict any cache backing `path`.
    ///
    /// Best effort by definition: lark is correct whether or not the
    /// hint is honored, so failures are swallowed. The default does
    /// nothing.
    fn drop_page_cache(&self, path: &Path) {
        let _ = path;
    }
}

/// A file opened for reading.
///
/// Positional only. lark never relies on a shared cursor, so an
/// implementation needs no seek state and no lock around one - which
/// is why an `SsTableReader` can serve concurrent readers without
/// serializing them.
pub trait ReadFile: Send + Sync {
    /// Fill `buf` with the bytes at `offset`, or fail.
    ///
    /// Reading past the end reports
    /// [`std::io::ErrorKind::UnexpectedEof`].
    fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()>;

    /// Length of the file in bytes.
    fn len(&self) -> io::Result<u64>;

    /// Whether the file holds no bytes.
    fn is_empty(&self) -> io::Result<bool> {
        Ok(self.len()? == 0)
    }
}

/// A file opened for writing. Append-only apart from
/// [`WriteFile::set_len`], which only manifest recovery uses.
pub trait WriteFile: Send {
    /// Append every byte of `buf`.
    fn write_all(&mut self, buf: &[u8]) -> io::Result<()>;

    /// Push buffered bytes to the host. Does not imply durability.
    fn flush(&mut self) -> io::Result<()>;

    /// Make every byte written so far durable.
    ///
    /// An environment whose [`Capabilities::durable_sync`] is `false`
    /// may return `Ok(())` without providing durability.
    fn sync_all(&mut self) -> io::Result<()>;

    /// Make the file's *data* durable, without necessarily flushing
    /// metadata that no reader depends on.
    ///
    /// This is `fdatasync` rather than `fsync`. A log that is appended
    /// to and fsynced on every commit otherwise pays an inode update per
    /// commit for a size field nothing reads back: recovery finds the
    /// end of the log from the records themselves, not from the length.
    /// On an NVMe device that is a second round trip to the drive on the
    /// critical path of every durable write.
    ///
    /// Defaults to [`WriteFile::sync_all`], which is always correct and
    /// merely slower, so an environment that cannot separate the two
    /// needs no implementation.
    fn sync_data(&mut self) -> io::Result<()> {
        self.sync_all()
    }

    /// Truncate or extend the file to exactly `len` bytes, leaving the
    /// write position at `len`.
    ///
    /// The position matters: WAL rollback truncates a partly written
    /// group and then appends again, so a writer that left its cursor
    /// past the new end would write into a hole. An implementation whose
    /// writes always append satisfies this for free; one that carries an
    /// explicit offset has to move it.
    fn set_len(&mut self, len: u64) -> io::Result<()>;

    /// Length of the file in bytes, counting only what has reached
    /// the host.
    fn len(&self) -> io::Result<u64>;

    /// Whether the file holds no bytes.
    fn is_empty(&self) -> io::Result<bool> {
        Ok(self.len()? == 0)
    }
}

/// A held directory lock. The lock is released when this is dropped.
pub trait FileLock: Send + Sync {}

/// A handle to a thread started by [`Env::spawn`].
pub trait JoinHandle: Send {
    /// Wait for the thread to finish. A thread that panicked is
    /// joined like any other; the panic does not propagate.
    fn join(self: Box<Self>);
}

/// The default [`Env`] for the target being compiled.
#[cfg(not(target_os = "wasi"))]
type DefaultEnv = StdEnv;

/// The default [`Env`] for the target being compiled.
#[cfg(target_os = "wasi")]
type DefaultEnv = WasiEnv;

/// The process-wide default environment: [`StdEnv`] everywhere except
/// WASI, where `WasiEnv` takes over. (`WasiEnv` only exists when
/// compiling for WASI, so it cannot be linked from a native build.)
///
/// Both are stateless, so one instance serves every database in the
/// process and [`crate::Options::default`] hands out a clone of this
/// `Arc` rather than allocating per open. A caller that never mentions
/// an [`Env`] therefore still gets the right one for the target,
/// including on `wasm32-wasip1`.
pub fn std_env() -> Arc<dyn Env> {
    static ENV: std::sync::OnceLock<Arc<DefaultEnv>> = std::sync::OnceLock::new();
    ENV.get_or_init(|| Arc::new(DefaultEnv::new())).clone()
}

/// Monotonic microseconds from the platform clock, or `None` where
/// the platform has none.
///
/// The opt-in timing paths (performance counters, statistics
/// histograms, the token-bucket rate limiter) use this rather than an
/// [`Env`], because a caller constructs them directly and they never
/// see one. `None` disables the timing; it never records a zero.
pub(crate) fn platform_micros() -> Option<u64> {
    platform_nanos().map(|nanos| nanos / 1_000)
}

/// The same clock as [`platform_micros`], in nanoseconds.
///
/// The performance counters report nanoseconds, so they read this
/// directly rather than losing three digits and reading zero for
/// every sub-microsecond scope.
pub(crate) fn platform_nanos() -> Option<u64> {
    #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
    {
        // `std::time::Instant::now` panics on this target, and a
        // panic on a path reachable from a public API is not an
        // acceptable way to report "no clock here".
        None
    }
    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    {
        static ORIGIN: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
        let origin = ORIGIN.get_or_init(std::time::Instant::now);
        Some(origin.elapsed().as_nanos() as u64)
    }
}

/// Microseconds elapsed since `start`, or `None` when `env` has no
/// monotonic clock.
///
/// `None` means "not measured". Callers skip the recording rather
/// than publishing a zero that reads like a measurement. This is the
/// one place that arithmetic lives, so the engine and the compaction
/// worker cannot disagree about it.
pub(crate) fn elapsed_micros(env: &dyn Env, start: Option<u64>) -> Option<u64> {
    let start = start?;
    Some(env.now_micros()?.saturating_sub(start))
}

/// Sync the directory that contains `path`.
///
/// A path with no parent (the filesystem root) has nothing to sync
/// and succeeds.
pub(crate) fn sync_parent_dir(env: &dyn Env, path: &Path) -> io::Result<()> {
    match path.parent() {
        Some(parent) => env.sync_dir(parent),
        None => Ok(()),
    }
}

/// Remove `path`, then make its removal durable.
pub(crate) fn remove_file_and_sync_parent(env: &dyn Env, path: &Path) -> io::Result<()> {
    env.remove_file(path)?;
    sync_parent_dir(env, path)
}

/// A sequential [`std::io::Read`] cursor over a [`ReadFile`].
///
/// Bridges the positional read model to the byte-stream helpers that
/// hash and copy whole files.
pub(crate) struct ReadFileCursor<F> {
    file: F,
    offset: u64,
    end: u64,
}

impl<F: std::ops::Deref<Target = dyn ReadFile>> ReadFileCursor<F> {
    /// Read the whole file, from byte zero.
    pub(crate) fn new(file: F) -> io::Result<Self> {
        let end = file.len()?;
        Ok(Self {
            file,
            offset: 0,
            end,
        })
    }

    /// Length of the underlying file, read once at construction.
    pub(crate) fn len(&self) -> u64 {
        self.end
    }
}

impl<F: std::ops::Deref<Target = dyn ReadFile>> io::Read for ReadFileCursor<F> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let remaining = self.end.saturating_sub(self.offset);
        if remaining == 0 || buf.is_empty() {
            return Ok(0);
        }
        let want = remaining.min(buf.len() as u64) as usize;
        self.file.read_exact_at(self.offset, &mut buf[..want])?;
        self.offset += want as u64;
        Ok(want)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn std_env_is_shared_not_reallocated() {
        assert!(Arc::ptr_eq(&std_env(), &std_env()));
    }

    #[test]
    fn platform_micros_is_monotonic_where_it_exists() {
        if let (Some(a), Some(b)) = (platform_micros(), platform_micros()) {
            assert!(b >= a);
        }
    }

    #[test]
    fn sync_parent_dir_accepts_existing_file() {
        let dir = tempfile::TempDir::new().unwrap();
        let env = std_env();
        let path = dir.path().join("file");
        env.write(&path, b"data").unwrap();

        sync_parent_dir(&*env, &path).unwrap();
    }

    #[test]
    fn remove_file_and_sync_parent_removes_file() {
        let dir = tempfile::TempDir::new().unwrap();
        let env = std_env();
        let path = dir.path().join("file");
        env.write(&path, b"data").unwrap();

        remove_file_and_sync_parent(&*env, &path).unwrap();

        assert!(!env.exists(&path));
    }

    #[test]
    fn read_file_cursor_streams_the_whole_file() {
        use std::io::Read;

        let dir = tempfile::TempDir::new().unwrap();
        let env = std_env();
        let path = dir.path().join("streamed");
        env.write(&path, b"0123456789").unwrap();

        let file = env.open_read(&path).unwrap();
        let mut cursor = ReadFileCursor::new(&*file).unwrap();
        let mut out = Vec::new();
        cursor.read_to_end(&mut out).unwrap();
        assert_eq!(out, b"0123456789");
    }

    #[test]
    fn elapsed_micros_is_none_without_a_clock() {
        let env = MemEnv::new();
        env.set_clocks(None, None);
        assert_eq!(elapsed_micros(&env, Some(5)), None);
        env.set_clocks(Some(1_000), Some(0));
        assert_eq!(elapsed_micros(&env, Some(400)), Some(600));
        assert_eq!(elapsed_micros(&env, None), None);
    }
}
