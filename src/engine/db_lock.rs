use std::fs::{File, OpenOptions};
use std::io;
use std::path::Path;
#[cfg(not(any(unix, windows)))]
use std::path::PathBuf;

const LOCK_FILE: &str = "LOCK";

pub(crate) struct DbDirectoryLock {
    file: File,
    #[cfg(not(any(unix, windows)))]
    path: PathBuf,
}

impl DbDirectoryLock {
    pub(crate) fn acquire_exclusive(db_dir: &Path) -> io::Result<Self> {
        std::fs::create_dir_all(db_dir)?;
        let path = db_dir.join(LOCK_FILE);
        let file = open_lock_file(&path).map_err(|e| lock_error(&path, e))?;

        lock_exclusive(&file).map_err(|e| lock_error(&path, e))?;

        Ok(Self {
            file,
            #[cfg(not(any(unix, windows)))]
            path,
        })
    }

    pub(crate) fn acquire_shared(db_dir: &Path) -> io::Result<Self> {
        let path = db_dir.join(LOCK_FILE);
        let file = open_lock_file_read_only(&path).map_err(|e| lock_error(&path, e))?;

        lock_shared(&file).map_err(|e| lock_error(&path, e))?;

        Ok(Self {
            file,
            #[cfg(not(any(unix, windows)))]
            path,
        })
    }
}

impl Drop for DbDirectoryLock {
    fn drop(&mut self) {
        unlock(&self.file);

        #[cfg(not(any(unix, windows)))]
        {
            let _ = std::fs::remove_file(&self.path);
        }
    }
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

#[cfg(not(any(unix, windows)))]
fn open_lock_file(path: &Path) -> io::Result<File> {
    OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(path)
}

fn open_lock_file_read_only(path: &Path) -> io::Result<File> {
    OpenOptions::new().read(true).open(path)
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

#[cfg(not(unix))]
fn lock_exclusive(_file: &File) -> io::Result<()> {
    Ok(())
}

#[cfg(not(unix))]
fn lock_shared(_file: &File) -> io::Result<()> {
    Ok(())
}

#[cfg(not(unix))]
fn unlock(_file: &File) {}
