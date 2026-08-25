use std::fs;
use std::io;
use std::path::Path;

/// Sync a directory entry on platforms where directory fsync is
/// available. This makes newly-created, renamed, or removed files
/// durable across OS crashes once their contents have been synced.
#[cfg(unix)]
pub(crate) fn sync_dir(path: &Path) -> io::Result<()> {
    fs::File::open(path)?.sync_all()
}

/// Best-effort no-op on platforms without portable directory fsync.
#[cfg(not(unix))]
pub(crate) fn sync_dir(path: &Path) -> io::Result<()> {
    let _ = path;
    Ok(())
}

pub(crate) fn sync_parent_dir(path: &Path) -> io::Result<()> {
    match path.parent() {
        Some(parent) => sync_dir(parent),
        None => Ok(()),
    }
}

pub(crate) fn remove_file_and_sync_parent(path: &Path) -> io::Result<()> {
    fs::remove_file(path)?;
    sync_parent_dir(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;

    #[test]
    fn sync_parent_dir_accepts_existing_file() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("file");
        File::create(&path).unwrap().write_all(b"data").unwrap();

        sync_parent_dir(&path).unwrap();
    }

    #[test]
    fn remove_file_and_sync_parent_removes_file() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("file");
        File::create(&path).unwrap().write_all(b"data").unwrap();

        remove_file_and_sync_parent(&path).unwrap();

        assert!(!path.exists());
    }
}
