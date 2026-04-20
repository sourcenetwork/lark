//! Platform-specific hints to the OS page cache.
//!
//! Background compaction reads and writes large SSTables
//! sequentially and will not re-read them any time soon — they
//! get moved to the block cache on demand during subsequent
//! foreground reads. Without a hint, the kernel keeps those
//! pages resident and evicts hot foreground data, which is the
//! opposite of what we want.
//!
//! # What this module does
//!
//! [`drop_page_cache`] calls `posix_fadvise(fd, 0, 0, DONTNEED)`
//! on Linux, which tells the kernel it can evict any cached
//! pages backing the file. It is a hint, not a guarantee, and
//! the return value is ignored on purpose — the engine is
//! correct whether or not the hint is honored, and the only
//! failure mode is "the cache stayed warmer than we wanted",
//! which is harmless.
//!
//! On non-Linux targets the helper is a no-op. macOS offers
//! `F_NOCACHE` but it must be set *before* any read happens,
//! which would require threading a flag through the SSTable
//! reader open path. A follow-up can add that once the Linux
//! path has settled.
//!
//! # When to call
//!
//! The compaction path calls this after a background read or
//! write completes. Foreground (user) reads are never hinted —
//! those are exactly the pages we want to keep warm.

#[cfg(target_os = "linux")]
pub(crate) fn drop_page_cache(file: &std::fs::File) {
    // `rustix::fs::fadvise` is a safe wrapper around
    // `posix_fadvise`; a range of `(0, 0)` means "the whole
    // file". We deliberately ignore the return — this is a
    // best-effort hint, and a failure (e.g. the file is backed
    // by a filesystem that doesn't implement the advice) is not
    // a correctness issue.
    let _ = rustix::fs::fadvise(file, 0, 0, rustix::fs::Advice::DontNeed);
}

#[cfg(not(target_os = "linux"))]
pub(crate) fn drop_page_cache(_file: &std::fs::File) {
    // No-op on non-Linux targets. See the module-level docs.
}

/// Convenience wrapper: open the file at `path` read-only and
/// call [`drop_page_cache`] on the resulting handle. Silently
/// ignores open errors — this is a best-effort hint on a file
/// that may already be gone by the time we reach it.
pub(crate) fn drop_page_cache_by_path(path: &std::path::Path) {
    if let Ok(file) = std::fs::OpenOptions::new().read(true).open(path) {
        drop_page_cache(&file);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn drop_page_cache_on_real_file_does_not_panic() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("hint.bin");
        std::fs::write(&path, vec![0u8; 4096]).unwrap();
        let file = std::fs::File::open(&path).unwrap();
        drop_page_cache(&file);
    }

    #[test]
    fn drop_page_cache_by_path_silently_ignores_missing_file() {
        let dir = TempDir::new().unwrap();
        drop_page_cache_by_path(&dir.path().join("never_existed"));
    }
}
