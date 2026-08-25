//! Shared helpers for lark-kv integration tests.
//!
//! Each file under `tests/` is compiled as its own integration-test
//! crate, so Rust has no inherent way to share code between them. The
//! idiomatic workaround is a `common/mod.rs` subdirectory: it is
//! *not* treated as a standalone test binary, but any test file can
//! pull it in with `mod common;`.
//!
//! Keep this module small and opinionated - helpers here get
//! recompiled once per test binary, so bloat costs compile time.

#![allow(dead_code)]

use std::fs;
use std::path::Path;

use lark_kv::{Db, Options};
use tempfile::TempDir;

/// Options with a small write buffer (4 KiB) so memtable flushes
/// trigger quickly - useful for exercising flush and compaction paths
/// in tests without having to write megabytes of data.
pub fn small_opts() -> Options {
    Options {
        write_buffer_size: 4 * 1024,
        ..Options::default()
    }
}

/// Open a fresh lark DB in `dir` using [`small_opts`].
pub fn open(dir: &TempDir) -> Db {
    Db::open(dir.path(), small_opts()).unwrap()
}

/// Put `count` sequential keys `key_000000..key_{count-1:06}` with
/// value `val_{i:06}`. Predictable filler for write-path tests.
pub fn fill_sequential(db: &Db, count: usize) {
    for i in 0..count {
        let k = format!("key_{:06}", i);
        let v = format!("val_{:06}", i);
        db.put(k.as_bytes(), v.as_bytes()).unwrap();
    }
}

/// Verify the `count` keys produced by [`fill_sequential`] are all
/// present with their expected values.
pub fn verify_sequential_keys(db: &Db, count: usize) {
    for i in 0..count {
        let k = format!("key_{:06}", i);
        let v = format!("val_{:06}", i);
        assert_eq!(
            db.get(k.as_bytes()).unwrap(),
            Some(v.as_bytes().to_vec()),
            "key {:?} missing or wrong value",
            k,
        );
    }
}

/// Count `*.sst` files under `<db_dir>/sst/`.
pub fn count_sst_files(db_dir: &Path) -> usize {
    count_with_extension(&db_dir.join("sst"), "sst")
}

/// Count `*.log` files under `<db_dir>/wal/`.
pub fn count_wal_files(db_dir: &Path) -> usize {
    count_with_extension(&db_dir.join("wal"), "log")
}

fn count_with_extension(dir: &Path, ext: &str) -> usize {
    if !dir.is_dir() {
        return 0;
    }
    fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some(ext))
        .count()
}

/// Force a full flush + compaction by calling `compact_range(None, None)`.
/// Blocks until the compaction scheduler reports completion.
pub fn force_compaction(db: &Db) {
    db.compact_range(None, None).unwrap();
}
