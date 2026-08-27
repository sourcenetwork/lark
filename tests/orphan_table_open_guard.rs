//! Cross-fix probe: the discarded-table open guard's "an orphan table that records no entry is a
//! crash artifact" rule reads the footer's `num_entries` and
//! `range_tombstone_size`. A V3/V4 footer is checksummed, so a damaged
//! one is refused. A V1/V2 footer is not, which is the metadata checksum fix's stated
//! deliberate hole, and here that hole feeds the discarded-table open guard: two zeroed
//! `u64`s in a legacy footer make a table that holds 200 keys claim to
//! hold none, and the guard then lets the open discard it.
//!
//! Both halves are exercised so the difference is the format version and
//! nothing else.

use std::fs;
use std::path::Path;

use lark_kv::{Db, DurabilityMode, Options, SstFileWriter};
use tempfile::TempDir;

/// A real V1 table written by the pre-change tree.
const LEGACY_V1: &[u8] = include_bytes!("fixtures/legacy_v1v2/legacy_flat.sst");

fn opts() -> Options {
    Options {
        write_buffer_size: 8 * 1024 * 1024,
        durability: DurabilityMode::Immediate,
        ..Options::default()
    }
}

fn copy_tree(from: &Path, to: &Path) {
    fs::create_dir_all(to).expect("mkdir");
    for e in fs::read_dir(from).expect("read_dir").flatten() {
        if e.file_name() == "LOCK" {
            continue;
        }
        let (src, dst) = (e.path(), to.join(e.file_name()));
        if src.is_dir() {
            copy_tree(&src, &dst);
        } else {
            fs::copy(&src, &dst).expect("copy");
        }
    }
}

/// A directory whose WAL holds `keys` fsynced writes and whose MANIFEST
/// is back at the zero length a cut before the first `AddFile` leaves.
fn wal_only_copy(keys: usize) -> TempDir {
    let live = TempDir::new().expect("tempdir");
    let db = Db::open(live.path(), opts()).expect("open");
    for i in 0..keys {
        db.put(format!("key_{i:06}").as_bytes(), b"v").expect("put");
    }
    let copy = TempDir::new().expect("tempdir");
    copy_tree(live.path(), copy.path());
    db.close().expect("close");
    drop(db);
    fs::write(copy.path().join("MANIFEST"), b"").expect("wipe manifest");
    copy
}

/// Zero the footer's `num_entries` and `range_tombstone_size`, the two
/// fields the guard reads. Footer offsets are documented in
/// `src/engine/sstable.rs`: the seven fixed fields start at the footer's
/// first byte in every version, and the magic is always the last eight.
fn lie_about_the_contents(bytes: &mut [u8], footer_size: usize) {
    let start = bytes.len() - footer_size;
    bytes[start + 8..start + 16].fill(0); // range_tombstone_size
    bytes[start + 48..start + 56].fill(0); // num_entries
}

fn modern_table(entries: usize) -> Vec<u8> {
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().join("modern.sst");
    let mut w = SstFileWriter::create(&path, &Options::default()).expect("create");
    for i in 0..entries {
        w.put(format!("ext_{i:06}").as_bytes(), b"v").expect("put");
    }
    w.finish().expect("finish");
    fs::read(&path).expect("read")
}

#[test]
fn a_modern_table_that_lies_about_its_entry_count_is_still_refused() {
    let mut bytes = modern_table(50);
    assert_eq!(
        &bytes[bytes.len() - 8..],
        &[0x03, 0x54, 0x53, 0x53, 0x4b, 0x52, 0x41, 0x4c],
        "this probe needs a V3 table",
    );
    lie_about_the_contents(&mut bytes, 72);

    let dir = wal_only_copy(10);
    fs::write(dir.path().join("sst").join("000042.sst"), &bytes).expect("write orphan");
    let err = match Db::open(dir.path(), opts()) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("a V3 footer with a forged entry count must not pass the guard"),
    };
    assert!(err.contains("000042.sst"), "got: {err}");
    println!("V3 lying footer: refused ({err})");
}

/// The same forgery on a legacy table. A failure here is not a new
/// defect on its own: it is the metadata checksum fix's un-checksummed legacy footer reaching
/// the discarded-table open guard's guard, and the consequence is that the open succeeds while a
/// table holding 200 keys is discarded.
#[test]
fn a_legacy_table_that_lies_about_its_entry_count_is_still_refused() {
    let mut bytes = LEGACY_V1.to_vec();
    assert_eq!(
        &bytes[bytes.len() - 8..],
        &[0x01, 0x54, 0x53, 0x53, 0x4b, 0x52, 0x41, 0x4c],
        "this probe needs the real V1 fixture",
    );
    lie_about_the_contents(&mut bytes, 64);

    let dir = wal_only_copy(10);
    let orphan = dir.path().join("sst").join("000042.sst");
    fs::write(&orphan, &bytes).expect("write orphan");
    match Db::open(dir.path(), opts()) {
        Err(e) => println!("V1 lying footer: refused ({e})"),
        Ok(db) => {
            let served = db.scan(None, None).expect("scan").len();
            panic!(
                "a legacy table holding 200 keys claimed to hold none and the open went \
                 through, serving {served} entries and leaving the table unreferenced. The \
                 legacy footer carries no checksum (the metadata checksum fix's stated hole), and the discarded-table open guard's guard \
                 trusts it."
            );
        }
    }
}
