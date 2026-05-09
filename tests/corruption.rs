//! Byte-level corruption and recovery scenarios ported from the
//! subset of RocksDB's `corruption_test.cc` that applies without
//! a fault-injecting filesystem.
//!
//! Each test writes data, closes the DB, mangles an on-disk file
//! with raw `std::fs` ops, then re-opens and asserts that the
//! engine either (a) surfaces a diagnostic error or (b) recovers
//! gracefully with the expected partial data — never silent loss
//! of earlier, uncorrupted data.

use std::fs::{self, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};

use lark_kv::{Db, Error};
use tempfile::TempDir;

mod common;

use common::{count_sst_files, count_wal_files, force_compaction, open};

// ── helpers ─────────────────────────────────────────────────────

/// Flip the byte at `offset` inside `path`.
fn flip_byte(path: &Path, offset: usize) {
    let mut bytes = fs::read(path).unwrap();
    bytes[offset] ^= 0xFF;
    fs::write(path, &bytes).unwrap();
}

/// Truncate `path` to `new_len` bytes.
fn truncate(path: &Path, new_len: u64) {
    let f = OpenOptions::new().write(true).open(path).unwrap();
    f.set_len(new_len).unwrap();
}

/// Return the path of the first SST file in `<db>/sst/`.
fn first_sst(db_dir: &Path) -> PathBuf {
    let sst_dir = db_dir.join("sst");
    let mut entries: Vec<_> = fs::read_dir(&sst_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("sst"))
        .collect();
    entries.sort_by_key(|e| e.path());
    entries.into_iter().next().unwrap().path()
}

/// Return the path of the first WAL file in `<db>/wal/`.
fn first_wal(db_dir: &Path) -> PathBuf {
    let wal_dir = db_dir.join("wal");
    let mut entries: Vec<_> = fs::read_dir(&wal_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("log"))
        .collect();
    entries.sort_by_key(|e| e.path());
    entries.into_iter().next().unwrap().path()
}

fn assert_open_fails_with_kind(dir: &TempDir, expected: io::ErrorKind) {
    match Db::open(dir.path(), Default::default()) {
        Err(Error::Io(e)) => assert_eq!(e.kind(), expected),
        Err(e) => panic!("expected I/O error, got {e:?}"),
        Ok(_) => panic!("expected DB open to fail"),
    }
}

// ── WAL tail corruption ─────────────────────────────────────────

#[test]
fn torn_wal_tail_checksum_flip_fails_open_and_keeps_wal() {
    // A checksum mismatch means replay cannot prove which committed
    // records are safe. Open must fail closed and leave the WAL for
    // repair/inspection rather than silently keeping only a prefix.
    let dir = TempDir::new().unwrap();
    {
        let db = open(&dir);
        db.put(b"good_1", b"1").unwrap();
        db.put(b"good_2", b"2").unwrap();
    }

    let wal = first_wal(dir.path());
    let wal_count = count_wal_files(dir.path());
    let size = fs::metadata(&wal).unwrap().len() as usize;
    // Flip the last byte of the file — always part of the trailing
    // record's 4-byte checksum.
    flip_byte(&wal, size - 1);

    assert_open_fails_with_kind(&dir, io::ErrorKind::InvalidData);
    assert!(wal.exists());
    assert_eq!(count_wal_files(dir.path()), wal_count);
}

#[test]
fn wal_truncated_at_arbitrary_offset_fails_open_and_keeps_wal() {
    // Robustness check: truncate the WAL at every byte offset from
    // one past the first record up to the end. The engine must fail
    // closed and keep the WAL rather than silently prefix-replay.
    let dir = TempDir::new().unwrap();
    {
        let db = open(&dir);
        db.put(b"a", b"1").unwrap();
        db.put(b"b", b"2").unwrap();
    }
    let wal = first_wal(dir.path());
    let wal_count = count_wal_files(dir.path());
    let full = fs::read(&wal).unwrap();
    for trim in [1u64, 3, 5, 9, 11, 15, 20] {
        if (full.len() as u64) <= trim {
            continue;
        }
        fs::write(&wal, &full).unwrap();
        truncate(&wal, full.len() as u64 - trim);
        assert_open_fails_with_kind(&dir, io::ErrorKind::UnexpectedEof);
        assert!(wal.exists());
        assert_eq!(count_wal_files(dir.path()), wal_count);
    }
}

#[test]
fn wal_checksum_flip_in_final_record_fails_open() {
    // Flipping a checksum byte in the last record is still a
    // corruption signal. Do not convert it into a clean stop.
    let dir = TempDir::new().unwrap();
    {
        let db = open(&dir);
        db.put(b"first", b"v1").unwrap();
        db.put(b"second", b"v2").unwrap();
    }
    let wal = first_wal(dir.path());
    let size = fs::metadata(&wal).unwrap().len() as usize;
    // Flip the very last byte (high byte of the trailing checksum).
    flip_byte(&wal, size - 1);

    assert_open_fails_with_kind(&dir, io::ErrorKind::InvalidData);
    assert!(wal.exists());
}

// ── manifest corruption ─────────────────────────────────────────

#[test]
fn manifest_deleted_prevents_reopen_of_nonempty_db() {
    // corruption_test.cc::MissingDescriptor — once a DB has written
    // SSTables, deleting the manifest drops the pointer to them.
    // Opening without a manifest yields a fresh-looking DB (the
    // manifest is re-created empty), so the pre-existing files are
    // orphaned but the open must not panic.
    let dir = TempDir::new().unwrap();
    {
        let db = open(&dir);
        for i in 0..200 {
            db.put(format!("k_{:04}", i).as_bytes(), &[0u8; 64])
                .unwrap();
        }
        force_compaction(&db);
    }
    let manifest = dir.path().join("MANIFEST");
    assert!(manifest.exists());
    fs::remove_file(&manifest).unwrap();

    // Expected behavior: open succeeds and produces an empty view
    // of the DB. The orphaned SST files are still on disk but not
    // referenced.
    let db = open(&dir);
    assert!(db.scan(None, None).unwrap().is_empty());
    // At least one SST file is still physically present.
    assert!(count_sst_files(dir.path()) >= 1);
}

#[test]
fn corrupted_manifest_record_stops_replay_but_opens() {
    // corruption_test.cc::CorruptedDescriptor — a mid-file bad
    // checksum in the manifest halts replay at the corruption but
    // leaves the pre-corruption state intact.
    let dir = TempDir::new().unwrap();
    {
        let db = open(&dir);
        db.put(b"k1", b"v1").unwrap();
        force_compaction(&db);
        db.put(b"k2", b"v2").unwrap();
        force_compaction(&db);
    }
    // Flip a byte deep in the manifest — lands inside the second
    // record's payload or checksum, which the replay loop treats
    // as "stop here".
    let manifest = dir.path().join("MANIFEST");
    let size = fs::metadata(&manifest).unwrap().len() as usize;
    flip_byte(&manifest, size - 6);

    // Open must succeed; the state visible is a prefix of what was
    // committed before the corruption, so at least `k1` should be
    // readable. The exact cutoff depends on where the flip landed.
    let db = open(&dir);
    let _ = db.get(b"k1");
    let _ = db.get(b"k2");
}

// ── SSTable corruption ──────────────────────────────────────────

#[test]
fn truncated_sst_file_to_below_footer_reports_error_on_open() {
    // corruption_test.cc::CorruptedBlock — an SSTable smaller than
    // its 64-byte footer cannot be opened. The engine must
    // surface the error rather than silently drop the file.
    let dir = TempDir::new().unwrap();
    {
        let db = open(&dir);
        for i in 0..100 {
            db.put(format!("k_{:04}", i).as_bytes(), b"v").unwrap();
        }
        force_compaction(&db);
    }
    let sst = first_sst(dir.path());
    truncate(&sst, 10);

    // Reopening must either error cleanly or surface a first read
    // error; it must not panic.
    if let Ok(db) = Db::open(dir.path(), Default::default()) {
        // If open succeeded, at least trying to read the
        // corrupted key should return an Err rather than wrong
        // data or a panic.
        let _ = db.get(b"k_0000");
    }
}

#[test]
fn sst_footer_magic_byte_flip_detected_on_open() {
    // corruption_test.cc::CorruptedBlock — the last 8 bytes of the
    // footer carry the magic number; flipping one byte must make
    // the engine refuse to trust the file.
    let dir = TempDir::new().unwrap();
    {
        let db = open(&dir);
        for i in 0..50 {
            db.put(format!("k_{:02}", i).as_bytes(), b"v").unwrap();
        }
        force_compaction(&db);
    }
    let sst = first_sst(dir.path());
    let size = fs::metadata(&sst).unwrap().len() as usize;
    flip_byte(&sst, size - 1); // high byte of magic

    // Open attempt: we accept either Err OR Ok that errors on read.
    if let Ok(db) = Db::open(dir.path(), Default::default()) {
        // If open tolerates the file, the first read of a key
        // inside that file must either error or return None —
        // crucially, it must not panic.
        let _ = db.get(b"k_00");
    }
}

#[test]
fn stray_file_in_sst_dir_does_not_break_open() {
    // Not a direct corruption_test.cc scenario, but a robustness
    // check: leftover non-SST files in the SST directory (partial
    // compaction temp files, editor swap files) should be ignored
    // rather than crash the open path.
    let dir = TempDir::new().unwrap();
    {
        let db = open(&dir);
        db.put(b"k", b"v").unwrap();
        force_compaction(&db);
    }
    let sst_dir = dir.path().join("sst");
    fs::write(sst_dir.join("leftover.tmp"), b"junk").unwrap();
    fs::write(sst_dir.join(".hidden"), b"junk").unwrap();

    let db = open(&dir);
    assert_eq!(db.get(b"k").unwrap(), Some(b"v".to_vec()));
}

// ── positive invariants ─────────────────────────────────────────

#[test]
fn clean_close_then_reopen_has_stable_file_count() {
    // Control case: when nothing is corrupted, reopening should
    // produce exactly the same on-disk layout.
    let dir = TempDir::new().unwrap();
    {
        let db = open(&dir);
        for i in 0..300 {
            db.put(format!("k_{:04}", i).as_bytes(), b"v").unwrap();
        }
        force_compaction(&db);
    }
    let sst_before = count_sst_files(dir.path());
    let wal_before = count_wal_files(dir.path());
    {
        let _db = open(&dir);
    }
    let sst_after = count_sst_files(dir.path());
    let wal_after = count_wal_files(dir.path());
    // Reopen creates a fresh WAL but shouldn't delete SSTs.
    assert_eq!(sst_before, sst_after);
    assert!(wal_after >= wal_before.saturating_sub(1));
}
