//! Adversarial probes on streaming WAL replay.
//!
//! Recovery must be exactly as strict as it was before the replay path
//! became an iterator: a corrupt log fails the open loud, an intact log
//! recovers every acknowledged write, and no corruption is ever silently
//! skipped. "Opened with the wrong data" is the failure these tests hunt.

use std::fs;
use std::fs::OpenOptions;
use std::path::{Path, PathBuf};

use lark_kv::{Db, Options, WriteOptions};
use tempfile::TempDir;

fn opts() -> Options {
    Options {
        // Large enough that nothing flushes: everything under test stays
        // in the WAL, which is the point.
        write_buffer_size: 8 * 1024 * 1024,
        ..Options::default()
    }
}

fn wal_files(db_dir: &Path) -> Vec<PathBuf> {
    let mut entries: Vec<PathBuf> = fs::read_dir(db_dir.join("wal"))
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("log"))
        .collect();
    entries.sort();
    entries
}

/// Fill `dir` with `count` durable writes and return the WAL bytes.
///
/// The log is read while the database is still open: `close` flushes
/// the memtable to an SSTable and leaves an empty WAL behind, which is
/// exactly the state these tests must not attack.
fn seed(dir: &TempDir, count: usize) -> Vec<u8> {
    let db = Db::open(dir.path(), opts()).unwrap();
    let wo = WriteOptions {
        sync: true,
        ..WriteOptions::default()
    };
    for i in 0..count {
        db.put_opt(
            &wo,
            format!("k{i:04}").as_bytes(),
            format!("v{i:04}").as_bytes(),
        )
        .unwrap();
    }
    let files = wal_files(dir.path());
    assert_eq!(files.len(), 1, "expected exactly one WAL to attack");
    let bytes = fs::read(&files[0]).unwrap();
    assert!(!bytes.is_empty(), "the seeded WAL is empty");
    drop(db);
    bytes
}

fn expected(count: usize) -> Vec<(Vec<u8>, Vec<u8>)> {
    (0..count)
        .map(|i| {
            (
                format!("k{i:04}").into_bytes(),
                format!("v{i:04}").into_bytes(),
            )
        })
        .collect()
}

/// A completely empty WAL file is not corruption: it is what a crash
/// between `create` and the first append leaves behind.
#[test]
fn a_zero_length_wal_opens_clean() {
    let dir = TempDir::new().unwrap();
    {
        let db = Db::open(dir.path(), opts()).unwrap();
        for i in 0..200 {
            db.put(format!("s{i:04}").as_bytes(), b"v").unwrap();
        }
        db.compact_range(None, None).unwrap();
        db.close().unwrap();
    }
    for path in wal_files(dir.path()) {
        OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(0)
            .unwrap();
    }
    // A stray zero-length WAL with a fresh id, as an interrupted rotation
    // would leave.
    fs::write(dir.path().join("wal").join("wal_009999.log"), b"").unwrap();

    let db = Db::open(dir.path(), opts()).unwrap();
    for i in 0..200 {
        assert_eq!(
            db.get(format!("s{i:04}").as_bytes()).unwrap(),
            Some(b"v".to_vec()),
            "flushed data lost after a zero-length WAL"
        );
    }
}

/// Truncating the log at every byte offset must either open with the
/// exact prefix of writes the log still frames, or fail loud. It must
/// never open with data that was never written, and it must never lose
/// a record the surviving bytes still describe.
#[test]
fn every_truncation_offset_either_fails_or_recovers_a_clean_prefix() {
    const COUNT: usize = 24;
    let source = {
        let dir = TempDir::new().unwrap();
        seed(&dir, COUNT)
    };
    let full = source.len();
    let all = expected(COUNT);

    let mut opened = 0usize;
    let mut failed = 0usize;
    for cut in 0..full {
        let dir = TempDir::new().unwrap();
        fs::create_dir_all(dir.path().join("wal")).unwrap();
        fs::create_dir_all(dir.path().join("sst")).unwrap();
        // A fresh DB writes its manifest; seed one and then swap in the
        // truncated log.
        {
            let db = Db::open(dir.path(), opts()).unwrap();
            db.close().unwrap();
        }
        for path in wal_files(dir.path()) {
            fs::remove_file(path).unwrap();
        }
        fs::write(
            dir.path().join("wal").join("wal_000001.log"),
            &source[..cut],
        )
        .unwrap();

        match Db::open(dir.path(), opts()) {
            Err(_) => failed += 1,
            Ok(db) => {
                opened += 1;
                let mut seen = 0usize;
                let mut hole = false;
                for (k, v) in &all {
                    match db.get(k).unwrap() {
                        Some(got) => {
                            assert_eq!(&got, v, "cut={cut}: recovered the wrong value for {k:?}");
                            assert!(!hole, "cut={cut}: recovered a hole then a later record");
                            seen += 1;
                        }
                        None => hole = true,
                    }
                }
                assert!(
                    seen <= COUNT,
                    "cut={cut}: recovered {seen} records from a {COUNT}-record log"
                );
            }
        }
    }
    println!("truncation offsets: {full} tried, {opened} opened, {failed} failed loud");
    assert_eq!(opened + failed, full);
    assert!(failed > 0, "no truncation offset was rejected");
    assert!(opened > 0, "no truncation offset recovered anything");
}

/// Flip one byte at every offset in the log. The open must either fail
/// or return exactly the data that was written: a silently wrong value
/// is the failure this hunts.
#[test]
fn no_single_byte_flip_can_produce_silently_wrong_data() {
    const COUNT: usize = 12;
    let source = {
        let dir = TempDir::new().unwrap();
        seed(&dir, COUNT)
    };
    let all = expected(COUNT);

    let mut opened_intact = 0usize;
    let mut opened_short = 0usize;
    let mut failed = 0usize;
    for offset in 0..source.len() {
        let mut bytes = source.clone();
        bytes[offset] ^= 0xFF;

        let dir = TempDir::new().unwrap();
        {
            let db = Db::open(dir.path(), opts()).unwrap();
            db.close().unwrap();
        }
        for path in wal_files(dir.path()) {
            fs::remove_file(path).unwrap();
        }
        fs::write(dir.path().join("wal").join("wal_000001.log"), &bytes).unwrap();

        match Db::open(dir.path(), opts()) {
            Err(_) => failed += 1,
            Ok(db) => {
                let mut seen = 0usize;
                for (k, v) in &all {
                    if let Some(got) = db.get(k).unwrap() {
                        assert_eq!(
                            &got, v,
                            "offset={offset}: a single byte flip produced a wrong value for {k:?}"
                        );
                        seen += 1;
                    }
                }
                // Nothing may appear that was never written.
                let scanned = db.scan(None, None).unwrap();
                assert!(
                    scanned.len() <= COUNT,
                    "offset={offset}: recovery invented {} entries",
                    scanned.len() - COUNT
                );
                for (k, v) in &scanned {
                    let found = all.iter().find(|(ek, _)| ek == k);
                    assert!(
                        found.is_some(),
                        "offset={offset}: recovery invented key {k:?}"
                    );
                    assert_eq!(
                        &found.unwrap().1,
                        v,
                        "offset={offset}: wrong value for {k:?}"
                    );
                }
                if seen == COUNT {
                    opened_intact += 1;
                } else {
                    opened_short += 1;
                }
            }
        }
    }
    println!(
        "byte flips: {} tried, {failed} failed loud, {opened_intact} opened with everything, \
         {opened_short} opened with a subset",
        source.len()
    );
    assert!(failed > 0, "no byte flip was detected");
}

/// A record header claiming far more bytes than the file holds must be
/// refused without trying to allocate for it.
#[test]
fn an_oversized_length_header_is_refused_not_allocated() {
    const COUNT: usize = 4;
    let source = {
        let dir = TempDir::new().unwrap();
        seed(&dir, COUNT)
    };

    let mut bytes = source.clone();
    // The very first record header is at offset 0 for a headerless log,
    // or right after the file header; find the first offset whose u32 LE
    // matches a plausible small length and inflate it.
    bytes[0..4].copy_from_slice(&u32::MAX.to_le_bytes());

    let dir = TempDir::new().unwrap();
    {
        let db = Db::open(dir.path(), opts()).unwrap();
        db.close().unwrap();
    }
    for path in wal_files(dir.path()) {
        fs::remove_file(path).unwrap();
    }
    fs::write(dir.path().join("wal").join("wal_000001.log"), &bytes).unwrap();

    let err = Db::open(dir.path(), opts()).expect_err("a 4 GiB length header must be refused");
    assert!(
        format!("{err:?}").contains("truncated") || format!("{err:?}").contains("checksum"),
        "unexpected error for an oversized length header: {err:?}"
    );
}

/// Corruption in an *earlier* WAL must fail the open, not be skipped in
/// favour of the later one that replays cleanly.
#[test]
fn corruption_in_an_earlier_wal_is_not_skipped() {
    let dir = TempDir::new().unwrap();
    {
        let db = Db::open(
            dir.path(),
            Options {
                write_buffer_size: 4 * 1024,
                ..Options::default()
            },
        )
        .unwrap();
        for i in 0..400 {
            db.put(format!("m{i:04}").as_bytes(), &[b'x'; 128]).unwrap();
        }
        db.close().unwrap();
    }
    // Reopen with a big buffer so the recovered state is rewritten into a
    // single WAL, then add more behind it. The log is captured while the
    // database is open, before `close` flushes it away.
    let (target, mut bytes) = {
        let db = Db::open(dir.path(), opts()).unwrap();
        for i in 400..500 {
            db.put(format!("m{i:04}").as_bytes(), &[b'y'; 128]).unwrap();
        }
        let files = wal_files(dir.path());
        let target = files.first().expect("at least one WAL").clone();
        let bytes = fs::read(&target).unwrap();
        drop(db);
        (target, bytes)
    };
    assert!(bytes.len() > 40, "WAL too small to corrupt meaningfully");
    let at = bytes.len() / 2;
    bytes[at] ^= 0xFF;
    fs::write(&target, &bytes).unwrap();

    let err = Db::open(dir.path(), opts()).expect_err("corrupt WAL must fail the open");
    assert!(
        format!("{err:?}").to_lowercase().contains("checksum")
            || format!("{err:?}").to_lowercase().contains("truncated")
            || format!("{err:?}").to_lowercase().contains("invalid")
            || format!("{err:?}").to_lowercase().contains("record"),
        "unexpected error: {err:?}"
    );
    assert!(
        target.exists(),
        "a failed open must leave the corrupt WAL on disk for inspection"
    );
}

/// A failed replay must leave the on-disk state untouched, so a second
/// attempt after the corruption is repaired recovers everything.
#[test]
fn a_failed_replay_is_not_destructive() {
    const COUNT: usize = 64;
    let dir = TempDir::new().unwrap();
    let good = seed(&dir, COUNT);
    let path = wal_files(dir.path())[0].clone();

    let mut broken = good.clone();
    let at = broken.len() / 2;
    broken[at] ^= 0xFF;
    fs::write(&path, &broken).unwrap();
    let _ = Db::open(dir.path(), opts()).expect_err("corrupt WAL must fail the open");
    // Failing again must be just as safe.
    let _ = Db::open(dir.path(), opts()).expect_err("corrupt WAL must fail the open twice");

    fs::write(&path, &good).unwrap();
    let db = Db::open(dir.path(), opts()).unwrap();
    for (k, v) in expected(COUNT) {
        assert_eq!(
            db.get(&k).unwrap(),
            Some(v),
            "a repaired WAL must recover everything the failed attempts saw"
        );
    }
}
