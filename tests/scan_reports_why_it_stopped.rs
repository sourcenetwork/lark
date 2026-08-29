//! A scan that fails mid-range must not read as one that finished.
//!
//! `Iterator` has nowhere to put a failure, so a scan that dies on a corrupt
//! block ends exactly like one that reached the end of its range. A caller
//! that only iterates sees a short answer and no reason for it. `status()`
//! is where the reason lives, and this holds it to that: after a scan over a
//! damaged store, the rows are a prefix and `status()` says so.

// Native-only. wasm-pack builds every test target for wasm32, and these use
// the filesystem, which does not exist there.
#![cfg(not(target_arch = "wasm32"))]

use std::io::{Read, Seek, SeekFrom, Write};

use regolith::{Db, Options, WriteBatch};

fn small_options() -> Options {
    Options {
        write_buffer_size: 64 * 1024,
        block_size: 4 * 1024,
        block_cache_size: 0,
        target_file_size: 256 * 1024,
        l0_compaction_trigger: 8,
        max_background_compactions: 0,
        ..Options::default()
    }
}

const KEYS: u64 = 20_000;

fn build(dir: &std::path::Path) {
    let db = Db::open(dir, small_options()).unwrap();
    let value = [b'v'; 128];
    let mut batch = WriteBatch::new();
    for i in 0..KEYS {
        batch.put(&i.to_be_bytes(), &value);
        if batch.buffered_bytes() >= 64 * 1024 {
            db.write(std::mem::take(&mut batch)).unwrap();
        }
    }
    db.write(batch).unwrap();
    db.flush().unwrap();
    db.close().unwrap();
}

/// Damage the middle of the largest SSTable, past its first data block, so a
/// scan gets going and then hits the damage rather than failing at open.
fn damage_a_data_block(dir: &std::path::Path) {
    let mut biggest: Option<(u64, std::path::PathBuf)> = None;
    for entry in walk(dir) {
        if entry.extension().and_then(|e| e.to_str()) != Some("sst") {
            continue;
        }
        let len = std::fs::metadata(&entry).unwrap().len();
        if biggest.as_ref().is_none_or(|(best, _)| len > *best) {
            biggest = Some((len, entry));
        }
    }
    let (len, path) = biggest.expect("the load must have produced an SSTable");
    let mut file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(&path)
        .unwrap();

    // A whole block's worth of garbage in the middle of the data area. The
    // checksum catches it, which is the point: the read fails rather than
    // returning wrong bytes.
    let offset = len / 3;
    file.seek(SeekFrom::Start(offset)).unwrap();
    let mut original = vec![0u8; 8 * 1024];
    file.read_exact(&mut original).unwrap();
    file.seek(SeekFrom::Start(offset)).unwrap();
    let flipped: Vec<u8> = original.iter().map(|b| !b).collect();
    file.write_all(&flipped).unwrap();
    file.sync_all().unwrap();
}

fn walk(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else {
            continue;
        };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else {
                out.push(p);
            }
        }
    }
    out
}

#[test]
fn a_scan_cut_short_by_damage_says_so_instead_of_looking_complete() {
    let dir = tempfile::tempdir().unwrap();
    build(dir.path());
    damage_a_data_block(dir.path());

    let db = Db::open(dir.path(), small_options()).unwrap();
    let mut scan = db.scan_stream(None, None).unwrap();
    let rows = scan.by_ref().count();

    // Either outcome is acceptable on its own; what is not acceptable is a
    // short scan that reports success. If the damage happened to land
    // somewhere a scan never reads, the scan is complete and Ok.
    match scan.status() {
        Err(_) => assert!(
            (rows as u64) < KEYS,
            "status reported a failure, so the scan must not also have returned every row"
        ),
        Ok(()) => assert_eq!(
            rows as u64, KEYS,
            "status reported success, so every row must be there: a short scan \
             reporting Ok is the exact failure this test exists to catch"
        ),
    }
}

#[test]
fn an_undamaged_scan_reports_success_and_returns_everything() {
    let dir = tempfile::tempdir().unwrap();
    build(dir.path());

    let db = Db::open(dir.path(), small_options()).unwrap();
    let mut scan = db.scan_stream(None, None).unwrap();
    let rows = scan.by_ref().count();
    scan.status().expect("an intact store must scan clean");
    assert_eq!(rows as u64, KEYS);
}

/// The same contract on the transaction side. A merged stream that loses its
/// snapshot cursor finishes on the buffered writes alone, which looks exactly
/// like a range that ended.
#[test]
fn a_transaction_scan_cut_short_says_so_too() {
    use regolith::TransactionDb;

    let dir = tempfile::tempdir().unwrap();
    let tdb = TransactionDb::open(dir.path(), small_options()).unwrap();
    for i in 0..10u64 {
        tdb.db().put(format!("k{i:02}").as_bytes(), b"v").unwrap();
    }

    let txn = tdb.begin_transaction();
    txn.put(b"zzz", b"buffered").unwrap();

    let mut scan = txn.scan_stream(None, None);
    let before = scan.by_ref().count();
    scan.status().expect("an intact snapshot must scan clean");
    assert_eq!(before, 11, "ten committed rows plus the buffered write");

    // Closing the database makes every iterator built afterwards carry a
    // terminal error. Without a status to consult, the merged stream would
    // return the one buffered write and look like a complete range.
    tdb.db().close().unwrap();

    let mut scan = txn.scan_stream(None, None);
    let after = scan.by_ref().count();
    assert!(
        scan.status().is_err(),
        "the snapshot side died, so a scan returning {after} of {before} rows \
         must report why rather than read as a complete range"
    );
}
