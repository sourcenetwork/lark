//! Independent backward-compatibility probe for G24, against bytes this
//! tree did not write.
//!
//! The fixtures under `tests/fixtures/legacy_from_base/` were produced by
//! building the pre-change tree (`git archive HEAD`, commit `d1ec2e7`)
//! out of tree and running its own `SstFileWriter` and `Db`. The same
//! pre-change build was then made to read the fixtures already checked in
//! under `legacy_v1v2/`, which is what proves those are legacy bytes
//! rather than a re-creation of the old layout.
//!
//! What this adds over the existing compat test: a legacy database whose
//! only SSTable carries a **range-tombstone block**, the one metadata
//! region the shipped compat fixtures never exercise, plus a partitioned
//! table with many more index leaves.

use std::fs;

use lark_kv::{Db, IngestOptions, Options};
use tempfile::TempDir;

const FRESH_V1: &[u8] = include_bytes!("fixtures/legacy_from_base/fresh_v1.sst");
const FRESH_V2: &[u8] = include_bytes!("fixtures/legacy_from_base/fresh_v2.sst");
const DB_MANIFEST: &[u8] = include_bytes!("fixtures/legacy_from_base/db/MANIFEST");
const DB_TABLE: &[u8] = include_bytes!("fixtures/legacy_from_base/db/sst/000020.sst");

const MAGIC_V1_LE: [u8; 8] = [0x01, 0x54, 0x53, 0x53, 0x4b, 0x52, 0x41, 0x4c];
const MAGIC_V2_LE: [u8; 8] = [0x02, 0x54, 0x53, 0x53, 0x4b, 0x52, 0x41, 0x4c];

fn assert_magic(bytes: &[u8], want: [u8; 8], what: &str) {
    assert_eq!(
        &bytes[bytes.len() - 8..],
        &want,
        "{what} is not legacy format any more; the fixture was regenerated",
    );
}

fn ingest(bytes: &[u8], name: &str) -> (TempDir, TempDir, Db) {
    let hold = TempDir::new().expect("tempdir");
    let sst = hold.path().join(name);
    fs::write(&sst, bytes).expect("write fixture");
    let db_dir = TempDir::new().expect("tempdir");
    let db = Db::open(db_dir.path(), Options::default()).expect("open");
    db.ingest_external_files(&[sst], IngestOptions::default())
        .expect("legacy table must ingest");
    (hold, db_dir, db)
}

#[test]
fn a_flat_legacy_table_written_by_the_pre_change_tree_reads_back_whole() {
    assert_magic(FRESH_V1, MAGIC_V1_LE, "fresh_v1.sst");
    let (_h, _d, db) = ingest(FRESH_V1, "fresh_v1.sst");
    for i in 0..300 {
        assert_eq!(
            db.get(format!("fk_{i:05}").as_bytes()).expect("get"),
            Some(format!("fv_{i:05}").into_bytes()),
            "fk_{i:05} lost",
        );
    }
    assert_eq!(db.scan(None, None).expect("scan").len(), 300);
}

#[test]
fn a_partitioned_legacy_table_written_by_the_pre_change_tree_reads_back_whole() {
    assert_magic(FRESH_V2, MAGIC_V2_LE, "fresh_v2.sst");
    let (_h, _d, db) = ingest(FRESH_V2, "fresh_v2.sst");
    for i in 0..1500 {
        assert_eq!(
            db.get(format!("pk_{i:05}").as_bytes()).expect("get"),
            Some(format!("pv_{i:05}").into_bytes()),
            "pk_{i:05} lost",
        );
    }
    let mut it = db.iter();
    it.seek_to_last();
    let mut n = 0usize;
    while it.valid() {
        n += 1;
        it.prev();
    }
    it.status().expect("reverse scan");
    assert_eq!(n, 1500, "reverse scan over a legacy partitioned table");
}

/// The legacy range-tombstone block: a V1 table written by the
/// pre-change tree that carries a `delete_range` and a point delete.
/// That block gained a 4-byte trailer in V3/V4, so a legacy one must
/// still parse with no trailer at all.
#[test]
fn a_legacy_database_with_a_range_tombstone_block_still_serves_it() {
    assert_magic(DB_TABLE, MAGIC_V1_LE, "the legacy database's SSTable");
    let root = TempDir::new().expect("tempdir");
    let db_dir = root.path().join("db");
    fs::create_dir_all(db_dir.join("sst")).expect("mkdir");
    fs::create_dir_all(db_dir.join("wal")).expect("mkdir");
    fs::write(db_dir.join("MANIFEST"), DB_MANIFEST).expect("manifest");
    fs::write(db_dir.join("sst").join("000020.sst"), DB_TABLE).expect("table");

    let db = Db::open(&db_dir, Options::default()).expect("legacy db must open");
    let mut wrong = Vec::new();
    for i in 0..3000usize {
        let key = format!("dk_{i:06}");
        let deleted = (100..200).contains(&i) || i == 2999;
        let want = if deleted {
            None
        } else {
            Some(format!("dv_{i:06}").into_bytes())
        };
        let got = db.get(key.as_bytes()).expect("get");
        if got != want {
            wrong.push(format!("{key}: want {want:?}, got {got:?}"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n  "));
    assert_eq!(
        db.scan(None, None).expect("scan").len(),
        3000 - 100 - 1,
        "the legacy range tombstone must still hide exactly its range",
    );
    println!("legacy range-tombstone block: 3000 keys checked, 101 correctly hidden");
}

/// Compaction must be able to rewrite a legacy table into the new format
/// without losing the legacy range tombstone's effect.
#[test]
fn compacting_a_legacy_range_tombstone_table_keeps_its_effect() {
    let root = TempDir::new().expect("tempdir");
    let db_dir = root.path().join("db");
    fs::create_dir_all(db_dir.join("sst")).expect("mkdir");
    fs::create_dir_all(db_dir.join("wal")).expect("mkdir");
    fs::write(db_dir.join("MANIFEST"), DB_MANIFEST).expect("manifest");
    fs::write(db_dir.join("sst").join("000020.sst"), DB_TABLE).expect("table");

    let db = Db::open(&db_dir, Options::default()).expect("open");
    db.put(b"zz_after", b"1").expect("put");
    db.compact_range(None, None).expect("compact");
    db.close().expect("close");
    drop(db);

    let mut legacy = 0usize;
    let mut upgraded = 0usize;
    for e in fs::read_dir(db_dir.join("sst"))
        .expect("read_dir")
        .flatten()
    {
        let b = fs::read(e.path()).expect("read");
        if b[b.len() - 8..] == MAGIC_V1_LE {
            legacy += 1;
        } else {
            upgraded += 1;
        }
    }

    let db = Db::open(&db_dir, Options::default()).expect("reopen");
    let mut wrong = Vec::new();
    for i in 0..3000usize {
        let key = format!("dk_{i:06}");
        let deleted = (100..200).contains(&i) || i == 2999;
        let want = if deleted {
            None
        } else {
            Some(format!("dv_{i:06}").into_bytes())
        };
        if db.get(key.as_bytes()).expect("get") != want {
            wrong.push(key);
        }
    }
    assert!(
        wrong.is_empty(),
        "{} keys wrong after compacting a legacy table: {:?}",
        wrong.len(),
        &wrong[..wrong.len().min(10)],
    );
    println!(
        "after compaction: {legacy} legacy table(s), {upgraded} upgraded, range tombstone intact"
    );
}
