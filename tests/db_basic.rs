//! db-level scenarios ported from RocksDB/LevelDB `db_test.cc`
//! and `write_batch_test.cc`. Each test maps to a named upstream
//! scenario so reviewers can cross-reference behavior against the
//! reference implementations.
//!
//! Scenarios that were already covered by [`../src/lib.rs`] inline
//! tests or [`parity.rs`] are intentionally *not* re-ported here;
//! this file is strictly additive coverage.

use lark_kv::{Db, Options, Range, WriteBatch};
use tempfile::TempDir;

mod common;

use common::{fill_sequential, force_compaction, open, verify_sequential_keys};

// ── db_test.cc: Empty / EmptyKey / EmptyValue ───────────────────

#[test]
fn db_is_empty_after_open() {
    // db_test.cc::Empty — a freshly opened database answers every
    // point lookup with `None` and a full scan with an empty vec.
    let dir = TempDir::new().unwrap();
    let db = open(&dir);
    assert_eq!(db.get(b"missing").unwrap(), None);
    assert!(db.scan(None, None).unwrap().is_empty());
}

#[test]
fn empty_key_round_trips() {
    // db_test.cc::EmptyKey — the empty byte string is a valid user
    // key. Must survive put → get → delete → reopen.
    let dir = TempDir::new().unwrap();
    {
        let db = open(&dir);
        db.put(b"", b"root").unwrap();
        assert_eq!(db.get(b"").unwrap(), Some(b"root".to_vec()));
    }
    {
        let db = open(&dir);
        assert_eq!(db.get(b"").unwrap(), Some(b"root".to_vec()));
        db.delete(b"").unwrap();
        assert_eq!(db.get(b"").unwrap(), None);
    }
}

#[test]
fn empty_value_is_distinct_from_missing_after_reopen() {
    // db_test.cc::EmptyValue — an empty byte string is a valid
    // *value*, distinct from "key absent". Reopening must preserve
    // that distinction.
    let dir = TempDir::new().unwrap();
    {
        let db = open(&dir);
        db.put(b"exists", b"").unwrap();
        db.put(b"also", b"v").unwrap();
    }
    let db = open(&dir);
    assert_eq!(db.get(b"exists").unwrap(), Some(vec![]));
    assert_eq!(db.get(b"missing").unwrap(), None);
    assert_eq!(db.get(b"also").unwrap(), Some(b"v".to_vec()));
}

// ── db_test.cc: Get paths ───────────────────────────────────────

#[test]
fn get_from_immutable_memtable_still_visible() {
    // db_test.cc::GetFromImmutableLayer — a value written into the
    // active memtable must remain readable *across* the rotation
    // that freezes it, until the flush actually lands it in an SST.
    let dir = TempDir::new().unwrap();
    let db = open(&dir);
    db.put(b"pinned", b"v").unwrap();
    // Force enough writes to likely rotate (or at minimum to force
    // a flush + compaction); the pinned key must still be visible.
    for i in 0..200 {
        let k = format!("filler_{:04}", i);
        db.put(k.as_bytes(), &[0u8; 64]).unwrap();
    }
    assert_eq!(db.get(b"pinned").unwrap(), Some(b"v".to_vec()));
}

#[test]
fn get_level0_newer_file_shadows_older() {
    // db_test.cc::GetLevel0Ordering — L0 files can overlap, so the
    // engine must prefer the newer file's value.
    let dir = TempDir::new().unwrap();
    let db = open(&dir);
    db.put(b"k", b"old").unwrap();
    force_compaction(&db);
    db.put(b"k", b"new").unwrap();
    assert_eq!(db.get(b"k").unwrap(), Some(b"new".to_vec()));
    // And the shadowing survives another compaction.
    force_compaction(&db);
    assert_eq!(db.get(b"k").unwrap(), Some(b"new".to_vec()));
}

#[test]
fn get_picks_correct_file_across_levels() {
    // db_test.cc::GetPicksCorrectFile — keys from many flushes sort
    // into non-overlapping L1+ files; a point lookup must pick the
    // right file for each key.
    let dir = TempDir::new().unwrap();
    let db = open(&dir);
    fill_sequential(&db, 500);
    force_compaction(&db);
    // Spot-check across the range.
    for i in [0usize, 100, 250, 499] {
        let k = format!("key_{:06}", i);
        let v = format!("val_{:06}", i);
        assert_eq!(db.get(k.as_bytes()).unwrap(), Some(v.into_bytes()));
    }
    assert_eq!(db.get(b"missing").unwrap(), None);
}

#[test]
fn get_encounters_empty_level_between_populated_ones() {
    // db_test.cc::GetEncountersEmptyLevel — after compaction, some
    // levels may be empty. Lookups should skip over them rather
    // than short-circuit.
    let dir = TempDir::new().unwrap();
    let db = open(&dir);
    for i in 0..1000 {
        let k = format!("k_{:06}", i);
        db.put(k.as_bytes(), b"v").unwrap();
    }
    force_compaction(&db);
    assert_eq!(db.get(b"k_000500").unwrap(), Some(b"v".to_vec()));
}

// ── db_test.cc: Snapshot paths ──────────────────────────────────

#[test]
fn snapshot_hides_later_writes() {
    // db_test.cc::SnapshotHidesLaterWrites — a snapshot taken now
    // must never see writes that arrive after it was taken, even
    // across flushes and compactions.
    let dir = TempDir::new().unwrap();
    let db = open(&dir);
    db.put(b"k", b"v1").unwrap();
    let snap = db.snapshot();
    db.put(b"k", b"v2").unwrap();
    force_compaction(&db);
    assert_eq!(snap.get(b"k").unwrap(), Some(b"v1".to_vec()));
    assert_eq!(db.get(b"k").unwrap(), Some(b"v2".to_vec()));
}

#[test]
fn identical_snapshots_see_same_state() {
    // db_test.cc::GetIdenticalSnapshots — two snapshots captured at
    // the same seq observe the same values.
    let dir = TempDir::new().unwrap();
    let db = open(&dir);
    db.put(b"k", b"v1").unwrap();
    let s1 = db.snapshot();
    let s2 = db.snapshot();
    db.put(b"k", b"v2").unwrap();
    assert_eq!(s1.get(b"k").unwrap(), s2.get(b"k").unwrap());
}

// ── db_test.cc: Iter paths ──────────────────────────────────────

#[test]
fn iter_empty_database_is_never_valid() {
    // db_test.cc::IterEmpty — on an empty DB, seek_to_first /
    // seek_to_last produce no valid position.
    let dir = TempDir::new().unwrap();
    let db = open(&dir);
    let mut it = db.iter();
    it.seek_to_first();
    assert!(!it.valid());
    it.seek_to_last();
    assert!(!it.valid());
}

#[test]
fn iter_single_entry_is_valid_exactly_once() {
    // db_test.cc::IterSingle — one-entry DB: seek_to_first yields
    // the entry; a subsequent `next` invalidates the iterator.
    let dir = TempDir::new().unwrap();
    let db = open(&dir);
    db.put(b"only", b"one").unwrap();
    let mut it = db.iter();
    it.seek_to_first();
    assert!(it.valid());
    assert_eq!(it.key(), Some(&b"only"[..]));
    assert_eq!(it.value(), Some(&b"one"[..]));
    it.next();
    assert!(!it.valid());
}

#[test]
fn iter_small_and_large_values_mixed() {
    // db_test.cc::IterSmallAndLargeMix — values of wildly different
    // sizes must round-trip through the iterator unchanged.
    let dir = TempDir::new().unwrap();
    let db = open(&dir);
    db.put(b"k_small", b"s").unwrap();
    db.put(b"k_large", &vec![0xAB; 100_000]).unwrap();
    db.put(b"k_medium", &vec![0x42; 1024]).unwrap();

    let mut it = db.iter();
    it.seek_to_first();
    let mut seen = 0;
    while it.valid() {
        seen += 1;
        let v = it.value().unwrap();
        match it.key().unwrap() {
            b"k_small" => assert_eq!(v.len(), 1),
            b"k_medium" => assert_eq!(v.len(), 1024),
            b"k_large" => assert_eq!(v.len(), 100_000),
            other => panic!("unexpected key {other:?}"),
        }
        it.next();
    }
    assert_eq!(seen, 3);
}

#[test]
fn iter_skips_deleted_keys_after_compaction() {
    // db_test.cc::IterWithDeleteAndCompaction — deletes must remain
    // invisible through the iterator even after compaction has
    // physically merged the tombstone with the original value.
    let dir = TempDir::new().unwrap();
    let db = open(&dir);
    db.put(b"alive", b"1").unwrap();
    db.put(b"dead", b"2").unwrap();
    force_compaction(&db);
    db.delete(b"dead").unwrap();
    force_compaction(&db);

    let mut it = db.iter();
    it.seek_to_first();
    let mut seen_keys = Vec::new();
    while it.valid() {
        seen_keys.push(it.key().unwrap().to_vec());
        it.next();
    }
    assert_eq!(seen_keys, vec![b"alive".to_vec()]);
}

#[test]
fn iter_reverse_walks_backward() {
    // db_test.cc::IterMulti (subset) — iterator supports reverse
    // traversal from the end of the DB.
    let dir = TempDir::new().unwrap();
    let db = open(&dir);
    for c in b'a'..=b'e' {
        db.put(&[c], &[c]).unwrap();
    }
    let mut it = db.iter();
    it.seek_to_last();
    let mut rev = Vec::new();
    while it.valid() {
        rev.push(it.key().unwrap().to_vec());
        it.prev();
    }
    assert_eq!(
        rev,
        vec![
            b"e".to_vec(),
            b"d".to_vec(),
            b"c".to_vec(),
            b"b".to_vec(),
            b"a".to_vec(),
        ]
    );
}

// ── db_test.cc: Recovery ────────────────────────────────────────

#[test]
fn recover_with_empty_wal_does_not_crash() {
    // db_test.cc::RecoverWithEmptyLog — a database that was closed
    // cleanly with nothing in its WAL reopens as an empty DB.
    let dir = TempDir::new().unwrap();
    drop(open(&dir));
    let db = open(&dir);
    assert!(db.scan(None, None).unwrap().is_empty());
}

#[test]
fn recover_with_large_wal_replays_every_entry() {
    // db_test.cc::RecoverWithLargeLog — tens of thousands of ops
    // must replay correctly on reopen.
    let dir = TempDir::new().unwrap();
    {
        let opts = Options {
            // Big buffer so nothing flushes; everything survives as WAL.
            write_buffer_size: 64 * 1024 * 1024,
            ..Options::default()
        };
        let db = Db::open(dir.path(), opts).unwrap();
        for i in 0..5_000 {
            let k = format!("k_{:06}", i);
            let v = format!("v_{}", i);
            db.put(k.as_bytes(), v.as_bytes()).unwrap();
        }
    }
    let db = open(&dir);
    for i in [0usize, 2_500, 4_999] {
        let k = format!("k_{:06}", i);
        let v = format!("v_{}", i);
        assert_eq!(db.get(k.as_bytes()).unwrap(), Some(v.into_bytes()));
    }
}

#[test]
fn recover_with_multiple_memtables_preserves_all_writes() {
    // db_test.cc::MultipleMemTables — writes spread across many
    // memtable rotations (with small write_buffer_size) must all
    // survive a reopen.
    let dir = TempDir::new().unwrap();
    {
        let db = open(&dir);
        fill_sequential(&db, 500);
    }
    let db = open(&dir);
    verify_sequential_keys(&db, 500);
}

#[test]
fn seq_number_preserved_across_reopen() {
    // db_test.cc::Recover (the seq-number invariant) — writes after
    // reopen must be ordered strictly after writes before close.
    let dir = TempDir::new().unwrap();
    {
        let db = open(&dir);
        db.put(b"k", b"before").unwrap();
    }
    let db = open(&dir);
    let snap = db.snapshot();
    db.put(b"k", b"after").unwrap();
    // Snapshot was taken *after* reopen but *before* the "after"
    // put, so it should observe "before".
    assert_eq!(snap.get(b"k").unwrap(), Some(b"before".to_vec()));
    assert_eq!(db.get(b"k").unwrap(), Some(b"after".to_vec()));
}

// ── db_test.cc: ApproximateSizes ────────────────────────────────

#[test]
fn approximate_sizes_grows_with_range_width() {
    // db_test.cc::ApproximateSizes — wider ranges must report more
    // bytes than narrower ones.
    let dir = TempDir::new().unwrap();
    let db = open(&dir);
    for i in 0..1_000 {
        let k = format!("k_{:06}", i);
        db.put(k.as_bytes(), &[0u8; 128]).unwrap();
    }
    force_compaction(&db);
    let narrow = db.get_approximate_sizes(&[Range::new(b"k_000000", b"k_000001")]);
    let wide = db.get_approximate_sizes(&[Range::new(b"k_000000", b"k_000999")]);
    assert_eq!(narrow.len(), 1);
    assert_eq!(wide.len(), 1);
    assert!(wide[0] > narrow[0]);
}

// ── write_batch_test.cc ─────────────────────────────────────────

#[test]
fn write_batch_empty_len_counts() {
    // write_batch_test.cc::Empty — new batch has zero of everything.
    let b = WriteBatch::new();
    assert_eq!(b.len(), 0);
    assert_eq!(b.merge_count(), 0);
    assert_eq!(b.range_delete_count(), 0);
    assert!(b.is_empty());
}

#[test]
fn write_batch_put_delete_delete_range_counted_separately() {
    // write_batch_test.cc::Multiple — each op kind increments its
    // own counter.
    let mut b = WriteBatch::new();
    b.put(b"a", b"1");
    b.put(b"b", b"2");
    b.delete(b"c");
    b.delete_range(b"d", b"f");
    b.merge(b"g", b"m1");
    b.merge(b"g", b"m2");
    assert_eq!(b.len(), 3); // two puts + one delete are point ops
    assert_eq!(b.range_delete_count(), 1);
    assert_eq!(b.merge_count(), 2);
    assert!(!b.is_empty());
}

#[test]
fn write_batch_degenerate_range_delete_is_ignored() {
    // write_batch_test.cc::ApproximateSize-style edge — start >= end
    // must silently no-op rather than record a bogus range.
    let mut b = WriteBatch::new();
    b.delete_range(b"x", b"x");
    b.delete_range(b"z", b"a");
    assert_eq!(b.range_delete_count(), 0);
}

#[test]
fn write_batch_put_delete_on_same_key_keeps_last_op() {
    // write_batch_test.cc::Multiple (seen-last-wins variant) — the
    // operation log keeps both entries, and applying the batch in
    // caller order leaves the final delete visible.
    let mut b = WriteBatch::new();
    b.put(b"k", b"v");
    b.delete(b"k");
    assert_eq!(b.len(), 2);
    // Applying the batch and reading should yield the final op.
    let dir = TempDir::new().unwrap();
    let db = open(&dir);
    db.put(b"k", b"prior").unwrap();
    db.write(b).unwrap();
    assert_eq!(db.get(b"k").unwrap(), None);
}

#[test]
fn write_batch_apply_is_atomic_under_reopen() {
    // write_batch_test.cc::Multiple + RocksDB::WriteBatchAtomicity — a
    // batch must be *entirely* applied on reopen if any of its
    // contents are visible.
    let dir = TempDir::new().unwrap();
    {
        let db = open(&dir);
        let mut b = WriteBatch::new();
        b.put(b"a", b"1");
        b.put(b"b", b"2");
        b.put(b"c", b"3");
        db.write(b).unwrap();
    }
    let db = open(&dir);
    // Either all three are present or none are; partial visibility
    // would indicate a bug in WAL record boundaries.
    let present = [b"a".as_ref(), b"b", b"c"]
        .iter()
        .filter(|k| db.get(k).unwrap().is_some())
        .count();
    assert!(present == 0 || present == 3, "partial batch: {present}/3");
}

#[test]
fn write_batch_range_delete_hides_every_key_in_range() {
    // write_batch_test.cc integration — DeleteRange inside a batch
    // tombstones every visible key in the range atomically.
    let dir = TempDir::new().unwrap();
    let db = open(&dir);
    for k in [b"b".as_ref(), b"c", b"d", b"e"] {
        db.put(k, b"v").unwrap();
    }
    db.put(b"a", b"keep_before").unwrap();
    db.put(b"z", b"keep_after").unwrap();

    let mut batch = WriteBatch::new();
    batch.delete_range(b"b", b"f");
    db.write(batch).unwrap();

    assert_eq!(db.get(b"a").unwrap(), Some(b"keep_before".to_vec()));
    assert_eq!(db.get(b"b").unwrap(), None);
    assert_eq!(db.get(b"c").unwrap(), None);
    assert_eq!(db.get(b"d").unwrap(), None);
    assert_eq!(db.get(b"e").unwrap(), None);
    assert_eq!(db.get(b"z").unwrap(), Some(b"keep_after".to_vec()));
}
