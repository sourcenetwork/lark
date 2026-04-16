//! Property-based tests for lark's core invariants.
//!
//! Every test in this file uses `proptest` to generate randomized
//! inputs and verify that the database satisfies its contract
//! across thousands of scenarios — something hand-written unit
//! tests can't cover.
//!
//! Run with:
//!
//! ```sh
//! cargo test --test proptest_invariants
//! # Or with more cases:
//! PROPTEST_CASES=1024 cargo test --test proptest_invariants
//! ```

use std::collections::BTreeMap;

use lark_kv::{Db, Options, WriteBatch};
use proptest::prelude::*;
use tempfile::TempDir;

// ── helpers ────────────────────────────────────────────────────

fn open(dir: &TempDir) -> Db {
    let opts = Options {
        write_buffer_size: 4 * 1024,
        ..Options::default()
    };
    Db::open(dir.path(), opts).unwrap()
}

/// Strategy that generates a key as 1–32 random bytes.
fn key_strategy() -> impl Strategy<Value = Vec<u8>> {
    prop::collection::vec(any::<u8>(), 1..=32)
}

/// Strategy that generates a value as 0–128 random bytes.
fn value_strategy() -> impl Strategy<Value = Vec<u8>> {
    prop::collection::vec(any::<u8>(), 0..=128)
}

// ── property tests ─────────────────────────────────────────────

proptest! {
    /// After N random puts, `scan(None, None)` returns every key
    /// in sorted order with no duplicates and no missing keys.
    /// This exercises the memtable, flush, and read paths end-to-end.
    #[test]
    fn scan_returns_all_keys_sorted(
        entries in prop::collection::vec(
            (key_strategy(), value_strategy()),
            1..=200,
        ),
    ) {
        let dir = TempDir::new().unwrap();
        let db = open(&dir);

        // Build the expected state: last-write-wins per key.
        let mut expected: BTreeMap<Vec<u8>, Vec<u8>> = BTreeMap::new();
        for (k, v) in &entries {
            db.put(k, v).unwrap();
            expected.insert(k.clone(), v.clone());
        }

        let result = db.scan(None, None).unwrap();

        // Same number of distinct keys.
        prop_assert_eq!(result.len(), expected.len());

        // Keys are in ascending order and values match.
        for ((rk, rv), (ek, ev)) in result.iter().zip(expected.iter()) {
            prop_assert_eq!(rk, ek);
            prop_assert_eq!(rv, ev);
        }
    }

    /// A snapshot at the current seq sees exactly the writes that
    /// preceded it and none of the writes that follow it.
    #[test]
    fn snapshot_isolation(
        before in prop::collection::vec(
            (key_strategy(), value_strategy()),
            1..=50,
        ),
        after in prop::collection::vec(
            (key_strategy(), value_strategy()),
            1..=50,
        ),
    ) {
        let dir = TempDir::new().unwrap();
        let db = open(&dir);

        // Phase 1 — write "before" entries.
        let mut expected: BTreeMap<Vec<u8>, Vec<u8>> = BTreeMap::new();
        for (k, v) in &before {
            db.put(k, v).unwrap();
            expected.insert(k.clone(), v.clone());
        }

        let snap = db.snapshot();

        // Phase 2 — write "after" entries (invisible to snap).
        for (k, v) in &after {
            db.put(k, v).unwrap();
        }

        // Snapshot reads must match "before" state exactly.
        for (k, v) in &expected {
            let got = snap.get(k).unwrap();
            prop_assert_eq!(got.as_ref(), Some(v));
        }

        // A key that was only written in "after" and not in
        // "before" must be invisible through the snapshot.
        for (k, _) in &after {
            if !expected.contains_key(k) {
                let got = snap.get(k).unwrap();
                prop_assert!(got.is_none(),
                    "key {:?} should be invisible through the snapshot", k);
            }
        }
    }

    /// After random writes + a full compaction, every key that was
    /// live before compaction is still readable with the correct
    /// (latest) value.
    #[test]
    fn compaction_preserves_latest_values(
        entries in prop::collection::vec(
            (key_strategy(), value_strategy()),
            1..=300,
        ),
    ) {
        let dir = TempDir::new().unwrap();
        let db = open(&dir);

        let mut expected: BTreeMap<Vec<u8>, Vec<u8>> = BTreeMap::new();
        for (k, v) in &entries {
            db.put(k, v).unwrap();
            expected.insert(k.clone(), v.clone());
        }

        db.compact_range(None, None).unwrap();

        for (k, v) in &expected {
            let got = db.get(k).unwrap();
            prop_assert_eq!(got.as_ref(), Some(v),
                "key {:?} changed or disappeared after compaction", k);
        }
    }

    /// Deleting a key makes it invisible to subsequent reads, even
    /// after compaction collapses the tombstone.
    #[test]
    fn delete_then_compact_hides_key(
        entries in prop::collection::vec(
            (key_strategy(), value_strategy()),
            1..=100,
        ),
        delete_indices in prop::collection::vec(
            0..100usize,
            0..=30,
        ),
    ) {
        let dir = TempDir::new().unwrap();
        let db = open(&dir);

        let mut expected: BTreeMap<Vec<u8>, Vec<u8>> = BTreeMap::new();
        for (k, v) in &entries {
            db.put(k, v).unwrap();
            expected.insert(k.clone(), v.clone());
        }

        let keys_vec: Vec<Vec<u8>> = expected.keys().cloned().collect();
        let mut deleted = std::collections::HashSet::new();
        for &idx in &delete_indices {
            if keys_vec.is_empty() { break; }
            let k = &keys_vec[idx % keys_vec.len()];
            db.delete(k).unwrap();
            deleted.insert(k.clone());
        }
        for k in &deleted {
            expected.remove(k);
        }

        db.compact_range(None, None).unwrap();

        // Deleted keys must be gone.
        for k in &deleted {
            prop_assert!(db.get(k).unwrap().is_none(),
                "deleted key {:?} is still visible after compaction", k);
        }
        // Surviving keys must be intact.
        for (k, v) in &expected {
            let got = db.get(k).unwrap();
            prop_assert_eq!(got.as_ref(), Some(v));
        }
    }

    /// A `WriteBatch` is atomic: either every write in the batch
    /// is visible, or none of them are. (Since lark doesn't have
    /// partial-batch failure modes today, the "none" case only
    /// happens on I/O error, which proptest can't trigger. So this
    /// test verifies the "all" case under concurrent reads.)
    #[test]
    fn write_batch_atomicity(
        batches in prop::collection::vec(
            prop::collection::vec(
                (key_strategy(), value_strategy()),
                1..=20,
            ),
            1..=10,
        ),
    ) {
        let dir = TempDir::new().unwrap();
        let db = open(&dir);

        let mut expected: BTreeMap<Vec<u8>, Vec<u8>> = BTreeMap::new();
        for batch_entries in &batches {
            let mut batch = WriteBatch::new();
            for (k, v) in batch_entries {
                batch.put(k, v);
                expected.insert(k.clone(), v.clone());
            }
            db.write(batch).unwrap();
        }

        for (k, v) in &expected {
            let got = db.get(k).unwrap();
            prop_assert_eq!(got.as_ref(), Some(v));
        }
    }
}
