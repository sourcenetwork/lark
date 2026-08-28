//! A blind write of the value a key already holds is not a conflict.
//!
//! Two transactions that both write `K = v`, the first committing, have a
//! serial equivalent: under "first, then second" the second writes `K = v` and
//! the state is `K = v`, exactly what letting both commit produces. Refusing
//! the second rejects a correct history.
//!
//! Content-addressed callers meet this constantly, because there the key is a
//! hash of the value, so "the same key" and "the same bytes" are one statement.
//! The cases below draw the line: byte equality on a key the transaction did
//! not read, and nothing else.

use regolith::{IsolationLevel, MergeOperator, OptimisticTransactionDb, Options};

/// Sums big-endian i64 deltas, so two `+1` operands make `+2` and the
/// operation is plainly not idempotent.
struct CounterMerge;

impl MergeOperator for CounterMerge {
    fn name(&self) -> &'static str {
        "counter"
    }

    fn full_merge(&self, _key: &[u8], base: Option<&[u8]>, operands: &[&[u8]]) -> Option<Vec<u8>> {
        let mut total: i64 = match base {
            Some(bytes) if bytes.len() == 8 => i64::from_be_bytes(bytes.try_into().unwrap()),
            Some(_) => return None,
            None => 0,
        };
        for operand in operands {
            if operand.len() != 8 {
                return None;
            }
            total = total.wrapping_add(i64::from_be_bytes((*operand).try_into().unwrap()));
        }
        Some(total.to_be_bytes().to_vec())
    }
}

fn db(dir: &std::path::Path) -> OptimisticTransactionDb {
    OptimisticTransactionDb::open(dir, Options::default()).unwrap()
}

fn levels() -> [IsolationLevel; 3] {
    [
        IsolationLevel::ReadCommitted,
        IsolationLevel::SnapshotIsolation,
        IsolationLevel::Serializable,
    ]
}

#[test]
fn concurrent_writes_of_identical_bytes_both_commit() {
    for level in levels() {
        let dir = tempfile::tempdir().unwrap();
        let db = db(dir.path());

        let first = db.begin_transaction_with(level);
        let second = db.begin_transaction_with(level);
        first.put(b"block", b"identical").unwrap();
        second.put(b"block", b"identical").unwrap();

        first.commit().unwrap();
        second
            .commit()
            .unwrap_or_else(|error| panic!("{level:?}: {error:?}"));

        assert_eq!(
            db.db().get(b"block").unwrap().as_deref(),
            Some(b"identical".as_slice())
        );
    }
}

/// The line: same key, different bytes, is a real lost update and still aborts.
#[test]
fn concurrent_writes_of_different_bytes_still_conflict() {
    for level in levels() {
        let dir = tempfile::tempdir().unwrap();
        let db = db(dir.path());

        let first = db.begin_transaction_with(level);
        let second = db.begin_transaction_with(level);
        first.put(b"key", b"one").unwrap();
        second.put(b"key", b"two").unwrap();

        first.commit().unwrap();
        assert!(
            second.commit().is_err(),
            "{level:?}: a differing blind write is a lost update"
        );
    }
}

/// Two deletes reach the same state, so the second is not a conflict either.
#[test]
fn concurrent_identical_deletes_both_commit() {
    for level in levels() {
        let dir = tempfile::tempdir().unwrap();
        let db = db(dir.path());
        db.db().put(b"key", b"value").unwrap();

        let first = db.begin_transaction_with(level);
        let second = db.begin_transaction_with(level);
        first.delete(b"key").unwrap();
        second.delete(b"key").unwrap();

        first.commit().unwrap();
        second
            .commit()
            .unwrap_or_else(|error| panic!("{level:?}: {error:?}"));

        assert_eq!(db.db().get(b"key").unwrap(), None);
    }
}

/// A delete against a concurrent put of the same key is not idempotent: the
/// two disagree about the final state.
#[test]
fn a_delete_against_a_concurrent_put_still_conflicts() {
    let dir = tempfile::tempdir().unwrap();
    let db = db(dir.path());
    db.db().put(b"key", b"value").unwrap();

    let first = db.begin_transaction_with(IsolationLevel::Serializable);
    let second = db.begin_transaction_with(IsolationLevel::Serializable);
    first.put(b"key", b"changed").unwrap();
    second.delete(b"key").unwrap();

    first.commit().unwrap();
    assert!(
        second.commit().is_err(),
        "delete over a differing put conflicts"
    );
}

/// **The safety boundary.** A transaction that *read* the key must still abort,
/// even when the value it would write matches, because the stale read may have
/// decided what it wrote.
#[test]
fn a_stale_read_still_conflicts_even_when_the_write_matches() {
    let dir = tempfile::tempdir().unwrap();
    let db = db(dir.path());

    let first = db.begin_transaction_with(IsolationLevel::Serializable);
    let second = db.begin_transaction_with(IsolationLevel::Serializable);

    // The second transaction observes the key as absent and acts on that.
    assert_eq!(second.get(b"key").unwrap(), None);
    second.put(b"key", b"same").unwrap();
    first.put(b"key", b"same").unwrap();

    first.commit().unwrap();
    assert!(
        second.commit().is_err(),
        "a read that no longer holds must abort whatever the write says"
    );
}

/// A plain read is only honoured under serializable. At the weaker levels it
/// is already unprotected, so a key present for that reason alone is a blind
/// write as far as this commit is concerned, and an identical write elides.
///
/// This is the counterpart to the serializable case above: the same schedule,
/// the opposite verdict, decided by what the level actually promises.
#[test]
fn a_plain_read_does_not_block_elision_below_serializable() {
    for level in [
        IsolationLevel::ReadCommitted,
        IsolationLevel::SnapshotIsolation,
    ] {
        let dir = tempfile::tempdir().unwrap();
        let db = db(dir.path());

        let first = db.begin_transaction_with(level);
        let second = db.begin_transaction_with(level);

        assert_eq!(second.get(b"key").unwrap(), None);
        second.put(b"key", b"same").unwrap();
        first.put(b"key", b"same").unwrap();

        first.commit().unwrap();
        second
            .commit()
            .unwrap_or_else(|error| panic!("{level:?}: {error:?}"));
    }
}

/// The same boundary through `get_for_update`, which is validated at every
/// level rather than only under serializable.
#[test]
fn a_stale_get_for_update_still_conflicts_when_the_write_matches() {
    for level in levels() {
        let dir = tempfile::tempdir().unwrap();
        let db = db(dir.path());

        let first = db.begin_transaction_with(level);
        let second = db.begin_transaction_with(level);

        assert_eq!(second.get_for_update(b"key").unwrap(), None);
        second.put(b"key", b"same").unwrap();
        first.put(b"key", b"same").unwrap();

        first.commit().unwrap();
        assert!(
            second.commit().is_err(),
            "{level:?}: get_for_update is a read and must still abort"
        );
    }
}

/// Merges are not idempotent: two `+1` operands are not one `+1`, so a merge
/// on a contended key keeps conflicting.
#[test]
fn concurrent_merges_still_conflict() {
    let dir = tempfile::tempdir().unwrap();
    let options = Options {
        merge_operator: Some(std::sync::Arc::new(CounterMerge)),
        ..Options::default()
    };
    let db = OptimisticTransactionDb::open(dir.path(), options).unwrap();

    let first = db.begin_transaction_with(IsolationLevel::Serializable);
    let second = db.begin_transaction_with(IsolationLevel::Serializable);
    first.merge(b"key", &1i64.to_be_bytes()).unwrap();
    second.merge(b"key", &1i64.to_be_bytes()).unwrap();

    first.commit().unwrap();
    assert!(second.commit().is_err(), "merges are not idempotent");
}

/// A sequential rewrite of the same bytes was never a conflict and still is
/// not, so the change does not quietly alter the uncontended path.
#[test]
fn sequential_identical_writes_commit() {
    let dir = tempfile::tempdir().unwrap();
    let db = db(dir.path());

    let first = db.begin_transaction_with(IsolationLevel::Serializable);
    first.put(b"key", b"value").unwrap();
    first.commit().unwrap();

    let second = db.begin_transaction_with(IsolationLevel::Serializable);
    second.put(b"key", b"value").unwrap();
    second.commit().unwrap();

    assert_eq!(
        db.db().get(b"key").unwrap().as_deref(),
        Some(b"value".as_slice())
    );
}
