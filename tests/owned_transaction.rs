//! Integration tests for the owned-transaction API.
//!
//! `Transaction<'db>` cannot be stored in a `'static` container, which is
//! what a storage abstraction layer needs when it hands transactions out
//! as boxed trait objects. `begin_transaction_owned` returns a
//! transaction that carries an `Arc` on its database instead, so the
//! tests here are about that boundary: the transaction still commits,
//! still validates, and still works after every named handle on the
//! database is gone.

// Native-only. wasm-pack builds every test target for wasm32, and these
// use the filesystem, which does not exist there. The browser suite lives
// in tests/wasm_opfs*.rs.
#![cfg(not(target_arch = "wasm32"))]

use std::sync::Arc;

use regolith::{
    DbSlice, IsolationLevel, OptimisticTransactionDb, Options, OwnedTransaction, TransactionDb,
    TransactionError,
};
use tempfile::TempDir;

fn optimistic(dir: &TempDir) -> Arc<OptimisticTransactionDb> {
    Arc::new(
        OptimisticTransactionDb::open(dir.path(), Options::default()).expect("open optimistic db"),
    )
}

#[test]
fn owned_transaction_commits() {
    let dir = TempDir::new().expect("tempdir");
    let db = optimistic(&dir);

    let txn = db.begin_transaction_owned(IsolationLevel::Serializable);
    txn.put(b"key", b"value").expect("put");
    txn.commit().expect("commit");

    assert_eq!(
        db.db().get(b"key").expect("get").as_deref(),
        Some(&b"value"[..])
    );
}

#[test]
fn owned_transaction_rollback_discards() {
    let dir = TempDir::new().expect("tempdir");
    let db = optimistic(&dir);

    let txn = db.begin_transaction_owned(IsolationLevel::Serializable);
    txn.put(b"key", b"value").expect("put");
    txn.rollback();

    assert_eq!(db.db().get(b"key").expect("get"), None);
}

#[test]
fn owned_transaction_reads_its_own_writes() {
    let dir = TempDir::new().expect("tempdir");
    let db = optimistic(&dir);

    let txn = db.begin_transaction_owned(IsolationLevel::Serializable);
    txn.put(b"key", b"buffered").expect("put");
    assert_eq!(
        txn.get(b"key").expect("get").as_deref(),
        Some(&b"buffered"[..])
    );
    txn.commit().expect("commit");
}

/// The point of the type: a transaction that outlives every named handle
/// on the database it came from. Dropping `db` here would invalidate a
/// borrowing `Transaction<'db>`, so this is what the `Arc` inside buys.
#[test]
fn owned_transaction_outlives_every_named_db_handle() {
    let dir = TempDir::new().expect("tempdir");

    let txn = {
        let db = optimistic(&dir);
        let txn = db.begin_transaction_owned(IsolationLevel::Serializable);
        txn.put(b"key", b"value").expect("put");
        txn
    };

    txn.put(b"second", b"value")
        .expect("put after db handle dropped");
    txn.commit().expect("commit after db handle dropped");

    let db = optimistic(&dir);
    assert_eq!(
        db.db().get(b"key").expect("get").as_deref(),
        Some(&b"value"[..])
    );
    assert_eq!(
        db.db().get(b"second").expect("get").as_deref(),
        Some(&b"value"[..])
    );
}

/// A boxed trait object is the shape a storage layer actually stores, and
/// it is exactly what the borrowing form cannot produce.
#[test]
fn owned_transaction_fits_a_static_trait_object() {
    trait Unit {
        fn run(self: Box<Self>) -> Result<(), TransactionError>;
    }

    impl Unit for OwnedTransaction {
        fn run(self: Box<Self>) -> Result<(), TransactionError> {
            self.put(b"boxed", b"value")?;
            (*self).commit()
        }
    }

    let dir = TempDir::new().expect("tempdir");
    let db = optimistic(&dir);

    let unit: Box<dyn Unit> = Box::new(db.begin_transaction_owned(IsolationLevel::Serializable));
    unit.run().expect("commit through trait object");

    assert_eq!(
        db.db().get(b"boxed").expect("get").as_deref(),
        Some(&b"value"[..])
    );
}

/// Serializable validation must still fire through the owned wrapper: it
/// is the same `Transaction`, so a read the other writer invalidated has
/// to lose at commit.
#[test]
fn owned_transaction_still_validates_serializable() {
    let dir = TempDir::new().expect("tempdir");
    let db = optimistic(&dir);
    db.db().put(b"key", b"first").expect("seed");

    let txn = db.begin_transaction_owned(IsolationLevel::Serializable);
    assert_eq!(
        txn.get(b"key").expect("get").as_deref(),
        Some(&b"first"[..])
    );

    db.db().put(b"key", b"second").expect("concurrent write");

    txn.put(b"other", b"value").expect("put");
    let outcome = txn.commit();
    assert!(
        matches!(outcome, Err(TransactionError::Conflict { .. })),
        "serializable commit must reject a read the concurrent write invalidated, got {outcome:?}"
    );
}

#[test]
fn owned_transaction_works_on_the_pessimistic_db() {
    let dir = TempDir::new().expect("tempdir");
    let db = Arc::new(TransactionDb::open(dir.path(), Options::default()).expect("open txn db"));

    let txn = db.begin_transaction_owned(IsolationLevel::Serializable);
    txn.put(b"key", b"value").expect("put");
    txn.commit().expect("commit");

    assert_eq!(
        db.db().get(b"key").expect("get").as_deref(),
        Some(&b"value"[..])
    );
}

#[test]
fn db_slice_adopts_an_owned_vec() {
    let slice: DbSlice = b"assembled elsewhere".to_vec().into();

    assert_eq!(&*slice, b"assembled elsewhere");
    assert_eq!(slice.len(), 19);
    assert_eq!(slice.to_vec(), b"assembled elsewhere".to_vec());
}

#[test]
fn db_slice_adopts_an_empty_vec() {
    let slice: DbSlice = Vec::new().into();

    assert!(slice.is_empty());
    assert_eq!(&*slice, b"");
}

/// Buffering now takes `&self`, so a shared transaction can be written
/// from several threads at once. Every write must survive: a lost
/// insert here would mean the concurrent write buffer drops data.
#[test]
fn concurrent_writes_through_a_shared_transaction_all_land() {
    const THREADS: usize = 8;
    const PER_THREAD: usize = 64;

    let dir = TempDir::new().expect("tempdir");
    let db = optimistic(&dir);
    let txn = Arc::new(db.begin_transaction_owned(IsolationLevel::Serializable));

    std::thread::scope(|scope| {
        for thread in 0..THREADS {
            let txn = Arc::clone(&txn);
            scope.spawn(move || {
                for i in 0..PER_THREAD {
                    let key = format!("k/{thread}/{i}");
                    txn.put(key.as_bytes(), key.as_bytes()).expect("put");
                }
            });
        }
    });

    Arc::into_inner(txn)
        .expect("sole owner after the scope joined")
        .commit()
        .expect("commit");

    for thread in 0..THREADS {
        for i in 0..PER_THREAD {
            let key = format!("k/{thread}/{i}");
            assert_eq!(
                db.db().get(key.as_bytes()).expect("get").as_deref(),
                Some(key.as_bytes()),
                "{key} was lost by the concurrent write buffer"
            );
        }
    }
}

/// Reads take `&self` too, and every one of them has to reach the
/// commit-time validation set under `Serializable`. If a concurrent
/// read were lost, the write that invalidated it would go undetected
/// and the transaction would commit when it must not.
#[test]
fn concurrent_reads_through_a_shared_transaction_are_all_validated() {
    const THREADS: usize = 8;
    const KEYS: usize = 32;

    let dir = TempDir::new().expect("tempdir");
    let db = optimistic(&dir);
    for i in 0..KEYS {
        db.db()
            .put(format!("k/{i}").as_bytes(), b"first")
            .expect("seed");
    }

    let txn = Arc::new(db.begin_transaction_owned(IsolationLevel::Serializable));
    std::thread::scope(|scope| {
        for _ in 0..THREADS {
            let txn = Arc::clone(&txn);
            scope.spawn(move || {
                for i in 0..KEYS {
                    txn.get(format!("k/{i}").as_bytes()).expect("get");
                }
            });
        }
    });

    // Invalidate one of the keys every thread read.
    db.db().put(b"k/17", b"second").expect("concurrent write");

    let txn = Arc::into_inner(txn).expect("sole owner after the scope joined");
    txn.put(b"unrelated", b"value").expect("put");
    let outcome = txn.commit();
    assert!(
        matches!(outcome, Err(TransactionError::Conflict { .. })),
        "a read recorded from another thread must still be validated, got {outcome:?}"
    );
}
