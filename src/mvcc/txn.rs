//! The transaction surface, driven by kovan-mvcc.
//!
//! Not yet reachable from the public API: `OptimisticTransactionDb` and
//! `TransactionDb` still run the native path. The allow below goes when
//! the default flips, and not before, so a genuinely unused item here is
//! still caught then.
//!
//! Everything about isolation, conflict detection, timestamps and the
//! two-phase commit belongs to kovan-mvcc. This module is the adapter:
//! byte keys straight through, `MvccError` mapped onto regolith's error
//! type, and the storage layer's out-of-band I/O failure checked before
//! a commit is reported as successful.

#![allow(dead_code)]

use std::sync::Arc;

use kovan_mvcc::{IsolationLevel as MvccIsolation, KovanMVCC, MvccError};

use super::storage::LarkStorage;
use crate::engine::{DurabilityMode, LarkEngine};
use crate::transaction::{IsolationLevel, TransactionError, TxResult};

/// Transactions over one database, executed by kovan-mvcc.
pub(crate) struct MvccTransactions {
    mvcc: KovanMVCC,
    storage: Arc<LarkStorage>,
}

impl MvccTransactions {
    pub(crate) fn new(engine: Arc<LarkEngine>, durability: DurabilityMode) -> Self {
        let storage = Arc::new(LarkStorage::new(engine, durability));
        Self {
            mvcc: KovanMVCC::with_storage(storage.clone() as Arc<dyn kovan_mvcc::Storage>),
            storage,
        }
    }

    pub(crate) fn begin(&self, isolation: IsolationLevel) -> MvccTxn<'_> {
        MvccTxn {
            inner: self.mvcc.begin_with_isolation(map_isolation(isolation)),
            storage: &self.storage,
        }
    }
}

fn map_isolation(level: IsolationLevel) -> MvccIsolation {
    match level {
        IsolationLevel::ReadCommitted => MvccIsolation::ReadCommitted,
        IsolationLevel::SnapshotIsolation => MvccIsolation::RepeatableRead,
        IsolationLevel::Serializable => MvccIsolation::Serializable,
    }
}

/// One in-flight transaction.
pub(crate) struct MvccTxn<'db> {
    inner: kovan_mvcc::Txn,
    storage: &'db Arc<LarkStorage>,
}

impl MvccTxn<'_> {
    pub(crate) fn get(&mut self, k: &[u8]) -> TxResult<Option<Vec<u8>>> {
        let value = self.inner.read(k);
        // `read` cannot report an I/O error through its signature, so a
        // miss might be a failure. Checking here turns a wrong answer
        // into a reported one.
        if let Some(err) = self.storage.take_failure() {
            return Err(TransactionError::Io(err));
        }
        Ok(value)
    }

    pub(crate) fn put(&mut self, k: &[u8], value: &[u8]) -> TxResult<()> {
        self.inner.write(k, value.to_vec()).map_err(map_error)
    }

    pub(crate) fn delete(&mut self, k: &[u8]) -> TxResult<()> {
        self.inner.delete(k).map_err(map_error)
    }

    /// Commit, reporting the sequence the transaction committed at.
    pub(crate) fn commit(self) -> TxResult<u64> {
        let storage = self.storage;
        let commit_ts = self.inner.commit().map_err(map_error)?;
        // A storage failure during the commit's own writes would
        // otherwise be invisible: the protocol saw every call succeed.
        if let Some(err) = storage.take_failure() {
            return Err(TransactionError::Io(err));
        }
        Ok(commit_ts)
    }
}

/// Map kovan-mvcc's failures onto regolith's.
///
/// The distinction that matters to a caller is whether retrying can
/// help. A lock or write conflict is another transaction winning a race,
/// so it is a conflict the caller may retry. Anything structural is not.
fn map_error(err: MvccError) -> TransactionError {
    match err {
        MvccError::LockConflict { key, .. } => TransactionError::Busy(key),
        MvccError::WriteConflict {
            key,
            conflicting_ts,
        }
        | MvccError::SerializationFailure {
            key,
            conflicting_ts,
        } => TransactionError::Conflict {
            key,
            observed_seq: 0,
            latest_seq: conflicting_ts,
        },
        MvccError::RollbackRecord { key } => TransactionError::Conflict {
            key,
            observed_seq: 0,
            latest_seq: 0,
        },
        MvccError::PrimaryLockMissing { .. } | MvccError::PrimaryLockMismatch => {
            TransactionError::Io(std::io::Error::other(
                "transaction lost its primary lock, which means another writer \
                 resolved it; retry the transaction",
            ))
        }
        MvccError::StorageError(msg) => TransactionError::Io(std::io::Error::other(msg)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn txns(dir: &TempDir) -> (crate::Db, MvccTransactions) {
        let db = crate::Db::open(dir.path(), crate::Options::default()).unwrap();
        let t = MvccTransactions::new(db.engine_arc(), DurabilityMode::Eventual);
        (db, t)
    }

    #[test]
    fn a_committed_write_is_visible_to_the_next_transaction() {
        let dir = TempDir::new().unwrap();
        let (_db, t) = txns(&dir);

        let mut w = t.begin(IsolationLevel::SnapshotIsolation);
        w.put(b"k", b"v").unwrap();
        w.commit().expect("commit");

        let mut r = t.begin(IsolationLevel::SnapshotIsolation);
        assert_eq!(r.get(b"k").unwrap().as_deref(), Some(&b"v"[..]));
    }

    #[test]
    fn a_transaction_reads_its_own_writes() {
        let dir = TempDir::new().unwrap();
        let (_db, t) = txns(&dir);
        let mut w = t.begin(IsolationLevel::SnapshotIsolation);
        w.put(b"k", b"first").unwrap();
        assert_eq!(w.get(b"k").unwrap().as_deref(), Some(&b"first"[..]));
        w.put(b"k", b"second").unwrap();
        assert_eq!(w.get(b"k").unwrap().as_deref(), Some(&b"second"[..]));
        w.commit().unwrap();
    }

    #[test]
    fn an_uncommitted_write_is_invisible_to_another_transaction() {
        let dir = TempDir::new().unwrap();
        let (_db, t) = txns(&dir);
        let mut w = t.begin(IsolationLevel::SnapshotIsolation);
        w.put(b"k", b"pending").unwrap();

        let mut r = t.begin(IsolationLevel::SnapshotIsolation);
        assert_eq!(
            r.get(b"k").unwrap(),
            None,
            "prewritten data must not be visible"
        );
        w.commit().unwrap();
    }

    #[test]
    fn a_delete_hides_a_committed_value() {
        let dir = TempDir::new().unwrap();
        let (_db, t) = txns(&dir);
        let mut w = t.begin(IsolationLevel::SnapshotIsolation);
        w.put(b"k", b"v").unwrap();
        w.commit().unwrap();

        let mut d = t.begin(IsolationLevel::SnapshotIsolation);
        d.delete(b"k").unwrap();
        d.commit().unwrap();

        let mut r = t.begin(IsolationLevel::SnapshotIsolation);
        assert_eq!(r.get(b"k").unwrap(), None);
    }

    /// Binary keys are the whole reason the key codec exists.
    #[test]
    fn a_binary_key_survives_the_round_trip_through_a_transaction() {
        let dir = TempDir::new().unwrap();
        let (_db, t) = txns(&dir);
        let k = [0x00u8, 0xff, 0x80, b'/', 0xc8];

        let mut w = t.begin(IsolationLevel::SnapshotIsolation);
        w.put(&k, b"binary").unwrap();
        w.commit().unwrap();

        let mut r = t.begin(IsolationLevel::SnapshotIsolation);
        assert_eq!(r.get(&k).unwrap().as_deref(), Some(&b"binary"[..]));
        // And it must not collide with the text of its own hex.
        assert_eq!(r.get(b"00ff802fc8").unwrap(), None);
    }

    #[test]
    fn two_transactions_writing_the_same_key_do_not_both_commit() {
        let dir = TempDir::new().unwrap();
        let (_db, t) = txns(&dir);

        let mut a = t.begin(IsolationLevel::SnapshotIsolation);
        let mut b = t.begin(IsolationLevel::SnapshotIsolation);
        a.put(b"k", b"a").unwrap();
        b.put(b"k", b"b").unwrap();

        let first = a.commit();
        let second = b.commit();
        assert!(
            first.is_ok() != second.is_ok(),
            "exactly one of two writers to the same key must commit; got {first:?} and {second:?}"
        );
    }

    #[test]
    fn a_commit_reports_a_sequence_that_moves_forward() {
        let dir = TempDir::new().unwrap();
        let (_db, t) = txns(&dir);
        let mut last = 0;
        for i in 0..8u8 {
            let mut w = t.begin(IsolationLevel::SnapshotIsolation);
            w.put(&[i], b"v").unwrap();
            let ts = w.commit().unwrap();
            assert!(ts > last, "commit {i} reported {ts}, not after {last}");
            last = ts;
        }
    }
}
