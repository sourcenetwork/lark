//! The transaction surface, driven by kovan-mvcc.
//!
//! The transaction path, driven by kovan-mvcc.
//!
//! **Not the default yet.** `OptimisticTransactionDb` and
//! `TransactionDb` still construct the native `Transaction`. The
//! adapter below is complete and tested; flipping those two entry
//! points over is the remaining step, and the `allow` under this
//! comment goes with it, so a genuinely unused item is still caught
//! once the module is reachable.
//!
//! Everything about isolation, conflict detection, timestamps and the
//! two-phase commit belongs to kovan-mvcc. Nothing here reimplements
//! any of it. This module is the adapter, and it does four things:
//!
//! - byte keys straight through, with no encoding layer;
//! - `MvccError` mapped onto regolith's error type;
//! - the storage layer's out-of-band I/O failure checked before a
//!   commit is reported as successful;
//! - the parts of regolith's surface Percolator has no verb for
//!   (savepoints, `delete_range`, `merge`) expressed *in terms of*
//!   kovan-mvcc's `read` / `write` / `delete`, never alongside them.
//!
//! # Why operations are staged
//!
//! kovan-mvcc's `Txn` owns its local write set and does not expose it,
//! so a savepoint cannot be taken by snapshotting it. Point writes are
//! therefore staged here and flushed into the transaction at commit.
//! A read consults the staging map first, so a transaction still sees
//! its own writes; a key in the staging map is written at commit and
//! so is conflict-checked as a write regardless, which is why
//! answering it locally costs no conflict coverage.
//!
//! # Serializable is BOCC
//!
//! At [`IsolationLevel::Serializable`] kovan-mvcc records every key a
//! transaction reads, including keys that were *absent*, and validates
//! the whole read set under its SSI commit lock before the commit is
//! allowed. That is backward-oriented optimistic concurrency control,
//! and it is what turns snapshot isolation into serializability by
//! closing the anti-dependency edge that admits write skew. It is
//! kovan-mvcc's, not regolith's: this adapter only has to make sure
//! every read a caller performs actually reaches `Txn::read`.

#![allow(dead_code)]

use std::collections::BTreeMap;
use std::sync::Arc;

use kovan_mvcc::{IsolationLevel as MvccIsolation, KovanMVCC, MvccError};

use super::storage::RegolithStorage;
use crate::engine::{DurabilityMode, RegolithEngine};
use crate::transaction::{IsolationLevel, TransactionError, TxResult};

/// Transactions over one database, executed by kovan-mvcc.
pub(crate) struct MvccTransactions {
    mvcc: KovanMVCC,
    storage: Arc<RegolithStorage>,
}

impl MvccTransactions {
    pub(crate) fn new(engine: Arc<RegolithEngine>, durability: DurabilityMode) -> Self {
        let storage = Arc::new(RegolithStorage::new(engine, durability));
        Self {
            mvcc: KovanMVCC::with_storage(storage.clone() as Arc<dyn kovan_mvcc::Storage>),
            storage,
        }
    }

    pub(crate) fn begin(&self, isolation: IsolationLevel) -> MvccTxn<'_> {
        MvccTxn {
            inner: self.mvcc.begin_with_isolation(map_isolation(isolation)),
            storage: &self.storage,
            staged: BTreeMap::new(),
            savepoints: Vec::new(),
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
    storage: &'db Arc<RegolithStorage>,
    /// Point writes not yet handed to kovan-mvcc. `Some` is a put,
    /// `None` a delete. Ordered so a multi-key failure always reports
    /// the same key. Flushed by [`MvccTxn::commit`].
    staged: BTreeMap<Vec<u8>, Option<Vec<u8>>>,
    /// Savepoint stack, each a copy of `staged` at the time it was set.
    savepoints: Vec<BTreeMap<Vec<u8>, Option<Vec<u8>>>>,
}

impl MvccTxn<'_> {
    /// The transaction's own staged write for `key`, if any.
    ///
    /// A staged key is written at commit and so is conflict-checked as
    /// a write; answering it from here rather than from `Txn::read`
    /// therefore costs no conflict coverage, and it is what makes a
    /// transaction see its own writes.
    fn staged(&self, key: &[u8]) -> Option<Option<Vec<u8>>> {
        self.staged.get(key).cloned()
    }

    pub(crate) fn get(&mut self, k: &[u8]) -> TxResult<Option<Vec<u8>>> {
        if let Some(local) = self.staged(k) {
            return Ok(local);
        }
        let value = self.inner.read(k);
        // `read` cannot report an I/O error through its signature, so a
        // miss might be a failure. Checking here turns a wrong answer
        // into a reported one.
        if let Some(err) = self.storage.take_failure() {
            return Err(TransactionError::Io(err));
        }
        Ok(value)
    }

    /// Read `key` and guarantee it is validated at commit whatever the
    /// isolation level.
    ///
    /// A plain [`MvccTxn::get`] enters the read set only at
    /// `Serializable`. This forces the key into the transaction's
    /// footprint at every level by staging its current value back as a
    /// write, so a concurrent writer of the same key loses the
    /// prewrite race. That is what stops a lost update at
    /// `SnapshotIsolation`.
    pub(crate) fn get_for_update(&mut self, k: &[u8]) -> TxResult<Option<Vec<u8>>> {
        let current = self.get(k)?;
        match &current {
            Some(v) => self.staged.insert(k.to_vec(), Some(v.clone())),
            // Re-staging an absent key as a delete keeps it in the
            // write set, so a concurrent insert is still a conflict.
            None => self.staged.insert(k.to_vec(), None),
        };
        Ok(current)
    }

    pub(crate) fn put(&mut self, k: &[u8], value: &[u8]) -> TxResult<()> {
        self.staged.insert(k.to_vec(), Some(value.to_vec()));
        Ok(())
    }

    pub(crate) fn delete(&mut self, k: &[u8]) -> TxResult<()> {
        self.staged.insert(k.to_vec(), None);
        Ok(())
    }

    /// Delete every key in `[start, end)`.
    ///
    /// Expanded into point deletes against the transaction's own read
    /// snapshot rather than handed down as a range. That is what makes
    /// a range delete visible to conflict detection: each key becomes
    /// an ordinary write, so kovan-mvcc's prewrite takes a lock on it
    /// and a concurrent writer of any key in the range conflicts. A
    /// range passed through whole would commit against a concurrent
    /// write silently.
    ///
    /// The cost is that the transaction holds one staged entry per key
    /// in the range, which is the price of the guarantee: a range
    /// delete that is not enumerated cannot be conflict-checked.
    pub(crate) fn delete_range(&mut self, start: &[u8], end: &[u8]) -> TxResult<()> {
        if start >= end {
            return Ok(());
        }
        let keys = self
            .storage
            .keys_in_range(start, end, self.inner.start_ts())
            .map_err(TransactionError::Io)?;
        for key in keys {
            self.staged.insert(key, None);
        }
        Ok(())
    }

    /// Apply `operand` to `key` through the configured merge operator.
    ///
    /// Resolved here, against the value the transaction can see, so
    /// what reaches kovan-mvcc is an ordinary write. The read is a real
    /// read, so at `Serializable` it enters the read set and the
    /// read-modify-write is validated as one.
    pub(crate) fn merge(
        &mut self,
        merge_operator: Option<&Arc<dyn crate::options::MergeOperator>>,
        k: &[u8],
        operand: &[u8],
    ) -> TxResult<()> {
        let op = merge_operator.ok_or_else(|| {
            TransactionError::Io(std::io::Error::other(
                "merge called with no merge operator configured; set Options::merge_operator",
            ))
        })?;
        let existing = self.get(k)?;
        let merged = op
            .full_merge(k, existing.as_deref(), std::slice::from_ref(&operand))
            .ok_or_else(|| {
                TransactionError::Io(std::io::Error::other("merge operator rejected the operand"))
            })?;
        self.staged.insert(k.to_vec(), Some(merged));
        Ok(())
    }

    /// Mark a point the transaction can be rewound to.
    pub(crate) fn set_savepoint(&mut self) {
        self.savepoints.push(self.staged.clone());
    }

    /// Undo every staged write back to the last savepoint.
    ///
    /// Reads are not rewound. A read that already happened is a read
    /// the transaction performed, and at `Serializable` it stays in
    /// kovan-mvcc's read set: rewinding it would hide a real
    /// anti-dependency and quietly weaken the isolation level.
    pub(crate) fn rollback_to_savepoint(&mut self) -> bool {
        match self.savepoints.pop() {
            Some(saved) => {
                self.staged = saved;
                true
            }
            None => false,
        }
    }

    /// Commit, reporting the sequence the transaction committed at.
    pub(crate) fn commit(mut self) -> TxResult<u64> {
        // Staged writes reach kovan-mvcc only now, in key order, so the
        // prewrite lock order is deterministic across transactions.
        let staged = std::mem::take(&mut self.staged);
        for (key, value) in staged {
            match value {
                Some(v) => self.inner.write(&key, v).map_err(map_error)?,
                None => self.inner.delete(&key).map_err(map_error)?,
            }
        }
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
    fn a_committed_transaction_is_visible_to_a_plain_db_read() {
        // kovan-mvcc keeps its versions in its own keyspace, so a
        // commit also projects the resolved value onto the ordinary
        // key. Without that projection the two APIs would answer
        // differently about the same database.
        let dir = TempDir::new().unwrap();
        let (db, txns) = txns(&dir);

        let mut tx = txns.begin(IsolationLevel::SnapshotIsolation);
        tx.put(b"projected", b"value").unwrap();
        tx.commit().unwrap();

        assert_eq!(
            db.get(b"projected").unwrap().as_deref(),
            Some(&b"value"[..]),
            "a committed transactional write must be visible to Db::get"
        );

        let mut tx = txns.begin(IsolationLevel::SnapshotIsolation);
        tx.delete(b"projected").unwrap();
        tx.commit().unwrap();

        assert_eq!(
            db.get(b"projected").unwrap(),
            None,
            "a committed transactional delete must be visible to Db::get"
        );
    }

    #[test]
    fn a_rollback_record_projects_nothing() {
        // A rollback is not a commit. Projecting it would publish a
        // value the transaction never committed.
        let dir = TempDir::new().unwrap();
        let (db, txns) = txns(&dir);
        let mut tx = txns.begin(IsolationLevel::SnapshotIsolation);
        tx.put(b"abandoned", b"value").unwrap();
        drop(tx);
        assert_eq!(db.get(b"abandoned").unwrap(), None);
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
