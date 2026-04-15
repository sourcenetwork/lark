//! ACID transactions on top of [`crate::Db`].
//!
//! Two flavors:
//!
//! - [`OptimisticTransactionDb`] — transactions take no locks,
//!   buffer writes in memory, and detect write-write conflicts at
//!   commit time by re-checking the visible seq of each touched key
//!   against the snapshot seq the transaction was anchored at.
//!   Best for low-contention workloads where most transactions
//!   commit on the first try.
//!
//! - [`TransactionDb`] — transactions acquire exclusive key locks
//!   as they write (or as [`Transaction::get_for_update`] is called)
//!   and hold them until commit / rollback. Conflict resolution is
//!   immediate (lock contention), not deferred to commit. A
//!   timeout-based deadlock defense returns [`TransactionError::Busy`]
//!   when a lock cannot be acquired in time. Best for workloads
//!   where contention is high and retry cost dominates.
//!
//! Both flavors share a single [`Transaction`] type. The type
//! carries the mode internally so user code can write helpers that
//! work against either db.
//!
//! # Isolation level
//!
//! Both flavors provide **snapshot isolation**: at [`Transaction`]
//! begin the current engine seq is captured, and every read within
//! the transaction sees the database as of that seq (except reads
//! that hit the transaction's own buffered writes — those see the
//! written value). This matches RocksDB's default transaction
//! isolation. Serializable isolation is out of scope for v1.
//!
//! # Out of scope (follow-ups)
//!
//! - Range-scan conflict tracking (only point writes / `get_for_update`
//!   participate in conflict detection).
//! - Wait-for graph deadlock detection (the pessimistic flavor ships
//!   with timeout-based detection only).
//! - Column-family-aware transactions (depends on CFs landing).
//! - Streaming iteration over a transaction's buffered-plus-snapshot
//!   view (`Transaction::iter` is not implemented — callers can
//!   commit then iterate, or use point lookups).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::{Condvar, Mutex};

use crate::column_family::{prefix_key, DEFAULT_CF_ID};
use crate::engine::{CommitOutcome, LarkEngine};
use crate::{Db, Error, Options, Result};

/// Default lock-acquisition timeout for [`TransactionDb`] when the
/// caller doesn't specify one on [`TransactionDb::with_lock_timeout`].
/// Tuned to "long enough that a fast transaction finishes, short
/// enough that a deadlock surfaces quickly".
const DEFAULT_LOCK_TIMEOUT: Duration = Duration::from_secs(1);

/// Reasons a transaction can fail to commit. Not a variant of
/// [`crate::Error`] — a conflict is a retry-able business outcome,
/// distinct from an I/O failure.
#[derive(Debug, thiserror::Error)]
pub enum TransactionError {
    /// Propagated I/O error from the underlying engine.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// An optimistic transaction observed a key that was written
    /// to by another transaction after the snapshot was captured.
    /// The caller should roll back and retry.
    #[error(
        "transaction conflict on key {key:?}: observed seq {observed_seq}, latest seq {latest_seq}"
    )]
    Conflict {
        /// The offending user key.
        key: Vec<u8>,
        /// The seq the transaction was anchored at.
        observed_seq: u64,
        /// The newest seq found for the key during the commit check.
        latest_seq: u64,
    },
    /// A pessimistic transaction could not acquire a key lock in
    /// time. Indicates either high contention or a deadlock; the
    /// caller should roll back and retry (possibly with a
    /// different operation order).
    #[error("transaction busy acquiring lock on key {0:?}")]
    Busy(Vec<u8>),
    /// The caller tried to use a savepoint that was never set.
    #[error("no savepoint to roll back to")]
    NoSavepoint,
}

/// Convenience alias for results returned by transaction methods.
pub type TxResult<T> = std::result::Result<T, TransactionError>;

impl From<Error> for TransactionError {
    fn from(e: Error) -> Self {
        match e {
            Error::Io(io) => TransactionError::Io(io),
            other => TransactionError::Io(std::io::Error::other(other.to_string())),
        }
    }
}

/// Optimistic-concurrency-control wrapper over a [`Db`].
///
/// `begin_transaction` returns a fresh [`Transaction`] whose
/// `commit` performs write-write conflict detection against the
/// seq captured at begin time. No locks are taken; other writers
/// proceed in parallel. Conflicts surface as
/// [`TransactionError::Conflict`].
pub struct OptimisticTransactionDb {
    inner: Db,
}

impl std::fmt::Debug for OptimisticTransactionDb {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OptimisticTransactionDb")
            .finish_non_exhaustive()
    }
}

impl OptimisticTransactionDb {
    /// Open or create an optimistic-transaction database at `path`.
    pub fn open<P: AsRef<Path>>(path: P, opts: Options) -> Result<Self> {
        Ok(Self {
            inner: Db::open(path, opts)?,
        })
    }

    /// Borrow the underlying [`Db`] for APIs that
    /// [`OptimisticTransactionDb`] doesn't wrap — snapshots, the
    /// streaming iterator, `compact_range`, `close`, etc.
    pub fn db(&self) -> &Db {
        &self.inner
    }

    /// Start a new optimistic transaction anchored at the engine's
    /// current sequence number. Reads see a consistent view as of
    /// that seq; writes buffer in memory until [`Transaction::commit`].
    pub fn begin_transaction(&self) -> Transaction<'_> {
        let engine = self.inner.engine_arc();
        let snapshot_seq = engine.snapshot_seq();
        engine.register_snapshot(snapshot_seq);
        Transaction::new(
            engine,
            snapshot_seq,
            self.inner.durability(),
            TxMode::Optimistic,
            None,
            DEFAULT_LOCK_TIMEOUT,
        )
    }
}

/// Pessimistic-concurrency-control wrapper over a [`Db`].
///
/// Transactions acquire exclusive locks on every key they touch
/// for update (via `put`, `delete`, or `get_for_update`) and hold
/// the locks until commit or rollback. Lock contention is resolved
/// immediately, not at commit. A timeout-based deadlock defense
/// surfaces as [`TransactionError::Busy`] when a lock cannot be
/// acquired in time.
pub struct TransactionDb {
    inner: Db,
    lock_manager: Arc<LockManager>,
    tx_id: AtomicU64,
    lock_timeout: Duration,
}

impl std::fmt::Debug for TransactionDb {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TransactionDb")
            .field("lock_timeout", &self.lock_timeout)
            .finish_non_exhaustive()
    }
}

impl TransactionDb {
    /// Open or create a pessimistic-transaction database at `path`.
    /// The lock-acquisition timeout defaults to `DEFAULT_LOCK_TIMEOUT`;
    /// customize it via [`TransactionDb::with_lock_timeout`].
    pub fn open<P: AsRef<Path>>(path: P, opts: Options) -> Result<Self> {
        Ok(Self {
            inner: Db::open(path, opts)?,
            lock_manager: Arc::new(LockManager::new()),
            tx_id: AtomicU64::new(1),
            lock_timeout: DEFAULT_LOCK_TIMEOUT,
        })
    }

    /// Borrow the underlying [`Db`].
    pub fn db(&self) -> &Db {
        &self.inner
    }

    /// Override the default lock-acquisition timeout for every
    /// future transaction created by this db.
    pub fn with_lock_timeout(mut self, timeout: Duration) -> Self {
        self.lock_timeout = timeout;
        self
    }

    /// Start a new pessimistic transaction. Reads see the engine
    /// state as of the current seq; writes acquire key locks that
    /// the caller retains until commit or rollback.
    pub fn begin_transaction(&self) -> Transaction<'_> {
        let engine = self.inner.engine_arc();
        let snapshot_seq = engine.snapshot_seq();
        engine.register_snapshot(snapshot_seq);
        let id = self.tx_id.fetch_add(1, Ordering::Relaxed);
        Transaction::new(
            engine,
            snapshot_seq,
            self.inner.durability(),
            TxMode::Pessimistic { tx_id: id },
            Some(Arc::clone(&self.lock_manager)),
            self.lock_timeout,
        )
    }
}

#[derive(Clone, Copy)]
enum TxMode {
    Optimistic,
    Pessimistic { tx_id: u64 },
}

/// An in-flight transaction. Created by
/// [`OptimisticTransactionDb::begin_transaction`] or
/// [`TransactionDb::begin_transaction`] and resolved by
/// [`Transaction::commit`] or [`Transaction::rollback`].
///
/// Reads within the transaction see a consistent snapshot captured
/// at begin time, except for keys that the transaction itself has
/// written — those always read back the buffered write.
///
/// Dropping a `Transaction` without committing is equivalent to
/// calling [`Transaction::rollback`]: buffered writes are
/// discarded and any held locks are released.
pub struct Transaction<'db> {
    engine: Arc<LarkEngine>,
    snapshot_seq: u64,
    durability: crate::engine::DurabilityMode,
    mode: TxMode,
    /// Ordered buffer of point writes. `Some(v)` is a put,
    /// `None` is a delete. Applied on commit.
    writes: BTreeMap<Vec<u8>, Option<Vec<u8>>>,
    /// Range deletes buffered for commit. Not tracked in the
    /// optimistic conflict set (initial impl limitation).
    range_deletes: Vec<(Vec<u8>, Vec<u8>)>,
    /// Merge operands buffered for commit.
    merges: Vec<(Vec<u8>, Vec<u8>)>,
    /// Keys explicitly flagged for conflict detection by
    /// `get_for_update`. Merged with the keys in `writes` at
    /// commit time.
    conflict_keys: HashSet<Vec<u8>>,
    /// Savepoint stack. Each entry captures the full write buffer
    /// and a count of locks held at that point.
    savepoints: Vec<Savepoint>,
    /// Keys for which this transaction holds a pessimistic lock.
    /// Released by `Drop` if not already released by `commit` or
    /// `rollback`.
    held_locks: Vec<Vec<u8>>,
    lock_manager: Option<Arc<LockManager>>,
    lock_timeout: Duration,
    resolved: bool,
    _phantom: std::marker::PhantomData<&'db ()>,
}

#[derive(Clone)]
struct Savepoint {
    writes: BTreeMap<Vec<u8>, Option<Vec<u8>>>,
    range_deletes: Vec<(Vec<u8>, Vec<u8>)>,
    merges: Vec<(Vec<u8>, Vec<u8>)>,
    conflict_keys: HashSet<Vec<u8>>,
    held_lock_count: usize,
}

impl<'db> Transaction<'db> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        engine: Arc<LarkEngine>,
        snapshot_seq: u64,
        durability: crate::engine::DurabilityMode,
        mode: TxMode,
        lock_manager: Option<Arc<LockManager>>,
        lock_timeout: Duration,
    ) -> Self {
        Self {
            engine,
            snapshot_seq,
            durability,
            mode,
            writes: BTreeMap::new(),
            range_deletes: Vec::new(),
            merges: Vec::new(),
            conflict_keys: HashSet::new(),
            savepoints: Vec::new(),
            held_locks: Vec::new(),
            lock_manager,
            lock_timeout,
            resolved: false,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Read `key` from the default column family. Returns the
    /// buffered write if the transaction has already written to
    /// `key`, otherwise the value visible at the transaction's
    /// snapshot. Does **not** register the key for conflict
    /// detection — use [`Transaction::get_for_update`] for that.
    pub fn get(&self, key: &[u8]) -> TxResult<Option<Vec<u8>>> {
        let prefixed = prefix_key(DEFAULT_CF_ID, key);
        if let Some(buffered) = self.writes.get(&prefixed) {
            return Ok(buffered.clone());
        }
        self.engine
            .get(&prefixed, self.snapshot_seq)
            .map_err(TransactionError::Io)
    }

    /// Read `key` and flag it for conflict detection on commit.
    /// For pessimistic transactions, also acquires an exclusive
    /// lock on the key for the duration of the transaction.
    pub fn get_for_update(&mut self, key: &[u8]) -> TxResult<Option<Vec<u8>>> {
        let prefixed = prefix_key(DEFAULT_CF_ID, key);
        self.acquire_lock_if_needed(&prefixed)?;
        self.conflict_keys.insert(prefixed.clone());
        if let Some(buffered) = self.writes.get(&prefixed) {
            return Ok(buffered.clone());
        }
        self.engine
            .get(&prefixed, self.snapshot_seq)
            .map_err(TransactionError::Io)
    }

    /// Buffer a put. For pessimistic transactions, acquires an
    /// exclusive lock on the key if not already held.
    pub fn put(&mut self, key: &[u8], value: &[u8]) -> TxResult<()> {
        let prefixed = prefix_key(DEFAULT_CF_ID, key);
        self.acquire_lock_if_needed(&prefixed)?;
        self.writes.insert(prefixed, Some(value.to_vec()));
        Ok(())
    }

    /// Buffer a delete. For pessimistic transactions, acquires an
    /// exclusive lock on the key if not already held.
    pub fn delete(&mut self, key: &[u8]) -> TxResult<()> {
        let prefixed = prefix_key(DEFAULT_CF_ID, key);
        self.acquire_lock_if_needed(&prefixed)?;
        self.writes.insert(prefixed, None);
        Ok(())
    }

    /// Buffer a range delete. Range deletes do **not** participate
    /// in optimistic conflict detection in this initial impl: a
    /// concurrent writer inserting a new key into the range between
    /// the tx's snapshot and commit will silently lose its write.
    /// For correctness-critical workloads prefer explicit
    /// `get_for_update` + per-key deletes over `delete_range`.
    pub fn delete_range(&mut self, start: &[u8], end: &[u8]) -> TxResult<()> {
        if start >= end {
            return Ok(());
        }
        // No lock acquisition for range deletes — the pessimistic
        // lock manager is keyed per user-key, which doesn't match
        // range semantics. A later iteration can add range locks.
        self.range_deletes.push((
            prefix_key(DEFAULT_CF_ID, start),
            prefix_key(DEFAULT_CF_ID, end),
        ));
        Ok(())
    }

    /// Buffer a merge operand. Merges are conflict-checked at the
    /// key level — two transactions cannot concurrently merge the
    /// same key under optimistic concurrency control.
    pub fn merge(&mut self, key: &[u8], operand: &[u8]) -> TxResult<()> {
        let prefixed = prefix_key(DEFAULT_CF_ID, key);
        self.acquire_lock_if_needed(&prefixed)?;
        self.conflict_keys.insert(prefixed.clone());
        self.merges.push((prefixed, operand.to_vec()));
        Ok(())
    }

    /// Save the current state of buffered writes. A later call to
    /// [`Transaction::rollback_to_savepoint`] reverts every buffered
    /// write made after this call.
    pub fn set_savepoint(&mut self) {
        self.savepoints.push(Savepoint {
            writes: self.writes.clone(),
            range_deletes: self.range_deletes.clone(),
            merges: self.merges.clone(),
            conflict_keys: self.conflict_keys.clone(),
            held_lock_count: self.held_locks.len(),
        });
    }

    /// Roll back to the most recent savepoint. Discards every
    /// buffered write made after the savepoint. Locks acquired
    /// after the savepoint stay held — lark's pessimistic lock
    /// manager does not release mid-transaction locks.
    pub fn rollback_to_savepoint(&mut self) -> TxResult<()> {
        let sp = self.savepoints.pop().ok_or(TransactionError::NoSavepoint)?;
        self.writes = sp.writes;
        self.range_deletes = sp.range_deletes;
        self.merges = sp.merges;
        self.conflict_keys = sp.conflict_keys;
        // Locks acquired after the savepoint remain held.
        let _ = sp.held_lock_count;
        Ok(())
    }

    /// Commit the transaction. For optimistic transactions this
    /// re-checks every key touched by `put` / `delete` / `merge` /
    /// `get_for_update` against the snapshot seq and surfaces any
    /// write-write conflict as [`TransactionError::Conflict`]. For
    /// pessimistic transactions the held locks already guarantee
    /// the absence of conflicts, so commit just applies the
    /// buffered writes.
    pub fn commit(mut self) -> TxResult<()> {
        let result = self.commit_inner();
        self.resolved = true;
        // Drop runs the cleanup (release locks, release snapshot).
        result
    }

    /// Discard the transaction's buffered writes and release any
    /// pessimistic locks. Equivalent to dropping the transaction,
    /// but surfaces as an explicit call in user code.
    pub fn rollback(mut self) {
        self.resolved = true;
        self.release_resources();
    }

    fn commit_inner(&mut self) -> TxResult<()> {
        let conflict_keys: Vec<Vec<u8>> = {
            // Every explicit conflict key plus every buffered
            // point-write key plus every merge key. We check once,
            // deduplicated.
            let mut set = self.conflict_keys.clone();
            for key in self.writes.keys() {
                set.insert(key.clone());
            }
            for (key, _) in &self.merges {
                set.insert(key.clone());
            }
            set.into_iter().collect()
        };

        let writes = std::mem::take(&mut self.writes);
        let range_deletes = std::mem::take(&mut self.range_deletes);
        let merges = std::mem::take(&mut self.merges);

        match self.mode {
            TxMode::Optimistic => {
                let outcome = self
                    .engine
                    .commit_optimistic(
                        &conflict_keys,
                        self.snapshot_seq,
                        writes,
                        range_deletes,
                        merges,
                        self.durability,
                    )
                    .map_err(TransactionError::Io)?;
                match outcome {
                    CommitOutcome::Ok => Ok(()),
                    CommitOutcome::Conflict {
                        key,
                        observed_seq,
                        latest_seq,
                    } => {
                        // Strip the 4-byte CF prefix so the error
                        // surfaces the user-visible key.
                        let user_key = if key.len() >= 4 {
                            key[4..].to_vec()
                        } else {
                            key
                        };
                        Err(TransactionError::Conflict {
                            key: user_key,
                            observed_seq,
                            latest_seq,
                        })
                    }
                }
            }
            TxMode::Pessimistic { .. } => {
                // Locks already guarantee no conflict; just apply.
                self.engine
                    .apply_batch(writes, range_deletes, merges, self.durability, false)
                    .map_err(TransactionError::Io)
            }
        }
    }

    fn acquire_lock_if_needed(&mut self, key: &[u8]) -> TxResult<()> {
        if let TxMode::Pessimistic { tx_id } = self.mode {
            if let Some(lm) = self.lock_manager.as_ref() {
                if !self.held_locks.iter().any(|k| k.as_slice() == key) {
                    lm.acquire(key, tx_id, self.lock_timeout).map_err(|_| {
                        // Strip the 4-byte CF prefix so the error
                        // surfaces the user-visible key.
                        let user_key = if key.len() >= 4 {
                            key[4..].to_vec()
                        } else {
                            key.to_vec()
                        };
                        TransactionError::Busy(user_key)
                    })?;
                    self.held_locks.push(key.to_vec());
                }
            }
        }
        Ok(())
    }

    fn release_resources(&mut self) {
        if let Some(lm) = self.lock_manager.as_ref() {
            if let TxMode::Pessimistic { tx_id } = self.mode {
                for key in self.held_locks.drain(..) {
                    lm.release(&key, tx_id);
                }
            }
        }
        self.engine.release_snapshot(self.snapshot_seq);
    }
}

impl Drop for Transaction<'_> {
    fn drop(&mut self) {
        if !self.resolved {
            self.resolved = true;
        }
        self.release_resources();
    }
}

// ─── LockManager ────────────────────────────────────────────────────────────

/// Single-shard exclusive lock manager used by [`TransactionDb`].
///
/// The hash map maps user keys to the tx id that currently holds
/// the lock. Acquires block on the condvar until the lock is free
/// or the deadline expires. Sharding the map for higher
/// concurrency is a future optimization — lark transactions today
/// expect low-to-moderate concurrency, so a single mutex is fine.
struct LockManager {
    locks: Mutex<HashMap<Vec<u8>, u64>>,
    cvar: Condvar,
}

impl LockManager {
    fn new() -> Self {
        Self {
            locks: Mutex::new(HashMap::new()),
            cvar: Condvar::new(),
        }
    }

    /// Acquire an exclusive lock on `key` for `tx_id`. Blocks up
    /// to `timeout`. Returns `Err(())` on timeout.
    fn acquire(&self, key: &[u8], tx_id: u64, timeout: Duration) -> std::result::Result<(), ()> {
        let deadline = Instant::now() + timeout;
        let mut guard = self.locks.lock();
        loop {
            match guard.get(key) {
                Some(&holder) if holder == tx_id => {
                    // Re-entrant: same transaction already holds the
                    // lock. Treat as a success.
                    return Ok(());
                }
                Some(_) => {
                    let now = Instant::now();
                    if now >= deadline {
                        return Err(());
                    }
                    let remaining = deadline - now;
                    let result = self.cvar.wait_for(&mut guard, remaining);
                    if result.timed_out() && guard.get(key).is_some_and(|&h| h != tx_id) {
                        return Err(());
                    }
                }
                None => {
                    guard.insert(key.to_vec(), tx_id);
                    return Ok(());
                }
            }
        }
    }

    /// Release `key`'s lock if held by `tx_id`. Notifies any
    /// waiters.
    fn release(&self, key: &[u8], tx_id: u64) {
        let mut guard = self.locks.lock();
        if let Some(&holder) = guard.get(key) {
            if holder == tx_id {
                guard.remove(key);
                self.cvar.notify_all();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn opt_db() -> (OptimisticTransactionDb, TempDir) {
        let dir = TempDir::new().unwrap();
        let db = OptimisticTransactionDb::open(dir.path(), Options::default()).unwrap();
        (db, dir)
    }

    fn pes_db() -> (TransactionDb, TempDir) {
        let dir = TempDir::new().unwrap();
        let db = TransactionDb::open(dir.path(), Options::default()).unwrap();
        (db, dir)
    }

    // ── Optimistic flavor ───────────────────────────────────────────────

    #[test]
    fn optimistic_basic_put_commit_read() {
        let (db, _dir) = opt_db();
        let mut tx = db.begin_transaction();
        tx.put(b"k", b"v").unwrap();
        tx.commit().unwrap();
        assert_eq!(db.db().get(b"k").unwrap(), Some(b"v".to_vec()));
    }

    #[test]
    fn optimistic_read_your_own_writes() {
        let (db, _dir) = opt_db();
        db.db().put(b"k", b"initial").unwrap();
        let mut tx = db.begin_transaction();
        assert_eq!(tx.get(b"k").unwrap(), Some(b"initial".to_vec()));
        tx.put(b"k", b"staged").unwrap();
        // Tx sees its own write.
        assert_eq!(tx.get(b"k").unwrap(), Some(b"staged".to_vec()));
        // Outside the tx the old value is still visible until commit.
        assert_eq!(db.db().get(b"k").unwrap(), Some(b"initial".to_vec()));
        tx.commit().unwrap();
        assert_eq!(db.db().get(b"k").unwrap(), Some(b"staged".to_vec()));
    }

    #[test]
    fn optimistic_rollback_discards_writes() {
        let (db, _dir) = opt_db();
        let mut tx = db.begin_transaction();
        tx.put(b"k", b"never").unwrap();
        tx.rollback();
        assert_eq!(db.db().get(b"k").unwrap(), None);
    }

    #[test]
    fn optimistic_conflict_detected() {
        let (db, _dir) = opt_db();
        db.db().put(b"k", b"v0").unwrap();
        let mut tx1 = db.begin_transaction();
        assert_eq!(tx1.get(b"k").unwrap(), Some(b"v0".to_vec()));
        // Concurrent writer bumps the key.
        db.db().put(b"k", b"v1").unwrap();
        tx1.put(b"k", b"v2").unwrap();
        match tx1.commit() {
            Err(TransactionError::Conflict { key, .. }) => {
                assert_eq!(key, b"k".to_vec());
            }
            other => panic!("expected conflict, got {other:?}"),
        }
        // db still reflects the concurrent writer's value.
        assert_eq!(db.db().get(b"k").unwrap(), Some(b"v1".to_vec()));
    }

    #[test]
    fn optimistic_get_for_update_tracks_conflicts() {
        let (db, _dir) = opt_db();
        db.db().put(b"k", b"v0").unwrap();
        let mut tx = db.begin_transaction();
        assert_eq!(tx.get_for_update(b"k").unwrap(), Some(b"v0".to_vec()));
        // Concurrent writer invalidates the read.
        db.db().put(b"k", b"v1").unwrap();
        // The tx didn't buffer a write on k, but it flagged k for
        // conflict detection — commit must still detect.
        tx.put(b"other", b"stuff").unwrap();
        match tx.commit() {
            Err(TransactionError::Conflict { key, .. }) => {
                assert_eq!(key, b"k".to_vec());
            }
            other => panic!("expected conflict, got {other:?}"),
        }
    }

    #[test]
    fn optimistic_no_conflict_passes() {
        let (db, _dir) = opt_db();
        db.db().put(b"k", b"v0").unwrap();
        let mut tx = db.begin_transaction();
        tx.put(b"other", b"stuff").unwrap();
        tx.commit().unwrap();
        assert_eq!(db.db().get(b"other").unwrap(), Some(b"stuff".to_vec()));
    }

    #[test]
    fn optimistic_snapshot_isolation_reads() {
        let (db, _dir) = opt_db();
        db.db().put(b"k", b"v0").unwrap();
        let tx = db.begin_transaction();
        db.db().put(b"k", b"v1").unwrap();
        // Tx is anchored at the seq before the second put.
        assert_eq!(tx.get(b"k").unwrap(), Some(b"v0".to_vec()));
    }

    #[test]
    fn optimistic_savepoint_rollback() {
        let (db, _dir) = opt_db();
        let mut tx = db.begin_transaction();
        tx.put(b"a", b"1").unwrap();
        tx.set_savepoint();
        tx.put(b"b", b"2").unwrap();
        tx.put(b"c", b"3").unwrap();
        tx.rollback_to_savepoint().unwrap();
        // a survives, b and c are rolled back.
        assert_eq!(tx.get(b"a").unwrap(), Some(b"1".to_vec()));
        assert_eq!(tx.get(b"b").unwrap(), None);
        assert_eq!(tx.get(b"c").unwrap(), None);
        tx.commit().unwrap();
        assert_eq!(db.db().get(b"a").unwrap(), Some(b"1".to_vec()));
        assert_eq!(db.db().get(b"b").unwrap(), None);
    }

    #[test]
    fn optimistic_rollback_to_savepoint_without_savepoint_errors() {
        let (db, _dir) = opt_db();
        let mut tx = db.begin_transaction();
        assert!(matches!(
            tx.rollback_to_savepoint(),
            Err(TransactionError::NoSavepoint)
        ));
    }

    #[test]
    fn optimistic_delete_commit() {
        let (db, _dir) = opt_db();
        db.db().put(b"k", b"v").unwrap();
        let mut tx = db.begin_transaction();
        tx.delete(b"k").unwrap();
        tx.commit().unwrap();
        assert_eq!(db.db().get(b"k").unwrap(), None);
    }

    // ── Pessimistic flavor ──────────────────────────────────────────────

    #[test]
    fn pessimistic_basic_put_commit_read() {
        let (db, _dir) = pes_db();
        let mut tx = db.begin_transaction();
        tx.put(b"k", b"v").unwrap();
        tx.commit().unwrap();
        assert_eq!(db.db().get(b"k").unwrap(), Some(b"v".to_vec()));
    }

    #[test]
    fn pessimistic_lock_blocks_second_writer() {
        let (db, _dir) = pes_db();
        let db = Arc::new(db);
        let mut tx1 = db.begin_transaction();
        tx1.put(b"k", b"v1").unwrap();
        // Second tx with a short timeout must fail to lock `k`.
        let db2 = Arc::clone(&db);
        let join = std::thread::spawn(move || {
            // The default lock timeout is 1s, which is too long for
            // the test — recreate the transaction with a shorter
            // manual lock acquisition via `get_for_update`. We
            // rely on acquire_lock_if_needed using the DB's
            // configured timeout; so we just do a normal put and
            // expect `Busy`.
            let mut tx2 = db2.begin_transaction();
            tx2.put(b"k", b"v2")
        });
        // Give tx2 some time to actually start waiting.
        std::thread::sleep(std::time::Duration::from_millis(50));
        // Commit tx1 — lock releases, tx2 should now succeed.
        tx1.commit().unwrap();
        let result = join.join().unwrap();
        assert!(result.is_ok(), "tx2 put should succeed once tx1 commits");
    }

    #[test]
    fn pessimistic_lock_timeout_returns_busy() {
        let dir = TempDir::new().unwrap();
        let db = TransactionDb::open(dir.path(), Options::default())
            .unwrap()
            .with_lock_timeout(Duration::from_millis(50));
        let db = Arc::new(db);
        // tx1 grabs the lock and holds it.
        let mut tx1 = db.begin_transaction();
        tx1.put(b"k", b"v1").unwrap();
        let db2 = Arc::clone(&db);
        let join = std::thread::spawn(move || {
            let mut tx2 = db2.begin_transaction();
            tx2.put(b"k", b"v2")
        });
        // tx2 should time out within ~50ms.
        let result = join.join().unwrap();
        assert!(matches!(result, Err(TransactionError::Busy(_))));
        // Cleanup.
        tx1.rollback();
    }

    #[test]
    fn pessimistic_reentrant_lock() {
        let (db, _dir) = pes_db();
        let mut tx = db.begin_transaction();
        tx.put(b"k", b"v1").unwrap();
        // Re-lock (via a second put) must not deadlock against
        // itself.
        tx.put(b"k", b"v2").unwrap();
        tx.get_for_update(b"k").unwrap();
        tx.commit().unwrap();
        assert_eq!(db.db().get(b"k").unwrap(), Some(b"v2".to_vec()));
    }

    #[test]
    fn pessimistic_rollback_releases_locks() {
        let (db, _dir) = pes_db();
        let db = Arc::new(db);
        let mut tx1 = db.begin_transaction();
        tx1.put(b"k", b"v1").unwrap();
        tx1.rollback();
        // New tx must now be able to grab the lock without blocking.
        let mut tx2 = db.begin_transaction();
        tx2.put(b"k", b"v2").unwrap();
        tx2.commit().unwrap();
        assert_eq!(db.db().get(b"k").unwrap(), Some(b"v2".to_vec()));
    }

    #[test]
    fn pessimistic_drop_releases_locks() {
        let (db, _dir) = pes_db();
        let db = Arc::new(db);
        {
            let mut tx1 = db.begin_transaction();
            tx1.put(b"k", b"v1").unwrap();
            // tx1 dropped here without explicit rollback.
        }
        let mut tx2 = db.begin_transaction();
        tx2.put(b"k", b"v2").unwrap();
        tx2.commit().unwrap();
        assert_eq!(db.db().get(b"k").unwrap(), Some(b"v2".to_vec()));
    }

    #[test]
    fn pessimistic_read_your_own_writes() {
        let (db, _dir) = pes_db();
        db.db().put(b"k", b"initial").unwrap();
        let mut tx = db.begin_transaction();
        tx.put(b"k", b"staged").unwrap();
        assert_eq!(tx.get(b"k").unwrap(), Some(b"staged".to_vec()));
        tx.commit().unwrap();
    }

    #[test]
    fn pessimistic_get_for_update_locks() {
        let dir = TempDir::new().unwrap();
        let db = Arc::new(
            TransactionDb::open(dir.path(), Options::default())
                .unwrap()
                .with_lock_timeout(Duration::from_millis(50)),
        );
        db.db().put(b"k", b"v0").unwrap();
        let mut tx1 = db.begin_transaction();
        assert_eq!(tx1.get_for_update(b"k").unwrap(), Some(b"v0".to_vec()));
        // Concurrent tx2 can't touch k.
        let db2 = Arc::clone(&db);
        let join = std::thread::spawn(move || {
            let mut tx2 = db2.begin_transaction();
            tx2.put(b"k", b"v1")
        });
        let result = join.join().unwrap();
        assert!(matches!(result, Err(TransactionError::Busy(_))));
        tx1.rollback();
    }

    #[test]
    fn pessimistic_savepoint_keeps_locks_but_rolls_back_writes() {
        let (db, _dir) = pes_db();
        let mut tx = db.begin_transaction();
        tx.put(b"a", b"1").unwrap();
        tx.set_savepoint();
        tx.put(b"b", b"2").unwrap();
        tx.rollback_to_savepoint().unwrap();
        // a's put survives, b's put is rolled back.
        assert_eq!(tx.get(b"a").unwrap(), Some(b"1".to_vec()));
        assert_eq!(tx.get(b"b").unwrap(), None);
        tx.commit().unwrap();
        assert_eq!(db.db().get(b"a").unwrap(), Some(b"1".to_vec()));
        assert_eq!(db.db().get(b"b").unwrap(), None);
    }

    // ── Shared behavior ─────────────────────────────────────────────────

    #[test]
    fn commit_is_atomic_with_respect_to_other_writers() {
        let (db, _dir) = opt_db();
        let mut tx = db.begin_transaction();
        tx.put(b"a", b"1").unwrap();
        tx.put(b"b", b"2").unwrap();
        tx.put(b"c", b"3").unwrap();
        tx.commit().unwrap();
        assert_eq!(db.db().get(b"a").unwrap(), Some(b"1".to_vec()));
        assert_eq!(db.db().get(b"b").unwrap(), Some(b"2".to_vec()));
        assert_eq!(db.db().get(b"c").unwrap(), Some(b"3".to_vec()));
    }

    #[test]
    fn multi_get_within_transaction() {
        let (db, _dir) = opt_db();
        db.db().put(b"a", b"1").unwrap();
        db.db().put(b"b", b"2").unwrap();
        let mut tx = db.begin_transaction();
        tx.put(b"a", b"staged").unwrap();
        assert_eq!(tx.get(b"a").unwrap(), Some(b"staged".to_vec()));
        assert_eq!(tx.get(b"b").unwrap(), Some(b"2".to_vec()));
        assert_eq!(tx.get(b"missing").unwrap(), None);
    }
}
