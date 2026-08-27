//! ACID transactions on top of [`crate::Db`].
//!
//! Two flavors:
//!
//! - [`OptimisticTransactionDb`]: transactions take no locks,
//!   buffer writes in memory, and detect write-write conflicts at
//!   commit time by re-checking the visible seq of each touched key
//!   against the snapshot seq the transaction was anchored at.
//!   Best for low-contention workloads where most transactions
//!   commit on the first try.
//!
//! - [`TransactionDb`]: transactions acquire exclusive key locks
//!   as they write (or as [`Transaction::get_for_update`] is called)
//!   and hold them until commit / rollback. Contention is resolved
//!   when the lock is taken rather than at commit, and a
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
//! Both flavors provide **snapshot isolation** and both prevent
//! lost updates, but they anchor their reads at different points.
//!
//! An optimistic transaction reads everything as of the engine seq
//! captured at begin. A pessimistic transaction reads a key it has
//! not locked at that same begin seq, and reads a key it locks
//! through [`Transaction::get_for_update`] as of the moment the
//! lock was acquired. The lock handoff orders it after every
//! transaction that committed while it waited, so a
//! read-modify-write through `get_for_update` sees the value the
//! previous lock holder committed instead of a stale one. Reads
//! that hit the transaction's own buffered writes always see the
//! written value. A read anchor only ever moves forward, so two
//! reads of the same key inside one transaction never travel
//! backwards in time.
//!
//! At commit each flavor validates a set of keys, every key against
//! the *earliest* sequence this transaction observed it at:
//!
//! - Optimistic: every key the transaction wrote or read through
//!   `get_for_update`, against the begin snapshot.
//! - Pessimistic: every key the transaction read and then wrote
//!   (the read-modify-write set) plus every key it read through
//!   `get_for_update`. A key written blind, without ever being
//!   read, is not validated: there is no read to invalidate, and the
//!   key lock already orders it against every other transaction.
//!
//! For a pessimistic transaction the check cannot fire while every
//! writer goes through the lock manager. It fires when a key the
//! transaction read was written around the lock manager (for example
//! through [`TransactionDb::db`]) or before the lock was taken,
//! which surfaces as [`TransactionError::Conflict`] rather than as
//! a lost update.
//!
//! Rolling back to a savepoint discards buffered writes; it does not
//! discard what the transaction has already read, so a read anchor
//! survives the rollback and still guards the commit.
//!
//! Serializable isolation is out of scope for v1: reads that are
//! never written are not validated, so a read-only key can change
//! underneath a transaction without aborting it.
//!
//! # Out of scope (follow-ups)
//!
//! - Range-scan conflict tracking (only point writes / `get_for_update`
//!   participate in conflict detection).
//! - Transactional range deletes. [`Transaction::delete_range`] rejects
//!   non-empty ranges until range conflict tracking or range locks land.
//! - Wait-for graph deadlock detection (the pessimistic flavor ships
//!   with timeout-based detection only).
//! - Column-family-aware transactions (depends on CFs landing).
//! - Streaming iteration over a transaction's buffered-plus-snapshot
//!   view (`Transaction::iter` is not implemented; callers can
//!   commit then iterate, or use point lookups).

use crate::portability::{AtomicU64, Ordering};
use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use parking_lot::{Condvar, Mutex};

use crate::column_family::{DEFAULT_CF_ID, prefix_key};
use crate::engine::{CommitOutcome, LarkEngine};
use crate::{Db, Error, Options, Result};

/// Default lock-acquisition timeout for [`TransactionDb`] when the
/// caller doesn't specify one on [`TransactionDb::with_lock_timeout`].
/// Tuned to "long enough that a fast transaction finishes, short
/// enough that a deadlock surfaces quickly".
const DEFAULT_LOCK_TIMEOUT: Duration = Duration::from_secs(1);

/// Reasons a transaction can fail to commit. Not a variant of
/// [`crate::Error`]: a conflict is a retry-able business outcome,
/// distinct from an I/O failure.
#[derive(Debug, thiserror::Error)]
pub enum TransactionError {
    /// Propagated I/O error from the underlying engine.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// A key in the transaction's validation set was written by
    /// someone else after this transaction first observed it: after
    /// the begin snapshot for an optimistic transaction, or after
    /// the read that the pessimistic transaction is about to
    /// overwrite. The caller should roll back and retry.
    #[error(
        "transaction conflict on key {key:?}: observed seq {observed_seq}, latest seq {latest_seq}"
    )]
    Conflict {
        /// The offending user key.
        key: Vec<u8>,
        /// The seq the transaction observed the key at.
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
    /// Transactional range deletes are disabled until they can
    /// participate in conflict detection or range locking.
    #[error("transactional range deletes are not supported")]
    UnsupportedRangeDelete,
}

/// Convenience alias for results returned by transaction methods.
pub type TxResult<T> = std::result::Result<T, TransactionError>;

impl From<Error> for TransactionError {
    fn from(e: Error) -> Self {
        match e {
            Error::Io(io) => TransactionError::Io(io),
            Error::Corruption(io) => TransactionError::Io(io),
            Error::InvalidArgument(message) | Error::InvalidColumnFamily(message) => {
                TransactionError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    message,
                ))
            }
            Error::ReadOnly => TransactionError::Io(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "database was opened read-only",
            )),
            Error::Closed => TransactionError::Io(std::io::Error::new(
                std::io::ErrorKind::NotConnected,
                "database is closed",
            )),
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
    /// [`OptimisticTransactionDb`] doesn't wrap: snapshots, the
    /// streaming iterator, `compact_range`, `close`, etc.
    pub fn db(&self) -> &Db {
        &self.inner
    }

    /// Start a new optimistic transaction anchored at the engine's
    /// current sequence number. Reads see a consistent view as of
    /// that seq; writes buffer in memory until [`Transaction::commit`].
    pub fn begin_transaction(&self) -> Transaction<'_> {
        let engine = self.inner.engine_arc();
        let snapshot_seq = engine.register_snapshot_at_horizon();
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
        let snapshot_seq = engine.register_snapshot_at_horizon();
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
/// written; those always read back the buffered write.
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
    /// What this transaction has observed about each key it read.
    /// Ordered so a multi-key conflict always reports the same key.
    /// Never rewound: a savepoint rollback undoes buffered writes,
    /// not reads that already happened.
    tracked: BTreeMap<Vec<u8>, KeyState>,
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
    resources_released: bool,
    _phantom: std::marker::PhantomData<&'db ()>,
}

#[derive(Clone)]
struct Savepoint {
    writes: BTreeMap<Vec<u8>, Option<Vec<u8>>>,
    range_deletes: Vec<(Vec<u8>, Vec<u8>)>,
    merges: Vec<(Vec<u8>, Vec<u8>)>,
    held_lock_count: usize,
}

/// What one transaction knows about one key it has read.
#[derive(Clone, Copy)]
struct KeyState {
    /// Earliest sequence this transaction observed the key at.
    /// Validation uses this one, because it is the read a later
    /// write of the same key would otherwise silently overwrite.
    first_read_seq: u64,
    /// Sequence later reads of the key are served at. Only ever
    /// moves forward, so `get_for_update` can promote a key that was
    /// already read at the begin snapshot to the lock horizon
    /// without any read of this transaction going backwards.
    read_seq: u64,
    /// The key was read through [`Transaction::get_for_update`], so
    /// it is validated at commit whether or not it is written.
    for_update: bool,
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
            tracked: BTreeMap::new(),
            savepoints: Vec::new(),
            held_locks: Vec::new(),
            lock_manager,
            lock_timeout,
            resolved: false,
            resources_released: false,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Read `key` from the default column family. Returns the
    /// buffered write if the transaction has already written to
    /// `key`, otherwise the value visible at the sequence this
    /// transaction observes `key` at: its begin snapshot, or, for a
    /// key a pessimistic transaction already holds a lock on, the
    /// horizon sampled when that lock was acquired.
    ///
    /// Takes no lock. The read is remembered, so writing the same
    /// key later turns it into a read-modify-write that is validated
    /// at commit and aborts with [`TransactionError::Conflict`]
    /// rather than losing the update. A key that is read and never
    /// written is not validated: use
    /// [`Transaction::get_for_update`] when a read must participate
    /// in conflict detection on its own.
    pub fn get(&mut self, key: &[u8]) -> TxResult<Option<Vec<u8>>> {
        let prefixed = prefix_key(DEFAULT_CF_ID, key);
        if let Some(buffered) = self.writes.get(&prefixed) {
            return Ok(buffered.clone());
        }
        let read_seq = self.observe(&prefixed, self.snapshot_seq, false);
        self.engine
            .get_at(&prefixed, read_seq)
            .map_err(TransactionError::Io)
    }

    /// Read `key`, flag it for conflict detection at commit, and,
    /// for pessimistic transactions, take an exclusive lock on it
    /// for the rest of the transaction.
    ///
    /// A pessimistic transaction reads the value as of the moment
    /// it acquired the lock, not as of its begin snapshot, so a
    /// read-modify-write under `get_for_update` observes every
    /// transaction that committed before the lock was released to
    /// it. An optimistic transaction reads at its begin snapshot
    /// and detects the conflict at commit instead.
    pub fn get_for_update(&mut self, key: &[u8]) -> TxResult<Option<Vec<u8>>> {
        let prefixed = prefix_key(DEFAULT_CF_ID, key);
        let already_held = self.lock_key(&prefixed)?;
        let horizon = self.read_horizon(&prefixed, already_held);
        let read_seq = self.observe(&prefixed, horizon, true);
        if let Some(buffered) = self.writes.get(&prefixed) {
            return Ok(buffered.clone());
        }
        self.engine
            .get_at(&prefixed, read_seq)
            .map_err(TransactionError::Io)
    }

    /// Buffer a put. For pessimistic transactions, acquires an
    /// exclusive lock on the key if not already held.
    pub fn put(&mut self, key: &[u8], value: &[u8]) -> TxResult<()> {
        let prefixed = prefix_key(DEFAULT_CF_ID, key);
        self.lock_key(&prefixed)?;
        self.writes.insert(prefixed, Some(value.to_vec()));
        Ok(())
    }

    /// Buffer a delete. For pessimistic transactions, acquires an
    /// exclusive lock on the key if not already held.
    pub fn delete(&mut self, key: &[u8]) -> TxResult<()> {
        let prefixed = prefix_key(DEFAULT_CF_ID, key);
        self.lock_key(&prefixed)?;
        self.writes.insert(prefixed, None);
        Ok(())
    }

    /// Attempt to delete every key in `[start, end)`.
    ///
    /// Non-empty transactional range deletes are rejected until
    /// they can participate in optimistic conflict detection or
    /// pessimistic range locking. For correctness-critical
    /// workloads, delete known keys individually with
    /// [`Transaction::delete`] and use [`Transaction::get_for_update`]
    /// when a read must also participate in conflict detection.
    ///
    /// Calls with `start >= end` are treated as no-ops and return
    /// `Ok(())`. That is deliberate and it is the one place where a
    /// transactional write's acceptance depends on the arguments being
    /// empty: `Db::write` rejects an empty batch on a read-only handle
    /// because handle state is not an argument, while here the
    /// rejection is a missing feature and an empty range asks for no
    /// work from it. Callers that want the unsupported-feature error
    /// unconditionally must check the range themselves.
    pub fn delete_range(&mut self, start: &[u8], end: &[u8]) -> TxResult<()> {
        if start >= end {
            return Ok(());
        }
        Err(TransactionError::UnsupportedRangeDelete)
    }

    /// Buffer a merge operand. Merges are conflict-checked at the
    /// key level: two transactions cannot concurrently merge the
    /// same key under optimistic concurrency control.
    pub fn merge(&mut self, key: &[u8], operand: &[u8]) -> TxResult<()> {
        let prefixed = prefix_key(DEFAULT_CF_ID, key);
        self.lock_key(&prefixed)?;
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
            held_lock_count: self.held_locks.len(),
        });
    }

    /// Roll back to the most recent savepoint. Discards every
    /// buffered write made after the savepoint. Locks acquired
    /// after the savepoint stay held: lark's pessimistic lock
    /// manager does not release mid-transaction locks.
    ///
    /// Reads are not rolled back. A key this transaction has already
    /// read keeps the sequence it was read at, so a rollback can
    /// neither rewind a later read of that key nor launder a write
    /// that landed around the lock manager in the meantime.
    pub fn rollback_to_savepoint(&mut self) -> TxResult<()> {
        let sp = self.savepoints.pop().ok_or(TransactionError::NoSavepoint)?;
        self.writes = sp.writes;
        self.range_deletes = sp.range_deletes;
        self.merges = sp.merges;
        // Locks acquired after the savepoint remain held.
        let _ = sp.held_lock_count;
        Ok(())
    }

    /// Commit the transaction. Every key in the validation set is
    /// re-checked against the earliest sequence this transaction
    /// observed it at, and any conflict surfaces as
    /// [`TransactionError::Conflict`]; otherwise the buffered
    /// writes are applied atomically.
    ///
    /// An optimistic transaction validates every key it wrote or
    /// read through [`Transaction::get_for_update`] against its
    /// begin snapshot, so the check catches any concurrent writer. A
    /// pessimistic transaction validates every key it read and then
    /// wrote, plus every key it read through `get_for_update`,
    /// against the sequence that read observed. The check passes
    /// whenever every writer went through the lock manager and fires
    /// for a write that bypassed it or that landed before the lock
    /// was taken. A pessimistic blind write is not validated: the
    /// key lock orders it, and there is no read to lose.
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
        let conflict_keys = self.validation_set();
        let writes = std::mem::take(&mut self.writes);
        let range_deletes = std::mem::take(&mut self.range_deletes);
        let merges = std::mem::take(&mut self.merges);

        let outcome = self
            .engine
            .commit_with_conflict_check(
                &conflict_keys,
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
            } => Err(TransactionError::Conflict {
                key: strip_cf_prefix(key),
                observed_seq,
                latest_seq,
            }),
        }
    }

    /// The keys this commit must validate, each mapped to the
    /// earliest sequence the transaction observed it at.
    ///
    /// Optimistic: every key written or merged, plus every key read
    /// through `get_for_update`, all anchored at the begin snapshot.
    ///
    /// Pessimistic: every key that was read and then written (the
    /// read-modify-write set), plus every key read through
    /// `get_for_update`. A key written without ever being read is
    /// left out: its lock already orders it against the other
    /// transactions, and there is no read for a concurrent writer to
    /// invalidate.
    fn validation_set(&mut self) -> Vec<(Vec<u8>, u64)> {
        let optimistic = matches!(self.mode, TxMode::Optimistic);
        let mut checks: BTreeMap<Vec<u8>, u64> = BTreeMap::new();
        for (key, state) in std::mem::take(&mut self.tracked) {
            let written = self.writes.contains_key(&key)
                || self.merges.iter().any(|(merged, _)| *merged == key);
            if state.for_update || written {
                checks.insert(key, state.first_read_seq);
            }
        }
        if optimistic {
            for key in self.writes.keys() {
                checks.entry(key.clone()).or_insert(self.snapshot_seq);
            }
            for (key, _) in &self.merges {
                checks.entry(key.clone()).or_insert(self.snapshot_seq);
            }
        }
        checks.into_iter().collect()
    }

    /// Take `key`'s exclusive lock in pessimistic mode. Returns
    /// `true` when this transaction already held it. A no-op for an
    /// optimistic transaction, which takes no locks.
    fn lock_key(&mut self, key: &[u8]) -> TxResult<bool> {
        match self.mode {
            TxMode::Optimistic => Ok(false),
            TxMode::Pessimistic { tx_id } => self.acquire_lock(key, tx_id),
        }
    }

    /// Sequence a fresh read of `key` is anchored at, given whether
    /// this transaction already held the key's lock.
    ///
    /// The pessimistic horizon is sampled with the lock held: a
    /// committing writer publishes the engine's visible seq before
    /// releasing the lock, and the lock manager's mutex is the
    /// release/acquire edge, so nothing newer can hide behind it.
    /// Under a lock this transaction already holds nothing can have
    /// advanced, so the anchor recorded then still stands.
    fn read_horizon(&self, key: &[u8], already_held: bool) -> u64 {
        match self.mode {
            TxMode::Optimistic => self.snapshot_seq,
            TxMode::Pessimistic { .. } => {
                if already_held && let Some(state) = self.tracked.get(key) {
                    return state.read_seq;
                }
                self.engine.snapshot_seq()
            }
        }
    }

    /// Record that this transaction observed `key` at `horizon` and
    /// return the sequence the read should be served at.
    ///
    /// `first_read_seq` keeps the earliest observation, because that
    /// is the read a later write would overwrite. `read_seq` only
    /// moves forward, so promoting a key from a plain `get` to
    /// `get_for_update` never makes a later read of the same key
    /// return an older value than an earlier one.
    fn observe(&mut self, key: &[u8], horizon: u64, for_update: bool) -> u64 {
        match self.tracked.get_mut(key) {
            Some(state) => {
                state.read_seq = state.read_seq.max(horizon);
                state.for_update |= for_update;
                state.read_seq
            }
            None => {
                self.tracked.insert(
                    key.to_vec(),
                    KeyState {
                        first_read_seq: horizon,
                        read_seq: horizon,
                        for_update,
                    },
                );
                horizon
            }
        }
    }

    /// Acquire `key`'s exclusive lock. Returns `true` when this
    /// transaction already held it.
    fn acquire_lock(&mut self, key: &[u8], tx_id: u64) -> TxResult<bool> {
        let Some(lm) = self.lock_manager.as_ref() else {
            return Ok(false);
        };
        if self.held_locks.iter().any(|k| k.as_slice() == key) {
            return Ok(true);
        }
        lm.acquire(key, tx_id, self.lock_timeout)
            .map_err(|_| TransactionError::Busy(strip_cf_prefix(key.to_vec())))?;
        self.held_locks.push(key.to_vec());
        Ok(false)
    }

    fn release_resources(&mut self) {
        if self.resources_released {
            return;
        }
        self.resources_released = true;
        if let Some(lm) = self.lock_manager.as_ref()
            && let TxMode::Pessimistic { tx_id } = self.mode
        {
            for key in self.held_locks.drain(..) {
                lm.release(&key, tx_id);
            }
        }
        self.engine.release_snapshot(self.snapshot_seq);
    }
}

/// Drop the 4-byte column-family prefix that `prefix_key` adds so
/// errors surface the key the caller passed in.
fn strip_cf_prefix(key: Vec<u8>) -> Vec<u8> {
    if key.len() >= 4 {
        key[4..].to_vec()
    } else {
        key
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
/// concurrency is a future optimization: lark transactions today
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
        // Microseconds from the platform clock. `None` means this
        // platform cannot measure a timeout at all, which is the same
        // single-threaded platform on which no other transaction can
        // be holding the lock, so the wait simply is not bounded by a
        // deadline there.
        let deadline =
            crate::env::platform_micros().map(|now| now.saturating_add(timeout.as_micros() as u64));
        let mut guard = self.locks.lock();
        loop {
            match guard.get(key) {
                Some(&holder) if holder == tx_id => {
                    // Re-entrant: same transaction already holds the
                    // lock. Treat as a success.
                    return Ok(());
                }
                Some(_) => {
                    let remaining = match (deadline, crate::env::platform_micros()) {
                        (Some(deadline), Some(now)) => {
                            if now >= deadline {
                                return Err(());
                            }
                            Duration::from_micros(deadline - now)
                        }
                        _ => timeout,
                    };
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
        if let Some(&holder) = guard.get(key)
            && holder == tx_id
        {
            guard.remove(key);
            self.cvar.notify_all();
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
    fn optimistic_rollback_releases_shared_snapshot_pin_once() {
        let (db, _dir) = opt_db();
        let tx1 = db.begin_transaction();
        let tx2 = db.begin_transaction();
        assert_eq!(db.db().get_int_property("lark.num-snapshots"), Some(2));

        tx1.rollback();
        assert_eq!(db.db().get_int_property("lark.num-snapshots"), Some(1));

        drop(tx2);
        assert_eq!(db.db().get_int_property("lark.num-snapshots"), Some(0));
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
        // conflict detection, so commit must still detect.
        tx.put(b"other", b"stuff").unwrap();
        match tx.commit() {
            Err(TransactionError::Conflict { key, .. }) => {
                assert_eq!(key, b"k".to_vec());
            }
            other => panic!("expected conflict, got {other:?}"),
        }
    }

    #[test]
    fn optimistic_conflict_reports_the_lowest_conflicting_key() {
        let (db, _dir) = opt_db();
        db.db().put(b"a", b"v0").unwrap();
        db.db().put(b"z", b"v0").unwrap();
        let mut tx = db.begin_transaction();
        // Tracked in the reverse of their sort order.
        tx.get_for_update(b"z").unwrap();
        tx.get_for_update(b"a").unwrap();
        db.db().put(b"z", b"v1").unwrap();
        db.db().put(b"a", b"v1").unwrap();
        match tx.commit() {
            Err(TransactionError::Conflict { key, .. }) => assert_eq!(key, b"a".to_vec()),
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
        let mut tx = db.begin_transaction();
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

    #[test]
    fn optimistic_range_delete_is_rejected() {
        let (db, _dir) = opt_db();
        let mut tx = db.begin_transaction();

        assert!(matches!(
            tx.delete_range(b"a", b"z"),
            Err(TransactionError::UnsupportedRangeDelete)
        ));
        assert!(tx.delete_range(b"z", b"a").is_ok());
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
    fn pessimistic_range_delete_is_rejected() {
        let (db, _dir) = pes_db();
        let mut tx = db.begin_transaction();

        assert!(matches!(
            tx.delete_range(b"a", b"z"),
            Err(TransactionError::UnsupportedRangeDelete)
        ));
        assert!(tx.delete_range(b"z", b"a").is_ok());
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
            // the test, so recreate the transaction with a shorter
            // manual lock acquisition via `get_for_update`. We
            // rely on acquire_lock using the DB's
            // configured timeout; so we just do a normal put and
            // expect `Busy`.
            let mut tx2 = db2.begin_transaction();
            tx2.put(b"k", b"v2")
        });
        // Give tx2 some time to actually start waiting.
        std::thread::sleep(std::time::Duration::from_millis(50));
        // Commit tx1: lock releases, tx2 should now succeed.
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
    fn pessimistic_rollback_releases_shared_snapshot_pin_once() {
        let (db, _dir) = pes_db();
        let tx1 = db.begin_transaction();
        let tx2 = db.begin_transaction();
        assert_eq!(db.db().get_int_property("lark.num-snapshots"), Some(2));

        tx1.rollback();
        assert_eq!(db.db().get_int_property("lark.num-snapshots"), Some(1));

        drop(tx2);
        assert_eq!(db.db().get_int_property("lark.num-snapshots"), Some(0));
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

    #[test]
    fn pessimistic_get_for_update_sees_writes_committed_after_begin() {
        let (db, _dir) = pes_db();
        db.db().put(b"k", b"v0").unwrap();
        let mut tx = db.begin_transaction();
        // Lands after the transaction began, before it locks `k`.
        db.db().put(b"k", b"v1").unwrap();
        assert_eq!(tx.get_for_update(b"k").unwrap(), Some(b"v1".to_vec()));
        // A plain `get` on the locked key reads at the same horizon.
        assert_eq!(tx.get(b"k").unwrap(), Some(b"v1".to_vec()));
        tx.put(b"k", b"v2").unwrap();
        tx.commit().unwrap();
        assert_eq!(db.db().get(b"k").unwrap(), Some(b"v2".to_vec()));
    }

    #[test]
    fn pessimistic_second_locker_does_not_observe_precommit_value() {
        let (db, _dir) = pes_db();
        db.db().put(b"k", b"v0").unwrap();
        // Both transactions begin before either one commits, so both
        // are anchored at the seq where `k` is still `v0`.
        let mut tx1 = db.begin_transaction();
        let mut tx2 = db.begin_transaction();
        assert_eq!(tx1.get_for_update(b"k").unwrap(), Some(b"v0".to_vec()));
        tx1.put(b"k", b"v1").unwrap();
        tx1.commit().unwrap();
        // tx2 only now gets the lock, so it must not see `v0`.
        assert_eq!(tx2.get_for_update(b"k").unwrap(), Some(b"v1".to_vec()));
        tx2.put(b"k", b"v2").unwrap();
        tx2.commit().unwrap();
        assert_eq!(db.db().get(b"k").unwrap(), Some(b"v2".to_vec()));
    }

    #[test]
    fn pessimistic_commit_detects_write_from_outside_the_lock_manager() {
        let (db, _dir) = pes_db();
        let mut tx = db.begin_transaction();
        tx.get_for_update(b"k").unwrap();
        // A raw `Db` write never touches the lock manager.
        db.db().put(b"k", b"racer").unwrap();
        tx.put(b"k", b"mine").unwrap();
        match tx.commit() {
            Err(TransactionError::Conflict { key, .. }) => assert_eq!(key, b"k".to_vec()),
            other => panic!("expected conflict, got {other:?}"),
        }
        assert_eq!(db.db().get(b"k").unwrap(), Some(b"racer".to_vec()));
    }

    #[test]
    fn pessimistic_sequential_transactions_do_not_conflict() {
        let (db, _dir) = pes_db();
        let mut tx1 = db.begin_transaction();
        tx1.put(b"k", b"v1").unwrap();
        tx1.commit().unwrap();
        let mut tx2 = db.begin_transaction();
        assert_eq!(tx2.get_for_update(b"k").unwrap(), Some(b"v1".to_vec()));
        tx2.put(b"k", b"v2").unwrap();
        tx2.commit().unwrap();
        assert_eq!(db.db().get(b"k").unwrap(), Some(b"v2".to_vec()));
    }

    #[test]
    fn pessimistic_blind_put_after_external_write_is_not_a_conflict() {
        let (db, _dir) = pes_db();
        db.db().put(b"k", b"v0").unwrap();
        let mut tx = db.begin_transaction();
        db.db().put(b"k", b"v1").unwrap();
        // Nothing was read, so there is no read to lose: a blind write
        // is last-writer-wins against a non-transactional writer.
        tx.put(b"k", b"v2").unwrap();
        tx.commit().unwrap();
        assert_eq!(db.db().get(b"k").unwrap(), Some(b"v2".to_vec()));
    }

    #[test]
    fn pessimistic_blind_put_before_external_write_is_not_a_conflict() {
        // The mirror ordering: the external write lands after the
        // transaction has already buffered its blind write.
        let (db, _dir) = pes_db();
        db.db().put(b"k", b"v0").unwrap();
        let mut tx = db.begin_transaction();
        tx.put(b"k", b"v2").unwrap();
        db.db().put(b"k", b"v1").unwrap();
        tx.commit().unwrap();
        assert_eq!(db.db().get(b"k").unwrap(), Some(b"v2".to_vec()));
    }

    #[test]
    fn pessimistic_savepoint_rollback_keeps_untouched_keys_unvalidated() {
        let (db, _dir) = pes_db();
        let mut tx = db.begin_transaction();
        tx.set_savepoint();
        tx.put(b"b", b"rolled-back").unwrap();
        tx.rollback_to_savepoint().unwrap();
        // `b` was written blind and the write was rolled back, so it
        // was never read and is not validated. The lock stays held.
        db.db().put(b"b", b"external").unwrap();
        tx.put(b"a", b"1").unwrap();
        tx.commit().unwrap();
        assert_eq!(db.db().get(b"a").unwrap(), Some(b"1".to_vec()));
        assert_eq!(db.db().get(b"b").unwrap(), Some(b"external".to_vec()));
    }

    #[test]
    fn pessimistic_savepoint_rollback_keeps_the_read_anchor() {
        // Rolling back to a savepoint used to restore the conflict map
        // and let the next write re-anchor at a newer horizon, which
        // laundered a write that had bypassed the lock manager.
        let (db, _dir) = pes_db();
        db.db().put(b"k", b"v0").unwrap();
        let mut tx = db.begin_transaction();
        assert_eq!(tx.get_for_update(b"k").unwrap(), Some(b"v0".to_vec()));
        tx.set_savepoint();
        tx.put(b"k", b"rolled-back").unwrap();
        tx.rollback_to_savepoint().unwrap();
        db.db().put(b"k", b"racer").unwrap();
        tx.put(b"k", b"mine").unwrap();
        let err = tx.commit().expect_err("the bypassing write must be caught");
        assert!(matches!(err, TransactionError::Conflict { .. }), "{err:?}");
        assert_eq!(db.db().get(b"k").unwrap(), Some(b"racer".to_vec()));
    }

    #[test]
    fn pessimistic_reads_do_not_travel_backwards_after_a_savepoint_rollback() {
        let (db, _dir) = pes_db();
        db.db().put(b"k", b"v0").unwrap();
        let mut tx = db.begin_transaction();
        db.db().put(b"k", b"v1").unwrap();
        let first = tx.get_for_update(b"k").unwrap();
        assert_eq!(first, Some(b"v1".to_vec()));
        tx.set_savepoint();
        tx.put(b"k", b"staged").unwrap();
        tx.rollback_to_savepoint().unwrap();
        // The lock is still held and the read anchor with it, so the
        // second read cannot return an older value than the first.
        assert_eq!(tx.get(b"k").unwrap(), first);
    }

    #[test]
    fn pessimistic_read_then_write_detects_a_concurrent_write() {
        // A read-modify-write through plain `get` takes no lock, so the
        // commit check is the only thing standing between it and a lost
        // update.
        let (db, _dir) = pes_db();
        db.db().put(b"k", b"v0").unwrap();
        let mut tx = db.begin_transaction();
        assert_eq!(tx.get(b"k").unwrap(), Some(b"v0".to_vec()));
        db.db().put(b"k", b"v1").unwrap();
        tx.put(b"k", b"derived-from-v0").unwrap();
        let err = tx.commit().expect_err("the stale read must be caught");
        assert!(matches!(err, TransactionError::Conflict { .. }), "{err:?}");
        assert_eq!(db.db().get(b"k").unwrap(), Some(b"v1".to_vec()));
    }

    #[test]
    fn pessimistic_read_without_a_write_is_not_validated() {
        let (db, _dir) = pes_db();
        db.db().put(b"read-only", b"v0").unwrap();
        let mut tx = db.begin_transaction();
        assert_eq!(tx.get(b"read-only").unwrap(), Some(b"v0".to_vec()));
        db.db().put(b"read-only", b"v1").unwrap();
        tx.put(b"other", b"1").unwrap();
        tx.commit().unwrap();
        assert_eq!(db.db().get(b"other").unwrap(), Some(b"1".to_vec()));
    }

    #[test]
    fn pessimistic_range_delete_over_a_tracked_key_is_a_conflict() {
        let (db, _dir) = pes_db();
        db.db().put(b"k", b"v0").unwrap();
        let mut tx = db.begin_transaction();
        assert_eq!(tx.get_for_update(b"k").unwrap(), Some(b"v0".to_vec()));
        db.db().delete_range(b"a", b"z").unwrap();
        tx.put(b"k", b"resurrected").unwrap();
        let err = tx
            .commit()
            .expect_err("a range delete over a tracked key is a conflict");
        assert!(matches!(err, TransactionError::Conflict { .. }), "{err:?}");
        assert_eq!(db.db().get(b"k").unwrap(), None);
    }

    #[test]
    fn optimistic_range_delete_over_a_tracked_key_is_a_conflict() {
        let (db, _dir) = opt_db();
        db.db().put(b"k", b"v0").unwrap();
        let mut tx = db.begin_transaction();
        assert_eq!(tx.get_for_update(b"k").unwrap(), Some(b"v0".to_vec()));
        db.db().delete_range(b"a", b"z").unwrap();
        tx.put(b"k", b"resurrected").unwrap();
        let err = tx
            .commit()
            .expect_err("a range delete over a tracked key is a conflict");
        assert!(matches!(err, TransactionError::Conflict { .. }), "{err:?}");
        assert_eq!(db.db().get(b"k").unwrap(), None);
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
