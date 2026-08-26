//! `kovan_mvcc::Storage` backed by the regolith engine.
//!
//! kovan-mvcc owns the transaction protocol. This owns only where the
//! bytes live, and its job is to give the protocol what it asks for at
//! the lowest cost the layout allows. See [`super::layout`] for the key
//! mapping and why locks are not durable.
//!
//! # Reporting failures across an infallible trait
//!
//! Most of the trait cannot fail: `put_write`, `put_data`, `delete_data`
//! return `()`, and `get_data` returns `Option`. A disk error inside one
//! has nowhere to go through the signature. Swallowing it would be the
//! worst outcome - a failed write would read back as a missing key and
//! the transaction would commit over it.
//!
//! So a failure is recorded in [`LarkStorage::failure`] and the
//! transaction layer checks it at commit, turning a silent loss into a
//! refused commit. The first failure is kept rather than the last: it is
//! the one that explains the rest.

use std::sync::Arc;

use kovan_mvcc::{LockInfo, Storage, Value, WriteInfo, WriteKind};
use parking_lot::Mutex;

use super::layout;
use crate::WriteBatchOp;
use crate::engine::{DurabilityMode, LarkEngine};

/// Ceiling on the keys one `delete_range` may cover.
///
/// A range delete is expanded into one staged write per key so that
/// kovan-mvcc's prewrite locks cover it, which is what makes it visible
/// to conflict detection. That expansion is linear in the range, so it
/// is bounded, and exceeding the bound is an error rather than a
/// truncation.
pub(crate) const MAX_RANGE_DELETE_KEYS: usize = 1 << 20;

/// Locks live here rather than on disk. See [`super::layout`].
type LockTable = kovan_map::HashMap<Vec<u8>, LockInfo>;

pub(crate) struct LarkStorage {
    engine: Arc<LarkEngine>,
    locks: LockTable,
    durability: DurabilityMode,
    /// The first I/O failure seen through an infallible trait method.
    failure: Mutex<Option<std::io::Error>>,
    /// Data versions staged by `put_data` and not yet durable.
    ///
    /// Percolator prewrites every key of a transaction and only then
    /// writes its commit records. Issuing one durable write per key
    /// would cost a group commit per key; staging them and flushing once
    /// costs one, and the ordering the protocol needs is preserved
    /// because [`Self::flush_staged`] runs before any write record is
    /// recorded.
    staged: Mutex<Vec<WriteBatchOp>>,
}

thread_local! {
    /// Scratch buffers for composing engine keys.
    ///
    /// Every `Storage` call builds a key from a prefix, the user key and
    /// a timestamp. Allocating a fresh `Vec` for each one puts a malloc
    /// and a free on the critical path of every read and every write; a
    /// per-thread buffer that grows once and is reused puts none there.
    /// Thread-local rather than shared so reuse costs no synchronisation
    /// between the very threads the engine exists to run concurrently.
    static KEY_BUF: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
    static PREFIX_BUF: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Compose a key into the thread's scratch buffer and hand it to `f`.
///
/// The buffer never escapes, so it can be reused for the next call.
fn with_key<R>(compose: impl FnOnce(&mut Vec<u8>), f: impl FnOnce(&[u8]) -> R) -> R {
    KEY_BUF.with(|b| {
        let mut buf = b.borrow_mut();
        compose(&mut buf);
        f(&buf)
    })
}

impl LarkStorage {
    pub(crate) fn new(engine: Arc<LarkEngine>, durability: DurabilityMode) -> Self {
        Self {
            engine,
            // Sized small: a lock table holds only in-flight transactions,
            // not the keyspace. kovan-map's default capacity is half a
            // million buckets, which would be the largest allocation in
            // an embedded build.
            locks: LockTable::with_capacity(1024),
            durability,
            failure: Mutex::new(None),
            staged: Mutex::new(Vec::new()),
        }
    }

    /// Take the first failure recorded since the last check.
    pub(crate) fn take_failure(&self) -> Option<std::io::Error> {
        self.failure.lock().take()
    }

    fn record(&self, err: std::io::Error) {
        let mut slot = self.failure.lock();
        if slot.is_none() {
            *slot = Some(err);
        }
    }

    /// Make everything staged by `put_data` durable.
    ///
    /// Called before the first write record of a commit, which is what
    /// keeps Percolator's ordering: a write record must never be visible
    /// before the data it points at.
    fn flush_staged(&self) {
        let ops: Vec<WriteBatchOp> = {
            let mut staged = self.staged.lock();
            if staged.is_empty() {
                return;
            }
            std::mem::take(&mut *staged)
        };
        if let Err(e) = self.engine.apply_batch(ops, self.durability, false) {
            self.record(e);
        }
    }

    fn stage(&self, op: WriteBatchOp) {
        self.staged.lock().push(op);
    }

    /// Read one key at the newest visible sequence.
    fn get(&self, key: &[u8]) -> Option<Vec<u8>> {
        match self.engine.get_latest(key) {
            Ok(v) => v,
            Err(e) => {
                self.record(e);
                None
            }
        }
    }
}

impl Storage for LarkStorage {
    fn get_lock(&self, key: &[u8]) -> Option<LockInfo> {
        self.locks.get(key)
    }

    /// Acquire, or report who holds it.
    ///
    /// This is a compare-and-swap, not an insert. Percolator relies on
    /// exactly one transaction winning a contended key: an insert that
    /// overwrote would let two prewrites both believe they hold the lock
    /// and the protocol would lose a write with no error anywhere. The
    /// concurrent map supplies the atomicity, which is the reason locks
    /// live here rather than in the engine, which has no CAS.
    fn put_lock(&self, key: &[u8], lock: LockInfo) -> Result<(), kovan_mvcc::MvccError> {
        match self.locks.insert_if_absent(key.to_vec(), lock.clone()) {
            None => Ok(()),
            Some(existing) => {
                if existing.txn_id == lock.txn_id {
                    // Already ours: re-prewriting the same key is allowed
                    // and refreshes the entry.
                    self.locks.insert(key.to_vec(), lock);
                    Ok(())
                } else {
                    Err(kovan_mvcc::MvccError::LockConflict {
                        key: key.to_vec(),
                        holder_txn: existing.txn_id,
                    })
                }
            }
        }
    }

    fn delete_lock(&self, key: &[u8]) {
        self.locks.remove(key);
    }

    fn get_latest_write(&self, key: &[u8], ts: u64) -> Option<(u64, WriteInfo)> {
        self.seek_write(key, ts, false)
    }

    fn get_latest_commit(&self, key: &[u8], ts: u64) -> Option<(u64, WriteInfo)> {
        self.seek_write(key, ts, true)
    }

    fn put_write(&self, key: &[u8], commit_ts: u64, info: WriteInfo) {
        // The data this record points at has to be durable first.
        self.flush_staged();
        let mut k = Vec::with_capacity(layout::composed_len(key));
        layout::write_key(key, commit_ts, &mut k);
        self.stage(WriteBatchOp::Put {
            key: k,
            value: encode_write_info(&info),
        });
        // A write record is the commit point, so it goes out now rather
        // than waiting for a later flush.
        self.flush_staged();
    }

    fn get_data(&self, key: &[u8], start_ts: u64) -> Option<Value> {
        // A staged write is this transaction's own and is not yet
        // durable; the protocol reads its own writes from its local
        // buffer, so only durable versions are consulted here.
        with_key(
            |b| layout::data_key(key, start_ts, b),
            |k| self.get(k).map(Arc::new),
        )
    }

    fn put_data(&self, key: &[u8], start_ts: u64, value: Value) {
        // This one has to own its key: the op is staged until the batch
        // is flushed, so it outlives any scratch buffer.
        let mut k = Vec::with_capacity(layout::composed_len(key));
        layout::data_key(key, start_ts, &mut k);
        self.stage(WriteBatchOp::Put {
            key: k,
            value: value.as_ref().clone(),
        });
    }

    fn delete_data(&self, key: &[u8], start_ts: u64) {
        let mut k = Vec::with_capacity(layout::composed_len(key));
        layout::data_key(key, start_ts, &mut k);
        self.stage(WriteBatchOp::Delete { key: k });
    }
}

impl LarkStorage {
    /// Distinct user keys holding a live value in `[start, end)` at `ts`.
    ///
    /// Walks the write records, not the engine's raw keys: the write
    /// records *are* the committed key set, and reading them at `ts`
    /// is what makes the answer the transaction's own snapshot rather
    /// than whatever is newest. A key whose newest write at or before
    /// `ts` is a delete or a rollback is absent and is left out.
    ///
    /// Bounded by [`MAX_RANGE_DELETE_KEYS`] and reported, not
    /// truncated: a range delete that silently covered part of its
    /// range would be worse than one that refuses.
    pub(crate) fn keys_in_range(
        &self,
        start: &[u8],
        end: &[u8],
        ts: u64,
    ) -> std::io::Result<Vec<Vec<u8>>> {
        let mut iter = self.engine.new_iter_latest();
        with_key(|b| layout::write_prefix(start, b), |target| iter.seek(target));

        let mut keys: Vec<Vec<u8>> = Vec::new();
        let mut last: Option<Vec<u8>> = None;
        while iter.valid() {
            let Some(composed) = iter.key() else { break };
            let Some(user_key) = layout::user_key_of(composed) else {
                break;
            };
            if user_key.as_slice() >= end {
                break;
            }
            // Records are newest-first within a key, so the first one
            // seen for a key is the one visible at `ts`; the rest are
            // older versions of a decision already made.
            if last.as_deref() != Some(user_key.as_slice()) {
                if let Some(ts_of) = layout::commit_ts_of(composed)
                    && ts_of <= ts
                    && let Some(info) = iter.value().and_then(decode_write_info)
                {
                    if info.kind == WriteKind::Put {
                        if keys.len() == MAX_RANGE_DELETE_KEYS {
                            return Err(std::io::Error::other(format!(
                                "delete_range covers more than {MAX_RANGE_DELETE_KEYS} keys; \
                                 every key is taken as a write so the range is \
                                 conflict-checked, so narrow the range or delete in batches"
                            )));
                        }
                        keys.push(user_key.clone());
                    }
                    last = Some(user_key);
                }
            }
            iter.next();
        }
        Ok(keys)
    }

    /// One seek for "newest write at or before `ts`".
    ///
    /// `skip_rollbacks` is the difference between `get_latest_write` and
    /// `get_latest_commit`: the first reports a rollback record, the
    /// second walks past it to the commit it superseded.
    fn seek_write(&self, key: &[u8], ts: u64, skip_rollbacks: bool) -> Option<(u64, WriteInfo)> {
        let mut iter = self.engine.new_iter_latest();
        with_key(
            |b| layout::write_seek(key, ts, b),
            |target| iter.seek(target),
        );
        PREFIX_BUF.with(|b| {
            let mut prefix = b.borrow_mut();
            layout::write_prefix(key, &mut prefix);
            self.walk_writes(&mut iter, &prefix, skip_rollbacks)
        })
    }

    /// Walk write records for one key, newest first.
    fn walk_writes(
        &self,
        iter: &mut crate::engine::iterator::LarkIterator,
        prefix: &[u8],
        skip_rollbacks: bool,
    ) -> Option<(u64, WriteInfo)> {
        while iter.valid() {
            let k = iter.key()?;
            if !k.starts_with(prefix) {
                return None;
            }
            let commit_ts = layout::commit_ts_of(k)?;
            let info = decode_write_info(iter.value()?)?;
            if !(skip_rollbacks && info.kind == WriteKind::Rollback) {
                return Some((commit_ts, info));
            }
            iter.next();
        }
        None
    }
}

/// `WriteInfo` on disk: the start timestamp it points at, then its kind.
fn encode_write_info(info: &WriteInfo) -> Vec<u8> {
    let mut out = Vec::with_capacity(9);
    out.extend_from_slice(&info.start_ts.to_be_bytes());
    out.push(match info.kind {
        WriteKind::Put => 0,
        WriteKind::Delete => 1,
        WriteKind::Rollback => 2,
    });
    out
}

fn decode_write_info(bytes: &[u8]) -> Option<WriteInfo> {
    if bytes.len() < 9 {
        return None;
    }
    Some(WriteInfo {
        start_ts: u64::from_be_bytes(bytes[0..8].try_into().ok()?),
        kind: match bytes[8] {
            0 => WriteKind::Put,
            1 => WriteKind::Delete,
            2 => WriteKind::Rollback,
            _ => return None,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use kovan_mvcc::LockType;
    use tempfile::TempDir;

    fn storage() -> (TempDir, Arc<LarkStorage>) {
        let dir = TempDir::new().unwrap();
        let db = crate::Db::open(dir.path(), crate::Options::default()).unwrap();
        let engine = db.engine_arc();
        std::mem::forget(db);
        (
            dir,
            Arc::new(LarkStorage::new(engine, DurabilityMode::Eventual)),
        )
    }

    fn lock(txn_id: u128, start_ts: u64) -> LockInfo {
        LockInfo {
            txn_id,
            start_ts,
            primary_key: std::sync::Arc::from(&b"tprimary"[..]),
            lock_type: LockType::Put,
            short_value: None,
        }
    }

    /// The bug this exists to prevent: an insert that overwrites lets two
    /// prewrites both believe they hold a contended key, and Percolator
    /// loses a write with no error raised anywhere.
    #[test]
    fn a_second_transaction_cannot_take_a_held_lock() {
        let (_d, s) = storage();
        s.put_lock(b"tk", lock(1, 10)).expect("first acquires");
        let err = s
            .put_lock(b"tk", lock(2, 11))
            .expect_err("a second transaction must be refused");
        match err {
            kovan_mvcc::MvccError::LockConflict { holder_txn, .. } => {
                assert_eq!(holder_txn, 1, "the error must name the real holder")
            }
            other => panic!("expected LockConflict, got {other:?}"),
        }
        assert_eq!(s.get_lock(b"tk").map(|l| l.txn_id), Some(1));
    }

    #[test]
    fn a_transaction_may_re_lock_its_own_key() {
        let (_d, s) = storage();
        s.put_lock(b"tk", lock(1, 10)).unwrap();
        s.put_lock(b"tk", lock(1, 10))
            .expect("re-prewriting our own key is allowed");
        s.delete_lock(b"tk");
        assert!(s.get_lock(b"tk").is_none());
        s.put_lock(b"tk", lock(2, 11))
            .expect("released, so another may take it");
    }

    #[test]
    fn data_round_trips_at_its_start_timestamp() {
        let (_d, s) = storage();
        s.put_data(b"tk", 7, Arc::new(b"seven".to_vec()));
        s.flush_staged();
        assert_eq!(s.get_data(b"tk", 7).as_deref(), Some(&b"seven".to_vec()));
        assert_eq!(
            s.get_data(b"tk", 8),
            None,
            "a different version must not answer"
        );
        assert!(s.take_failure().is_none());
    }

    /// `get_latest_write` must answer with the newest commit at or before
    /// the timestamp, in one seek.
    #[test]
    fn the_newest_write_at_or_before_a_timestamp_is_found() {
        let (_d, s) = storage();
        for (commit_ts, start_ts) in [(10u64, 9u64), (20, 19), (30, 29)] {
            s.put_write(
                b"tk",
                commit_ts,
                WriteInfo {
                    start_ts,
                    kind: WriteKind::Put,
                },
            );
        }
        assert_eq!(s.get_latest_write(b"tk", 25).map(|(ts, _)| ts), Some(20));
        assert_eq!(s.get_latest_write(b"tk", 30).map(|(ts, _)| ts), Some(30));
        assert!(
            s.get_latest_write(b"tk", 5).is_none(),
            "nothing committed that early"
        );
    }

    /// A rollback is visible to `get_latest_write` and skipped by
    /// `get_latest_commit`. Conflating them would make a rolled-back
    /// transaction's key read as its previous value at the wrong time.
    #[test]
    fn a_rollback_is_reported_by_one_query_and_skipped_by_the_other() {
        let (_d, s) = storage();
        s.put_write(
            b"tk",
            10,
            WriteInfo {
                start_ts: 9,
                kind: WriteKind::Put,
            },
        );
        s.put_write(
            b"tk",
            20,
            WriteInfo {
                start_ts: 19,
                kind: WriteKind::Rollback,
            },
        );

        assert_eq!(s.get_latest_write(b"tk", 25).map(|(ts, _)| ts), Some(20));
        assert_eq!(
            s.get_latest_commit(b"tk", 25).map(|(ts, _)| ts),
            Some(10),
            "get_latest_commit must walk past the rollback"
        );
    }

    #[test]
    fn one_keys_writes_do_not_answer_for_another() {
        let (_d, s) = storage();
        s.put_write(
            b"tk",
            10,
            WriteInfo {
                start_ts: 9,
                kind: WriteKind::Put,
            },
        );
        assert!(
            s.get_latest_write(b"tkk", 99).is_none(),
            "a longer key must not match"
        );
        assert!(
            s.get_latest_write(b"t", 99).is_none(),
            "a shorter key must not match"
        );
    }

    #[test]
    fn write_records_survive_a_reopen() {
        let dir = TempDir::new().unwrap();
        {
            let db = crate::Db::open(dir.path(), crate::Options::default()).unwrap();
            let s = LarkStorage::new(db.engine_arc(), DurabilityMode::Immediate);
            s.put_data(b"tk", 9, Arc::new(b"v".to_vec()));
            s.put_write(
                b"tk",
                10,
                WriteInfo {
                    start_ts: 9,
                    kind: WriteKind::Put,
                },
            );
            db.close().unwrap();
        }
        let db = crate::Db::open(dir.path(), crate::Options::default()).unwrap();
        let s = LarkStorage::new(db.engine_arc(), DurabilityMode::Immediate);
        assert_eq!(s.get_latest_write(b"tk", 99).map(|(ts, _)| ts), Some(10));
        assert_eq!(s.get_data(b"tk", 9).as_deref(), Some(&b"v".to_vec()));
        // Locks are deliberately not durable.
        assert!(
            s.get_lock(b"tk").is_none(),
            "locks must not survive a reopen"
        );
    }

    #[test]
    fn write_info_round_trips_through_its_encoding() {
        for kind in [WriteKind::Put, WriteKind::Delete, WriteKind::Rollback] {
            let info = WriteInfo {
                start_ts: 12345,
                kind,
            };
            let back = decode_write_info(&encode_write_info(&info)).expect("decodes");
            assert_eq!(back.start_ts, 12345);
            assert_eq!(back.kind, kind);
        }
        assert!(decode_write_info(b"short").is_none());
        assert!(
            decode_write_info(&[0; 8]).is_none(),
            "missing the kind byte"
        );
    }
}
