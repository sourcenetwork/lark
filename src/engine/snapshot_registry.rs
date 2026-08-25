//! Tracks the sequence numbers of every live `Snapshot` handed out by
//! the engine so that compaction can decide which older versions of a
//! user key are safe to drop.
//!
//! # Pin semantics
//!
//! Every call to [`Db::snapshot`](crate::Db::snapshot) registers a
//! sequence number, and the corresponding [`Snapshot`](crate::Snapshot)
//! releases it on drop. Multiple snapshots can pin the same seq; the
//! registry keeps a refcount per seq and removes the entry when the
//! refcount reaches zero.
//!
//! [`SnapshotRegistry::oldest_live_seq`] returns the smallest currently
//! registered seq - or `u64::MAX` when no snapshot is live. Compaction
//! uses this as the **pin seq**: any version with a smaller seq than
//! the largest visible version at `pin_seq` is invisible to every
//! live snapshot and to current reads, and can be discarded.

use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

use parking_lot::Mutex;

/// Thread-safe registry of live snapshot sequence numbers.
pub(crate) struct SnapshotRegistry {
    /// `seq -> (refcount, earliest_register_unix)`. `BTreeMap`
    /// keeps the smallest seq at the front so `oldest_live_seq`
    /// is O(log n). `earliest_register_unix` is captured on the
    /// first `register` call at a given seq and reused by
    /// subsequent increments so the property
    /// `lark.oldest-snapshot-time` is stable across refcount
    /// changes.
    active: Mutex<BTreeMap<u64, SlotState>>,
}

#[derive(Debug, Clone, Copy)]
struct SlotState {
    refcount: usize,
    registered_at_unix: u64,
}

impl SnapshotRegistry {
    pub(crate) fn new() -> Self {
        Self {
            active: Mutex::new(BTreeMap::new()),
        }
    }

    /// Register a new pin at `seq`. Must be balanced by a later
    /// [`release`](Self::release) call - typically from the
    /// `Drop` impl of the owning snapshot type.
    pub(crate) fn register(&self, seq: u64) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let mut active = self.active.lock();
        active
            .entry(seq)
            .or_insert(SlotState {
                refcount: 0,
                registered_at_unix: now,
            })
            .refcount += 1;
    }

    /// Release one pin at `seq`. A no-op if no pin at that seq is
    /// currently registered, which should only happen in test
    /// teardown races.
    pub(crate) fn release(&self, seq: u64) {
        let mut active = self.active.lock();
        if let Some(slot) = active.get_mut(&seq) {
            slot.refcount -= 1;
            if slot.refcount == 0 {
                active.remove(&seq);
            }
        }
    }

    /// Return the smallest currently registered seq, or `u64::MAX`
    /// if no snapshot is live. Compaction uses this as its GC
    /// horizon: entries older than the largest version visible to
    /// this seq are safe to drop.
    pub(crate) fn oldest_live_seq(&self) -> u64 {
        self.active
            .lock()
            .keys()
            .next()
            .copied()
            .unwrap_or(u64::MAX)
    }

    /// Number of distinct live snapshots. Counts pins, not
    /// distinct seqs - two snapshots taken at the same seq
    /// contribute two.
    pub(crate) fn live_count(&self) -> u64 {
        self.active
            .lock()
            .values()
            .map(|slot| slot.refcount as u64)
            .sum()
    }

    /// Unix-seconds timestamp when the oldest currently-live
    /// snapshot was registered, or `None` when no snapshot is
    /// alive. Used to populate the
    /// `lark.oldest-snapshot-time` property.
    pub(crate) fn oldest_snapshot_time_unix(&self) -> Option<u64> {
        self.active
            .lock()
            .values()
            .next()
            .map(|slot| slot.registered_at_unix)
    }

    /// Test-only: number of distinct seq values currently pinned.
    #[cfg(test)]
    pub(crate) fn pin_count(&self) -> usize {
        self.active.lock().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_release_refcounts_correctly() {
        let r = SnapshotRegistry::new();
        assert_eq!(r.oldest_live_seq(), u64::MAX);
        assert_eq!(r.pin_count(), 0);

        r.register(10);
        r.register(10);
        r.register(5);
        r.register(20);

        assert_eq!(r.oldest_live_seq(), 5);
        assert_eq!(r.pin_count(), 3);

        r.release(5);
        assert_eq!(r.oldest_live_seq(), 10);
        assert_eq!(r.pin_count(), 2);

        r.release(10);
        // Second pin at 10 still alive.
        assert_eq!(r.oldest_live_seq(), 10);
        assert_eq!(r.pin_count(), 2);

        r.release(10);
        assert_eq!(r.oldest_live_seq(), 20);
        assert_eq!(r.pin_count(), 1);

        r.release(20);
        assert_eq!(r.oldest_live_seq(), u64::MAX);
        assert_eq!(r.pin_count(), 0);
    }

    #[test]
    fn release_unknown_seq_is_noop() {
        let r = SnapshotRegistry::new();
        r.release(42);
        assert_eq!(r.oldest_live_seq(), u64::MAX);
    }

    #[test]
    fn live_count_tracks_refcount_total_not_distinct_seqs() {
        let r = SnapshotRegistry::new();
        r.register(5);
        r.register(5);
        r.register(9);
        assert_eq!(r.live_count(), 3);
        assert_eq!(r.pin_count(), 2);
        r.release(5);
        assert_eq!(r.live_count(), 2);
    }

    #[test]
    fn oldest_snapshot_time_is_none_when_empty_and_some_when_pinned() {
        let r = SnapshotRegistry::new();
        assert!(r.oldest_snapshot_time_unix().is_none());
        r.register(7);
        assert!(r.oldest_snapshot_time_unix().is_some());
        r.release(7);
        assert!(r.oldest_snapshot_time_unix().is_none());
    }

    #[test]
    fn concurrent_register_release_stays_consistent() {
        use std::sync::Arc;
        use std::thread;
        let r = Arc::new(SnapshotRegistry::new());
        let mut handles = Vec::new();
        for worker in 0..4u64 {
            let r = Arc::clone(&r);
            handles.push(thread::spawn(move || {
                for i in 0..200u64 {
                    let seq = worker * 200 + i + 1;
                    r.register(seq);
                    r.release(seq);
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(r.pin_count(), 0);
        assert_eq!(r.live_count(), 0);
        assert_eq!(r.oldest_live_seq(), u64::MAX);
    }
}
