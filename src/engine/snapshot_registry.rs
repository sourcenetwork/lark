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
//! registered seq — or `u64::MAX` when no snapshot is live. Compaction
//! uses this as the **pin seq**: any version with a smaller seq than
//! the largest visible version at `pin_seq` is invisible to every
//! live snapshot and to current reads, and can be discarded.

use std::collections::BTreeMap;

use parking_lot::Mutex;

/// Thread-safe registry of live snapshot sequence numbers.
pub(crate) struct SnapshotRegistry {
    /// `seq -> refcount`. `BTreeMap` keeps the smallest seq at the
    /// front so `oldest_live_seq` is O(log n).
    active: Mutex<BTreeMap<u64, usize>>,
}

impl SnapshotRegistry {
    pub(crate) fn new() -> Self {
        Self {
            active: Mutex::new(BTreeMap::new()),
        }
    }

    /// Register a new pin at `seq`. Must be balanced by a later
    /// [`release`](Self::release) call — typically from the
    /// `Drop` impl of the owning snapshot type.
    pub(crate) fn register(&self, seq: u64) {
        let mut active = self.active.lock();
        *active.entry(seq).or_insert(0) += 1;
    }

    /// Release one pin at `seq`. A no-op if no pin at that seq is
    /// currently registered, which should only happen in test
    /// teardown races.
    pub(crate) fn release(&self, seq: u64) {
        let mut active = self.active.lock();
        if let Some(count) = active.get_mut(&seq) {
            *count -= 1;
            if *count == 0 {
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
}
