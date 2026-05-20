//! Tailing iterator: a forward-only iterator that keeps producing
//! new keys as writers append to the database, without closing and
//! reopening.
//!
//! # How it works
//!
//! A standard [`crate::Iter`] captures an `Arc<Version>` at creation
//! time, plus the set of active and frozen memtables. That pin
//! gives it snapshot isolation — any flush, compaction, or new
//! write after creation is invisible — at the cost of missing
//! later activity.
//!
//! A [`TailingIter`] holds an engine handle and, on top of that,
//! tracks the last user key it returned. When its current view is
//! exhausted, it asks the engine for a fresh
//! `(active_memtable, frozen_memtables, version)` tuple, rebuilds
//! its underlying cursor at `seq = u64::MAX`, and seeks strictly
//! past the last key it already emitted. This has three
//! consequences:
//!
//! * **New writes that sort after the current position are
//!   visible.** A memtable rotation, a flush to L0, or a fresh
//!   `put` at a larger key all show up on the next tail.
//! * **No key is ever re-emitted.** The seek-past-last-key step
//!   guarantees forward progress even after a refresh.
//! * **Writes to keys sorted _before_ the current position are
//!   permanently skipped.** That matches the typical tailing
//!   workload — appending to a log at monotonically increasing
//!   keys — and preserves the strict-ordering guarantee.
//!
//! Tailing iterators are forward-only. There is no `prev` /
//! `seek_for_prev` / `seek_to_last`, since reverse iteration
//! doesn't have a coherent "what's new since I was here"
//! semantics against a concurrently-written database.
//!
//! # Explicit refresh
//!
//! Callers that want to see writes that arrived between the
//! iterator's current position and the end of its current view
//! can call [`TailingIter::refresh`] to force a rebuild. This is
//! cheap — a few `Arc::clone` calls plus an in-memory merge-iter
//! reconstruction — but not free, so tight polling loops should
//! prefer to drain to exhaustion first and then refresh.

use std::sync::Arc;

use crate::column_family::{cf_upper_bound, prefix_key, ColumnFamilyHandle, DEFAULT_CF_ID};
use crate::engine::iterator::LarkIterator;
use crate::engine::LarkEngine;
use crate::statistics::{Histogram, Statistics, Ticker, TimeScope};
use crate::Result;

/// Forward-only tailing iterator. See the module docs.
///
/// Created via [`crate::Db::iter_tailing`] or
/// [`crate::Db::iter_tailing_cf`]. Owns an `Arc<LarkEngine>` so it
/// can be moved between threads and outlives the originating `Db`
/// handle.
pub struct TailingIter {
    engine: Arc<LarkEngine>,
    inner: LarkIterator,
    /// User key (with the CF prefix still attached — we match on
    /// the internal representation) of the most recently emitted
    /// entry. Used by [`Self::refresh`] to position the rebuilt
    /// cursor strictly past what the caller has already seen.
    last_returned: Option<Vec<u8>>,
    /// Column-family scope. Every seek target is prefixed with
    /// `cf_id.to_be_bytes()`, and `valid` enforces the upper
    /// bound so the iterator never bleeds across CF boundaries.
    cf_id: u32,
    /// Exclusive upper bound of the CF scope. Pre-computed so
    /// `valid` can compare without re-deriving the bound on every
    /// step.
    cf_upper: Vec<u8>,
    valid_cf: bool,
    /// Statistics sink captured at construction so seek/next
    /// instrumentation borrows only this field, not the whole
    /// `&self` (which would conflict with the mutable borrows in
    /// `refresh_and_reseek`).
    stats: Option<Arc<Statistics>>,
}

impl TailingIter {
    pub(crate) fn new(engine: Arc<LarkEngine>, cf_id: u32) -> Self {
        let inner = engine.new_iter(u64::MAX);
        let stats = engine.statistics_arc();
        Self {
            engine,
            inner,
            last_returned: None,
            cf_id,
            cf_upper: cf_upper_bound(cf_id),
            valid_cf: true,
            stats,
        }
    }

    pub(crate) fn empty(engine: Arc<LarkEngine>) -> Self {
        let mut iter = Self::new(engine, DEFAULT_CF_ID);
        iter.valid_cf = false;
        iter
    }

    fn tick_seek(&self) {
        if let Some(s) = self.stats.as_deref() {
            s.add(Ticker::IterSeekCount, 1);
        }
    }

    fn tick_next(&self) {
        if let Some(s) = self.stats.as_deref() {
            s.add(Ticker::IterNextCount, 1);
        }
    }

    /// Position the cursor at the first user key `>= target`
    /// within this iterator's column family. Resets the
    /// "last returned" bookkeeping so a subsequent refresh uses
    /// the new position as its floor.
    pub fn seek(&mut self, target: &[u8]) {
        if !self.valid_cf {
            return;
        }
        self.tick_seek();
        {
            let _t = TimeScope::new(self.stats.as_deref(), Histogram::DbIterSeek);
            self.last_returned = None;
            self.inner.seek(&prefix_key(self.cf_id, target));
        }
        self.record_current();
    }

    /// Position the cursor at the first key in this CF.
    pub fn seek_to_first(&mut self) {
        if !self.valid_cf {
            return;
        }
        self.tick_seek();
        {
            let _t = TimeScope::new(self.stats.as_deref(), Histogram::DbIterSeek);
            self.last_returned = None;
            self.inner.seek(&self.cf_id.to_be_bytes());
        }
        self.record_current();
    }

    /// Advance to the next user key. On exhaustion, automatically
    /// refreshes the underlying view once and re-seeks strictly
    /// past the last returned key. If still nothing new is
    /// available, the iterator becomes invalid; callers can poll
    /// by calling [`Self::refresh`] again later.
    pub fn next(&mut self) {
        if !self.valid_cf {
            return;
        }
        // Scope the timing handle so its borrow on `self.stats`
        // ends before we call the `&mut self` helpers below.
        {
            let _t = TimeScope::new(self.stats.as_deref(), Histogram::DbIterNext);
            if self.inner.valid() {
                self.inner.next();
            }
        }

        if !self.inner.valid() || !self.within_cf() {
            // Current view exhausted. Try a refresh — this picks
            // up memtable rotations, L0 flushes, and anything
            // else that landed after the previous view was
            // captured. `refresh_and_reseek` handles the
            // "seek strictly past last_returned" step so we
            // never re-emit a key we already produced.
            self.refresh_and_reseek();
        }

        self.record_current();
        if self.inner.valid() && self.within_cf() {
            self.tick_next();
        }
    }

    /// Force a refresh of the underlying view and re-seek to the
    /// first key strictly greater than the last one returned.
    /// Useful in tight polling loops that want to check for new
    /// data without waiting for the current view to be fully
    /// exhausted.
    pub fn refresh(&mut self) {
        if !self.valid_cf {
            return;
        }
        self.refresh_and_reseek();
        self.record_current();
    }

    fn refresh_and_reseek(&mut self) {
        // Rebuild the merging iterator against the engine's
        // latest `(active, frozen, version)` tuple.
        self.inner = self.engine.new_iter(u64::MAX);

        // Position strictly past the last key we already emitted.
        // If we haven't emitted anything yet, seek to the start
        // of the CF so we pick up everything in the new view.
        match self.last_returned.clone() {
            Some(k) => {
                // `LarkIterator::seek` interprets its argument as
                // a user key and lands on the newest visible
                // version, so a literal seek(k) returns k itself
                // (or the next user key if k was tombstoned).
                // Advance past k explicitly when seek landed on
                // it so the caller never sees a re-emission.
                self.inner.seek(&k);
                if self.inner.valid() && self.inner.key() == Some(k.as_slice()) {
                    self.inner.next();
                }
            }
            None => {
                self.inner.seek(&self.cf_id.to_be_bytes());
            }
        }
    }

    fn record_current(&mut self) {
        if self.inner.valid() && self.within_cf() {
            if let Some(k) = self.inner.key() {
                self.last_returned = Some(k.to_vec());
            }
        }
    }

    fn within_cf(&self) -> bool {
        if !self.valid_cf {
            return false;
        }
        match self.inner.key() {
            Some(k) => k >= self.cf_id.to_be_bytes().as_slice() && k < self.cf_upper.as_slice(),
            None => false,
        }
    }

    /// Whether the cursor is currently positioned on a visible
    /// key within this iterator's column family. Becomes `false`
    /// when the current view is exhausted; a subsequent
    /// [`Self::next`] or [`Self::refresh`] may re-validate it if
    /// new writes have arrived.
    pub fn valid(&self) -> bool {
        self.inner.valid() && self.within_cf()
    }

    /// Current user key with the CF prefix stripped, or `None` if
    /// the iterator is not valid.
    pub fn key(&self) -> Option<&[u8]> {
        if !self.valid() {
            return None;
        }
        self.inner.key().and_then(|k| k.get(4..))
    }

    /// Current value, or `None` if the iterator is not valid.
    pub fn value(&self) -> Option<&[u8]> {
        if !self.valid() {
            return None;
        }
        self.inner.value()
    }

    /// Propagate any I/O error from the underlying cursor.
    pub fn status(&self) -> Result<()> {
        self.inner.status().map_err(crate::Error::from)
    }
}

/// Helper called from `Db::iter_tailing` — defaults to the default
/// column family so the common case doesn't need to pass a handle.
pub(crate) fn new_default(engine: Arc<LarkEngine>) -> TailingIter {
    TailingIter::new(engine, DEFAULT_CF_ID)
}

/// Helper called from `Db::iter_tailing_cf`.
pub(crate) fn new_for_cf(engine: Arc<LarkEngine>, cf: &ColumnFamilyHandle) -> TailingIter {
    TailingIter::new(engine, cf.id())
}

/// Helper called when a stale CF handle is used with `Db::iter_tailing_cf`.
pub(crate) fn new_empty(engine: Arc<LarkEngine>) -> TailingIter {
    TailingIter::empty(engine)
}
