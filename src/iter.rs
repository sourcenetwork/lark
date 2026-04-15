//! Public streaming iterator over a consistent snapshot of the database.
//!
//! An [`Iter`] is obtained from [`Db::iter`](crate::Db::iter) or
//! [`Snapshot::iter`](crate::Snapshot::iter) and yields user keys in
//! ascending order. It honors MVCC snapshot visibility — keys written
//! after the iterator was created are invisible — and hides tombstoned
//! keys from the stream. The iterator is safe against concurrent
//! background compaction: it holds pinned references to every SSTable
//! file that existed at creation time, so compaction can unlink files
//! without invalidating in-flight reads.

use std::marker::PhantomData;
use std::sync::Arc;

use crate::engine::iterator::LarkIterator;
use crate::engine::LarkEngine;
use crate::statistics::{Histogram, Statistics, Ticker, TimeScope};
use crate::Result;

/// Streaming iterator over a consistent view of the database.
///
/// Unlike [`Db::scan`](crate::Db::scan), which materializes an entire
/// result set at once, `Iter` yields entries on demand and bounds memory
/// by the size of a single block.
///
/// # Lifecycle
///
/// A fresh iterator is not positioned; call one of [`seek`](Self::seek),
/// [`seek_for_prev`](Self::seek_for_prev), or
/// [`seek_to_first`](Self::seek_to_first) before reading.
///
/// # Example
///
/// ```no_run
/// use lark_kv::{Db, Options};
///
/// let db = Db::open("/tmp/lark_iter", Options::default()).unwrap();
/// db.put(b"apple", b"red").unwrap();
/// db.put(b"banana", b"yellow").unwrap();
/// db.put(b"cherry", b"red").unwrap();
///
/// let mut it = db.iter();
/// it.seek(b"b");
/// while it.valid() {
///     println!("{:?} = {:?}", it.key(), it.value());
///     it.next();
/// }
/// ```
pub struct Iter<'a> {
    inner: LarkIterator,
    /// Optional statistics sink captured from the parent `Db`'s
    /// options at construction time. Seek / next / prev all fire
    /// ticker increments and timing histograms through this
    /// handle, guarded by a single `Option` branch.
    stats: Option<Arc<Statistics>>,
    // Ties the iterator's lifetime to its parent `Db` / `Snapshot` so the
    // borrow checker prevents the iterator from outliving the engine it
    // was built from.
    _marker: PhantomData<&'a LarkEngine>,
}

impl<'a> Iter<'a> {
    pub(crate) fn from_internal(inner: LarkIterator) -> Self {
        Self {
            inner,
            stats: None,
            _marker: PhantomData,
        }
    }

    pub(crate) fn with_stats(mut self, stats: Option<Arc<Statistics>>) -> Self {
        self.stats = stats;
        self
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

    /// Position the iterator at the first user key `>= target`. Sets the
    /// scan direction to forward, so subsequent [`next`](Self::next)
    /// calls advance alphabetically.
    pub fn seek(&mut self, target: &[u8]) {
        self.tick_seek();
        let _t = TimeScope::new(self.stats.as_deref(), Histogram::DbIterSeek);
        self.inner.seek(target);
    }

    /// Position the iterator at the largest user key `<= target`. Sets
    /// the scan direction to reverse, so subsequent [`prev`](Self::prev)
    /// calls walk backward. Calling [`next`](Self::next) after
    /// `seek_for_prev` flips direction and moves alphabetically forward.
    pub fn seek_for_prev(&mut self, target: &[u8]) {
        self.tick_seek();
        let _t = TimeScope::new(self.stats.as_deref(), Histogram::DbIterSeek);
        self.inner.seek_for_prev(target);
    }

    /// Position the iterator at the first user key starting with
    /// `prefix` and confine forward iteration to keys that share that
    /// prefix. The scan ends (iterator becomes invalid) as soon as the
    /// next candidate key falls outside the prefix range.
    ///
    /// When the database was built with a [`PrefixExtractor`] matching
    /// the prefix width, SSTables that demonstrably cannot contain the
    /// prefix are skipped via the prefix bloom filter. Files without a
    /// prefix bloom are consulted normally (conservative superset).
    /// Point lookups are unaffected.
    ///
    /// [`PrefixExtractor`]: crate::PrefixExtractor
    pub fn seek_prefix(&mut self, prefix: &[u8]) {
        self.tick_seek();
        let _t = TimeScope::new(self.stats.as_deref(), Histogram::DbIterSeek);
        self.inner.seek_prefix(prefix);
    }

    /// Position the iterator at the smallest user key in the database.
    /// Sets the scan direction to forward.
    pub fn seek_to_first(&mut self) {
        self.tick_seek();
        let _t = TimeScope::new(self.stats.as_deref(), Histogram::DbIterSeek);
        self.inner.seek_to_first();
    }

    /// Position the iterator at the largest user key in the database.
    /// Sets the scan direction to reverse.
    pub fn seek_to_last(&mut self) {
        self.tick_seek();
        let _t = TimeScope::new(self.stats.as_deref(), Histogram::DbIterSeek);
        self.inner.seek_to_last();
    }

    /// Advance to the next user key alphabetically. If the iterator was
    /// walking backward, direction flips before the advance. A no-op if
    /// the iterator is not valid.
    pub fn next(&mut self) {
        let _t = TimeScope::new(self.stats.as_deref(), Histogram::DbIterNext);
        self.inner.next();
        // Only count the step when it produced a key — an
        // end-of-stream `next` that invalidates the iterator
        // should not inflate the counter.
        if self.inner.valid() {
            self.tick_next();
        }
    }

    /// Step back to the previous user key alphabetically. If the
    /// iterator was walking forward, direction flips before the step.
    /// A no-op if the iterator is not valid.
    pub fn prev(&mut self) {
        let _t = TimeScope::new(self.stats.as_deref(), Histogram::DbIterNext);
        self.inner.prev();
        if self.inner.valid() {
            self.tick_next();
        }
    }

    /// Whether the iterator currently points at a valid `(key, value)`
    /// pair. Becomes `false` once the end of the stream is reached or an
    /// error is encountered.
    pub fn valid(&self) -> bool {
        self.inner.valid()
    }

    /// Returns the current user key, or `None` if the iterator isn't
    /// positioned on a live entry.
    pub fn key(&self) -> Option<&[u8]> {
        self.inner.key()
    }

    /// Returns the current value, or `None` if the iterator isn't
    /// positioned on a live entry.
    pub fn value(&self) -> Option<&[u8]> {
        self.inner.value()
    }

    /// Returns `Ok(())` if the iterator has not encountered an I/O error,
    /// or the most recent error otherwise. An iterator that reports an
    /// error stops yielding entries (`valid()` returns `false`).
    pub fn status(&self) -> Result<()> {
        self.inner.status().map_err(crate::Error::Io)
    }
}
