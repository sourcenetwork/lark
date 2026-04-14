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

use crate::engine::iterator::LarkIterator;
use crate::engine::LarkEngine;
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
/// let mut it = db.iter().unwrap();
/// it.seek(b"b");
/// while it.valid() {
///     println!("{:?} = {:?}", it.key(), it.value());
///     it.next();
/// }
/// ```
pub struct Iter<'a> {
    inner: LarkIterator,
    // Ties the iterator's lifetime to its parent `Db` / `Snapshot` so the
    // borrow checker prevents the iterator from outliving the engine it
    // was built from.
    _marker: PhantomData<&'a LarkEngine>,
}

impl<'a> Iter<'a> {
    pub(crate) fn from_internal(inner: LarkIterator) -> Self {
        Self {
            inner,
            _marker: PhantomData,
        }
    }

    /// Position the iterator at the first user key `>= target`.
    pub fn seek(&mut self, target: &[u8]) {
        self.inner.seek(target);
    }

    /// Position the iterator at the largest user key `<= target`. The
    /// iterator can then be advanced forward with [`next`](Self::next).
    pub fn seek_for_prev(&mut self, target: &[u8]) {
        self.inner.seek_for_prev(target);
    }

    /// Position the iterator at the smallest user key in the database.
    pub fn seek_to_first(&mut self) {
        self.inner.seek_to_first();
    }

    /// Advance to the next user key. A no-op if the iterator is not valid.
    pub fn next(&mut self) {
        self.inner.next();
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
