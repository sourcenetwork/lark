//! The published read view: the one object a reader loads to obtain a
//! consistent set of LSM sources.
//!
//! # Invariants
//!
//! * A [`ReadView`] is immutable once published. Every mutation of the
//!   memtable set or of the current version publishes a **new** view;
//!   nothing ever mutates a view a reader may be holding.
//! * The view is the single source of truth for the active memtable and
//!   the frozen memtable list. `LarkEngine` owns no separate copy.
//! * Successive published views only ever move data in the "older"
//!   direction (active -> frozen -> version) and never lose it, so a
//!   reader holding an older view sees a subset of the data a newer
//!   view exposes, never a different one.
//! * The `Arc<Version>` a view holds pins every `Arc<LiveSst>` in it,
//!   and each `LiveSst` holds the SSTable's open file descriptor. A
//!   compaction may unlink a table's path the instant it leaves the
//!   current version; a reader holding an older view keeps reading it
//!   through that descriptor. This view composes with that existing pin
//!   chain by adding one level to it, and does not duplicate it.
//! * Lock order: [`VersionStore`] mutex -> [`ReadViewCell::publish`]
//!   mutex -> `ReadViewCell::current` write lock. Nothing acquires the
//!   `VersionStore` mutex while holding either of the latter two.
//!   Readers acquire only `current`, shared.
//!
//! # Why the version half is published by the store, not by callers
//!
//! Every version change goes through [`VersionSet::apply`], which is
//! only reachable through a [`VersionGuard`]. The guard compares the
//! version it entered with against the one it leaves with and publishes
//! the difference, so a foreground flush, an ingest, a `drop_all` and a
//! background compaction all refresh the view without any of them
//! having to remember to.

use std::ops::{Deref, DerefMut};
use std::sync::{Arc, OnceLock};

use parking_lot::{Mutex, MutexGuard, RwLock};

use super::manifest::{Version, VersionSet};
use super::memtable::MemTable;

/// The set of sources one read resolves against, plus nothing else.
pub(crate) struct ReadView {
    /// The memtable writers are currently appending to.
    pub(crate) active: Arc<MemTable>,
    /// Memtables sealed and awaiting flush, oldest first.
    pub(crate) frozen: Vec<Arc<MemTable>>,
    /// The LSM version: the SSTables at every level, with their readers
    /// already open.
    pub(crate) version: Arc<Version>,
}

/// Holds the currently published [`ReadView`].
pub(crate) struct ReadViewCell {
    current: RwLock<Arc<ReadView>>,
    /// Serializes publishers so two of them cannot each build a next
    /// view from the same current one and lose the other's change.
    /// Readers never take this.
    publish: Mutex<()>,
}

impl ReadViewCell {
    /// Publish an initial view. Called once per engine open.
    pub(crate) fn new(view: ReadView) -> Self {
        Self {
            current: RwLock::new(Arc::new(view)),
            publish: Mutex::new(()),
        }
    }

    /// Load the currently published view. This is the read path: one
    /// shared lock acquisition and one `Arc` clone, with the guard
    /// dropped before the caller does any work.
    #[inline]
    pub(crate) fn load(&self) -> Arc<ReadView> {
        Arc::clone(&self.current.read())
    }

    /// Atomically replace the memtable half of the view. `mutate`
    /// receives the current `(active, frozen)` and returns the next
    /// pair plus a value for the caller. Runs under the publish mutex,
    /// so a rotation and a flush retirement serialize instead of
    /// racing, and both halves change in one publication rather than
    /// two observable steps.
    pub(crate) fn update_memtables<R>(
        &self,
        mutate: impl FnOnce(&Arc<MemTable>, &[Arc<MemTable>]) -> (Arc<MemTable>, Vec<Arc<MemTable>>, R),
    ) -> R {
        let _publishing = self.publish.lock();
        let current = self.load();
        let (active, frozen, out) = mutate(&current.active, &current.frozen);
        let next = Arc::new(ReadView {
            active,
            frozen,
            version: Arc::clone(&current.version),
        });
        *self.current.write() = next;
        out
    }

    /// Replace the version half of the view. Called by
    /// [`VersionGuard::drop`] and by nothing else.
    fn publish_version(&self, version: Arc<Version>) {
        let _publishing = self.publish.lock();
        let current = self.load();
        let next = Arc::new(ReadView {
            active: Arc::clone(&current.active),
            frozen: current.frozen.clone(),
            version,
        });
        *self.current.write() = next;
    }
}

/// The [`VersionSet`] plus the read view its edits publish into.
///
/// Every caller reaches the version set through [`Self::lock`], and the
/// guard that returns publishes any version the critical section
/// installed. That is what keeps a reader's view of the SSTables from
/// lagging behind a background compaction.
pub(crate) struct VersionStore {
    inner: Mutex<VersionSet>,
    /// Attached after construction: the view needs a memtable, and the
    /// version set is built before one exists. Empty only between
    /// [`Self::new`] and [`Self::attach_view`], a window with no
    /// concurrent readers.
    view: OnceLock<Arc<ReadViewCell>>,
}

impl VersionStore {
    pub(crate) fn new(versions: VersionSet) -> Self {
        Self {
            inner: Mutex::new(versions),
            view: OnceLock::new(),
        }
    }

    /// Attach the read-view cell this store publishes into. Called once,
    /// during engine open, before the engine is handed out.
    pub(crate) fn attach_view(&self, cell: Arc<ReadViewCell>) {
        let _ = self.view.set(cell);
    }

    /// Lock the version set. The returned guard derefs to
    /// [`VersionSet`] and publishes the resulting version on drop when
    /// the critical section changed it.
    pub(crate) fn lock(&self) -> VersionGuard<'_> {
        let guard = self.inner.lock();
        let entry_version = guard.current();
        VersionGuard {
            entry_version,
            guard,
            view: self.view.get(),
        }
    }
}

/// Exclusive access to the [`VersionSet`], publishing on release.
pub(crate) struct VersionGuard<'a> {
    guard: MutexGuard<'a, VersionSet>,
    view: Option<&'a Arc<ReadViewCell>>,
    entry_version: Arc<Version>,
}

impl Deref for VersionGuard<'_> {
    type Target = VersionSet;

    fn deref(&self) -> &VersionSet {
        &self.guard
    }
}

impl DerefMut for VersionGuard<'_> {
    fn deref_mut(&mut self) -> &mut VersionSet {
        &mut self.guard
    }
}

impl Drop for VersionGuard<'_> {
    fn drop(&mut self) {
        let Some(view) = self.view else {
            return;
        };
        let current = self.guard.current();
        if Arc::ptr_eq(&current, &self.entry_version) {
            return;
        }
        // Published while the version-set mutex is still held (this
        // runs before the guard field drops), so publications land in
        // the same order the edits did.
        view.publish_version(current);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::manifest::VersionEdit;

    fn store_with_view() -> (tempfile::TempDir, Arc<VersionStore>, Arc<ReadViewCell>) {
        let dir = tempfile::tempdir().unwrap();
        let sst_dir = dir.path().join("sst");
        std::fs::create_dir_all(&sst_dir).unwrap();
        let versions = VersionSet::open(dir.path(), &sst_dir).unwrap();
        let store = Arc::new(VersionStore::new(versions));
        let cell = Arc::new(ReadViewCell::new(ReadView {
            active: Arc::new(MemTable::new()),
            frozen: Vec::new(),
            version: store.lock().current(),
        }));
        store.attach_view(Arc::clone(&cell));
        (dir, store, cell)
    }

    #[test]
    fn a_rotation_publishes_the_sealed_memtable_and_the_fresh_one_together() {
        let (_dir, _store, cell) = store_with_view();
        let before = cell.load();
        before.active.put(b"k", b"v", 1);

        cell.update_memtables(|active, frozen| {
            let mut next = frozen.to_vec();
            next.push(Arc::clone(active));
            (Arc::new(MemTable::new()), next, ())
        });

        let after = cell.load();
        assert!(after.active.is_empty(), "writers got a fresh memtable");
        assert_eq!(after.frozen.len(), 1);
        assert!(
            Arc::ptr_eq(&after.frozen[0], &before.active),
            "the sealed memtable is the one writers were using",
        );
        assert!(
            Arc::ptr_eq(&after.version, &before.version),
            "a memtable publication leaves the version alone",
        );
        assert!(
            !before.active.is_empty(),
            "the view a reader still holds keeps its data",
        );
    }

    #[test]
    fn a_version_edit_publishes_a_new_view_that_keeps_the_memtables() {
        let (_dir, store, cell) = store_with_view();
        let before = cell.load();

        store
            .lock()
            .apply(&[VersionEdit::SetNextFileId(7)])
            .unwrap();

        let after = cell.load();
        assert_eq!(after.version.next_file_id, 7);
        assert!(Arc::ptr_eq(&after.active, &before.active));
        assert_eq!(after.frozen.len(), before.frozen.len());
    }

    #[test]
    fn a_critical_section_that_changes_no_version_publishes_nothing() {
        let (_dir, store, cell) = store_with_view();
        let before = cell.load();

        {
            let guard = store.lock();
            let _ = guard.current();
        }

        assert!(
            Arc::ptr_eq(&before, &cell.load()),
            "a read-only critical section must not churn the published view",
        );
    }
}
