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
//!   chain by adding one level to it, and does not duplicate it. The
//!   top level of that chain is an epoch pin rather than a refcount: a
//!   published view that is replaced is retired to kovan, which frees
//!   it only once every reader that could still reach it has released
//!   its guard, and the `Arc`s the view holds are released at that
//!   reclamation rather than at the swap.
//! * Publication invariant: every publication is a compare-exchange
//!   from the exact view it derived its next value from, so the
//!   published views form a total order in which each is a function of
//!   its immediate predecessor and no publication can be lost. That is
//!   what replaced the publish mutex: publishers do not exclude each
//!   other, they rebuild on whatever won.
//! * Readers take no lock and touch no refcount. The only lock left in
//!   this file is the [`VersionStore`] mutex, which readers never
//!   acquire.
//!
//! # What reclamation costs, stated plainly
//!
//! Publication is lock-free, not wait-free: only the load is wait-free,
//! and a publisher still retries when another one wins. A retry only
//! happens because some other publisher succeeded, so the loop cannot
//! livelock, but it is not a bounded number of the publisher's own
//! steps and this module does not claim it is.
//!
//! A retired view is not dropped at the swap. It is handed to kovan,
//! which frees it once every reservation slot that could still reach it
//! has released. That deferral is the price of a read that touches no
//! refcount, and it is not only a memory question: a retired view owns
//! an `Arc<Version>`, and a `Version` owns the open descriptor of every
//! SSTable in it, including ones a compaction has already unlinked. A
//! deferred drop is therefore a held file descriptor and a held disk
//! block, not just a held allocation. The refcounted predecessor
//! released both at the swap.
//!
//! [`quiesce`] is what closes that gap, and every thread lark owns
//! calls it: at every version publication (where the dropped view may
//! own a whole obsolete `Version`), at every memtable retirement and
//! `drop_all` (where it may own a whole `write_buffer_size`
//! allocation), and in the compaction worker before it parks. Nothing
//! on the read path calls it.
//!
//! Reclamation is process-global, so a thread lark does **not** own
//! matters too. A thread that reads once and then idles keeps its
//! reservation slot published until its next pin, so it holds the
//! batches containing nodes born at or before its last pinned epoch.
//! The bound is that this is a one-time hold: the reclaimer's scan
//! skips a slot whose published epoch predates every node in a batch,
//! so a batch built entirely after that thread went idle is released on
//! schedule and the retention does not accumulate. That
//! bound is asserted, at the descriptor level, by
//! `tests/adv_atom_reclaim.rs`. An embedder whose threads pin and then
//! park for a long time can shorten the hold only by reading again or
//! by exiting; lark exposes no public drain today.
//!
//! # The deterministic counts, and where they do not hold
//!
//! Per read, on the fast path: one acquire load of the published
//! pointer, one thread-local pin, zero atomic read-modify-writes and
//! zero lock acquisitions. The refcounted predecessor cost one
//! `parking_lot::RwLock` shared acquisition plus four read-modify-writes
//! (lock acquire, lock release, `Arc::clone`, `Arc` drop).
//!
//! "Zero lock acquisitions" is exact on `x86_64`, `aarch64` and
//! `s390x`, where kovan's reservation slot is a native 128-bit DCAS and
//! its sub-word reads are plain `AtomicU64` loads. On every other
//! target - `armv7`, both `wasm32`s, `i686`, `riscv64` - the slot
//! routes through a `portable_atomic::AtomicU128` that may be emulated
//! with a spinlock, and `pin()` reads the slot's epoch on every
//! outermost pin. Reads there take that lock, reclamation loses its
//! lock-free progress guarantee, and this module claims neither. On a
//! single-threaded `wasm32` build the whole change is a wash: the
//! uncontended `RwLock` it replaced was already close to free.
//!
//! Per version edit: one [`quiesce`], and therefore one global epoch
//! advance. Every version edit has just written a manifest record, and
//! the destructors that advance runs are the same ones the refcounted
//! predecessor ran at the same point under the same mutex, so what is
//! new is the epoch advance alone. Throughput effect UNMEASURED.
//!
//! Reads hold their guard for the whole read, cold block fetches
//! included, which pins the reading thread's epoch across file I/O. A
//! reader stalled there defers younger batches until it resumes. That
//! is a deliberate trade for the consistent-triple guarantee, and the
//! write paths that must not pin across I/O drop their guard
//! explicitly rather than relying on scope.
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

use kovan::{Atom, AtomGuard};
use parking_lot::{Mutex, MutexGuard};

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
///
/// There is no publish mutex. A mutex-serialized load-then-store and a
/// compare-exchange do not compose - a publisher holding the mutex can
/// still overwrite a version a CAS publisher installed after its own
/// load - so the two publish paths are either both serialized or both
/// on the CAS loop. They are both on the CAS loop, which also removes
/// the `VersionStore` mutex -> publish mutex nesting.
pub(crate) struct ReadViewCell {
    current: Atom<ReadView>,
}

impl ReadViewCell {
    /// Publish an initial view. Called once per engine open.
    pub(crate) fn new(view: ReadView) -> Self {
        Self {
            current: Atom::new(view),
        }
    }

    /// Load the currently published view. This is the read path: one
    /// acquire load of the published pointer plus an epoch check, with
    /// no refcount traffic and - on the targets named in the module
    /// doc's counts section - no lock acquisition.
    ///
    /// The returned guard pins the view for as long as it is held, so
    /// hold it for exactly the read that needs it. A guard held across
    /// a flush or a compaction keeps that thread's epoch pinned and
    /// defers reclamation of everything retired under it.
    #[inline]
    pub(crate) fn load(&self) -> AtomGuard<'_, ReadView> {
        self.current.load()
    }

    /// Atomically replace the memtable half of the view. `mutate`
    /// receives the current `(active, frozen)` and returns the next
    /// pair, and both halves change in one publication rather than two
    /// observable steps.
    ///
    /// The transformation is applied inside a compare-exchange retry
    /// loop, so a rotation and a flush retirement each rebuild on
    /// whatever the other published instead of serializing behind it.
    /// `mutate` must therefore be a pure function of its arguments:
    /// it runs once per attempt, and every attempt but the last is
    /// discarded. Expensive or effectful work - allocating the fresh
    /// memtable, creating the replacement WAL - happens before the
    /// call, never inside the closure.
    ///
    /// The `Fn` bound is a partial fence and nothing more. It rejects a
    /// closure that moves a captured value out (handing off a `Wal`,
    /// retiring a memtable by moving it) or takes a unique borrow of
    /// one, which covers the mistakes that cost data. It does not
    /// reject interior mutability, so a `Cell` or an atomic bumped
    /// inside the closure compiles and is applied once per attempt;
    /// `tests::the_fn_bound_does_not_stop_an_effectful_retry` pins that
    /// down so the bound is not mistaken for a proof of purity.
    pub(crate) fn update_memtables(
        &self,
        mutate: impl Fn(&Arc<MemTable>, &[Arc<MemTable>]) -> (Arc<MemTable>, Vec<Arc<MemTable>>),
    ) {
        self.current.rcu(|current| {
            let (active, frozen) = mutate(&current.active, &current.frozen);
            ReadView {
                active,
                frozen,
                version: Arc::clone(&current.version),
            }
        });
    }

    /// Replace the version half of the view. Called by
    /// [`VersionGuard::drop`] and by nothing else.
    ///
    /// The closure is three refcount increments and a `Vec`
    /// allocation; a losing attempt drops the `ReadView` it built and
    /// returns every one of them, so a retry is exactly balanced.
    fn publish_version(&self, version: Arc<Version>) {
        self.current.rcu(|current| ReadView {
            active: Arc::clone(&current.active),
            frozen: current.frozen.clone(),
            version: Arc::clone(&version),
        });
        // The view this replaced owned the previous `Arc<Version>`, and
        // a `Version` owns the open descriptor of every SSTable in it
        // including the ones the edit just unlinked. Submitting the
        // batch here runs those drops at the point the refcounted
        // predecessor ran them: inside the version-set critical
        // section, which already holds a manifest write.
        quiesce();
    }
}

/// Release the retired views this thread is still gating, and submit
/// the ones it has retired but not yet handed to the reclaimer.
///
/// kovan keeps a thread's reservation slot published after its last
/// guard drops, so a thread that pinned and then went idle keeps every
/// batch born at or before its last pinned epoch alive. A retired
/// [`ReadView`] owns an `Arc<Version>`, which owns the open descriptor
/// of every SSTable in that version, so what is retained is not only
/// memory: it is file descriptors against the process rlimit and the
/// disk blocks of inodes a compaction has already unlinked. Every
/// thread lark owns calls this before it idles; see the module doc for
/// the bound that applies to threads it does not own.
pub(crate) fn quiesce() {
    kovan::flush();
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
    use std::sync::atomic::{AtomicU64, Ordering};

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

        let fresh = Arc::new(MemTable::new());
        cell.update_memtables(|active, frozen| {
            let mut next = frozen.to_vec();
            next.push(Arc::clone(active));
            (Arc::clone(&fresh), next)
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

    /// One publication, nothing racing it: the closure runs exactly
    /// once. Deterministic, so it holds on a loaded machine.
    #[test]
    fn an_uncontended_publication_runs_its_closure_once() {
        let (_dir, _store, cell) = store_with_view();
        let attempts = AtomicU64::new(0);
        let fresh = Arc::new(MemTable::new());

        cell.update_memtables(|active, frozen| {
            attempts.fetch_add(1, Ordering::SeqCst);
            let mut next = frozen.to_vec();
            next.push(Arc::clone(active));
            (Arc::clone(&fresh), next)
        });

        assert_eq!(attempts.load(Ordering::SeqCst), 1);
    }

    /// The lost-update property, forced deterministically: publishing
    /// from inside the first attempt guarantees the enclosing
    /// compare-exchange fails, so the retry has to rebuild on what won
    /// instead of clobbering it. No threads and no timing, so the
    /// counts below are exact rather than probable.
    ///
    /// This is one schedule, chosen because it is the one that matters,
    /// not an exhaustive check of every interleaving. Proving the
    /// publication protocol across all of them needs a model checker,
    /// which this crate does not yet carry.
    #[test]
    fn a_publication_that_loses_the_race_rebuilds_on_the_winner() {
        let (_dir, _store, cell) = store_with_view();
        let attempts = AtomicU64::new(0);
        let fresh = Arc::new(MemTable::new());
        let interloper = Arc::new(MemTable::new());

        cell.update_memtables(|active, frozen| {
            if attempts.fetch_add(1, Ordering::SeqCst) == 0 {
                cell.current.rcu(|current| ReadView {
                    active: Arc::clone(&interloper),
                    frozen: current.frozen.clone(),
                    version: Arc::clone(&current.version),
                });
            }
            let mut next = frozen.to_vec();
            next.push(Arc::clone(active));
            (Arc::clone(&fresh), next)
        });

        assert_eq!(
            attempts.load(Ordering::SeqCst),
            2,
            "a losing publication retries exactly once against one winner",
        );

        let after = cell.load();
        assert!(
            Arc::ptr_eq(&after.active, &fresh),
            "every attempt installs the same hoisted memtable",
        );
        assert_eq!(after.frozen.len(), 1);
        assert!(
            Arc::ptr_eq(&after.frozen[0], &interloper),
            "the retry sealed the memtable the winner published, not the one \
             the first attempt read - the interloper's change is not lost",
        );
    }

    /// A guard held across two publications keeps its view readable.
    /// `kovan::flush` is the strongest reclamation trigger kovan
    /// exposes, so if a retired view could be freed out from under a
    /// live reader, this is where it would happen. It is called
    /// directly rather than through [`quiesce`] so the check does not
    /// depend on what `quiesce` happens to be defined as.
    #[test]
    fn a_held_view_stays_readable_across_publications_and_a_flush() {
        let (_dir, _store, cell) = store_with_view();
        let pinned = cell.load();
        pinned.active.put(b"k", b"v", 1);
        let pinned_active = Arc::clone(&pinned.active);

        for _ in 0..2 {
            let fresh = Arc::new(MemTable::new());
            cell.update_memtables(|_, _| (Arc::clone(&fresh), Vec::new()));
        }
        kovan::flush();

        assert!(
            Arc::ptr_eq(&pinned.active, &pinned_active),
            "the pinned view still names the memtable it was published with",
        );
        assert_eq!(
            pinned.active.get(b"k", u64::MAX).and_then(|(_, v)| v),
            Some(b"v".to_vec())
        );
        assert!(cell.load().active.is_empty(), "the published view moved on");
    }

    /// The module doc claims the `Fn` bound is "the fence that makes a
    /// retry safe" and that it "refuses to compile a closure that ...
    /// mutates one (bumping a counter)". It does not. `Fn` forbids a
    /// unique borrow of a capture; it says nothing about interior
    /// mutability, so an effectful closure compiles and its effect is
    /// applied once per attempt.
    #[test]
    fn the_fn_bound_does_not_stop_an_effectful_retry() {
        use std::cell::Cell;

        let (_dir, _store, cell) = store_with_view();
        let side_effect = Cell::new(0u64);
        let attempts = AtomicU64::new(0);
        let fresh = Arc::new(MemTable::new());
        let interloper = Arc::new(MemTable::new());

        cell.update_memtables(|active, frozen| {
            side_effect.set(side_effect.get() + 1);
            if attempts.fetch_add(1, Ordering::SeqCst) == 0 {
                cell.current.rcu(|current| ReadView {
                    active: Arc::clone(&interloper),
                    frozen: current.frozen.clone(),
                    version: Arc::clone(&current.version),
                });
            }
            let mut next = frozen.to_vec();
            next.push(Arc::clone(active));
            (Arc::clone(&fresh), next)
        });

        assert_eq!(
            side_effect.get(),
            2,
            "an `Fn` closure applied its side effect once per attempt",
        );
    }

    /// A version publication that wins the race against a rotation must
    /// survive the rotation's retry, and vice versa. The existing race
    /// test only covers memtable-versus-memtable; these are the two
    /// cross-path orderings that the deleted publish mutex used to
    /// serialize.
    #[test]
    fn a_version_published_mid_retry_survives_the_rotation_that_lost() {
        let (_dir, store, cell) = store_with_view();
        let attempts = AtomicU64::new(0);
        let fresh = Arc::new(MemTable::new());

        cell.update_memtables(|active, frozen| {
            if attempts.fetch_add(1, Ordering::SeqCst) == 0 {
                store
                    .lock()
                    .apply(&[VersionEdit::SetNextFileId(11)])
                    .unwrap();
            }
            let mut next = frozen.to_vec();
            next.push(Arc::clone(active));
            (Arc::clone(&fresh), next)
        });

        let after = cell.load();
        assert_eq!(attempts.load(Ordering::SeqCst), 2);
        assert_eq!(
            after.version.next_file_id, 11,
            "the version the interloper published was clobbered by the retry",
        );
        assert!(Arc::ptr_eq(&after.active, &fresh));
        assert_eq!(after.frozen.len(), 1);
    }

    #[test]
    fn a_rotation_published_mid_retry_survives_the_version_edit_that_lost() {
        let (_dir, store, cell) = store_with_view();
        let sealed = Arc::clone(&cell.load().active);
        let fresh = Arc::new(MemTable::new());

        // Publish the rotation from inside the version-set critical
        // section, after the guard sampled its entry version. The
        // guard's publication on drop therefore starts from a view it
        // did not build on.
        {
            let mut guard = store.lock();
            cell.update_memtables(|active, frozen| {
                let mut next = frozen.to_vec();
                next.push(Arc::clone(active));
                (Arc::clone(&fresh), next)
            });
            guard.apply(&[VersionEdit::SetNextFileId(13)]).unwrap();
        }

        let after = cell.load();
        assert_eq!(after.version.next_file_id, 13);
        assert!(
            Arc::ptr_eq(&after.active, &fresh),
            "the version publication clobbered the rotation",
        );
        assert_eq!(after.frozen.len(), 1);
        assert!(Arc::ptr_eq(&after.frozen[0], &sealed));
    }

    /// The hold half of the reclamation contract, with
    /// `Arc::strong_count` as the probe rather than a timing loop: a
    /// memtable named by a live guard must NOT be released, however
    /// hard the publishing thread is pushed to reclaim.
    ///
    /// A second thread is held alive but idle for the duration, so the
    /// outcome does not depend on the harness happening to run this
    /// test alone in the process.
    #[test]
    fn a_pinned_memtable_is_held_while_the_guard_lives() {
        let (_dir, _store, cell) = store_with_view();
        let cell_for_helper = Arc::clone(&cell);
        let ready = Arc::new(std::sync::Barrier::new(2));
        let release = Arc::new(std::sync::Barrier::new(2));
        let helper = {
            let (ready, release) = (Arc::clone(&ready), Arc::clone(&release));
            std::thread::spawn(move || {
                drop(cell_for_helper.load());
                ready.wait();
                release.wait();
            })
        };
        ready.wait();

        let pinned = cell.load();
        let probe = Arc::clone(&pinned.active);
        assert_eq!(
            Arc::strong_count(&probe),
            2,
            "probe plus the published view"
        );

        for _ in 0..4 {
            let fresh = Arc::new(MemTable::new());
            cell.update_memtables(|_, _| (Arc::clone(&fresh), Vec::new()));
        }
        quiesce();
        assert!(
            Arc::strong_count(&probe) >= 2,
            "a retired view was released while a guard still named it",
        );

        drop(pinned);
        release.wait();
        helper.join().unwrap();
    }

    // The release half of the contract, and the bound on an idle
    // pinner, are process-global: a batch is freed only when every
    // reservation slot in the process releases it, and this binary
    // runs 500+ tests in parallel threads that are themselves idle
    // pinners. Both are asserted in `tests/adv_atom_reclaim.rs`, which
    // runs sequentially and observes the descriptors directly.

    #[test]
    fn a_critical_section_that_changes_no_version_publishes_nothing() {
        let (_dir, store, cell) = store_with_view();
        let before = cell.load();

        {
            let guard = store.lock();
            let _ = guard.current();
        }

        assert!(
            std::ptr::eq(&*before, &*cell.load()),
            "a read-only critical section must not churn the published view",
        );
    }
}
