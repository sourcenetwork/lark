//! The atomics, locks and gates the engine builds on.
//!
//! Everything here has two implementations selected by `cfg(loom)`. A
//! normal build takes `std::sync`; a `--cfg loom` build takes
//! `loom::sync`, whose instrumented mocks are the only accesses loom
//! can see. A `std::sync::atomic::AtomicPtr` is invisible to the model
//! checker, so anything that wants to be model-checked imports from
//! here rather than from `std` directly: the arena, the skip list, the
//! memtable, the read horizon and the commit pipeline all do.
//!
//! The loom mocks are only usable inside a `loom::model` call, so
//! `cfg(loom)` is a model-checking build and nothing else.
//!
//! # Why the wrappers exist
//!
//! [`Mutex`] and [`RwLock`] are thin newtypes rather than re-exports
//! because `std` poisons a lock whose holder panicked and hands back a
//! `LockResult`, while the engine has no use for poisoning: a panic in
//! one thread must not convert every later lock acquisition in every
//! other thread into a second panic, which is how a single failing
//! assertion turns into a torn shutdown. Each wrapper recovers the
//! guard with [`std::sync::PoisonError::into_inner`] and hands it back
//! directly, so a caller sees the lock's contents exactly as the
//! panicking thread left them and deals with it as ordinary state.
//!
//! [`Gate`] is the third primitive and the only one with no `std`
//! counterpart: a reader-writer gate whose exclusive guard owns its
//! claim instead of borrowing the gate, so it can outlive the call that
//! took it. `checkpoint_capture` needs exactly that, returning a
//! snapshot that pins every referenced SSTable against a concurrent
//! unlink for as long as the caller holds it.

#[cfg(loom)]
pub(crate) use loom::sync::Arc;
#[cfg(not(loom))]
pub(crate) use std::sync::Arc;

#[cfg(loom)]
pub(crate) use loom::sync::atomic::{AtomicPtr, AtomicU64, AtomicUsize, Ordering};
#[cfg(not(loom))]
pub(crate) use std::sync::atomic::{AtomicPtr, AtomicU64, AtomicUsize, Ordering};

#[cfg(all(loom, debug_assertions))]
pub(crate) use loom::sync::atomic::AtomicBool;
#[cfg(all(not(loom), debug_assertions))]
pub(crate) use std::sync::atomic::AtomicBool;

#[cfg(loom)]
use loom::sync as imp;
#[cfg(not(loom))]
use std::sync as imp;

pub(crate) use imp::{Condvar, MutexGuard, RwLockReadGuard, RwLockWriteGuard};

use std::sync::{PoisonError, TryLockError};

/// A mutex that hands back its guard rather than a `LockResult`.
///
/// See the module docs for why poisoning is absorbed rather than
/// propagated.
#[derive(Default)]
pub(crate) struct Mutex<T>(imp::Mutex<T>);

impl<T> Mutex<T> {
    /// A new mutex holding `value`.
    #[cfg(not(loom))]
    pub(crate) const fn new(value: T) -> Self {
        Self(imp::Mutex::new(value))
    }

    /// A new mutex holding `value`.
    ///
    /// Not `const` under loom: the mock allocates its own state.
    #[cfg(loom)]
    pub(crate) fn new(value: T) -> Self {
        Self(imp::Mutex::new(value))
    }

    /// Consume the mutex and return what it held.
    #[allow(dead_code)]
    pub(crate) fn into_inner(self) -> T {
        self.0.into_inner().unwrap_or_else(PoisonError::into_inner)
    }
}

impl<T> Mutex<T> {
    /// Lock, blocking until the mutex is available.
    pub(crate) fn lock(&self) -> MutexGuard<'_, T> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Lock if it is free right now, otherwise `None`.
    pub(crate) fn try_lock(&self) -> Option<MutexGuard<'_, T>> {
        match self.0.try_lock() {
            Ok(guard) => Some(guard),
            Err(TryLockError::Poisoned(poisoned)) => Some(poisoned.into_inner()),
            Err(TryLockError::WouldBlock) => None,
        }
    }

    /// Borrow the contents mutably. Exclusive access is proven by
    /// `&mut self`, so this takes no lock.
    #[allow(dead_code)]
    pub(crate) fn get_mut(&mut self) -> &mut T {
        self.0.get_mut().unwrap_or_else(PoisonError::into_inner)
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for Mutex<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// A reader-writer lock that hands back its guard rather than a
/// `LockResult`.
///
/// See the module docs for why poisoning is absorbed rather than
/// propagated.
#[derive(Default)]
pub(crate) struct RwLock<T>(imp::RwLock<T>);

impl<T> RwLock<T> {
    /// A new lock holding `value`.
    #[cfg(not(loom))]
    pub(crate) const fn new(value: T) -> Self {
        Self(imp::RwLock::new(value))
    }

    /// A new lock holding `value`.
    ///
    /// Not `const` under loom: the mock allocates its own state.
    #[cfg(loom)]
    pub(crate) fn new(value: T) -> Self {
        Self(imp::RwLock::new(value))
    }
}

impl<T> RwLock<T> {
    /// Take shared access, blocking until no writer holds the lock.
    pub(crate) fn read(&self) -> RwLockReadGuard<'_, T> {
        self.0.read().unwrap_or_else(PoisonError::into_inner)
    }

    /// Take exclusive access, blocking until the lock is free.
    pub(crate) fn write(&self) -> RwLockWriteGuard<'_, T> {
        self.0.write().unwrap_or_else(PoisonError::into_inner)
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for RwLock<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// State behind a [`Gate`]: how many readers hold it, whether a writer
/// does, and how many writers are queued.
#[derive(Default)]
struct GateState {
    readers: usize,
    writing: bool,
    writers_waiting: usize,
}

/// A reader-writer gate whose exclusive guard can own its claim.
///
/// The engine uses one to serialize compaction against the operations
/// that must not see a file unlinked underneath them. Compaction and
/// ordinary reads enter shared; a flush, a manual compaction and a
/// checkpoint enter exclusive.
///
/// Writer-preferring: a waiting writer blocks new readers, so a steady
/// stream of compaction passes cannot starve the checkpoint waiting
/// behind them. Readers do not recurse, which is what makes that
/// preference safe: a reader that re-entered while a writer waited
/// would deadlock.
pub(crate) struct Gate {
    state: Mutex<GateState>,
    cv: Condvar,
}

impl Gate {
    /// A gate nobody holds.
    pub(crate) fn new() -> Self {
        Self {
            state: Mutex::new(GateState::default()),
            cv: Condvar::new(),
        }
    }

    /// Enter shared, blocking while a writer holds or waits for the
    /// gate.
    pub(crate) fn read(&self) -> GateReadGuard<'_> {
        let mut state = self.state.lock();
        while state.writing || state.writers_waiting > 0 {
            state = self.cv.wait(state).unwrap_or_else(PoisonError::into_inner);
        }
        state.readers += 1;
        drop(state);
        GateReadGuard { gate: self }
    }

    /// Enter exclusive, blocking until no reader and no writer holds
    /// the gate.
    pub(crate) fn write(&self) -> GateWriteGuard<'_> {
        self.acquire_write();
        GateWriteGuard { gate: self }
    }

    /// Enter exclusive, returning a guard that keeps the gate alive
    /// itself and so may outlive this call.
    ///
    /// An associated function taking `std::sync::Arc` rather than a
    /// method: only `std::sync::Arc` is a legal `self` receiver, and the
    /// gate is not model-checked, so it uses the real `Arc` in both
    /// builds and agrees with the engine's.
    pub(crate) fn write_owned(gate: &std::sync::Arc<Self>) -> OwnedGateWriteGuard {
        gate.acquire_write();
        OwnedGateWriteGuard {
            gate: std::sync::Arc::clone(gate),
        }
    }

    fn acquire_write(&self) {
        let mut state = self.state.lock();
        state.writers_waiting += 1;
        while state.writing || state.readers > 0 {
            state = self.cv.wait(state).unwrap_or_else(PoisonError::into_inner);
        }
        state.writers_waiting -= 1;
        state.writing = true;
    }

    fn release_read(&self) {
        let mut state = self.state.lock();
        state.readers -= 1;
        let idle = state.readers == 0;
        drop(state);
        if idle {
            self.cv.notify_all();
        }
    }

    fn release_write(&self) {
        let mut state = self.state.lock();
        state.writing = false;
        drop(state);
        self.cv.notify_all();
    }
}

impl Default for Gate {
    fn default() -> Self {
        Self::new()
    }
}

/// Shared claim on a [`Gate`], released on drop.
pub(crate) struct GateReadGuard<'a> {
    gate: &'a Gate,
}

impl Drop for GateReadGuard<'_> {
    fn drop(&mut self) {
        self.gate.release_read();
    }
}

/// Exclusive claim on a [`Gate`], released on drop.
pub(crate) struct GateWriteGuard<'a> {
    gate: &'a Gate,
}

impl Drop for GateWriteGuard<'_> {
    fn drop(&mut self) {
        self.gate.release_write();
    }
}

/// Exclusive claim on a [`Gate`] that owns its share of the gate, so it
/// can be returned from the call that took it.
pub(crate) struct OwnedGateWriteGuard {
    gate: std::sync::Arc<Gate>,
}

impl Drop for OwnedGateWriteGuard {
    fn drop(&mut self) {
        self.gate.release_write();
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize as StdAtomicUsize, Ordering as StdOrdering};

    #[test]
    fn a_lock_whose_holder_panicked_still_hands_back_its_contents() {
        let mutex = Arc::new(Mutex::new(vec![1u32, 2, 3]));
        let poisoner = Arc::clone(&mutex);
        let panicked = std::thread::spawn(move || {
            let mut guard = poisoner.lock();
            guard.push(4);
            panic!("the holder dies mid-update");
        })
        .join();
        assert!(panicked.is_err(), "the probe needs the thread to panic");

        assert_eq!(
            *mutex.lock(),
            vec![1, 2, 3, 4],
            "poisoning must not hide what the panicking thread had already written",
        );
        assert!(mutex.try_lock().is_some(), "try_lock must recover too");
    }

    #[test]
    fn an_rwlock_whose_writer_panicked_still_reads() {
        let lock = Arc::new(RwLock::new(7u32));
        let poisoner = Arc::clone(&lock);
        let _ = std::thread::spawn(move || {
            let mut guard = poisoner.write();
            *guard = 9;
            panic!("the writer dies mid-update");
        })
        .join();

        assert_eq!(*lock.read(), 9);
        assert_eq!(*lock.write(), 9);
    }

    #[test]
    fn a_gate_admits_many_readers_but_only_one_writer() {
        let gate = Arc::new(Gate::new());
        let live = Arc::new(StdAtomicUsize::new(0));
        let peak = Arc::new(StdAtomicUsize::new(0));
        let exclusive_overlaps = Arc::new(StdAtomicUsize::new(0));

        std::thread::scope(|scope| {
            for i in 0..8 {
                let (gate, live, peak, overlaps) = (
                    Arc::clone(&gate),
                    Arc::clone(&live),
                    Arc::clone(&peak),
                    Arc::clone(&exclusive_overlaps),
                );
                scope.spawn(move || {
                    for _ in 0..200 {
                        if i % 4 == 0 {
                            let _w = gate.write();
                            if live.load(StdOrdering::SeqCst) != 0 {
                                overlaps.fetch_add(1, StdOrdering::SeqCst);
                            }
                            std::thread::yield_now();
                            if live.load(StdOrdering::SeqCst) != 0 {
                                overlaps.fetch_add(1, StdOrdering::SeqCst);
                            }
                        } else {
                            let _r = gate.read();
                            let now = live.fetch_add(1, StdOrdering::SeqCst) + 1;
                            peak.fetch_max(now, StdOrdering::SeqCst);
                            std::thread::yield_now();
                            live.fetch_sub(1, StdOrdering::SeqCst);
                        }
                    }
                });
            }
        });

        assert_eq!(
            exclusive_overlaps.load(StdOrdering::SeqCst),
            0,
            "a writer saw a reader inside the gate",
        );
        assert!(
            peak.load(StdOrdering::SeqCst) > 1,
            "readers never overlapped, so the gate is serializing them like a mutex",
        );
    }

    #[test]
    fn an_owned_write_guard_outlives_the_call_that_took_it() {
        let gate = Arc::new(Gate::new());
        let guard = held_by_a_returned_value(&gate);
        assert!(
            std::thread::scope(|scope| {
                let gate = Arc::clone(&gate);
                let probe = scope.spawn(move || gate.state.try_lock().is_some());
                probe.join().expect("probe")
            }),
            "the gate's own mutex must not be held between operations",
        );
        drop(guard);
        // The gate is free again, so an exclusive entry completes.
        drop(gate.write());
    }

    fn held_by_a_returned_value(gate: &Arc<Gate>) -> OwnedGateWriteGuard {
        Gate::write_owned(gate)
    }

    #[test]
    fn a_waiting_writer_blocks_new_readers() {
        let gate = Arc::new(Gate::new());
        let held = gate.read();

        let entered = Arc::new(StdAtomicUsize::new(0));
        std::thread::scope(|scope| {
            let (writer_gate, writer_entered) = (Arc::clone(&gate), Arc::clone(&entered));
            let writer = scope.spawn(move || {
                let _w = writer_gate.write();
                writer_entered.fetch_add(1, StdOrdering::SeqCst);
            });

            // Give the writer time to register itself as waiting, then
            // check that a fresh reader queues behind it rather than
            // joining the reader already inside.
            while gate.state.lock().writers_waiting == 0 {
                std::thread::yield_now();
            }
            assert!(
                gate.state.lock().readers == 1,
                "the probe needs exactly the one reader it took",
            );
            drop(held);
            writer.join().expect("writer");
        });
        assert_eq!(entered.load(StdOrdering::SeqCst), 1);
    }
}
