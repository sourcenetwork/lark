//! Concurrency primitives, taken from `loom` in a `--cfg loom` build and
//! from `std` / `parking_lot` otherwise.
//!
//! Loom replays a test under every thread interleaving the C11 memory
//! model permits, but it can only see accesses that go through its own
//! instrumented mocks: a `std::sync::atomic::AtomicPtr` is invisible to
//! it. Anything that wants to be model-checked therefore imports its
//! atomics and locks from here instead of from `std` directly. The
//! arena, the skip list, the memtable and the engine's read horizon all
//! do.
//!
//! The mocks are only usable inside a `loom::model` call, so `cfg(loom)`
//! is a model-checking build and nothing else. `tests/loom_memtable.rs`
//! is its only consumer; a normal build never compiles the `loom` arm
//! and never links the crate.

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

#[cfg(not(loom))]
pub(crate) use parking_lot::Mutex;

#[cfg(loom)]
pub(crate) use self::loom_mutex::Mutex;

#[cfg(loom)]
mod loom_mutex {
    /// `parking_lot::Mutex`'s surface over `loom::sync::Mutex`.
    ///
    /// Loom mocks `std`, so its `lock` hands back a `LockResult`;
    /// parking_lot does not poison and hands back the guard. Absorbing
    /// that difference here is what lets the arena and the memtable
    /// compile unchanged in both builds. A poisoned lock is recovered
    /// rather than unwrapped: a model-checking build must not turn a
    /// failing assertion in one thread into a panic in another.
    #[derive(Debug)]
    pub(crate) struct Mutex<T>(loom::sync::Mutex<T>);

    impl<T> Mutex<T> {
        /// A new mutex holding `value`.
        pub(crate) fn new(value: T) -> Self {
            Self(loom::sync::Mutex::new(value))
        }

        /// Lock, blocking until the mutex is available.
        pub(crate) fn lock(&self) -> loom::sync::MutexGuard<'_, T> {
            self.0
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
        }

        /// Borrow the contents mutably, without locking.
        pub(crate) fn get_mut(&mut self) -> &mut T {
            self.0
                .get_mut()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
        }
    }
}
