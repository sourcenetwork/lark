//! The atomic types the engine uses, and the portability tier map.
//!
//! Every atomic in lark's production paths comes from here rather than
//! from `core::sync::atomic` directly. `portable-atomic` emits the
//! native instruction on any target that has one and substitutes an
//! implementation where the hardware does not, so a server build is
//! unchanged and a target without the instruction still builds.
//!
//! # Why not `core::sync::atomic`
//!
//! Two independent gaps, and only the second is about contention.
//!
//! * **`AtomicU64` does not exist on 32-bit bare metal.** `thumbv7em`
//!   and `riscv32imac` have no 64-bit atomic instruction, so
//!   `core::sync::atomic::AtomicU64` is simply absent on those
//!   targets. lark has six production `AtomicU64` sites and two of
//!   them, `latest_seq` and `visible_seq`, are the MVCC core.
//!   `portable-atomic`'s `fallback` feature supplies the type there.
//! * **`thumbv6m` has no compare-and-swap at all.** Not slow: absent.
//!   Every `fetch_add`, on any width, needs one. The
//!   `critical-section` feature of this crate forwards to
//!   `portable-atomic/critical-section`, which implements the missing
//!   operations inside a critical section. Enabling it requires the
//!   *binary* to also depend on `critical-section` and select a
//!   platform implementation; a library cannot choose one on the
//!   integrator's behalf. Without either hardware CAS or that feature
//!   the build stops at a named error rather than an obscure one,
//!   because `require-cas` is enabled.
//!
//! # What portability does not cost
//!
//! On a target with the native instruction `portable-atomic` is a
//! `#[repr(transparent)]` wrapper that lowers to the same code, so no
//! server or tier-A build pays for the fallback paths it never takes.
//!
//! On single-threaded wasm the atomics were already close to free:
//! without the `atomics` target feature a wasm module has exactly one
//! thread, and a read-modify-write lowers to a plain load, an
//! arithmetic op, and a store, with no lock prefix and no fence. That
//! is also why converting lark's remaining locks to lock-free
//! structures would be a pessimisation there rather than a win:
//! nothing contends, so every extra atomic read-modify-write a
//! lock-free algorithm performs is pure overhead.
//!
//! # Tiers
//!
//! | tier | example targets | threads | atomic CAS | state |
//! |---|---|---|---|---|
//! | server | `x86_64-unknown-linux-gnu`, `aarch64-unknown-linux-gnu` | many | native | supported |
//! | wasm | `wasm32-wasip1`, `wasm32-unknown-unknown` | one | native, never contended | supported |
//! | embedded A (`std`) | `armv7-unknown-linux-gnueabihf`, `aarch64-unknown-linux-gnu`, esp-idf | few | native | supported |
//! | embedded B (`no_std`) | `thumbv7em-none-eabi`, `thumbv6m-none-eabi`, `riscv32imac-unknown-none-elf` | one | absent on `thumbv6m` | **not supported**, see below |
//!
//! # `no_std` blockers
//!
//! Tier B does not build today and this module does not change that.
//! The atomics were one prerequisite; these are the rest, each with
//! what it costs to clear. The list is here rather than in a design
//! document so it cannot drift away from the dependency set it
//! describes.
//!
//! | dependency | state | cost to clear |
//! |---|---|---|
//! | `xxhash-rust` | already `#![no_std]` | none |
//! | `lru` | already `#![no_std]`, hashbrown-backed | none |
//! | `rustix` | already `cfg(unix)`-gated | none |
//! | `tracing` | `no_std` via `default-features = false` | low |
//! | `crossbeam-skiplist` | has an `alloc` feature | low: `default-features = false, features = ["alloc"]` |
//! | `lz4_flex` | `no_std` covers the block format, which is all lark uses | low: `default-features = false` |
//! | `thiserror` 1.0 | no `no_std` support | medium: bump to 2.x |
//! | `snap` | no `no_std` path at all; lark uses only `snap::raw` | medium: upstream a feature, vendor `raw.rs`, or drop Snappy on `no_std` |
//! | **`parking_lot`** | **`std`-only, and its wasm parker panics** | **high, see below** |
//!
//! ## `parking_lot`, specifically
//!
//! `parking_lot` is the largest single dependency blocker for tier B
//! and it fails in two distinct ways.
//!
//! * **It requires `std`.** `parking_lot_core` is one of the five
//!   crates that fail to compile for `thumbv7em-none-eabi` before
//!   lark's own code is even reached. It needs thread parking, which
//!   means a thread registry and a futex or equivalent, neither of
//!   which exists on bare metal.
//! * **Its wasm parker panics unconditionally.** `Condvar::wait` on
//!   `wasm32` reaches `thread_parker::imp::ThreadParker::prepare_park`
//!   and panics with "Parking not supported on this platform". That is
//!   a panic on a production path, not a degraded mode.
//!
//! Cost to clear: `Mutex` and `RwLock` map onto `std::sync` on a `std`
//! target and onto `spin` or `critical-section` on `no_std`, which is
//! a mechanical substitution across roughly 60 call sites. The
//! `arc_lock` feature lark enables today has no `std::sync`
//! equivalent, so the handful of `ArcRwLockReadGuard` users have to be
//! restructured to hold a plain guard or an owned `Arc` instead. The
//! `Condvar` half is more subtle than the `Mutex` half: on a
//! single-threaded target there is no other thread to wake, so a
//! blocking wait has to be replaced by the caller doing the work
//! itself rather than by a different condvar.

pub(crate) use portable_atomic::{AtomicBool, AtomicU64, AtomicU8, AtomicUsize, Ordering};
