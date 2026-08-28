//! A transaction commit must hand its buffered writes to the engine, not
//! copy them.
//!
//! The buffers are concurrent so that buffering can take `&self`. Reading
//! one back out through its borrowing iterator clones every key and every
//! value, which at commit is pure waste: the transaction is being
//! consumed, so nothing needs the map's own copies afterwards. The defect
//! this guards costs two allocations and a full byte copy of the write
//! buffer per commit, which on an `Options::embedded` budget of a few MiB
//! is enough to double peak footprint at the worst possible moment.
//!
//! Measured with a counting global allocator rather than a timer: the
//! copy is a byte-for-byte duplicate of the buffer, so counting bytes is
//! exact and load-independent where wall clock is neither.

// Native-only. wasm-pack builds every test target for wasm32, and this
// uses the filesystem.
#![cfg(not(target_arch = "wasm32"))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use regolith::{IsolationLevel, OptimisticTransactionDb, Options};
use tempfile::TempDir;

static ARMED: AtomicBool = AtomicBool::new(false);
static BYTES: AtomicUsize = AtomicUsize::new(0);
static COUNT: AtomicUsize = AtomicUsize::new(0);

struct Counting;

// SAFETY: every method forwards to `System`, which is a correct
// allocator, and only adds relaxed counter updates around it. No pointer
// or layout is altered, so the `GlobalAlloc` contract is exactly
// `System`'s.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        unsafe {
            let p = System.alloc(layout);
            if !p.is_null() && ARMED.load(Ordering::Relaxed) {
                BYTES.fetch_add(layout.size(), Ordering::Relaxed);
                COUNT.fetch_add(1, Ordering::Relaxed);
            }
            p
        }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        unsafe {
            let p = System.realloc(ptr, layout, new_size);
            if !p.is_null() && ARMED.load(Ordering::Relaxed) {
                BYTES.fetch_add(new_size.saturating_sub(layout.size()), Ordering::Relaxed);
                COUNT.fetch_add(1, Ordering::Relaxed);
            }
            p
        }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        unsafe {
            let p = System.alloc_zeroed(layout);
            if !p.is_null() && ARMED.load(Ordering::Relaxed) {
                BYTES.fetch_add(layout.size(), Ordering::Relaxed);
                COUNT.fetch_add(1, Ordering::Relaxed);
            }
            p
        }
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

const ENTRIES: usize = 512;
const VALUE_LEN: usize = 1024;

/// Buffer `ENTRIES` writes, then measure only the commit.
fn commit_alloc_bytes() -> (usize, usize) {
    let dir = TempDir::new().expect("tempdir");
    let db = Arc::new(OptimisticTransactionDb::open(dir.path(), Options::default()).expect("open"));
    let txn = db.begin_transaction_owned(IsolationLevel::Serializable);

    let value = vec![0xA5u8; VALUE_LEN];
    for i in 0..ENTRIES {
        txn.put(format!("key/{i:06}").as_bytes(), &value)
            .expect("put");
    }

    BYTES.store(0, Ordering::Relaxed);
    COUNT.store(0, Ordering::Relaxed);
    ARMED.store(true, Ordering::Relaxed);
    txn.commit().expect("commit");
    ARMED.store(false, Ordering::Relaxed);

    (BYTES.load(Ordering::Relaxed), COUNT.load(Ordering::Relaxed))
}

/// The commit must not allocate a per-entry copy of the write buffer.
///
/// Cloning the buffer out of the map costs two allocations per entry, one
/// for the key and one for the value, on top of whatever the WAL record
/// itself needs. Moving it costs none. Measured on this workload, 512
/// entries of 1 KiB:
///
/// ```text
/// clone (iter):      2_217_472 bytes, 1_699 allocations
/// move  (into_iter): 1_689_232 bytes,   679 allocations
/// ```
///
/// The 528_240 byte difference is one full duplicate of the buffer and
/// the 1_020 allocation difference is the two-per-entry clone. The bound
/// is on the allocation count rather than bytes, because the count is
/// what separates the two forms cleanly: the WAL encoding dominates the
/// byte figure and moves with unrelated tuning, while a per-entry copy
/// shows up as a fixed multiple of `ENTRIES` no matter how the record is
/// laid out.
#[test]
fn commit_does_not_copy_the_write_buffer() {
    let payload = ENTRIES * VALUE_LEN;
    let (bytes, count) = commit_alloc_bytes();

    eprintln!(
        "commit of {ENTRIES} x {VALUE_LEN}B: {bytes} bytes in {count} allocations \
         ({payload} bytes of values buffered)",
    );

    assert!(
        count < ENTRIES * 2,
        "commit made {count} allocations for {ENTRIES} buffered writes, which is at \
         least two per entry: the buffer is being cloned out of the map rather than \
         moved"
    );
}
