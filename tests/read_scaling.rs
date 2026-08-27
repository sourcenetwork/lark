//! Point-read throughput and allocation harness.
//!
//! Both tests are `#[ignore]`d: they are measurements, not gates. Run
//! them with
//!
//! ```text
//! cargo test --release --test read_scaling -- --ignored --nocapture
//! ```
//!
//! `point_read_scaling` reports random warm-cache `Db::get` throughput
//! at 1, 2, 4 and 8 threads plus the 8-thread scaling factor.
//! `allocations_per_warm_cache_get` reports the mean number of heap
//! allocations one warm-cache `Db::get` performs. Keys are built once,
//! before the timed loop, so neither number measures `format!`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};

use lark_kv::{Db, Options};
use tempfile::TempDir;

/// Counts every allocation the process makes. The counter is a relaxed
/// atomic increment on top of the system allocator, which is cheap
/// enough to leave in place for the throughput test too.
struct CountingAlloc;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: `layout` upholds `GlobalAlloc::alloc`'s contract by
        // this function's own contract, and `System` is a valid
        // allocator to forward it to.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: by this function's contract `ptr` came from a
        // `System` allocation made with exactly `layout`, which is the
        // invariant `System::dealloc` requires.
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: by this function's contract `ptr` came from a
        // `System` allocation made with exactly `layout`, and
        // `new_size` yields a valid layout when paired with
        // `layout.align()`.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAlloc = CountingAlloc;

/// Process CPU seconds (user + system) so far, read from
/// `/proc/self/stat`. Fields 14 and 15 are `utime` and `stime` in clock
/// ticks, and `USER_HZ` is 100 on every Linux target. Wall throughput
/// on a shared box measures the neighbours as much as the engine; CPU
/// time per operation does not, so both are reported.
fn cpu_seconds() -> f64 {
    let stat = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
    // The second field is the comm, which may contain spaces; skip past
    // its closing parenthesis before splitting.
    let rest = match stat.rfind(") ") {
        Some(i) => &stat[i + 2..],
        None => return 0.0,
    };
    let fields: Vec<&str> = rest.split_whitespace().collect();
    let tick = |i: usize| -> f64 {
        fields
            .get(i)
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.0)
    };
    // After the comm, `state` is index 0, so utime is index 11 and
    // stime index 12.
    (tick(11) + tick(12)) / 100.0
}

const KEYS: usize = 200_000;
const SECONDS: u64 = 3;
const THREAD_COUNTS: [usize; 4] = [1, 2, 4, 8];

fn keys() -> Arc<Vec<Vec<u8>>> {
    Arc::new(
        (0..KEYS)
            .map(|i| format!("key_{i:09}").into_bytes())
            .collect(),
    )
}

/// A database holding `KEYS` compacted keys with a warm block cache.
fn warm_db(dir: &TempDir, keys: &[Vec<u8>]) -> Arc<Db> {
    let opts = Options {
        write_buffer_size: 8 * 1024 * 1024,
        block_cache_size: 512 * 1024 * 1024,
        ..Options::default()
    };
    let db = Arc::new(Db::open(dir.path(), opts).unwrap());
    let value = vec![b'v'; 100];
    for key in keys {
        db.put(key, &value).unwrap();
    }
    db.compact_range(None, None).unwrap();
    for key in keys {
        assert!(db.get(key).unwrap().is_some());
    }
    db
}

#[test]
#[ignore = "throughput harness, not a gate; run with --ignored --nocapture"]
fn point_read_scaling() {
    let dir = TempDir::new().unwrap();
    let keys = keys();
    let db = warm_db(&dir, &keys);

    let mut rates = Vec::new();
    let mut line = String::new();
    for threads in THREAD_COUNTS {
        let stop = Arc::new(AtomicBool::new(false));
        let total = Arc::new(AtomicU64::new(0));
        let gate = Arc::new(Barrier::new(threads + 1));
        let mut handles = Vec::new();
        for t in 0..threads {
            let db = Arc::clone(&db);
            let keys = Arc::clone(&keys);
            let stop = Arc::clone(&stop);
            let total = Arc::clone(&total);
            let gate = Arc::clone(&gate);
            handles.push(thread::spawn(move || {
                let mut x = 0x9E37_79B9_7F4A_7C15u64 ^ (t as u64).wrapping_mul(0x1234_5678);
                let mut n = 0u64;
                gate.wait();
                while !stop.load(Ordering::Relaxed) {
                    for _ in 0..256 {
                        x ^= x << 13;
                        x ^= x >> 7;
                        x ^= x << 17;
                        let i = (x % KEYS as u64) as usize;
                        assert!(db.get(&keys[i]).unwrap().is_some());
                        n += 1;
                    }
                }
                total.fetch_add(n, Ordering::Relaxed);
            }));
        }
        gate.wait();
        let start = Instant::now();
        let cpu_start = cpu_seconds();
        thread::sleep(Duration::from_secs(SECONDS));
        stop.store(true, Ordering::Relaxed);
        for h in handles {
            h.join().unwrap();
        }
        let elapsed = start.elapsed().as_secs_f64();
        let cpu = cpu_seconds() - cpu_start;
        let done = total.load(Ordering::Relaxed) as f64;
        let ops = done / elapsed;
        let cpu_per_m = cpu / (done / 1e6);
        rates.push(ops);
        line.push_str(&format!("{threads}t={ops:.0}/{cpu_per_m:.2}cpu "));
    }
    let scaling = rates[rates.len() - 1] / rates[0];
    println!(
        "POINT_READ_SCALING {line}scaling_8t={scaling:.2}x (ops/s per thread count, and CPU seconds per million reads)"
    );
}

#[test]
#[ignore = "allocation harness, not a gate; run with --ignored --nocapture"]
fn allocations_per_warm_cache_get() {
    let dir = TempDir::new().unwrap();
    let keys = keys();
    let db = warm_db(&dir, &keys);

    const SAMPLES: usize = 100_000;
    let mut x = 0x243F_6A88_85A3_08D3u64;
    // One warm pass so every touched block is cached before counting.
    for i in 0..SAMPLES {
        assert!(db.get(&keys[i % KEYS]).unwrap().is_some());
    }
    let before = ALLOCS.load(Ordering::Relaxed);
    for _ in 0..SAMPLES {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        let i = (x % KEYS as u64) as usize;
        assert!(db.get(&keys[i]).unwrap().is_some());
    }
    let after = ALLOCS.load(Ordering::Relaxed);
    let per_get = (after - before) as f64 / SAMPLES as f64;
    println!("ALLOCS_PER_WARM_GET {per_get:.2}");
}
