//! `ENTRY_OVERHEAD` is the whole reason the byte budget bounds memory
//! rather than payload bytes. This measures what a cached entry really
//! costs the allocator and checks the budget still covers it.
//!
//! Two fresh child processes run the identical workload over the
//! identical data directory, one with a real block cache and one with
//! `block_cache_size = 0`. The difference in live heap is what the
//! block cache costs; `lark.block-cache-usage` is what it claims to
//! cost. If the claim is under the cost, the budget stops bounding
//! memory and the over-allocation defect is back in a new costume.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicIsize, Ordering};

use std::sync::Arc;

use lark_kv::{Db, Options, Statistics, Ticker};

static LIVE: AtomicIsize = AtomicIsize::new(0);

struct Counting;

// SAFETY: every method forwards to `System` unchanged and only adds
// relaxed counter updates, so the `GlobalAlloc` contract is exactly
// `System`'s.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        unsafe {
            let p = System.alloc(layout);
            if !p.is_null() {
                LIVE.fetch_add(layout.size() as isize, Ordering::Relaxed);
            }
            p
        }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe {
            LIVE.fetch_sub(layout.size() as isize, Ordering::Relaxed);
            System.dealloc(ptr, layout)
        }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        unsafe {
            let p = System.realloc(ptr, layout, new_size);
            if !p.is_null() {
                LIVE.fetch_add(
                    new_size as isize - layout.size() as isize,
                    Ordering::Relaxed,
                );
            }
            p
        }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        unsafe {
            let p = System.alloc_zeroed(layout);
            if !p.is_null() {
                LIVE.fetch_add(layout.size() as isize, Ordering::Relaxed);
            }
            p
        }
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

const KEYS: u32 = 60_000;
const DIR_ENV: &str = "LARK_ADV_OVERHEAD_DIR";
const CACHE_ENV: &str = "LARK_ADV_OVERHEAD_CACHE";
const BS_ENV: &str = "LARK_ADV_OVERHEAD_BS";

fn key(i: u32) -> Vec<u8> {
    format!("k{i:07}").into_bytes()
}

/// Child: open the prepared DB with or without a cache, read every
/// key twice, and report live heap against the cache's own claim.
#[test]
fn adv_overhead_child() {
    let (Ok(dir), Ok(cache)) = (std::env::var(DIR_ENV), std::env::var(CACHE_ENV)) else {
        return;
    };
    let cache_bytes: usize = cache.parse().expect("cache size");
    let block_size: usize = std::env::var(BS_ENV)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1024);
    let stats = Arc::new(Statistics::new());
    let db = Db::open(
        &dir,
        Options {
            block_cache_size: cache_bytes,
            block_cache_num_shard_bits: 0,
            block_size,
            statistics: Some(Arc::clone(&stats)),
            ..Options::default()
        },
    )
    .unwrap();

    let heap_open = LIVE.load(Ordering::Relaxed);
    for _ in 0..2 {
        for i in 0..KEYS {
            assert!(db.get(&key(i)).unwrap().is_some());
        }
    }
    let heap_warm = LIVE.load(Ordering::Relaxed);
    let usage = db.get_int_property("lark.block-cache-usage").unwrap();
    let capacity = db.get_int_property("lark.block-cache-capacity").unwrap();
    let adds = stats.get_ticker(Ticker::BlockCacheAdd);
    let hits = stats.get_ticker(Ticker::BlockCacheHit);
    let misses = stats.get_ticker(Ticker::BlockCacheMiss);
    println!(
        "ADVRESULT cache={cache_bytes} heap={} usage={usage} capacity={capacity} \
adds={adds} hits={hits} misses={misses}",
        heap_warm - heap_open
    );
    drop(db);
}

#[test]
fn a_cached_entry_costs_no_more_heap_than_it_is_charged() {
    if std::env::var(DIR_ENV).is_ok() {
        return;
    }
    let prepare = |block_size: usize| -> tempfile::TempDir {
        let dir = tempfile::TempDir::new().unwrap();
        let db = Db::open(
            dir.path(),
            Options {
                block_cache_size: 0,
                block_size,
                write_buffer_size: 4 * 1024 * 1024,
                ..Options::default()
            },
        )
        .unwrap();
        for i in 0..KEYS {
            db.put(&key(i), &[(i % 251) as u8; 96]).unwrap();
        }
        db.compact_range(None, None).unwrap();
        db.close().unwrap();
        dir
    };
    let dir = prepare(1024);
    let small_dir = prepare(256);

    let exe = std::env::current_exe().expect("test binary");
    struct Row {
        heap: i64,
        usage: i64,
        capacity: i64,
        adds: i64,
    }
    let run_in = |data: &std::path::Path, cache_bytes: usize| -> Row {
        let out = std::process::Command::new(&exe)
            .args(["--exact", "adv_overhead_child", "--nocapture"])
            .env(DIR_ENV, data)
            .env(CACHE_ENV, cache_bytes.to_string())
            .output()
            .expect("spawn child");
        let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
        assert!(
            out.status.success(),
            "child with cache={cache_bytes} failed:\n{stdout}{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let line = stdout
            .lines()
            .find(|l| l.starts_with("ADVRESULT"))
            .unwrap_or_else(|| panic!("no result from child:\n{stdout}"))
            .to_string();
        println!("{line}");
        let field = |name: &str| -> i64 {
            line.split_whitespace()
                .find_map(|f| f.strip_prefix(name))
                .and_then(|v| v.parse().ok())
                .unwrap_or_else(|| panic!("missing {name} in {line}"))
        };
        Row {
            heap: field("heap="),
            usage: field("usage="),
            capacity: field("capacity="),
            adds: field("adds="),
        }
    };
    let run = |cache_bytes: usize| run_in(dir.path(), cache_bytes);

    // Baseline: the same reads with no cache at all. A disabled cache
    // must cost nothing on the read path.
    let off = run(0);
    assert_eq!(off.usage, 0, "a zero budget cached bytes");
    assert!(
        off.heap.abs() < 64 * 1024,
        "a disabled cache grew the heap by {} bytes over a read workload",
        off.heap
    );

    // Roomy: everything the workload touches fits, so nothing is
    // evicted and the delta is the pure per-entry cost.
    let roomy = run(64 * 1024 * 1024);
    assert!(
        roomy.adds > 0 && roomy.usage > 0,
        "the workload never filled the cache"
    );
    let roomy_real = roomy.heap - off.heap;
    let roomy_excess = roomy_real - roomy.usage;
    println!(
        "ADVOVERHEAD roomy real={roomy_real} charged={} excess={roomy_excess} \
entries={} excess_per_entry={:.2} charged_per_entry={:.1}",
        roomy.usage,
        roomy.adds,
        roomy_excess as f64 / roomy.adds as f64,
        roomy.usage as f64 / roomy.adds as f64
    );

    // Saturated: the hand runs constantly, so this also carries
    // whatever crossbeam-epoch has retired but not yet reclaimed.
    let tight = run(2 * 1024 * 1024);
    let tight_real = tight.heap - off.heap;
    println!(
        "ADVOVERHEAD saturated real={tight_real} charged={} capacity={} adds={} \
over_capacity={} ratio={:.4}",
        tight.usage,
        tight.capacity,
        tight.adds,
        tight_real - tight.capacity,
        tight_real as f64 / tight.capacity as f64
    );

    // Worst case for the charge: the smallest blocks the workload can
    // produce, so per-entry bookkeeping is the largest share of each
    // entry and any under-charge shows up as the largest fraction of
    // the budget.
    let tiny_off = run_in(small_dir.path(), 0);
    let tiny = run_in(small_dir.path(), 2 * 1024 * 1024);
    let tiny_real = tiny.heap - tiny_off.heap;
    println!(
        "ADVOVERHEAD small_blocks real={tiny_real} charged={} capacity={} adds={} \
charged_per_entry={:.1} over_capacity={} ratio={:.4}",
        tiny.usage,
        tiny.capacity,
        tiny.adds,
        tiny.usage as f64 / tiny.adds.max(1) as f64,
        tiny_real - tiny.capacity,
        tiny_real as f64 / tiny.capacity as f64
    );

    assert!(
        roomy_real <= roomy.usage + roomy.adds * 16,
        "an unevicted entry costs {} heap bytes more than the charge it is charged",
        roomy_excess as f64 / roomy.adds as f64
    );
    for (label, real, capacity) in [
        ("1 KiB blocks", tight_real, tight.capacity),
        ("256 B blocks", tiny_real, tiny.capacity),
    ] {
        assert!(
            real <= capacity * 5 / 4,
            "a saturated cache with {label} holds {real} heap bytes against a {capacity}-byte budget"
        );
    }
}
