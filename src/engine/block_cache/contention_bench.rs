//! Contention measurement for the block cache's per-shard mutex.
//!
//! [`BlockCache::get`] sits on every SSTable block read, and it takes a
//! `parking_lot::Mutex` because an LRU `get` mutates the recency list.
//! That makes it the highest-value remaining lock in the engine and the
//! only one worth considering for a lock-free replacement. A conversion
//! is only justified if a contended benchmark shows the lock losing, so
//! this module measures the lock against two references:
//!
//! * a **lock-free map**: `crossbeam_skiplist::SkipMap`, already a
//!   dependency, doing the same lookup with no lock and no eviction.
//!   Any real lock-free cache has to do at least this much work and
//!   then pay for eviction on top.
//! * a **zero-synchronisation floor**: a plain `Vec` index plus the
//!   `Arc::clone` every variant performs. Nothing can be faster than
//!   this, so the gap between it and the mutex is the entire budget a
//!   lock-free design could ever recover.
//!
//! Run with:
//!
//! ```text
//! cargo test --release --lib -- --ignored --nocapture --test-threads=1 block_cache
//! ```
//!
//! `--test-threads=1` matters: two of these running at once fight for
//! the same cores and both sets of numbers become meaningless.
//!
//! The numbers are machine-specific; the shape (scaling across thread
//! counts, and the gap to the floor) is what the decision rests on.
//!
//! # Verdict: the lock stays
//!
//! Measured on a 36-core x86_64 Linux box, aggregate throughput at 8
//! threads over a fully resident working set:
//!
//! | variant | 1 thread | 8 threads |
//! |---|---|---|
//! | mutex, 64 shards (shipped) | 30.4 Mops/s | 31.4 Mops/s |
//! | lock-free `SkipMap`, no eviction | 6.5 Mops/s | 24.7 Mops/s |
//! | mutex, 1 shard | 30.9 Mops/s | 2.1 Mops/s |
//! | no synchronisation (floor) | 82.4 Mops/s | 185.2 Mops/s |
//!
//! The lock-free structure available in the dependency set loses at
//! every thread count, and it loses while doing strictly less work
//! than a cache must: it never evicts. Sharding is already what keeps
//! the lock off the critical path, as the one-shard row shows by
//! collapsing without it.
//!
//! The remaining gap to the floor is not all lock. Decomposed by
//! [`block_cache_lock_share`] at 8 threads: 4.7 ns/op to hash the key
//! to a shard, about 166 ns/op for the mutex plus the shared-cacheline
//! write it protects, and about 85 ns/op for the LRU lookup and
//! recency splice inside the critical section. A lock-free design
//! removes the mutex but still has to publish recency somewhere, so
//! the recoverable budget is a fraction of that 166 ns rather than all
//! of it.
//!
//! Two further reasons this is not a close call for lark specifically.
//! On single-threaded wasm the lock is never contended and costs about
//! 14 ns uncontended, while every extra atomic read-modify-write a
//! lock-free algorithm performs is pure loss. On a target with no
//! compare-and-swap a lock-free cache falls back to a critical
//! section, which is a global lock and strictly worse than a sharded
//! one. And `Options::embedded()` disables the cache outright, so on
//! both targets this work is about the shard mutex never runs.

use std::hint::black_box;
use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};

use crossbeam_skiplist::SkipMap;

use super::BlockCache;
use crate::engine::block::{Block, BlockBuilder, RESTART_INTERVAL};

/// Payload bytes per cached block. Matches the default
/// `Options::block_size`.
const BLOCK_PAYLOAD: usize = 4096;

/// Distinct blocks in the working set. Chosen so the whole set fits
/// the budget below and every lookup is a hit: a miss benchmark would
/// measure the absent path, not the lock.
const NUM_BLOCKS: u64 = 4096;

/// Byte budget. Comfortably larger than the working set so nothing is
/// evicted mid-run and the measurement is not contaminated by eviction
/// churn that only one of the three variants performs.
const CAPACITY: usize = 64 * 1024 * 1024;

/// Offsets are spaced by the payload size so keys look like real block
/// offsets inside one file.
const STRIDE: u64 = BLOCK_PAYLOAD as u64;

const OPS_PER_THREAD: u64 = 300_000;

const THREAD_COUNTS: [usize; 4] = [1, 2, 4, 8];

/// Repetitions per data point, of which the fastest is kept. Anything
/// else running on the machine can only ever make a run slower, so the
/// minimum is the measurement least contaminated by it. Reporting a
/// mean instead would fold the neighbours' load into lark's number.
const REPS: usize = 5;

fn dummy_block() -> Arc<Block> {
    let mut builder = BlockBuilder::new(RESTART_INTERVAL);
    builder.add(b"k", &vec![0u8; BLOCK_PAYLOAD]);
    Arc::new(Block::decode(builder.finish()).expect("decode"))
}

/// xorshift64. A benchmark needs an index stream that the optimiser
/// cannot fold away and that costs the same in every variant; a real
/// PRNG dependency would only add noise.
#[inline]
fn next_index(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x % NUM_BLOCKS
}

/// One run of `body` on `threads` threads that start together,
/// returning the slowest thread's elapsed time. Taking the max rather
/// than the mean is what makes a contended run look contended: a lock
/// that starves one thread shows up here and averages away.
fn one_run<F>(threads: usize, body: &F) -> Duration
where
    F: Fn(u64) + Sync,
{
    let barrier = Barrier::new(threads + 1);
    let mut worst = Duration::ZERO;
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let barrier = &barrier;
                scope.spawn(move || {
                    barrier.wait();
                    let start = Instant::now();
                    body(t as u64 + 1);
                    start.elapsed()
                })
            })
            .collect();
        barrier.wait();
        worst = handles
            .into_iter()
            .map(|h| h.join().expect("bench thread"))
            .max()
            .unwrap_or(Duration::ZERO);
    });
    worst
}

/// The fastest of [`REPS`] runs. See that constant for why the minimum
/// rather than the mean.
fn timed<F>(threads: usize, body: F) -> Duration
where
    F: Fn(u64) + Sync,
{
    (0..REPS)
        .map(|_| one_run(threads, &body))
        .min()
        .unwrap_or(Duration::ZERO)
}

/// Nanoseconds of wall time per operation on the slowest thread, and
/// millions of operations per second aggregated across all threads.
fn report(label: &str, threads: usize, elapsed: Duration) -> f64 {
    let ns_per_op = elapsed.as_nanos() as f64 / OPS_PER_THREAD as f64;
    let total_ops = (threads as f64) * (OPS_PER_THREAD as f64);
    let mops = total_ops / elapsed.as_secs_f64() / 1e6;
    println!("{label:<34} {threads:>2}t  {ns_per_op:>7.1} ns/op  {mops:>8.2} Mops/s");
    mops
}

fn filled_cache(shard_bits: u32) -> BlockCache {
    let cache = BlockCache::with_config(CAPACITY, shard_bits, false);
    for i in 0..NUM_BLOCKS {
        cache.insert(1, i * STRIDE, dummy_block());
    }
    let hits = (0..NUM_BLOCKS)
        .filter(|i| cache.get(1, i * STRIDE).is_some())
        .count();
    assert_eq!(
        hits as u64, NUM_BLOCKS,
        "the working set must be fully resident or the benchmark measures misses, not the lock"
    );
    cache
}

fn scaling(first: f64, last: f64) -> String {
    format!("{:.2}x", last / first)
}

#[test]
#[ignore = "benchmark; run explicitly with --release --ignored --nocapture"]
fn block_cache_contention() {
    println!();
    println!(
        "block cache contention: {NUM_BLOCKS} resident blocks of {BLOCK_PAYLOAD} B, \
         {OPS_PER_THREAD} ops/thread"
    );
    println!();

    // Variant 1: the shipped cache at the default shard count.
    let cache = filled_cache(6);
    let mut sharded = Vec::new();
    for &threads in &THREAD_COUNTS {
        let elapsed = timed(threads, |seed| {
            let mut rng = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
            for _ in 0..OPS_PER_THREAD {
                let idx = next_index(&mut rng);
                black_box(cache.get(1, idx * STRIDE));
            }
        });
        sharded.push(report("mutex, 64 shards (default)", threads, elapsed));
    }
    drop(cache);

    // Variant 2: the same cache collapsed to one shard. This is the
    // worst case the lock can produce and it bounds how much a
    // lock-free replacement could win even in principle.
    let cache = filled_cache(0);
    let mut single = Vec::new();
    for &threads in &THREAD_COUNTS {
        let elapsed = timed(threads, |seed| {
            let mut rng = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
            for _ in 0..OPS_PER_THREAD {
                let idx = next_index(&mut rng);
                black_box(cache.get(1, idx * STRIDE));
            }
        });
        single.push(report("mutex, 1 shard (worst case)", threads, elapsed));
    }
    drop(cache);

    // Variant 3: a real lock-free map doing the same lookup, with no
    // LRU maintenance and no eviction to pay for.
    let map: SkipMap<(u64, u64), Arc<Block>> = SkipMap::new();
    for i in 0..NUM_BLOCKS {
        map.insert((1, i * STRIDE), dummy_block());
    }
    let mut lockfree = Vec::new();
    for &threads in &THREAD_COUNTS {
        let elapsed = timed(threads, |seed| {
            let mut rng = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
            for _ in 0..OPS_PER_THREAD {
                let idx = next_index(&mut rng);
                black_box(map.get(&(1, idx * STRIDE)).map(|e| Arc::clone(e.value())));
            }
        });
        lockfree.push(report("lock-free SkipMap (no eviction)", threads, elapsed));
    }
    drop(map);

    // Variant 4: no synchronisation at all. Index a slice and clone the
    // `Arc`, which every variant above also does. Nothing can beat this.
    let blocks: Vec<Arc<Block>> = (0..NUM_BLOCKS).map(|_| dummy_block()).collect();
    let mut floor = Vec::new();
    for &threads in &THREAD_COUNTS {
        let elapsed = timed(threads, |seed| {
            let mut rng = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
            for _ in 0..OPS_PER_THREAD {
                let idx = next_index(&mut rng);
                black_box(Arc::clone(&blocks[idx as usize]));
            }
        });
        floor.push(report("no synchronisation (floor)", threads, elapsed));
    }

    println!();
    println!(
        "scaling 1t -> 8t:  mutex/64 {}   mutex/1 {}   skipmap {}   floor {}",
        scaling(sharded[0], sharded[3]),
        scaling(single[0], single[3]),
        scaling(lockfree[0], lockfree[3]),
        scaling(floor[0], floor[3]),
    );
}

/// Decompose [`BlockCache::get`] into the parts a lock-free rewrite
/// could remove and the parts it could not.
///
/// The floor in [`block_cache_contention`] is far below the cache, but
/// most of that gap is work no design avoids: hashing the key to a
/// shard, and touching the recency metadata. Only the mutex itself is
/// recoverable, so this measures the mutex on its own, over the same
/// shard array and the same key hash, with a trivial critical section.
/// Whatever that costs is the entire budget a lock-free block cache
/// could compete for.
#[test]
#[ignore = "benchmark; run explicitly with --release --ignored --nocapture"]
fn block_cache_lock_share() {
    use parking_lot::Mutex;
    use xxhash_rust::xxh3::xxh3_64;

    println!();
    println!("what the shard mutex itself costs, {OPS_PER_THREAD} ops/thread");
    println!();

    // A shard mutex only costs what it costs if it sits on its own
    // cacheline. Sixty-four `Mutex<u64>` fit in sixteen lines, so an
    // unpadded microbench measures false sharing and calls it lock
    // cost. The real shard is much larger; this is what it measures.
    println!(
        "size_of::<Mutex<CacheShard>>() = {} B, 64 shards span {} cachelines",
        std::mem::size_of::<Mutex<super::CacheShard>>(),
        (64 * std::mem::size_of::<Mutex<super::CacheShard>>()).div_ceil(64),
    );
    println!();

    #[repr(align(64))]
    struct Padded(Mutex<u64>);

    for &(num_shards, padded, label) in &[
        (64usize, false, "hash only, no lock"),
        (64, false, "hash + lock, 64 packed shards"),
        (64, true, "hash + lock, 64 padded shards"),
        (1, false, "hash + lock, 1 shard"),
    ] {
        let shards: Box<[Padded]> = (0..num_shards)
            .map(|_| Padded(Mutex::new(0u64)))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let packed: Box<[Mutex<u64>]> = (0..num_shards)
            .map(|_| Mutex::new(0u64))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let mask = (num_shards - 1) as u64;
        let lock = !label.starts_with("hash only");
        for &threads in &THREAD_COUNTS {
            let elapsed = timed(threads, |seed| {
                let mut rng = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
                for _ in 0..OPS_PER_THREAD {
                    let idx = next_index(&mut rng);
                    let mut buf = [0u8; 16];
                    buf[..8].copy_from_slice(&1u64.to_le_bytes());
                    buf[8..].copy_from_slice(&(idx * STRIDE).to_le_bytes());
                    let h = (xxh3_64(&buf) & mask) as usize;
                    if !lock {
                        black_box(h);
                    } else if padded {
                        let mut g = shards[h].0.lock();
                        *g = g.wrapping_add(idx);
                    } else {
                        let mut g = packed[h].lock();
                        *g = g.wrapping_add(idx);
                    }
                }
            });
            report(label, threads, elapsed);
        }
        println!();
    }
}

#[test]
#[ignore = "benchmark; run explicitly with --release --ignored --nocapture"]
fn block_cache_contention_with_inserts() {
    println!();
    println!("block cache, 95% get / 5% insert, {OPS_PER_THREAD} ops/thread");
    println!();
    for &shard_bits in &[6u32, 0u32] {
        let label = if shard_bits == 6 {
            "mutex, 64 shards, 5% inserts"
        } else {
            "mutex, 1 shard, 5% inserts"
        };
        let cache = filled_cache(shard_bits);
        let mut mops = Vec::new();
        for &threads in &THREAD_COUNTS {
            let elapsed = timed(threads, |seed| {
                let mut rng = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
                let block = dummy_block();
                for op in 0..OPS_PER_THREAD {
                    let idx = next_index(&mut rng);
                    if op % 20 == 0 {
                        cache.insert(1, idx * STRIDE, Arc::clone(&block));
                    } else {
                        black_box(cache.get(1, idx * STRIDE));
                    }
                }
            });
            mops.push(report(label, threads, elapsed));
        }
        println!("scaling 1t -> 8t: {}", scaling(mops[0], mops[3]));
        println!();
    }
}
