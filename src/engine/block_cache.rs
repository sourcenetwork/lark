use std::ops::Bound;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use crossbeam_epoch::{self as epoch, Guard};
use crossbeam_skiplist::SkipList;
use parking_lot::Mutex;
use xxhash_rust::xxh3::xxh3_64;

use super::block::Block;
use crate::options::MAX_BLOCK_CACHE_SHARD_BITS;
use crate::statistics::{Statistics, Ticker};

/// Cache key: (file_id, block_offset).
///
/// Ordered on `(file_id, offset)`, so every block of one file is a
/// single contiguous range of the shard map and [`BlockCache::evict_file`]
/// can walk that file's entries instead of the whole shard.
#[derive(Eq, PartialEq, PartialOrd, Ord, Clone, Copy)]
struct CacheKey {
    file_id: u64,
    offset: u64,
}

/// Hard upper bound on the number of shards the cache will ever
/// create. A 32-bit shard-bit config of 8 → 256 shards is plenty
/// for a single-process embedded store.
const MAX_SHARD_BITS: u32 = MAX_BLOCK_CACHE_SHARD_BITS;

/// Minimum per-shard capacity. Tiny caches with many shards
/// would otherwise produce shards with 0 bytes of capacity, which
/// is almost certainly a misconfiguration: fall back to a single
/// shard in that case.
const MIN_SHARD_CAPACITY: usize = 64 * 1024;

/// Bookkeeping bytes an entry costs beyond [`Block::charge`]: the
/// skip-list node and its tower, the [`ClockEntry`] allocation, the ring
/// slot, and the `Arc` reference counts. Charged against the byte budget
/// so the budget alone bounds everything the cache holds, with no
/// separate entry-count cap that could bind invisibly below it.
///
/// This is an under-estimate, not a ceiling. A counting global allocator
/// over 6,000 live entries with no eviction in flight measured 138.6
/// bytes of live heap per entry against the 128 charged here, so the
/// bookkeeping runs about 8% past its own charge; the allocator counts
/// requested layout sizes, so the real figure is higher again by
/// whatever the size classes round up. Raising the constant would admit
/// fewer entries per budget and move the hit rate, so it is left where
/// it is deliberately and the shortfall is stated on
/// [`BlockCache`] instead of hidden here.
// vertexia: 128 under-charges measured bookkeeping by ~8%; raising it is
// a cache-behaviour change and wants its own hit-rate measurement.
const ENTRY_OVERHEAD: usize = 128;

/// Bytes one cached entry costs the cache.
fn entry_charge(block: &Block) -> usize {
    block.charge() + ENTRY_OVERHEAD
}

/// One cached block plus the CLOCK bookkeeping the hand reads.
///
/// The shard map and the shard ring hold `Arc`s to the same allocation,
/// so a reader that reaches the entry through the map sets the same
/// reference bit the hand later clears.
struct ClockEntry {
    block: Arc<Block>,
    /// The map key, so the hand can unlink the entry it evicts without
    /// a reverse lookup.
    key: CacheKey,
    /// [`entry_charge`] at insert time, so an eviction subtracts exactly
    /// what the insert added.
    charge: usize,
    /// Ring index, fixed for the entry's lifetime: a replace or an
    /// `evict_file` reaches the slot in O(1) instead of scanning.
    slot: usize,
    /// Set by every reader through `&self`, cleared by the hand.
    /// Advisory only, so `Relaxed` is enough: no data is published
    /// through the bit, and a lost update costs one extra miss rather
    /// than correctness.
    referenced: AtomicBool,
}

/// The CLOCK ring: one slot per live entry, plus the hand.
///
/// Grows on demand and reuses freed slots, so its length is the peak
/// live entry count, which the byte budget already bounds. Nothing here
/// is sized from `capacity` or from the shard count.
///
/// A shard drained by `evict_file` keeps the two vectors' capacity as a
/// reserve for the blocks that replace the ones it just dropped, so
/// `used` can read 0 while the ring still holds 16 bytes per
/// peak-live-entry. That reserve is bounded by the budget the entries
/// were charged against, and 16 of each entry's [`ENTRY_OVERHEAD`]
/// bytes are those two slots, so it is paid for while the entries live.
/// [`ClockRing::reset`], which is what `BlockCache::clear` runs, hands
/// it back.
struct ClockRing {
    slots: Vec<Option<Arc<ClockEntry>>>,
    free: Vec<usize>,
    /// Slot the hand inspects next. Kept below `slots.len()`; `reset`
    /// clears both together.
    hand: usize,
    /// Bytes currently held by this shard, per [`entry_charge`]. A plain
    /// field rather than an atomic because every mutation happens under
    /// the ring mutex anyway.
    used: usize,
}

impl ClockRing {
    const fn new() -> Self {
        Self {
            slots: Vec::new(),
            free: Vec::new(),
            hand: 0,
            used: 0,
        }
    }

    /// Reserve a ring slot, reusing a freed one before growing.
    /// The returned index is always in bounds for `slots`.
    fn alloc_slot(&mut self) -> usize {
        if let Some(slot) = self.free.pop() {
            return slot;
        }
        self.slots.push(None);
        self.slots.len() - 1
    }

    /// Release `slot` and the bytes the entry it held charged.
    fn release(&mut self, slot: usize, charge: usize) {
        if let Some(held) = self.slots.get_mut(slot) {
            if held.take().is_some() {
                self.free.push(slot);
                self.used = self.used.saturating_sub(charge);
            }
        }
    }

    /// Drop every entry and hand the slot vectors' capacity back to the
    /// allocator. Assigning fresh vectors rather than clearing in place
    /// is the difference between `clear` releasing the reserve and
    /// merely emptying it.
    fn reset(&mut self) {
        self.slots = Vec::new();
        self.free = Vec::new();
        self.hand = 0;
        self.used = 0;
    }
}

/// Per-shard state: a lock-free map the readers share, and a ring the
/// writers take a mutex for.
///
/// Every mutation of `map`, of `ring`, and of `ring.used` happens under
/// `ring`'s mutex, so a shard has one writer at a time and any number of
/// concurrent readers. [`CacheShard::get`] takes no lock at all.
struct CacheShard {
    /// Created on the shard's first insert. A `SkipList` header is 512
    /// bytes, so allocating one per shard up front would put the cache's
    /// own footprint back under the shard count; an empty shard costs
    /// the `OnceLock` and the ring instead. A reader on an
    /// uninitialized shard sees `None` and misses, which is correct: the
    /// shard holds nothing.
    map: OnceLock<Box<SkipList<CacheKey, Arc<ClockEntry>>>>,
    /// Byte budget for this shard: total capacity / num_shards.
    capacity: usize,
    ring: Mutex<ClockRing>,
}

impl CacheShard {
    fn new(capacity: usize) -> Self {
        Self {
            map: OnceLock::new(),
            capacity,
            ring: Mutex::new(ClockRing::new()),
        }
    }

    /// This shard's map, created on first use.
    fn map(&self) -> &SkipList<CacheKey, Arc<ClockEntry>> {
        self.map
            .get_or_init(|| Box::new(SkipList::new(epoch::default_collector().clone())))
    }

    /// Look up `key`, giving the entry a second chance against the hand.
    ///
    /// Lock-free: the map read is an epoch-protected traversal and the
    /// reference bit is a plain atomic store, so a reader never waits on
    /// an insert that is freeing blocks.
    fn get(&self, key: &CacheKey) -> Option<Arc<Block>> {
        let map = self.map.get()?;
        let guard = epoch::pin();
        let found = map.get(key, &guard)?;
        let entry = found.value();
        entry.referenced.store(true, Ordering::Relaxed);
        Some(Arc::clone(&entry.block))
    }

    /// Drop any entry already stored at `key` so a re-insert replaces
    /// rather than double-counts, returning its ring slot to the free
    /// list.
    fn take_existing(&self, ring: &mut ClockRing, key: &CacheKey, guard: &Guard) {
        let Some(map) = self.map.get() else {
            return;
        };
        let Some(found) = map.get(key, guard) else {
            return;
        };
        let entry = found.value();
        ring.release(entry.slot, entry.charge);
        found.remove();
    }

    /// Advance the CLOCK hand to the next evictable entry and drop it.
    /// Returns `false` only when the ring holds no live entry.
    ///
    /// The first revolution clears every reference bit it passes; from
    /// the second the hand takes whatever it lands on. A reader that
    /// keeps re-setting bits therefore costs at most one extra miss and
    /// can never stall an insert, and the search is bounded at
    /// `2 * slots.len() + 1` steps. Refusing the insert instead would
    /// make admission depend on reader timing, which is a silent
    /// hit-rate cliff rather than one extra miss.
    fn evict_one(&self, ring: &mut ClockRing, guard: &Guard) -> bool {
        let len = ring.slots.len();
        if len == 0 {
            return false;
        }
        if ring.hand >= len {
            ring.hand = 0;
        }
        for step in 0..=2 * len {
            let hand = ring.hand;
            ring.hand = if hand + 1 == len { 0 } else { hand + 1 };
            let forced = step >= len;
            let Some(entry) = ring.slots[hand]
                .take_if(|entry| forced || !entry.referenced.swap(false, Ordering::Relaxed))
            else {
                continue;
            };
            ring.free.push(hand);
            ring.used = ring.used.saturating_sub(entry.charge);
            if let Some(map) = self.map.get() {
                if let Some(found) = map.get(&entry.key, guard) {
                    found.remove();
                }
            }
            return true;
        }
        false
    }

    /// Insert `(key, block)` within this shard's byte budget, running
    /// the hand to make room. Returns `false` without storing anything
    /// when `size` exceeds the whole shard budget; the caller then
    /// decides whether the cache-wide budget can still absorb it.
    fn insert_within_budget(
        &self,
        ring: &mut ClockRing,
        key: CacheKey,
        block: &Arc<Block>,
        size: usize,
    ) -> bool {
        // Checked before the replace-path removal so a refusal leaves
        // `used` untouched and the caller's byte accounting exact.
        if size > self.capacity {
            return false;
        }
        let guard = epoch::pin();
        self.take_existing(ring, &key, &guard);
        while ring.used + size > self.capacity {
            if !self.evict_one(ring, &guard) {
                break;
            }
        }
        self.store(ring, key, Arc::clone(block), size, &guard);
        true
    }

    /// Publish one entry into the ring and the map together.
    fn store(
        &self,
        ring: &mut ClockRing,
        key: CacheKey,
        block: Arc<Block>,
        size: usize,
        guard: &Guard,
    ) {
        let slot = ring.alloc_slot();
        let entry = Arc::new(ClockEntry {
            block,
            key,
            charge: size,
            slot,
            // A fresh entry starts unreferenced. Inserting with the bit
            // already set is what classic VM CLOCK does and it costs
            // 0.9 to 2.0 points of hit rate against LRU on every trace
            // replayed for this change; starting it clear gains 0.5 to
            // 1.4 points instead.
            referenced: AtomicBool::new(false),
        });
        ring.slots[slot] = Some(Arc::clone(&entry));
        ring.used += size;
        self.map().insert(key, entry, guard).release(guard);
    }

    /// Replace the whole shard with one entry that is larger than the
    /// shard's own share of the budget. Only reached once the caller has
    /// reserved `size` against the cache-wide budget.
    fn replace_all_with(
        &self,
        ring: &mut ClockRing,
        key: CacheKey,
        block: Arc<Block>,
        size: usize,
    ) {
        let mut guard = epoch::pin();
        self.clear(ring, &mut guard);
        self.store(ring, key, block, size, &guard);
    }

    /// Drop every entry belonging to `file_id`.
    ///
    /// `CacheKey` orders on `(file_id, offset)`, so the file's blocks are
    /// one contiguous run: this costs the run plus a search rather than
    /// a walk of the whole shard, and allocates nothing. The successor
    /// is taken before each removal because a removed entry is unlinked
    /// from the list it was reached through.
    fn evict_file(&self, ring: &mut ClockRing, file_id: u64) {
        let Some(map) = self.map.get() else {
            return;
        };
        let guard = epoch::pin();
        let first = CacheKey { file_id, offset: 0 };
        let mut cursor = map.lower_bound(Bound::Included(&first), &guard);
        while let Some(found) = cursor {
            if found.key().file_id != file_id {
                break;
            }
            cursor = found.next();
            let entry = found.value();
            ring.release(entry.slot, entry.charge);
            found.remove();
        }
    }

    fn clear(&self, ring: &mut ClockRing, guard: &mut Guard) {
        if let Some(map) = self.map.get() {
            map.clear(guard);
        }
        ring.reset();
    }
}

/// Sharded CLOCK block cache for decompressed SSTable data blocks.
///
/// The cache is split into `2^shard_bits` independent shards keyed
/// by `xxh3(file_id, offset)`. Each shard holds a lock-free ordered map
/// of its entries plus a mutex-guarded CLOCK ring and byte counter.
///
/// # Reads take no lock
///
/// A hit is an epoch-protected map traversal followed by one relaxed
/// atomic store of the entry's reference bit, so readers never block
/// each other and never wait behind an insert that is freeing evicted
/// blocks. That is what CLOCK buys over a true LRU, whose `get` has to
/// reorder a recency list and therefore needs `&mut`.
///
/// CLOCK is an approximation of LRU and the hit rate differs: the
/// reference bit ranks entries into "touched since the hand last
/// passed" or not, where LRU ranks them exactly. On the traces replayed
/// for this change (zipfian point reads, zipfian plus a compaction
/// sweep, and an LSM level-shaped mix, at four budgets each) it lands
/// 0.46 to 1.25 points above LRU on every one of them, because an
/// unre-read block admitted by a scan is dropped a revolution later
/// instead of being promoted to the head of the list. It does not fix
/// the cyclic-sweep pathology: on a working set 1.5x the budget both
/// policies score zero.
///
/// # Capacity
///
/// `Options::block_cache_size` is the total byte budget. It is a hard
/// bound on [`BlockCache::usage`], the figure the cache accounts and
/// the `lark.block-cache-usage` property publishes: that total never
/// exceeds the budget, whatever the shard count, block size, or value
/// size, and every shard enforces its share exactly under its own ring
/// mutex. It is a close bound, not a hard one, on resident memory. See
/// the allocation section below for the measured gap and where it comes
/// from. The budget is split evenly across shards; each shard runs its
/// own hand as inserts would push it over its share.
///
/// An entry larger than one shard's share is handled by
/// [`Options::strict_capacity_limit`]:
///
/// * `false` (default): the per-shard split is a soft target. The
///   shard is emptied and the entry admitted, but only once the
///   entry has been reserved against the cache-wide budget, so no
///   number of shards can add up past `block_cache_size`. An entry
///   larger than the whole budget is never cached.
/// * `true`: the shard refuses the insert and leaves the caller to
///   use the block directly; nothing is cached.
///
/// The cache-wide reservation is checked against the published total,
/// which can lag inserts still in flight on other threads; the
/// per-shard budget is always exact because it is enforced under the
/// shard's own ring mutex.
///
/// A budget of 0 disables the cache: no shard is allocated, every
/// `get` misses, every `insert` is dropped, and the block-cache
/// tickers stay at zero.
///
/// # Allocation
///
/// Everything the cache allocates is driven by the byte budget, never
/// by the shard count: a shard's map and ring start empty and grow with
/// the working set, and each entry is charged [`Block::charge`] plus
/// [`ENTRY_OVERHEAD`] for the skip-list node, the ring slot, and the
/// `Arc` headers that `Block::charge` cannot see. An empty shard costs
/// 96 bytes, measured with a counting global allocator at 1, 4, 16, 64
/// and 128 shards, so the empty-cache footprint is flat in the shard
/// count rather than proportional to it.
///
/// Two costs `usage()` does not cover, both measured, neither growing
/// without bound:
///
/// * [`ENTRY_OVERHEAD`] under-charges the real bookkeeping by about 8%
///   (138.6 bytes measured against 128 charged).
/// * The map reclaims an evicted entry's node through crossbeam's epoch
///   collector, so a block's bytes return to the allocator a while
///   after `usage()` stops counting them, and an `evict_file` walk
///   holds one guard across the whole range so nothing it retires can
///   be reclaimed until that guard drops.
///
/// Together those put live heap a few percent above the budget at
/// saturation: 3.9% measured with 1 KiB blocks, 2.5% with 256-byte
/// blocks. The overshoot is flat in the number of entries churned
/// through the cache (1x, 3x, 10x and 30x the budget all measure the
/// same), so it is a constant margin rather than a leak.
///
/// The shards share the process-wide epoch collector with the
/// memtable's skip list rather than owning a private one. Two
/// consequences: reclamation latency for either subsystem depends on
/// the other's pin windows, and one pin in every 128 on any thread
/// tries to advance the global epoch and collect, which puts a little
/// shared-cache-line work on the read path. Every pin taken here is
/// scoped to a single map operation, so no guard is ever held across a
/// wait.
pub(crate) struct BlockCache {
    shards: Box<[CacheShard]>,
    /// Total capacity across all shards, in bytes. Kept
    /// separately so `usage_and_capacity` can answer quickly
    /// without summing per-shard.
    capacity: usize,
    /// Shard mask = `num_shards - 1`. `num_shards` is always a
    /// power of two so `hash & mask` picks the shard.
    shard_mask: u64,
    /// Number of shards (always `shard_mask + 1`). Only referenced
    /// by tests that want to confirm the configured shard count;
    /// production paths go through `shard_mask` directly.
    #[cfg(test)]
    num_shards: usize,
    /// Approximate total bytes currently held across all shards.
    /// Updated under each shard's ring mutex via atomic ops so
    /// `usage()` can be called without taking any lock.
    total_used: AtomicUsize,
    /// Whether strict capacity is enforced. See struct doc.
    strict: bool,
    /// Optional statistics sink. When set, every `get` and
    /// `insert` call increments the corresponding tickers.
    stats: Option<Arc<Statistics>>,
}

impl BlockCache {
    /// Create a new block cache with the given capacity in bytes
    /// and default sharding and strictness.
    #[cfg(test)]
    pub(crate) fn new(capacity_bytes: usize) -> Self {
        Self::with_config(capacity_bytes, 6, false)
    }

    /// Create a new block cache with an explicit byte budget,
    /// shard-bits, and strictness configuration. `shard_bits` is
    /// clamped to `[0, MAX_SHARD_BITS]`.
    ///
    /// A `capacity_bytes` of 0 builds a disabled cache: no shard is
    /// allocated, nothing is stored, and `get` always misses.
    pub(crate) fn with_config(
        capacity_bytes: usize,
        shard_bits: u32,
        strict_capacity_limit: bool,
    ) -> Self {
        if capacity_bytes == 0 {
            return Self {
                shards: Vec::new().into_boxed_slice(),
                capacity: 0,
                shard_mask: 0,
                #[cfg(test)]
                num_shards: 0,
                total_used: AtomicUsize::new(0),
                strict: strict_capacity_limit,
                stats: None,
            };
        }
        let shard_bits = shard_bits.min(MAX_SHARD_BITS);
        let mut num_shards: usize = 1usize << shard_bits;
        // Fall back to fewer shards if splitting would leave
        // every shard below the minimum useful capacity.
        while num_shards > 1 && capacity_bytes / num_shards < MIN_SHARD_CAPACITY {
            num_shards /= 2;
        }
        let per_shard = capacity_bytes / num_shards;
        let shards: Box<[CacheShard]> = (0..num_shards)
            .map(|_| CacheShard::new(per_shard))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self {
            shards,
            capacity: per_shard * num_shards,
            shard_mask: (num_shards - 1) as u64,
            #[cfg(test)]
            num_shards,
            total_used: AtomicUsize::new(0),
            strict: strict_capacity_limit,
            stats: None,
        }
    }

    /// Attach an optional statistics sink. Called once at engine
    /// open after the cache has been constructed; subsequent
    /// `get` / `insert` calls will update the provided tickers.
    pub(crate) fn with_stats(mut self, stats: Option<Arc<Statistics>>) -> Self {
        self.stats = stats;
        self
    }

    /// Hash a cache key down to a shard index.
    fn shard_index(&self, key: &CacheKey) -> usize {
        let mut buf = [0u8; 16];
        buf[..8].copy_from_slice(&key.file_id.to_le_bytes());
        buf[8..].copy_from_slice(&key.offset.to_le_bytes());
        (xxh3_64(&buf) & self.shard_mask) as usize
    }

    /// Try to get a block from the cache. A disabled cache
    /// (`block_cache_size` of 0) always misses and records nothing:
    /// there was no cache lookup to count.
    pub(crate) fn get(&self, file_id: u64, offset: u64) -> Option<Arc<Block>> {
        if self.shards.is_empty() {
            return None;
        }
        let key = CacheKey { file_id, offset };
        let idx = self.shard_index(&key);
        let hit = self.shards[idx].get(&key);
        if let Some(s) = self.stats.as_deref() {
            if hit.is_some() {
                s.add(Ticker::BlockCacheHit, 1);
            } else {
                s.add(Ticker::BlockCacheMiss, 1);
            }
        }
        crate::perf_context::record_block_cache_lookup(hit.is_some());
        hit
    }

    /// Insert a block into the cache. The block may be evicted
    /// before it is next read, especially under memory pressure.
    /// The function signature deliberately takes ownership of the
    /// `Arc`: the caller's clone is the one they continue to
    /// use, and the cache's copy is managed internally. A disabled
    /// cache (`block_cache_size` of 0) drops the block.
    pub(crate) fn insert(&self, file_id: u64, offset: u64, block: Arc<Block>) {
        if self.shards.is_empty() {
            return;
        }
        let key = CacheKey { file_id, offset };
        let size = entry_charge(&block);
        let idx = self.shard_index(&key);
        let stored = {
            let shard = &self.shards[idx];
            let mut ring = shard.ring.lock();
            let before = ring.used;
            if shard.insert_within_budget(&mut ring, key, &block, size) {
                self.publish(before, ring.used);
                true
            } else if self.strict || size > self.capacity {
                // Too big for one shard, and either strict mode or too
                // big for the whole cache. `insert_within_budget`
                // refuses before it touches the ring, so `used` is
                // still `before` and there is no delta to publish.
                false
            } else {
                // Non-strict oversized. Reserve the entry against the
                // cache-wide budget before touching the shard, so the
                // total cannot creep up with the shard count the way an
                // unchecked per-shard overshoot would. The winning
                // exchange is itself the publish: it writes the total
                // this shard will hold once `replace_all_with` has
                // dropped `before` bytes and stored `size`.
                loop {
                    let current = self.total_used.load(Ordering::Acquire);
                    let after = current.saturating_sub(before).saturating_add(size);
                    if after > self.capacity {
                        return;
                    }
                    if self
                        .total_used
                        .compare_exchange_weak(current, after, Ordering::AcqRel, Ordering::Acquire)
                        .is_ok()
                    {
                        break;
                    }
                }
                shard.replace_all_with(&mut ring, key, block, size);
                true
            }
        };
        // Counted only when the block was actually cached: the ticker
        // documents itself as one per miss that populated the cache, and
        // a refusal populates nothing.
        if stored {
            if let Some(s) = self.stats.as_deref() {
                s.add(Ticker::BlockCacheAdd, 1);
            }
        }
    }

    /// Publish a shard's byte delta to the lock-free running total.
    /// Adds and subtracts commute, so deltas from different shards can
    /// land in any order without drifting.
    fn publish(&self, before: usize, after: usize) {
        if after >= before {
            self.total_used.fetch_add(after - before, Ordering::Relaxed);
        } else {
            self.total_used.fetch_sub(before - after, Ordering::Relaxed);
        }
    }

    /// Record a "useful" bloom-filter hit - the filter correctly
    /// returned "not present" and spared a block read. Called
    /// from SSTable reader paths that have already consulted the
    /// cache and know they're about to short-circuit the lookup.
    pub(crate) fn record_bloom_useful(&self) {
        if let Some(s) = self.stats.as_deref() {
            s.add(Ticker::BloomFilterUseful, 1);
        }
        crate::perf_context::record_bloom_check(true);
    }

    /// Record a "full positive" bloom-filter hit - the filter
    /// said "maybe", the reader went to the block, and the key
    /// was actually present.
    pub(crate) fn record_bloom_full_positive(&self) {
        if let Some(s) = self.stats.as_deref() {
            s.add(Ticker::BloomFilterFullPositive, 1);
        }
        crate::perf_context::record_bloom_check(false);
    }

    /// Evict all blocks belonging to a specific file.
    pub(crate) fn evict_file(&self, file_id: u64) {
        for shard in self.shards.iter() {
            let (before, after) = {
                let mut ring = shard.ring.lock();
                let before = ring.used;
                shard.evict_file(&mut ring, file_id);
                (before, ring.used)
            };
            if before > after {
                self.total_used.fetch_sub(before - after, Ordering::Relaxed);
            }
        }
    }

    /// Clear the entire cache.
    ///
    /// Each shard publishes exactly the bytes it dropped, under its own
    /// lock. Storing a flat zero into the running total instead would
    /// race an `insert` whose delta has not landed yet and leave
    /// `usage()` permanently under-reporting what the shards hold.
    pub(crate) fn clear(&self) {
        for shard in self.shards.iter() {
            let freed = {
                let mut ring = shard.ring.lock();
                let freed = ring.used;
                let mut guard = epoch::pin();
                shard.clear(&mut ring, &mut guard);
                freed
            };
            if freed > 0 {
                self.total_used.fetch_sub(freed, Ordering::Relaxed);
            }
        }
    }

    /// Total bytes currently held across every shard, counting each
    /// entry's [`Block::charge`] plus [`ENTRY_OVERHEAD`]. Used by the
    /// `lark.block-cache-usage` property and by unit tests to verify
    /// eviction. Lock-free, so it can lag an insert in flight on
    /// another thread by that insert's charge.
    pub(crate) fn usage(&self) -> usize {
        self.total_used.load(Ordering::Relaxed)
    }

    /// Total byte capacity: the sum of every shard's budget.
    /// This may be slightly smaller than the
    /// `Options::block_cache_size` the user requested because the
    /// total is rounded down to an integer multiple of the shard
    /// count.
    pub(crate) fn capacity(&self) -> usize {
        self.capacity
    }

    /// Number of shards in this cache. Exposed for tests that
    /// want to verify multi-shard distribution.
    #[cfg(test)]
    pub(crate) fn num_shards(&self) -> usize {
        self.num_shards
    }

    /// Number of shards currently holding at least one entry.
    /// Used by tests to confirm sharding actually distributes
    /// inserts across the shard array.
    #[cfg(test)]
    pub(crate) fn populated_shards(&self) -> usize {
        self.shards
            .iter()
            .filter(|s| s.ring.lock().used > 0)
            .count()
    }

    /// Bytes actually held, recomputed from the shards under their
    /// own locks. The ground truth `usage()`'s lock-free atomic is
    /// supposed to track.
    #[cfg(test)]
    pub(crate) fn true_usage(&self) -> usize {
        self.shards.iter().map(|s| s.ring.lock().used).sum()
    }

    /// Entries currently held across every shard.
    #[cfg(test)]
    pub(crate) fn entry_count(&self) -> usize {
        self.shards
            .iter()
            .map(|s| s.map.get().map_or(0, |m| m.len()))
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::block::{BlockBuilder, RESTART_INTERVAL};

    fn dummy_block(size: usize) -> Arc<Block> {
        let mut builder = BlockBuilder::new(RESTART_INTERVAL);
        let value = vec![0u8; size];
        builder.add(b"k", &value);
        Arc::new(Block::decode(builder.finish()).expect("decode"))
    }

    #[test]
    fn single_insert_then_get() {
        let cache = BlockCache::new(1024 * 1024);
        let blk = dummy_block(256);
        cache.insert(1, 0, blk.clone());
        assert!(cache.get(1, 0).is_some());
        assert!(cache.usage() >= 256);
    }

    #[test]
    fn eviction_bounds_total_usage() {
        // 4 KB capacity, 1 shard (MIN_SHARD_CAPACITY fallback
        // collapses to 1 since 4 KB / 64 = 64 bytes per shard).
        let cache = BlockCache::with_config(4 * 1024, 6, false);
        // Insert many 1 KB blocks. Only a handful should survive.
        for i in 0..32u64 {
            cache.insert(1, i * 100, dummy_block(1024));
        }
        let usage = cache.usage();
        assert!(
            usage <= cache.capacity(),
            "usage {usage} exceeded capacity {}",
            cache.capacity()
        );
        // Oldest entry should have been evicted.
        assert!(cache.get(1, 0).is_none());
    }

    #[test]
    fn strict_capacity_rejects_oversized_entry() {
        let cache = BlockCache::with_config(64 * 1024, 0, true);
        // Single shard, 64 KB capacity. A 128 KB block won't fit.
        let big = dummy_block(128 * 1024);
        cache.insert(1, 0, big);
        assert!(
            cache.get(1, 0).is_none(),
            "strict cache must reject oversized entries"
        );
        assert_eq!(cache.usage(), 0);
    }

    #[test]
    fn non_strict_cache_admits_an_entry_bigger_than_one_shard() {
        // 8 shards of 64 KiB. A 128 KiB block does not fit its own
        // shard but fits the 512 KiB cache-wide budget, so the
        // non-strict cache empties the shard and takes it.
        let cache = BlockCache::with_config(512 * 1024, 3, false);
        assert_eq!(cache.num_shards(), 8);
        cache.insert(1, 0, dummy_block(128 * 1024));
        assert!(
            cache.get(1, 0).is_some(),
            "non-strict cache should admit an entry larger than one shard"
        );
        assert!(cache.usage() <= cache.capacity());
    }

    #[test]
    fn non_strict_cache_refuses_an_entry_bigger_than_the_whole_budget() {
        let cache = BlockCache::with_config(64 * 1024, 0, false);
        cache.insert(1, 0, dummy_block(128 * 1024));
        assert!(
            cache.get(1, 0).is_none(),
            "a block larger than the entire budget must not be cached"
        );
        assert_eq!(cache.usage(), 0);
    }

    #[test]
    fn oversized_admissions_stay_inside_the_budget_at_every_shard_count() {
        // One oversized entry per shard used to be admitted with no
        // cache-wide check, so resident bytes scaled with the shard
        // count instead of the budget.
        let budget = 256 * 64 * 1024;
        let mut usages = Vec::new();
        for bits in [0u32, 4, 8] {
            let cache = BlockCache::with_config(budget, bits, false);
            for file_id in 0..4096u64 {
                cache.insert(file_id, 0, dummy_block(256 * 1024));
            }
            assert!(
                cache.usage() <= cache.capacity(),
                "shard_bits {bits}: usage {} over capacity {}",
                cache.usage(),
                cache.capacity()
            );
            usages.push(cache.usage());
        }
        assert_eq!(
            usages[0], usages[2],
            "resident bytes still track the shard count"
        );
    }

    #[test]
    fn sharding_distributes_inserts_across_shards() {
        // 64 MB so we get the full 64-shard default. Insert 1024
        // entries across many different (file_id, offset) pairs
        // and verify more than one shard ends up populated.
        let cache = BlockCache::with_config(64 * 1024 * 1024, 6, false);
        for i in 0..1024u64 {
            cache.insert(i, i * 4096, dummy_block(1024));
        }
        let populated = cache.populated_shards();
        assert_eq!(cache.num_shards(), 64);
        assert!(
            populated > 16,
            "expected inserts to fan out across shards, populated = {populated}"
        );
    }

    #[test]
    fn evict_file_removes_only_that_files_blocks() {
        let cache = BlockCache::with_config(64 * 1024 * 1024, 6, false);
        for off in 0..16u64 {
            cache.insert(1, off * 4096, dummy_block(1024));
            cache.insert(2, off * 4096, dummy_block(1024));
        }
        cache.evict_file(1);
        // File 1 is gone.
        for off in 0..16u64 {
            assert!(cache.get(1, off * 4096).is_none());
            assert!(cache.get(2, off * 4096).is_some());
        }
    }

    #[test]
    fn clear_zeroes_usage() {
        let cache = BlockCache::with_config(64 * 1024 * 1024, 6, false);
        for i in 0..64u64 {
            cache.insert(1, i * 4096, dummy_block(1024));
        }
        assert!(cache.usage() > 0);
        cache.clear();
        assert_eq!(cache.usage(), 0);
        assert!(cache.get(1, 0).is_none());
    }

    #[test]
    fn repeated_insert_at_same_key_does_not_double_count() {
        let cache = BlockCache::with_config(64 * 1024, 0, false);
        cache.insert(1, 0, dummy_block(1024));
        let first_usage = cache.usage();
        cache.insert(1, 0, dummy_block(1024));
        cache.insert(1, 0, dummy_block(1024));
        // Re-inserting the same key replaces rather than accumulating.
        let final_usage = cache.usage();
        assert_eq!(first_usage, final_usage);
    }

    #[test]
    fn miss_on_absent_key_returns_none() {
        let cache = BlockCache::with_config(64 * 1024, 0, false);
        assert!(cache.get(99, 999).is_none());
    }

    #[test]
    fn capacity_reflects_rounded_budget() {
        // 100 KB / 64 shards would drop below MIN_SHARD_CAPACITY, so
        // the constructor collapses to fewer shards. Capacity is the
        // actual rounded budget after collapse, not the request.
        let cache = BlockCache::with_config(100_000, 6, false);
        assert!(cache.capacity() <= 100_000);
        assert!(cache.capacity() > 0);
    }

    #[test]
    fn resident_bytes_track_the_byte_budget_not_the_shard_count() {
        // The defect this guards: every shard used to preallocate a
        // fixed 1,000,000-entry map, so the cache's own footprint
        // scaled with the shard count and ignored the byte budget.
        // Nothing is allocated up front now, and the budget is the
        // only bound at any shard count.
        let budget = 8 * 1024 * 1024;
        let mut usages = Vec::new();
        for bits in [0u32, 2, 4, 6] {
            let cache = BlockCache::with_config(budget, bits, false);
            assert_eq!(cache.usage(), 0, "a fresh cache holds nothing");
            for i in 0..8192u64 {
                cache.insert(1, i * 4096, dummy_block(4096));
            }
            assert!(
                cache.usage() <= cache.capacity(),
                "shard_bits {bits}: usage {} over capacity {}",
                cache.usage(),
                cache.capacity()
            );
            usages.push(cache.usage());
        }
        // Every configuration converges on the same budget, within one
        // entry per shard of rounding.
        let spread =
            usages.iter().max().copied().unwrap_or(0) - usages.iter().min().copied().unwrap_or(0);
        assert!(
            spread <= budget / 16,
            "resident bytes moved with shard_bits: {usages:?}"
        );
    }

    #[test]
    fn per_entry_overhead_is_charged_against_the_budget() {
        // A budget filled with tiny blocks is bounded by the entry
        // overhead, not just by payload bytes: without charging it, a
        // 1 MiB budget would hold millions of 64-byte blocks.
        let cache = BlockCache::with_config(1024 * 1024, 0, false);
        for i in 0..100_000u64 {
            cache.insert(1, i * 64, dummy_block(0));
        }
        assert!(cache.usage() <= cache.capacity());
        assert!(
            cache.entry_count() <= cache.capacity() / ENTRY_OVERHEAD,
            "held {} entries against a {}-byte budget",
            cache.entry_count(),
            cache.capacity()
        );
    }

    #[test]
    fn a_working_set_that_fits_the_budget_is_kept_whole() {
        // The regression this guards: an entry-count cap derived from
        // the configured `block_size` evicted entries that fit inside
        // the byte budget, silently shrinking the cache.
        let cache = BlockCache::with_config(8 * 1024 * 1024, 0, false);
        let mut offered = 0usize;
        for i in 0..3500u64 {
            let blk = dummy_block(1024);
            offered += entry_charge(&blk);
            cache.insert(1, i * 4096, blk);
        }
        assert!(
            offered <= cache.capacity(),
            "test setup: the working set must fit the byte budget"
        );
        assert_eq!(
            cache.entry_count(),
            3500,
            "the cache evicted entries that fit inside its byte budget"
        );
        assert_eq!(cache.usage(), offered);
    }

    #[test]
    fn zero_budget_disables_the_cache() {
        let cache = BlockCache::with_config(0, 6, false);
        assert_eq!(cache.num_shards(), 0);
        assert_eq!(cache.capacity(), 0);
        cache.insert(1, 0, dummy_block(256));
        assert!(cache.get(1, 0).is_none());
        assert_eq!(cache.usage(), 0);
        cache.evict_file(1);
        cache.clear();
        assert_eq!(cache.usage(), 0);
    }

    #[test]
    fn zero_budget_strict_cache_is_also_disabled() {
        let cache = BlockCache::with_config(0, 0, true);
        cache.insert(1, 0, dummy_block(256));
        assert!(cache.get(1, 0).is_none());
        assert_eq!(cache.usage(), 0);
    }

    #[test]
    fn tiny_budget_still_admits_a_block_that_fits() {
        let cache = BlockCache::with_config(4096, 6, false);
        cache.insert(1, 0, dummy_block(128));
        assert!(cache.get(1, 0).is_some());
        assert!(cache.usage() <= cache.capacity());
    }

    /// Byte accounting is exact: `usage()` is the sum of every live
    /// entry's charge, which backs the `lark.block-cache-usage`
    /// property.
    #[test]
    fn byte_accounting_is_exact() {
        let cache = BlockCache::with_config(64 * 1024 * 1024, 0, false);
        let mut expected = 0usize;
        for i in 0..64u64 {
            let blk = dummy_block(512);
            expected += entry_charge(&blk);
            cache.insert(1, i * 4096, blk);
        }
        assert_eq!(cache.usage(), expected);
    }

    /// `clear()` used to store a flat zero into the running total
    /// outside the shard locks, so a concurrent `insert` could add its
    /// delta afterwards and leave `usage()` reporting bytes the cache
    /// does not hold, permanently.
    #[test]
    fn usage_does_not_drift_when_clear_races_insert() {
        use std::sync::atomic::AtomicBool;
        for _ in 0..50 {
            let cache = Arc::new(BlockCache::with_config(64 * 1024 * 1024, 6, false));
            let stop = Arc::new(AtomicBool::new(false));
            let writer = {
                let cache = Arc::clone(&cache);
                let stop = Arc::clone(&stop);
                std::thread::spawn(move || {
                    let mut i = 0u64;
                    while !stop.load(Ordering::Relaxed) {
                        cache.insert(i % 97, i * 4096, dummy_block(256));
                        i += 1;
                    }
                })
            };
            for _ in 0..300 {
                cache.clear();
            }
            stop.store(true, Ordering::Relaxed);
            writer.join().expect("writer");
            assert_eq!(
                cache.usage(),
                cache.true_usage(),
                "usage() drifted away from the real byte total"
            );
        }
    }

    /// Concurrent readers and writers racing eviction: the byte budget
    /// holds under contention.
    #[test]
    fn concurrent_inserts_respect_the_budget() {
        let cache = Arc::new(BlockCache::with_config(1024 * 1024, 2, false));
        let mut handles = Vec::new();
        for t in 0..8u64 {
            let cache = Arc::clone(&cache);
            handles.push(std::thread::spawn(move || {
                for i in 0..4000u64 {
                    cache.insert(t, i * 64, dummy_block(64));
                    let _ = cache.get(t, (i / 2) * 64);
                }
            }));
        }
        for h in handles {
            h.join().expect("worker");
        }
        assert!(
            cache.true_usage() <= cache.capacity(),
            "usage {} over capacity {}",
            cache.true_usage(),
            cache.capacity()
        );
    }

    #[test]
    fn evict_file_does_not_touch_other_files() {
        let cache = BlockCache::with_config(64 * 1024 * 1024, 6, false);
        cache.insert(7, 0, dummy_block(1024));
        cache.insert(8, 0, dummy_block(1024));
        let before = cache.usage();
        cache.evict_file(99); // a file id that was never inserted
        assert_eq!(cache.usage(), before);
        assert!(cache.get(7, 0).is_some());
        assert!(cache.get(8, 0).is_some());
    }
}

#[cfg(test)]
#[path = "block_cache_adversarial.rs"]
mod adversarial;
