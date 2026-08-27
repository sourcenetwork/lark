use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use parking_lot::Mutex;
use xxhash_rust::xxh3::xxh3_64;

use super::block::Block;
use crate::options::MAX_BLOCK_CACHE_SHARD_BITS;
use crate::statistics::{Statistics, Ticker};

/// Cache key: (file_id, block_offset).
#[derive(Hash, Eq, PartialEq, Clone, Copy)]
struct CacheKey {
    file_id: u64,
    offset: u64,
}

// `deny.toml` ignores RUSTSEC-2026-0253 on the grounds that this key
// cannot have a panicking destructor. `Copy` and `Drop` are mutually
// exclusive in Rust, so requiring `Copy` here is the whole proof, and
// it is checked at compile time: give `CacheKey` a field with a
// destructor and this stops compiling rather than silently making the
// ignore a lie.
const _: fn() = || {
    fn assert_copy<T: Copy>() {}
    assert_copy::<CacheKey>();
};

/// Hard upper bound on the number of shards the cache will ever
/// create. A 32-bit shard-bit config of 8 → 256 shards is plenty
/// for a single-process embedded store.
const MAX_SHARD_BITS: u32 = MAX_BLOCK_CACHE_SHARD_BITS;

/// Minimum per-shard capacity. Tiny caches with many shards
/// would otherwise produce shards with 0 bytes of capacity, which
/// is almost certainly a misconfiguration: fall back to a single
/// shard in that case.
const MIN_SHARD_CAPACITY: usize = 64 * 1024;

/// Bookkeeping bytes an entry costs beyond [`Block::charge`]: the LRU
/// list node, the hash-table slot, the `Arc` reference counts, and
/// allocator rounding. Charged against the byte budget so the budget
/// alone bounds everything the cache holds, with no separate
/// entry-count cap that could bind invisibly below it.
const ENTRY_OVERHEAD: usize = 128;

/// Bytes one cached entry costs the cache.
fn entry_charge(block: &Block) -> usize {
    block.charge() + ENTRY_OVERHEAD
}

/// Per-shard state.
struct CacheShard {
    /// LRU ordering plus the block payload. Constructed unbounded so
    /// the map grows with the working set instead of preallocating
    /// for a full cache; `capacity` is the only bound and it is
    /// enforced in `insert`, so the LRU never evicts behind `used`'s
    /// back.
    lru: lru::LruCache<CacheKey, Arc<Block>>,
    /// Byte budget for this shard: total capacity / num_shards.
    capacity: usize,
    /// Bytes currently held by entries in `lru`, per
    /// [`entry_charge`]. Kept as a plain field (not atomic) because
    /// every mutation is done under the shard mutex anyway.
    used: usize,
}

impl CacheShard {
    fn new(capacity: usize) -> Self {
        Self {
            lru: lru::LruCache::unbounded(),
            capacity,
            used: 0,
        }
    }

    /// Drop any entry already stored at `key` so a re-insert replaces
    /// rather than double-counts.
    fn take_existing(&mut self, key: &CacheKey) {
        if let Some(old) = self.lru.pop(key) {
            self.used = self.used.saturating_sub(entry_charge(&old));
        }
    }

    /// Insert `(key, block)` within this shard's byte budget,
    /// evicting the LRU tail to make room. Returns `false` without
    /// storing anything when `size` exceeds the whole shard budget;
    /// the caller then decides whether the cache-wide budget can
    /// still absorb it.
    fn insert_within_budget(&mut self, key: CacheKey, block: &Arc<Block>, size: usize) -> bool {
        // Checked before the replace-path pop so a refusal leaves
        // `used` untouched and the caller's byte accounting exact.
        if size > self.capacity {
            return false;
        }
        self.take_existing(&key);
        while self.used + size > self.capacity {
            match self.lru.pop_lru() {
                Some((_, evicted)) => {
                    self.used = self.used.saturating_sub(entry_charge(&evicted));
                }
                None => break,
            }
        }
        self.used += size;
        self.lru.put(key, Arc::clone(block));
        true
    }

    /// Replace the whole shard with one entry that is larger than the
    /// shard's own share of the budget. Only reached once the caller
    /// has reserved `size` against the cache-wide budget.
    fn replace_all_with(&mut self, key: CacheKey, block: Arc<Block>, size: usize) {
        self.lru.clear();
        self.used = size;
        self.lru.put(key, block);
    }

    fn get(&mut self, key: &CacheKey) -> Option<Arc<Block>> {
        self.lru.get(key).cloned()
    }

    fn evict_file(&mut self, file_id: u64) {
        let keys: Vec<CacheKey> = self
            .lru
            .iter()
            .filter(|(k, _)| k.file_id == file_id)
            .map(|(k, _)| *k)
            .collect();
        for key in keys {
            if let Some(block) = self.lru.pop(&key) {
                self.used = self.used.saturating_sub(entry_charge(&block));
            }
        }
    }

    fn clear(&mut self) {
        self.lru.clear();
        self.used = 0;
    }
}

/// Sharded LRU block cache for decompressed SSTable data blocks.
///
/// The cache is split into `2^shard_bits` independent shards keyed
/// by `xxh3(file_id, offset)`. Each shard holds its own mutex, LRU
/// list, and byte counter, so concurrent readers contend only with
/// other readers that happen to hash to the same shard.
///
/// # Capacity
///
/// `Options::block_cache_size` is the total byte budget and it is a
/// hard bound: the cache never holds more than it, whatever the shard
/// count, block size, or value size. The budget is split evenly
/// across shards; each shard evicts its own LRU tail as inserts would
/// push it over its share.
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
/// shard's own mutex.
///
/// A budget of 0 disables the cache: no shard is allocated, every
/// `get` misses, every `insert` is dropped, and the block-cache
/// tickers stay at zero.
///
/// # Allocation
///
/// Everything the cache allocates is driven by the byte budget,
/// never by the shard count: the per-shard maps start empty and grow
/// with the working set, and each entry is charged
/// [`Block::charge`] plus [`ENTRY_OVERHEAD`], which covers the LRU
/// node, the hash slot, and the `Arc` header that `Block::charge`
/// cannot see. `usage()` reports that same total, so the number the
/// `lark.block-cache-usage` property publishes is what the cache
/// actually costs rather than payload bytes alone.
pub(crate) struct BlockCache {
    shards: Box<[Mutex<CacheShard>]>,
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
    /// Updated under each shard's mutex via atomic ops so
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
        let shards: Box<[Mutex<CacheShard>]> = (0..num_shards)
            .map(|_| Mutex::new(CacheShard::new(per_shard)))
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
        let hit = self.shards[idx].lock().get(&key);
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
        {
            let mut shard = self.shards[idx].lock();
            let before = shard.used;
            if shard.insert_within_budget(key, &block, size) {
                self.publish(before, shard.used);
            } else if self.strict || size > self.capacity {
                // Too big for one shard and either strict mode or too
                // big for the entire cache: nothing is stored, but the
                // replace-path pop may still have freed bytes.
                self.publish(before, shard.used);
            } else {
                // Non-strict oversized. Reserve the entry against the
                // cache-wide budget before touching the shard, so the
                // total cannot creep up with the shard count the way
                // an unchecked per-shard overshoot would.
                let freed = shard.used;
                loop {
                    let current = self.total_used.load(Ordering::Acquire);
                    let after = current.saturating_sub(freed).saturating_add(size);
                    if after > self.capacity {
                        self.publish(before, shard.used);
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
                shard.replace_all_with(key, block, size);
            }
        }
        if let Some(s) = self.stats.as_deref() {
            s.add(Ticker::BlockCacheAdd, 1);
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
                let mut shard = shard.lock();
                let before = shard.used;
                shard.evict_file(file_id);
                (before, shard.used)
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
                let mut shard = shard.lock();
                let freed = shard.used;
                shard.clear();
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
        self.shards.iter().filter(|s| s.lock().used > 0).count()
    }

    /// Bytes actually held, recomputed from the shards under their
    /// own locks. The ground truth `usage()`'s lock-free atomic is
    /// supposed to track.
    #[cfg(test)]
    pub(crate) fn true_usage(&self) -> usize {
        self.shards.iter().map(|s| s.lock().used).sum()
    }

    /// Entries currently held across every shard.
    #[cfg(test)]
    pub(crate) fn entry_count(&self) -> usize {
        self.shards.iter().map(|s| s.lock().lru.len()).sum()
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
