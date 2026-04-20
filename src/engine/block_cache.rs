use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;
use xxhash_rust::xxh3::xxh3_64;

use super::block::Block;
use crate::statistics::{Statistics, Ticker};

/// Cache key: (file_id, block_offset).
#[derive(Hash, Eq, PartialEq, Clone, Copy)]
struct CacheKey {
    file_id: u64,
    offset: u64,
}

/// Hard upper bound on the number of shards the cache will ever
/// create. A 32-bit shard-bit config of 8 → 256 shards is plenty
/// for a single-process embedded store.
const MAX_SHARD_BITS: u32 = 8;

/// Minimum per-shard capacity. Tiny caches with many shards
/// would otherwise produce shards with 0 bytes of capacity, which
/// is almost certainly a misconfiguration — fall back to a single
/// shard in that case.
const MIN_SHARD_CAPACITY: usize = 64 * 1024;

/// Per-shard state.
struct CacheShard {
    /// LRU ordering plus the block payload. The LRU itself is
    /// bounded by a huge entry count so in practice the byte
    /// budget below is the real cap.
    lru: lru::LruCache<CacheKey, Arc<Block>>,
    /// Byte budget for this shard — capacity / num_shards.
    capacity: usize,
    /// Bytes currently held by entries in `lru`. Kept as a plain
    /// field (not atomic) because every mutation is done under
    /// the shard mutex anyway.
    used: usize,
}

impl CacheShard {
    fn new(capacity: usize) -> Self {
        Self {
            // 1M is larger than any realistic per-shard entry
            // count; the byte accounting is what actually bounds
            // the cache.
            lru: lru::LruCache::new(NonZeroUsize::new(1_000_000).unwrap()),
            capacity,
            used: 0,
        }
    }

    /// Try to insert `(key, block)` into this shard, evicting LRU
    /// entries as needed. Returns `true` if the insert succeeded,
    /// `false` if the single entry is larger than the shard
    /// capacity — in that case the caller sees the raw `Arc` and
    /// `strict_capacity_limit` controls whether we cache anyway.
    fn insert(&mut self, key: CacheKey, block: Arc<Block>, strict: bool) -> bool {
        let size = block.charge();
        if size == 0 {
            return false;
        }

        // A block larger than the whole shard can only be cached
        // when strict_capacity_limit is off — in strict mode we
        // refuse rather than blow past the budget.
        if size > self.capacity {
            if strict {
                return false;
            }
            // Non-strict: evict everything, accept the oversized
            // entry, live with the overshoot.
            self.lru.clear();
            self.used = 0;
            self.used += size;
            self.lru.put(key, block);
            return true;
        }

        // Replace-path: drop any existing entry at this key so we
        // don't double-count.
        if let Some(old) = self.lru.pop(&key) {
            self.used = self.used.saturating_sub(old.charge());
        }

        // Evict LRU entries until the new one fits.
        while self.used + size > self.capacity {
            match self.lru.pop_lru() {
                Some((_, evicted)) => {
                    self.used = self.used.saturating_sub(evicted.charge());
                }
                None => break,
            }
        }

        self.used += size;
        self.lru.put(key, block);
        true
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
                self.used = self.used.saturating_sub(block.charge());
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
/// `Options::block_cache_size` is the total byte budget. It is
/// split evenly across shards; each shard evicts its own LRU tail
/// as inserts would push it over its share. If a single entry is
/// larger than one shard's capacity, behavior depends on
/// [`Options::strict_capacity_limit`]:
///
/// * `false` (default): the shard evicts everything and admits
///   the oversized entry, accepting a one-entry overshoot.
/// * `true`: the shard refuses the insert and leaves the caller
///   to use the block directly; nothing is cached.
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
    /// and default sharding / strictness.
    #[cfg(test)]
    pub(crate) fn new(capacity_bytes: usize) -> Self {
        Self::with_config(capacity_bytes, 6, false)
    }

    /// Create a new block cache with an explicit shard-bits and
    /// strictness configuration. `shard_bits` is clamped to
    /// `[0, MAX_SHARD_BITS]`.
    pub(crate) fn with_config(
        capacity_bytes: usize,
        shard_bits: u32,
        strict_capacity_limit: bool,
    ) -> Self {
        let shard_bits = shard_bits.min(MAX_SHARD_BITS);
        let mut num_shards: usize = 1usize << shard_bits;
        // Fall back to a single shard if splitting would leave
        // every shard below the minimum useful capacity.
        while num_shards > 1 && capacity_bytes / num_shards < MIN_SHARD_CAPACITY {
            num_shards /= 2;
        }
        let per_shard = capacity_bytes / num_shards.max(1);
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

    /// Try to get a block from the cache.
    pub(crate) fn get(&self, file_id: u64, offset: u64) -> Option<Arc<Block>> {
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
    /// `Arc` — the caller's clone is the one they continue to
    /// use, and the cache's copy is managed internally.
    pub(crate) fn insert(&self, file_id: u64, offset: u64, block: Arc<Block>) {
        let size = block.charge();
        let key = CacheKey { file_id, offset };
        let idx = self.shard_index(&key);
        let (before_used, after_used) = {
            let mut shard = self.shards[idx].lock();
            let before = shard.used;
            shard.insert(key, block, self.strict);
            (before, shard.used)
        };
        // Keep the atomic total in sync with the per-shard delta.
        if after_used >= before_used {
            self.total_used
                .fetch_add(after_used - before_used, Ordering::Relaxed);
        } else {
            self.total_used
                .fetch_sub(before_used - after_used, Ordering::Relaxed);
        }
        let _ = size;
        if let Some(s) = self.stats.as_deref() {
            s.add(Ticker::BlockCacheAdd, 1);
        }
    }

    /// Record a "useful" bloom-filter hit — the filter correctly
    /// returned "not present" and spared a block read. Called
    /// from SSTable reader paths that have already consulted the
    /// cache and know they're about to short-circuit the lookup.
    pub(crate) fn record_bloom_useful(&self) {
        if let Some(s) = self.stats.as_deref() {
            s.add(Ticker::BloomFilterUseful, 1);
        }
        crate::perf_context::record_bloom_check(true);
    }

    /// Record a "full positive" bloom-filter hit — the filter
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
    pub(crate) fn clear(&self) {
        for shard in self.shards.iter() {
            shard.lock().clear();
        }
        self.total_used.store(0, Ordering::Relaxed);
    }

    /// Approximate total bytes currently held across every shard.
    /// Used by the `lark.block-cache-usage` property and by
    /// unit tests to verify eviction.
    pub(crate) fn usage(&self) -> usize {
        self.total_used.load(Ordering::Relaxed)
    }

    /// Total byte capacity — the sum of every shard's budget.
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
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_block(size: usize) -> Arc<Block> {
        // Minimal encoded block: a single restart at offset 0 +
        // the 4-byte restart count footer, padded with zeros to
        // reach approximately `size` bytes. The decoder only
        // checks trailing format, not content.
        let mut data = vec![0u8; size.max(8)];
        let restart_count: u32 = 1;
        let data_len = data.len();
        // restarts: single u32 at position data_len - 8
        data[data_len - 8..data_len - 4].copy_from_slice(&0u32.to_le_bytes());
        data[data_len - 4..].copy_from_slice(&restart_count.to_le_bytes());
        Arc::new(Block::decode(data).expect("decode"))
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
            usage <= cache.capacity() + 2048,
            "usage {usage} exceeded capacity {} by more than one block",
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
    fn non_strict_cache_admits_oversized_entry() {
        let cache = BlockCache::with_config(64 * 1024, 0, false);
        let big = dummy_block(128 * 1024);
        cache.insert(1, 0, big);
        assert!(
            cache.get(1, 0).is_some(),
            "non-strict cache should admit oversized entries"
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
