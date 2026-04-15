use std::num::NonZeroUsize;
use std::sync::Arc;

use parking_lot::Mutex;

use super::block::Block;
use crate::statistics::{Statistics, Ticker};

/// Cache key: (file_id, block_offset).
#[derive(Hash, Eq, PartialEq, Clone)]
struct CacheKey {
    file_id: u64,
    offset: u64,
}

/// LRU block cache for decompressed SSTable data blocks.
///
/// Stores `Arc<Block>` so concurrent readers can share cached blocks
/// without copying.
pub(crate) struct BlockCache {
    cache: Mutex<lru::LruCache<CacheKey, Arc<Block>>>,
    /// Optional statistics sink. When set, every `get` and
    /// `insert` call increments the corresponding tickers. The
    /// cache holds an `Arc` so the engine can replace it on
    /// `drop_all` without caring about the cache's lifetime.
    stats: Option<Arc<Statistics>>,
}

impl BlockCache {
    /// Create a new block cache with the given capacity in bytes.
    /// The actual number of entries is estimated from capacity / average_block_size.
    pub(crate) fn new(capacity_bytes: usize) -> Self {
        // Estimate: average block is 16KB, so capacity / 16KB entries
        let estimated_entries = std::cmp::max(capacity_bytes / (16 * 1024), 16);
        Self {
            cache: Mutex::new(lru::LruCache::new(
                NonZeroUsize::new(estimated_entries).unwrap(),
            )),
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

    /// Try to get a block from the cache.
    pub(crate) fn get(&self, file_id: u64, offset: u64) -> Option<Arc<Block>> {
        let key = CacheKey { file_id, offset };
        let hit = self.cache.lock().get(&key).cloned();
        if let Some(s) = self.stats.as_deref() {
            if hit.is_some() {
                s.add(Ticker::BlockCacheHit, 1);
            } else {
                s.add(Ticker::BlockCacheMiss, 1);
            }
        }
        hit
    }

    /// Insert a block into the cache.
    pub(crate) fn insert(&self, file_id: u64, offset: u64, block: Arc<Block>) {
        let key = CacheKey { file_id, offset };
        self.cache.lock().put(key, block);
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
    }

    /// Record a "full positive" bloom-filter hit — the filter
    /// said "maybe", the reader went to the block, and the key
    /// was actually present.
    pub(crate) fn record_bloom_full_positive(&self) {
        if let Some(s) = self.stats.as_deref() {
            s.add(Ticker::BloomFilterFullPositive, 1);
        }
    }

    /// Evict all blocks belonging to a specific file.
    pub(crate) fn evict_file(&self, file_id: u64) {
        let mut cache = self.cache.lock();
        let keys: Vec<CacheKey> = cache
            .iter()
            .filter(|(k, _): &(&CacheKey, &Arc<Block>)| k.file_id == file_id)
            .map(|(k, _): (&CacheKey, &Arc<Block>)| k.clone())
            .collect();
        for key in keys {
            cache.pop(&key);
        }
    }

    /// Clear the entire cache.
    pub(crate) fn clear(&self) {
        self.cache.lock().clear();
    }
}
