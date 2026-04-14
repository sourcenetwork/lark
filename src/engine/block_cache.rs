use std::num::NonZeroUsize;
use std::sync::Arc;

use parking_lot::Mutex;

use super::block::Block;

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
        }
    }

    /// Try to get a block from the cache.
    pub(crate) fn get(&self, file_id: u64, offset: u64) -> Option<Arc<Block>> {
        let key = CacheKey { file_id, offset };
        self.cache.lock().get(&key).cloned()
    }

    /// Insert a block into the cache.
    pub(crate) fn insert(&self, file_id: u64, offset: u64, block: Arc<Block>) {
        let key = CacheKey { file_id, offset };
        self.cache.lock().put(key, block);
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
