use std::sync::atomic::{AtomicUsize, Ordering};

use crossbeam_skiplist::SkipMap;

use super::internal_key::{
    decode_internal_key, encode_internal_key, lookup_key, VALUE_TYPE_DELETION, VALUE_TYPE_VALUE,
};

/// Concurrent in-memory sorted table backed by a lock-free skip list.
///
/// Supports multiple concurrent readers and a single writer (serialized
/// externally by the engine's write lock).
pub(crate) struct MemTable {
    data: SkipMap<Vec<u8>, Vec<u8>>,
    approximate_size: AtomicUsize,
}

impl MemTable {
    pub(crate) fn new() -> Self {
        Self {
            data: SkipMap::new(),
            approximate_size: AtomicUsize::new(0),
        }
    }

    /// Insert a key-value pair with the given sequence number.
    pub(crate) fn put(&self, key: &[u8], value: &[u8], seq: u64) {
        let internal_key = encode_internal_key(key, seq, VALUE_TYPE_VALUE);
        let size = internal_key.len() + value.len();
        self.data.insert(internal_key, value.to_vec());
        self.approximate_size.fetch_add(size, Ordering::Relaxed);
    }

    /// Insert a deletion tombstone for the given key.
    pub(crate) fn delete(&self, key: &[u8], seq: u64) {
        let internal_key = encode_internal_key(key, seq, VALUE_TYPE_DELETION);
        let size = internal_key.len();
        self.data.insert(internal_key, Vec::new());
        self.approximate_size.fetch_add(size, Ordering::Relaxed);
    }

    /// Look up a key visible at the given snapshot sequence number.
    ///
    /// Returns:
    /// - `Some(Some(value))` if found with a value
    /// - `Some(None)` if found as a tombstone (deleted)
    /// - `None` if not present in this memtable at this snapshot
    pub(crate) fn get(&self, key: &[u8], snapshot_seq: u64) -> Option<Option<Vec<u8>>> {
        let search_key = lookup_key(key, snapshot_seq);

        for entry in self.data.range(search_key..) {
            let (user_key, seq, value_type) = decode_internal_key(entry.key());

            if user_key != key {
                return None;
            }

            if seq <= snapshot_seq {
                return if value_type == VALUE_TYPE_DELETION {
                    Some(None)
                } else {
                    Some(Some(entry.value().clone()))
                };
            }
        }

        None
    }

    /// Iterate entries visible at `snapshot_seq`, deduplicated by user key.
    /// Returns `(user_key, Option<value>)` — `None` is a tombstone.
    pub(crate) fn iter(&self, snapshot_seq: u64) -> Vec<(Vec<u8>, Option<Vec<u8>>)> {
        let mut result = Vec::new();
        let mut last_user_key: Option<Vec<u8>> = None;

        for entry in self.data.iter() {
            let (user_key, seq, value_type) = decode_internal_key(entry.key());

            if seq > snapshot_seq {
                continue;
            }

            if let Some(ref last) = last_user_key {
                if last.as_slice() == user_key {
                    continue;
                }
            }

            last_user_key = Some(user_key.to_vec());

            if value_type == VALUE_TYPE_DELETION {
                result.push((user_key.to_vec(), None));
            } else {
                result.push((user_key.to_vec(), Some(entry.value().clone())));
            }
        }

        result
    }

    /// Iterate **all** raw entries in internal-key order, preserving every
    /// version and tombstone. Used by flush and compaction; the returned
    /// pairs are `(internal_key, value_bytes)` with value_bytes empty for
    /// tombstones.
    pub(crate) fn iter_internal(&self) -> Vec<(Vec<u8>, Vec<u8>)> {
        self.data
            .iter()
            .map(|e| (e.key().clone(), e.value().clone()))
            .collect()
    }

    pub(crate) fn approximate_size(&self) -> usize {
        self.approximate_size.load(Ordering::Relaxed)
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_put_get() {
        let mt = MemTable::new();
        mt.put(b"key1", b"value1", 1);
        assert_eq!(mt.get(b"key1", 1), Some(Some(b"value1".to_vec())));
        assert_eq!(mt.get(b"key1", 0), None);
    }

    #[test]
    fn test_delete() {
        let mt = MemTable::new();
        mt.put(b"key1", b"value1", 1);
        mt.delete(b"key1", 2);

        assert_eq!(mt.get(b"key1", 2), Some(None));
        assert_eq!(mt.get(b"key1", 1), Some(Some(b"value1".to_vec())));
    }

    #[test]
    fn test_overwrite() {
        let mt = MemTable::new();
        mt.put(b"key1", b"v1", 1);
        mt.put(b"key1", b"v2", 2);

        assert_eq!(mt.get(b"key1", 2), Some(Some(b"v2".to_vec())));
        assert_eq!(mt.get(b"key1", 1), Some(Some(b"v1".to_vec())));
    }

    #[test]
    fn test_iter_dedup() {
        let mt = MemTable::new();
        mt.put(b"a", b"v1", 1);
        mt.put(b"a", b"v2", 2);
        mt.put(b"b", b"v3", 3);

        let items = mt.iter(3);
        assert_eq!(items.len(), 2);
        assert_eq!(items[0], (b"a".to_vec(), Some(b"v2".to_vec())));
        assert_eq!(items[1], (b"b".to_vec(), Some(b"v3".to_vec())));
    }

    #[test]
    fn test_iter_internal_preserves_versions() {
        let mt = MemTable::new();
        mt.put(b"a", b"v1", 1);
        mt.put(b"a", b"v2", 2);
        mt.delete(b"a", 3);
        let items = mt.iter_internal();
        assert_eq!(items.len(), 3);
    }
}
