use std::sync::atomic::{AtomicUsize, Ordering};

use crossbeam_skiplist::SkipMap;
use parking_lot::Mutex;

use super::internal_key::{
    decode_internal_key, encode_internal_key, lookup_key, VALUE_TYPE_DELETION, VALUE_TYPE_MERGE,
    VALUE_TYPE_VALUE,
};
use super::range_tombstone::{max_covering_seq, RangeTombstone};

/// Concurrent in-memory sorted table backed by a lock-free skip list.
///
/// Supports multiple concurrent readers and a single writer (serialized
/// externally by the engine's write lock).
///
/// Range tombstones are stored in a separate `Mutex<Vec<_>>` rather
/// than interleaved with point entries. Range deletes are orders of
/// magnitude rarer than point writes so the lock is cheap, and
/// keeping them separate lets point-entry lookups stay lock-free.
pub(crate) struct MemTable {
    data: SkipMap<Vec<u8>, Vec<u8>>,
    range_tombstones: Mutex<Vec<RangeTombstone>>,
    approximate_size: AtomicUsize,
}

impl MemTable {
    pub(crate) fn new() -> Self {
        Self {
            data: SkipMap::new(),
            range_tombstones: Mutex::new(Vec::new()),
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

    /// Insert a merge operand for the given key. The operand will
    /// be combined with any older base value (or other operands) at
    /// read time via the configured [`crate::MergeOperator`].
    pub(crate) fn merge(&self, key: &[u8], operand: &[u8], seq: u64) {
        let internal_key = encode_internal_key(key, seq, VALUE_TYPE_MERGE);
        let size = internal_key.len() + operand.len();
        self.data.insert(internal_key, operand.to_vec());
        self.approximate_size.fetch_add(size, Ordering::Relaxed);
    }

    /// Record a range tombstone — every user key in `[start, end)`
    /// is considered deleted as of `seq`.
    pub(crate) fn delete_range(&self, start: &[u8], end: &[u8], seq: u64) {
        let size = start.len() + end.len() + 8;
        self.range_tombstones
            .lock()
            .push(RangeTombstone::new(start.to_vec(), end.to_vec(), seq));
        self.approximate_size.fetch_add(size, Ordering::Relaxed);
    }

    /// Return a snapshot of every range tombstone currently held.
    /// Used by flush (to persist them into the produced SSTable)
    /// and by the iterator / scan paths to query cover info.
    pub(crate) fn clone_range_tombstones(&self) -> Vec<RangeTombstone> {
        self.range_tombstones.lock().clone()
    }

    /// Largest seq of any range tombstone covering `user_key` that is
    /// visible at `snapshot_seq`. Returns `0` if no such tombstone
    /// exists — `0` is a safe sentinel because real seqs start at 1.
    pub(crate) fn covering_range_tombstone_seq(&self, user_key: &[u8], snapshot_seq: u64) -> u64 {
        max_covering_seq(&self.range_tombstones.lock(), user_key, snapshot_seq)
    }

    /// Look up the newest point entry for `key` visible at
    /// `snapshot_seq`. Returns `Some((seq, value_opt))` — `value_opt`
    /// is `Some(..)` for a live value and `None` for a tombstone —
    /// or `None` if the memtable has no entry for `key` at or below
    /// `snapshot_seq`.
    ///
    /// This method intentionally ignores range tombstones; the caller
    /// is responsible for merging range-tombstone coverage across
    /// sources and comparing seqs.
    pub(crate) fn get(&self, key: &[u8], snapshot_seq: u64) -> Option<(u64, Option<Vec<u8>>)> {
        let search_key = lookup_key(key, snapshot_seq);

        for entry in self.data.range(search_key..) {
            let (user_key, seq, value_type) = decode_internal_key(entry.key());

            if user_key != key {
                return None;
            }

            if seq <= snapshot_seq {
                return if value_type == VALUE_TYPE_DELETION {
                    Some((seq, None))
                } else {
                    Some((seq, Some(entry.value().clone())))
                };
            }
        }

        None
    }

    /// Walk every visible entry for `key` at `snapshot_seq` in
    /// newest-seq-first order, appending `(seq, value_type, bytes)`
    /// tuples onto `out` and stopping at (and including) the first
    /// terminator (`VALUE_TYPE_VALUE` or `VALUE_TYPE_DELETION`).
    /// Returns `true` when a terminator was reached — callers walking
    /// multiple sources use this to decide whether to continue the
    /// walk into the next source.
    ///
    /// Used by the merge-operator read path to collect a chain of
    /// merge operands layered on top of the underlying base value.
    pub(crate) fn collect_merge_chain(
        &self,
        key: &[u8],
        snapshot_seq: u64,
        out: &mut Vec<(u64, u8, Vec<u8>)>,
    ) -> bool {
        let search_key = lookup_key(key, snapshot_seq);

        for entry in self.data.range(search_key..) {
            let (user_key, seq, value_type) = decode_internal_key(entry.key());
            if user_key != key {
                return false;
            }
            if seq > snapshot_seq {
                continue;
            }
            out.push((seq, value_type, entry.value().clone()));
            if value_type != VALUE_TYPE_MERGE {
                return true;
            }
        }
        false
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

    /// Return the first `(internal_key, value)` pair whose key is in the
    /// half-open range `[lower, ..)`. Used by the streaming iterator to walk
    /// the memtable statelessly — each call does a fresh `O(log N)` seek in
    /// the skip list.
    pub(crate) fn first_entry_from(
        &self,
        lower: std::ops::Bound<&[u8]>,
    ) -> Option<(Vec<u8>, Vec<u8>)> {
        self.data
            .range::<[u8], _>((lower, std::ops::Bound::Unbounded))
            .next()
            .map(|e| (e.key().clone(), e.value().clone()))
    }

    /// Return the last `(internal_key, value)` pair whose key is in the
    /// half-open range `(.., upper]`. The companion of [`first_entry_from`]
    /// used for reverse seeks.
    pub(crate) fn last_entry_before(
        &self,
        upper: std::ops::Bound<&[u8]>,
    ) -> Option<(Vec<u8>, Vec<u8>)> {
        self.data
            .range::<[u8], _>((std::ops::Bound::Unbounded, upper))
            .next_back()
            .map(|e| (e.key().clone(), e.value().clone()))
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
        assert_eq!(mt.get(b"key1", 1), Some((1, Some(b"value1".to_vec()))));
        assert_eq!(mt.get(b"key1", 0), None);
    }

    #[test]
    fn test_delete() {
        let mt = MemTable::new();
        mt.put(b"key1", b"value1", 1);
        mt.delete(b"key1", 2);

        assert_eq!(mt.get(b"key1", 2), Some((2, None)));
        assert_eq!(mt.get(b"key1", 1), Some((1, Some(b"value1".to_vec()))));
    }

    #[test]
    fn test_overwrite() {
        let mt = MemTable::new();
        mt.put(b"key1", b"v1", 1);
        mt.put(b"key1", b"v2", 2);

        assert_eq!(mt.get(b"key1", 2), Some((2, Some(b"v2".to_vec()))));
        assert_eq!(mt.get(b"key1", 1), Some((1, Some(b"v1".to_vec()))));
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

    #[test]
    fn test_range_tombstone_basic() {
        let mt = MemTable::new();
        mt.delete_range(b"b", b"d", 5);
        assert_eq!(mt.covering_range_tombstone_seq(b"a", 10), 0);
        assert_eq!(mt.covering_range_tombstone_seq(b"b", 10), 5);
        assert_eq!(mt.covering_range_tombstone_seq(b"c", 10), 5);
        assert_eq!(mt.covering_range_tombstone_seq(b"d", 10), 0); // end exclusive
                                                                  // Invisible to snapshot older than the tombstone.
        assert_eq!(mt.covering_range_tombstone_seq(b"c", 4), 0);
    }

    #[test]
    fn test_range_tombstone_clone() {
        let mt = MemTable::new();
        mt.delete_range(b"a", b"c", 1);
        mt.delete_range(b"e", b"g", 2);
        let rts = mt.clone_range_tombstones();
        assert_eq!(rts.len(), 2);
    }
}
