use std::sync::atomic::{AtomicUsize, Ordering};

use crossbeam_skiplist::SkipMap;
use parking_lot::Mutex;

use super::internal_key::{
    decode_internal_key, encode_internal_key, lookup_key, InternalKey, VALUE_TYPE_DELETION,
    VALUE_TYPE_MERGE, VALUE_TYPE_VALUE,
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
    data: SkipMap<InternalKey, Vec<u8>>,
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
        let ik = encode_internal_key(key, seq, VALUE_TYPE_VALUE);
        let size = ik.len() + value.len();
        self.data.insert(InternalKey(ik), value.to_vec());
        self.approximate_size.fetch_add(size, Ordering::Relaxed);
    }

    /// Insert a deletion tombstone for the given key.
    pub(crate) fn delete(&self, key: &[u8], seq: u64) {
        let ik = encode_internal_key(key, seq, VALUE_TYPE_DELETION);
        let size = ik.len();
        self.data.insert(InternalKey(ik), Vec::new());
        self.approximate_size.fetch_add(size, Ordering::Relaxed);
    }

    /// Insert a merge operand for the given key. The operand will
    /// be combined with any older base value (or other operands) at
    /// read time via the configured [`crate::MergeOperator`].
    pub(crate) fn merge(&self, key: &[u8], operand: &[u8], seq: u64) {
        let ik = encode_internal_key(key, seq, VALUE_TYPE_MERGE);
        let size = ik.len() + operand.len();
        self.data.insert(InternalKey(ik), operand.to_vec());
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
        let search_key = InternalKey(lookup_key(key, snapshot_seq));

        for entry in self.data.range(search_key..) {
            let (user_key, seq, value_type) = decode_internal_key(entry.key().as_slice());

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
        let search_key = InternalKey(lookup_key(key, snapshot_seq));

        for entry in self.data.range(search_key..) {
            let (user_key, seq, value_type) = decode_internal_key(entry.key().as_slice());
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
            .map(|e| (e.key().0.clone(), e.value().clone()))
            .collect()
    }

    /// Walk every raw entry whose user key falls in `[start, end)` and
    /// return the count and approximate total size (sum of internal-key
    /// length + value length). Every version and every tombstone is
    /// counted — this is a raw-entry stat, not a distinct-user-key stat.
    ///
    /// Used by [`crate::Db::get_approximate_memtable_stats`] to give
    /// callers a cheap-ish estimate of how big a range is inside the
    /// active memtable without doing a full visible scan.
    pub(crate) fn approximate_stats_for_range(&self, start: &[u8], end: &[u8]) -> (u64, u64) {
        if start >= end {
            return (0, 0);
        }
        // Walk from the smallest possible internal key for `start`
        // (seq=MAX, value_type=0) to the smallest for `end`. Every
        // entry in between has user key in `[start, end)`.
        let lo = InternalKey(lookup_key(start, u64::MAX));
        let hi = InternalKey(lookup_key(end, u64::MAX));
        let mut count: u64 = 0;
        let mut size: u64 = 0;
        for entry in self
            .data
            .range((std::ops::Bound::Included(lo), std::ops::Bound::Excluded(hi)))
        {
            count += 1;
            size += (entry.key().len() + entry.value().len()) as u64;
        }
        (count, size)
    }

    /// Return the first `(internal_key, value)` pair whose key is in the
    /// half-open range `[lower, ..)`. Used by the streaming iterator to walk
    /// the memtable statelessly — each call does a fresh `O(log N)` seek in
    /// the skip list.
    pub(crate) fn first_entry_from(
        &self,
        lower: std::ops::Bound<&[u8]>,
    ) -> Option<(Vec<u8>, Vec<u8>)> {
        let bound = match lower {
            std::ops::Bound::Included(b) => std::ops::Bound::Included(InternalKey(b.to_vec())),
            std::ops::Bound::Excluded(b) => std::ops::Bound::Excluded(InternalKey(b.to_vec())),
            std::ops::Bound::Unbounded => std::ops::Bound::Unbounded,
        };
        self.data
            .range((bound, std::ops::Bound::<InternalKey>::Unbounded))
            .next()
            .map(|e| (e.key().0.clone(), e.value().clone()))
    }

    /// Return the last `(internal_key, value)` pair whose key is in the
    /// half-open range `(.., upper]`. The companion of [`first_entry_from`]
    /// used for reverse seeks.
    pub(crate) fn last_entry_before(
        &self,
        upper: std::ops::Bound<&[u8]>,
    ) -> Option<(Vec<u8>, Vec<u8>)> {
        let bound = match upper {
            std::ops::Bound::Included(b) => std::ops::Bound::Included(InternalKey(b.to_vec())),
            std::ops::Bound::Excluded(b) => std::ops::Bound::Excluded(InternalKey(b.to_vec())),
            std::ops::Bound::Unbounded => std::ops::Bound::Unbounded,
        };
        self.data
            .range((std::ops::Bound::<InternalKey>::Unbounded, bound))
            .next_back()
            .map(|e| (e.key().0.clone(), e.value().clone()))
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

    #[test]
    fn empty_memtable_reports_empty_and_zero_size() {
        let mt = MemTable::new();
        assert!(mt.is_empty());
        assert_eq!(mt.approximate_size(), 0);
        assert_eq!(mt.get(b"k", u64::MAX), None);
        assert!(mt.iter_internal().is_empty());
    }

    #[test]
    fn approximate_size_grows_monotonically() {
        let mt = MemTable::new();
        let s0 = mt.approximate_size();
        mt.put(b"k", b"v", 1);
        let s1 = mt.approximate_size();
        mt.put(b"k2", b"vv", 2);
        let s2 = mt.approximate_size();
        mt.delete(b"k3", 3);
        let s3 = mt.approximate_size();
        mt.delete_range(b"a", b"z", 4);
        let s4 = mt.approximate_size();
        assert!(s1 > s0 && s2 > s1 && s3 > s2 && s4 > s3);
    }

    #[test]
    fn merge_and_get_returns_operand_as_value() {
        // `get` returns the newest entry regardless of value_type; merge
        // operands appear as `Some(bytes)` just like values. The merge
        // resolution itself happens one level up.
        let mt = MemTable::new();
        mt.merge(b"k", b"op1", 1);
        let (seq, val) = mt.get(b"k", 1).expect("should find operand");
        assert_eq!(seq, 1);
        assert_eq!(val, Some(b"op1".to_vec()));
    }

    #[test]
    fn collect_merge_chain_walks_until_terminator() {
        let mt = MemTable::new();
        mt.put(b"k", b"base", 1);
        mt.merge(b"k", b"a", 2);
        mt.merge(b"k", b"b", 3);

        let mut chain = Vec::new();
        let reached_term = mt.collect_merge_chain(b"k", 3, &mut chain);
        assert!(reached_term);
        // Newest seq first: b, a, base (terminator).
        assert_eq!(chain.len(), 3);
        assert_eq!(chain[0].0, 3);
        assert_eq!(chain[2].0, 1);
        assert_eq!(chain[2].1, VALUE_TYPE_VALUE);
    }

    #[test]
    fn collect_merge_chain_stops_at_tombstone() {
        let mt = MemTable::new();
        mt.delete(b"k", 1);
        mt.merge(b"k", b"a", 2);

        let mut chain = Vec::new();
        let reached_term = mt.collect_merge_chain(b"k", 2, &mut chain);
        assert!(reached_term);
        assert_eq!(chain.len(), 2);
        assert_eq!(chain[1].1, VALUE_TYPE_DELETION);
    }

    #[test]
    fn collect_merge_chain_returns_false_when_only_merges_visible() {
        let mt = MemTable::new();
        mt.merge(b"k", b"a", 1);
        mt.merge(b"k", b"b", 2);
        let mut chain = Vec::new();
        let terminated = mt.collect_merge_chain(b"k", 2, &mut chain);
        assert!(!terminated, "pure-merge chain must return false");
        assert_eq!(chain.len(), 2);
    }

    #[test]
    fn approximate_stats_for_range_counts_every_version() {
        let mt = MemTable::new();
        mt.put(b"a", b"1", 1);
        mt.put(b"a", b"2", 2);
        mt.put(b"b", b"x", 3);
        mt.put(b"z", b"outside", 4);

        let (count, _size) = mt.approximate_stats_for_range(b"a", b"c");
        assert_eq!(count, 3, "two versions of 'a' + one 'b'");

        let (count_empty, _) = mt.approximate_stats_for_range(b"x", b"a");
        assert_eq!(count_empty, 0, "reversed range yields zero");
    }

    #[test]
    fn first_and_last_entry_bracket_the_memtable() {
        let mt = MemTable::new();
        mt.put(b"b", b"1", 1);
        mt.put(b"m", b"2", 2);
        mt.put(b"y", b"3", 3);

        let first = mt
            .first_entry_from(std::ops::Bound::Unbounded)
            .expect("has first");
        let last = mt
            .last_entry_before(std::ops::Bound::Unbounded)
            .expect("has last");
        assert_eq!(user_key_of_v(&first.0), b"b");
        assert_eq!(user_key_of_v(&last.0), b"y");

        // Bounded from above "k" — first >= k is "m".
        let m_first = mt
            .first_entry_from(std::ops::Bound::Included(&encode_internal_key(
                b"k",
                u64::MAX,
                VALUE_TYPE_VALUE,
            )))
            .expect("has entry");
        assert_eq!(user_key_of_v(&m_first.0), b"m");
    }

    fn user_key_of_v(ik: &[u8]) -> &[u8] {
        &ik[..ik.len() - 9]
    }
}
