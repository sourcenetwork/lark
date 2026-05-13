//! Range tombstones: `(start, end, seq)` triples that delete every
//! user key in `[start, end)` as of `seq`.
//!
//! Memtables and SSTables each hold a collection of range tombstones
//! alongside their point entries. A read at `snapshot_seq` treats a
//! user key `k` as deleted when the newest covering range tombstone
//! visible at that snapshot has a `seq` strictly greater than the
//! newest point entry for `k` visible at that snapshot.
//!
//! Tombstones are indexed by start key with a prefix maximum of end
//! keys. A point coverage check first binary-searches the last
//! tombstone whose start is `<= user_key`, then walks backward only
//! while some earlier tombstone can still reach the key.

/// A single range tombstone.
#[derive(Debug, Clone)]
pub(crate) struct RangeTombstone {
    pub(crate) start: Vec<u8>,
    pub(crate) end: Vec<u8>,
    pub(crate) seq: u64,
}

impl RangeTombstone {
    pub(crate) fn new(start: Vec<u8>, end: Vec<u8>, seq: u64) -> Self {
        Self { start, end, seq }
    }

    /// Does this tombstone cover `user_key`? The range is half-open:
    /// `start <= user_key < end`.
    pub(crate) fn covers(&self, user_key: &[u8]) -> bool {
        self.start.as_slice() <= user_key && user_key < self.end.as_slice()
    }

    pub(crate) fn overlaps(&self, start: &[u8], end: &[u8]) -> bool {
        self.start.as_slice() < end && start < self.end.as_slice()
    }

    pub(crate) fn clip_to(&self, start: &[u8], end: &[u8]) -> Option<Self> {
        if !self.overlaps(start, end) {
            return None;
        }

        let clipped_start = if self.start.as_slice() < start {
            start.to_vec()
        } else {
            self.start.clone()
        };
        let clipped_end = if self.end.as_slice() > end {
            end.to_vec()
        } else {
            self.end.clone()
        };

        (clipped_start < clipped_end).then(|| Self::new(clipped_start, clipped_end, self.seq))
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct RangeTombstoneSet {
    tombstones: Vec<RangeTombstone>,
    prefix_max_end: Vec<Vec<u8>>,
}

impl RangeTombstoneSet {
    pub(crate) fn from_vec(mut tombstones: Vec<RangeTombstone>) -> Self {
        sort_dedup_tombstones(&mut tombstones);
        let prefix_max_end = build_prefix_max_end(&tombstones);
        Self {
            tombstones,
            prefix_max_end,
        }
    }

    pub(crate) fn push(&mut self, tombstone: RangeTombstone) {
        self.tombstones.push(tombstone);
        sort_dedup_tombstones(&mut self.tombstones);
        self.prefix_max_end = build_prefix_max_end(&self.tombstones);
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.tombstones.is_empty()
    }

    pub(crate) fn as_slice(&self) -> &[RangeTombstone] {
        &self.tombstones
    }

    pub(crate) fn iter(&self) -> std::slice::Iter<'_, RangeTombstone> {
        self.tombstones.iter()
    }

    /// Return the largest `seq` of any range tombstone that covers
    /// `user_key` and is visible at `snapshot_seq`.
    pub(crate) fn max_covering_seq(&self, user_key: &[u8], snapshot_seq: u64) -> u64 {
        if self.tombstones.is_empty() {
            return 0;
        }

        let mut idx = self
            .tombstones
            .partition_point(|rt| rt.start.as_slice() <= user_key);
        let mut best = 0;

        while idx > 0 {
            idx -= 1;
            if self.prefix_max_end[idx].as_slice() <= user_key {
                break;
            }

            let rt = &self.tombstones[idx];
            if rt.seq <= snapshot_seq && rt.covers(user_key) && rt.seq > best {
                best = rt.seq;
            }
        }

        best
    }

    pub(crate) fn clipped_overlaps(&self, start: &[u8], end: &[u8]) -> Vec<RangeTombstone> {
        let mut result = Vec::new();
        if start >= end {
            return result;
        }

        let mut idx = self
            .tombstones
            .partition_point(|rt| rt.start.as_slice() < end);

        while idx > 0 {
            idx -= 1;
            if self.prefix_max_end[idx].as_slice() <= start {
                break;
            }

            if let Some(clipped) = self.tombstones[idx].clip_to(start, end) {
                result.push(clipped);
            }
        }

        sort_dedup_tombstones(&mut result);
        result
    }
}

/// Convenience wrapper for one-shot checks over an unindexed slice.
#[cfg(test)]
pub(crate) fn max_covering_seq(
    tombstones: &[RangeTombstone],
    user_key: &[u8],
    snapshot_seq: u64,
) -> u64 {
    RangeTombstoneSet::from_vec(tombstones.to_vec()).max_covering_seq(user_key, snapshot_seq)
}

/// Return an exclusive upper bound that contains exactly `key` among
/// byte strings sharing `key` as a prefix.
pub(crate) fn exclusive_successor(key: &[u8]) -> Vec<u8> {
    let mut end = Vec::with_capacity(key.len() + 1);
    end.extend_from_slice(key);
    end.push(0);
    end
}

pub(crate) fn sort_dedup_tombstones(tombstones: &mut Vec<RangeTombstone>) {
    tombstones.sort_by(|a, b| {
        a.start
            .cmp(&b.start)
            .then_with(|| a.end.cmp(&b.end))
            .then_with(|| b.seq.cmp(&a.seq))
    });
    tombstones.dedup_by(|a, b| a.start == b.start && a.end == b.end && a.seq == b.seq);
}

fn build_prefix_max_end(tombstones: &[RangeTombstone]) -> Vec<Vec<u8>> {
    let mut prefix = Vec::with_capacity(tombstones.len());
    let mut max_end: Option<Vec<u8>> = None;
    for rt in tombstones {
        if max_end
            .as_ref()
            .is_none_or(|current| current.as_slice() < rt.end.as_slice())
        {
            max_end = Some(rt.end.clone());
        }
        prefix.push(max_end.as_ref().expect("just initialized").clone());
    }
    prefix
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn covers_half_open() {
        let rt = RangeTombstone::new(b"b".to_vec(), b"d".to_vec(), 10);
        assert!(!rt.covers(b"a"));
        assert!(rt.covers(b"b"));
        assert!(rt.covers(b"c"));
        assert!(!rt.covers(b"d")); // end is exclusive
        assert!(!rt.covers(b"e"));
    }

    #[test]
    fn max_covering_seq_picks_largest_visible() {
        let tombstones = vec![
            RangeTombstone::new(b"a".to_vec(), b"z".to_vec(), 5),
            RangeTombstone::new(b"a".to_vec(), b"z".to_vec(), 20), // invisible at snap=10
            RangeTombstone::new(b"c".to_vec(), b"e".to_vec(), 8),
        ];
        assert_eq!(max_covering_seq(&tombstones, b"d", 10), 8);
        assert_eq!(max_covering_seq(&tombstones, b"b", 10), 5);
        assert_eq!(max_covering_seq(&tombstones, b"m", 10), 5);
        // At snap=100, the seq=20 tombstone becomes visible.
        assert_eq!(max_covering_seq(&tombstones, b"m", 100), 20);
        // No coverage.
        assert_eq!(max_covering_seq(&tombstones, b"0", 100), 0);
    }

    #[test]
    fn max_covering_seq_empty_list_returns_zero() {
        assert_eq!(max_covering_seq(&[], b"k", u64::MAX), 0);
    }

    #[test]
    fn empty_range_covers_nothing() {
        // A degenerate `[x, x)` range has no keys in it.
        let rt = RangeTombstone::new(b"m".to_vec(), b"m".to_vec(), 1);
        assert!(!rt.covers(b"m"));
        assert!(!rt.covers(b"n"));
        assert!(!rt.covers(b"l"));
    }

    #[test]
    fn snapshot_below_all_tombstone_seqs_sees_no_coverage() {
        let rts = vec![
            RangeTombstone::new(b"a".to_vec(), b"z".to_vec(), 10),
            RangeTombstone::new(b"a".to_vec(), b"z".to_vec(), 20),
        ];
        assert_eq!(max_covering_seq(&rts, b"m", 5), 0);
        assert_eq!(max_covering_seq(&rts, b"m", 9), 0);
        assert_eq!(max_covering_seq(&rts, b"m", 10), 10);
    }

    #[test]
    fn tombstone_set_prunes_disjoint_ranges() {
        let set = RangeTombstoneSet::from_vec(vec![
            RangeTombstone::new(b"a".to_vec(), b"b".to_vec(), 1),
            RangeTombstone::new(b"d".to_vec(), b"e".to_vec(), 2),
            RangeTombstone::new(b"g".to_vec(), b"h".to_vec(), 3),
        ]);
        assert_eq!(set.max_covering_seq(b"f", u64::MAX), 0);
        assert_eq!(set.max_covering_seq(b"g", u64::MAX), 3);
    }

    #[test]
    fn clipped_overlaps_returns_fragments_in_order() {
        let set = RangeTombstoneSet::from_vec(vec![
            RangeTombstone::new(b"a".to_vec(), b"z".to_vec(), 5),
            RangeTombstone::new(b"c".to_vec(), b"e".to_vec(), 7),
        ]);
        let clipped = set.clipped_overlaps(b"b", b"f");
        assert_eq!(clipped.len(), 2);
        assert_eq!(clipped[0].start, b"b");
        assert_eq!(clipped[0].end, b"f");
        assert_eq!(clipped[0].seq, 5);
        assert_eq!(clipped[1].start, b"c");
        assert_eq!(clipped[1].end, b"e");
        assert_eq!(clipped[1].seq, 7);
    }

    #[test]
    fn exclusive_successor_contains_exact_key_only() {
        assert_eq!(exclusive_successor(b"a"), b"a\0");
        assert!(b"a".as_slice() < exclusive_successor(b"a").as_slice());
        assert!(exclusive_successor(b"a").as_slice() < b"aa".as_slice());
    }
}
