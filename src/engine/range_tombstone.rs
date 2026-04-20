//! Range tombstones: `(start, end, seq)` triples that delete every
//! user key in `[start, end)` as of `seq`.
//!
//! Memtables and SSTables each hold a collection of range tombstones
//! alongside their point entries. A read at `snapshot_seq` treats a
//! user key `k` as deleted when the newest covering range tombstone
//! visible at that snapshot has a `seq` strictly greater than the
//! newest point entry for `k` visible at that snapshot.
//!
//! The representation is deliberately a plain list — range deletes
//! are expected to be orders of magnitude rarer than point writes,
//! so an O(N) scan per lookup is acceptable for v1. We can upgrade
//! to an interval tree or a sorted "fragmented range tombstone map"
//! later if benchmarks show the need.

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
}

/// Scan `tombstones` and return the largest `seq` of any range
/// tombstone that covers `user_key` and is visible at `snapshot_seq`
/// (i.e. its seq is `<= snapshot_seq`). Returns `0` if no such
/// tombstone exists — `0` is a safe "no-op" sentinel because real
/// seqs start at 1.
pub(crate) fn max_covering_seq(
    tombstones: &[RangeTombstone],
    user_key: &[u8],
    snapshot_seq: u64,
) -> u64 {
    let mut best = 0;
    for rt in tombstones {
        if rt.seq > snapshot_seq {
            continue;
        }
        if rt.covers(user_key) && rt.seq > best {
            best = rt.seq;
        }
    }
    best
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
}
