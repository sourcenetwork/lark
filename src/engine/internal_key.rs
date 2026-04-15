//! Internal key encoding shared by the memtable and SSTables.
//!
//! Format: `[user_key][!seq as u64 BE][value_type: u8]`
//!
//! The bitwise-NOT of the sequence number ensures that for a given user key,
//! newer entries (higher seq) sort *first* in lexicographic order. That lets
//! both the memtable's skip list and an SSTable's sorted blocks answer
//! "most recent version visible at snapshot_seq" with a single forward seek:
//! seek to `user_key || !snapshot_seq || 0x00`, then the first entry with the
//! matching user key has the largest seq ≤ snapshot_seq.

/// Entry is a live value.
pub(crate) const VALUE_TYPE_VALUE: u8 = 1;
/// Entry is a deletion tombstone.
pub(crate) const VALUE_TYPE_DELETION: u8 = 0;
/// Entry is a merge operand — a piece of data that will be combined
/// with an older base value (or other operands) at read time via the
/// configured [`crate::MergeOperator`].
pub(crate) const VALUE_TYPE_MERGE: u8 = 2;

/// Size of the trailing `(seq, value_type)` suffix in an internal key.
pub(crate) const INTERNAL_KEY_SUFFIX_LEN: usize = 9;

pub(crate) fn encode_internal_key(user_key: &[u8], seq: u64, value_type: u8) -> Vec<u8> {
    let mut key = Vec::with_capacity(user_key.len() + INTERNAL_KEY_SUFFIX_LEN);
    key.extend_from_slice(user_key);
    key.extend_from_slice(&(!seq).to_be_bytes());
    key.push(value_type);
    key
}

pub(crate) fn decode_internal_key(internal_key: &[u8]) -> (&[u8], u64, u8) {
    let len = internal_key.len();
    let value_type = internal_key[len - 1];
    let seq_bytes: [u8; 8] = internal_key[len - 9..len - 1].try_into().unwrap();
    let seq = !u64::from_be_bytes(seq_bytes);
    let user_key = &internal_key[..len - 9];
    (user_key, seq, value_type)
}

/// Extract the user key portion of an internal key without decoding seq/type.
pub(crate) fn user_key_of(internal_key: &[u8]) -> &[u8] {
    &internal_key[..internal_key.len() - INTERNAL_KEY_SUFFIX_LEN]
}

/// Build the search key used to locate the most-recent visible entry for
/// `user_key` at `snapshot_seq`. The first entry with matching user key found
/// at or after this key is the answer (if any).
pub(crate) fn lookup_key(user_key: &[u8], snapshot_seq: u64) -> Vec<u8> {
    encode_internal_key(user_key, snapshot_seq, VALUE_TYPE_DELETION)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let ik = encode_internal_key(b"hello", 42, VALUE_TYPE_VALUE);
        let (uk, seq, vt) = decode_internal_key(&ik);
        assert_eq!(uk, b"hello");
        assert_eq!(seq, 42);
        assert_eq!(vt, VALUE_TYPE_VALUE);
    }

    #[test]
    fn newer_seq_sorts_first() {
        let older = encode_internal_key(b"k", 5, VALUE_TYPE_VALUE);
        let newer = encode_internal_key(b"k", 10, VALUE_TYPE_VALUE);
        assert!(newer < older, "higher seq must sort before lower seq");
    }

    #[test]
    fn lookup_key_matches_first_visible() {
        // At snapshot_seq=7, the first entry >= lookup_key with matching user
        // key should be the one with seq=5 (largest seq ≤ 7).
        let seq_10 = encode_internal_key(b"k", 10, VALUE_TYPE_VALUE);
        let seq_5 = encode_internal_key(b"k", 5, VALUE_TYPE_VALUE);
        let seq_3 = encode_internal_key(b"k", 3, VALUE_TYPE_VALUE);
        let probe = lookup_key(b"k", 7);
        assert!(seq_10 < probe);
        assert!(seq_5 >= probe);
        assert!(seq_3 >= probe);
    }
}
