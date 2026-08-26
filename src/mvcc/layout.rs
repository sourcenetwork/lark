//! How Percolator's three key spaces sit inside one regolith keyspace.
//!
//! Percolator wants a lock column family, a write column family and a
//! data column family. Backing each with its own store would mean three
//! physical reads for one transactional read, on an engine that already
//! resolves a version in a single seek. So the mapping is chosen to give
//! the protocol what it asks for while costing as little as the layout
//! allows:
//!
//! ```text
//! locks   in memory, never written        0 reads, 0 writes
//! writes  W | key | 0x00 | !commit_ts     1 seek  (seek_for_prev)
//! data    D | key | 0x00 |  start_ts      1 point read
//! ```
//!
//! # Locks are not durable, on purpose
//!
//! Percolator persists locks so that a *different node* can resolve a
//! coordinator that died on its own. regolith is embedded and holds an
//! exclusive directory lock, so there is no such node: if the process
//! dies, no transaction is in flight to resolve, and prewritten data
//! with no write record is already invisible to every reader. Keeping
//! locks in memory therefore removes a durable write and a durable read
//! from the hot path without weakening anything a single process can
//! observe. It also supplies the compare-and-swap `put_lock` wants,
//! which the engine itself does not have.
//!
//! What it gives up is garbage: prewritten data whose transaction died
//! is unreachable but not yet reclaimed. That is the GC's problem, not
//! a correctness one, and `gc_data` is where it is collected.
//!
//! # Why the write records are keyed by an inverted timestamp
//!
//! `get_latest_write(key, ts)` asks for the newest commit at or before
//! `ts`. With `!commit_ts` the newest commit sorts *first*, so the query
//! is one forward seek to `W | key | 0x00 | !ts` and taking what it
//! lands on, exactly as the engine resolves its own MVCC.

/// Upper bound on the bytes a composed key occupies: prefix, the key
/// with every `0x00` escaped to two bytes, the two-byte terminator and
/// the timestamp. Used to size the scratch buffer, so an over-estimate
/// costs nothing and an under-estimate would cost a reallocation.
pub(crate) fn composed_len(key: &[u8]) -> usize {
    1 + 2 * key.len() + 2 + 8
}

/// Prefix for a write record.
pub(crate) const WRITE: u8 = b'W';
/// Prefix for a data version.
pub(crate) const DATA: u8 = b'D';

/// Escape byte. A `0x00` inside a key is written as `SEP, ESCAPE`.
///
/// Keys are raw bytes: the tagged UTF-8 encoding this module used to
/// assume was removed once kovan-mvcc took bytes directly, and with it
/// the guarantee that `0x00` never appears inside a key. A bare `0x00`
/// separator stopped separating at that point, and it was not
/// theoretical: `write_prefix(b"a")` is `W a 0x00`, and the composed
/// key for `b"a\0b"` begins with exactly those bytes, so a read of
/// `a` could walk onto `a\0b`'s write record and return its value.
///
/// The fix is order-preserving escaping. Inside a key `0x00` becomes
/// `0x00 0xFF`; the key is terminated by `0x00 0x00`. Ordering is
/// unchanged because the terminator is smaller than both an escaped
/// `0x00` (`0x00 0xFF`) and any non-zero byte, so a shorter key still
/// sorts before every key it is a prefix of. No key can be a prefix of
/// another once terminated, which is the property the seek relies on.
pub(crate) const ESCAPE: u8 = 0xFF;

/// Separator byte. Doubled it terminates a key; followed by [`ESCAPE`]
/// it is a literal `0x00` within one.
pub(crate) const SEP: u8 = 0x00;

/// Append `key` with every `0x00` escaped, then the terminator.
fn push_escaped(key: &[u8], out: &mut Vec<u8>) {
    // The common case has no zero byte at all, so the whole key moves
    // in one copy rather than a byte at a time.
    let mut rest = key;
    while let Some(at) = rest.iter().position(|&b| b == SEP) {
        out.extend_from_slice(&rest[..at]);
        out.push(SEP);
        out.push(ESCAPE);
        rest = &rest[at + 1..];
    }
    out.extend_from_slice(rest);
    out.push(SEP);
    out.push(SEP);
}

fn compose(prefix: u8, key: &[u8], ts_bytes: [u8; 8], out: &mut Vec<u8>) {
    out.clear();
    out.reserve(composed_len(key));
    out.push(prefix);
    push_escaped(key, out);
    out.extend_from_slice(&ts_bytes);
}

/// The key a write record is stored at. Newest sorts first.
pub(crate) fn write_key(key: &[u8], commit_ts: u64, out: &mut Vec<u8>) {
    compose(WRITE, key, (!commit_ts).to_be_bytes(), out);
}

/// The seek target for "newest commit at or before `ts`".
pub(crate) fn write_seek(key: &[u8], ts: u64, out: &mut Vec<u8>) {
    compose(WRITE, key, (!ts).to_be_bytes(), out);
}

/// The prefix every write record for `key` shares, used to tell a hit
/// for this key from the next key's records.
pub(crate) fn write_prefix(key: &[u8], out: &mut Vec<u8>) {
    out.clear();
    out.reserve(composed_len(key));
    out.push(WRITE);
    push_escaped(key, out);
}

/// The key a data version is stored at.
pub(crate) fn data_key(key: &[u8], start_ts: u64, out: &mut Vec<u8>) {
    compose(DATA, key, start_ts.to_be_bytes(), out);
}

/// Recover the user key from a composed write record key, undoing the
/// escaping [`push_escaped`] applied.
///
/// `None` when the bytes are not a write record: wrong prefix, no
/// terminator, or too short to carry a timestamp.
pub(crate) fn user_key_of(composed: &[u8]) -> Option<Vec<u8>> {
    let body = composed.strip_prefix(&[WRITE])?;
    let body = body.get(..body.len().checked_sub(8)?)?;
    let mut out = Vec::with_capacity(body.len());
    let mut i = 0;
    while i < body.len() {
        if body[i] != SEP {
            out.push(body[i]);
            i += 1;
            continue;
        }
        match body.get(i + 1)? {
            &SEP => return (i + 2 == body.len()).then_some(out),
            &ESCAPE => out.push(SEP),
            _ => return None,
        }
        i += 2;
    }
    None
}

/// Recover the commit timestamp from a write record's key.
pub(crate) fn commit_ts_of(write_key: &[u8]) -> Option<u64> {
    let ts = write_key.get(write_key.len().checked_sub(8)?..)?;
    Some(!u64::from_be_bytes(ts.try_into().ok()?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn newer_commits_sort_before_older_ones() {
        let (mut a, mut b) = (Vec::new(), Vec::new());
        write_key(b"tk", 10, &mut a);
        write_key(b"tk", 5, &mut b);
        assert!(a < b, "a newer commit must sort first so one seek finds it");
    }

    #[test]
    fn a_seek_lands_on_the_newest_commit_at_or_before_the_timestamp() {
        let mut seek = Vec::new();
        write_seek(b"tk", 7, &mut seek);
        let (mut at10, mut at5) = (Vec::new(), Vec::new());
        write_key(b"tk", 10, &mut at10);
        write_key(b"tk", 5, &mut at5);
        assert!(
            at10 < seek,
            "a commit after the read timestamp sorts before the seek"
        );
        assert!(
            at5 >= seek,
            "the newest commit at or before it sorts at or after"
        );
    }

    #[test]
    fn the_separator_keeps_one_key_from_being_a_prefix_of_another() {
        // Without the separator, "tk" at ts 0 and the key "tk\0..." would
        // be indistinguishable.
        let (mut short, mut long) = (Vec::new(), Vec::new());
        write_key(b"tk", 1, &mut short);
        write_key(b"tkk", 1, &mut long);
        assert_ne!(short, long);
        let mut prefix = Vec::new();
        write_prefix(b"tk", &mut prefix);
        assert!(short.starts_with(&prefix));
        assert!(
            !long.starts_with(&prefix),
            "a longer key must not match the shorter's prefix"
        );
    }

    #[test]
    fn a_commit_timestamp_survives_the_round_trip() {
        for ts in [0u64, 1, 42, u64::MAX - 1, u64::MAX] {
            let mut k = Vec::new();
            write_key(b"tk", ts, &mut k);
            assert_eq!(commit_ts_of(&k), Some(ts), "ts {ts}");
        }
    }

    #[test]
    fn write_and_data_never_collide() {
        let (mut w, mut d) = (Vec::new(), Vec::new());
        write_key(b"tk", 1, &mut w);
        data_key(b"tk", 1, &mut d);
        assert_ne!(w, d, "the two spaces must not share a key");
        assert_eq!(w[0], WRITE);
        assert_eq!(d[0], DATA);
    }

    #[test]
    fn the_buffer_is_reused_rather_than_reallocated() {
        let mut buf = Vec::new();
        write_key(b"tk", 1, &mut buf);
        let cap = buf.capacity();
        for ts in 0..64 {
            write_key(b"tk", ts, &mut buf);
        }
        assert_eq!(buf.capacity(), cap, "composing a key must not reallocate");
    }
}

#[cfg(test)]
mod nul_key_tests {
    use super::*;

    /// Keys chosen to sit on every boundary the escaping touches:
    /// empty, bare separators, a separator next to the escape byte,
    /// and ordinary bytes either side.
    const AWKWARD: &[&[u8]] = &[
        b"",
        b"\x00",
        b"\x00\x00",
        b"\x00\xff",
        b"\xff",
        b"\xff\x00",
        b"a",
        b"a\x00",
        b"a\x00b",
        b"a\x00\x00b",
        b"a\x00\xffb",
        b"ab",
        b"b",
    ];

    #[test]
    fn escaping_preserves_the_order_of_the_raw_keys() {
        // The seek depends on this: `write_seek` lands where it does
        // only because composing a key keeps the ordering the raw
        // bytes had. An escaping that reordered keys would send a read
        // to the wrong place without ever tripping a prefix check.
        for a in AWKWARD {
            for b in AWKWARD {
                let (mut ca, mut cb) = (Vec::new(), Vec::new());
                write_key(a, 1, &mut ca);
                write_key(b, 1, &mut cb);
                assert_eq!(
                    ca.cmp(&cb),
                    a.cmp(b),
                    "composed order disagrees with raw order for {a:?} vs {b:?}"
                );
            }
        }
    }

    #[test]
    fn a_user_key_survives_the_round_trip_through_a_composed_key() {
        for k in AWKWARD {
            let mut composed = Vec::new();
            write_key(k, 7, &mut composed);
            assert_eq!(
                user_key_of(&composed).as_deref(),
                Some(&k[..]),
                "{k:?} did not survive compose then decompose"
            );
            assert_eq!(commit_ts_of(&composed), Some(7));
        }
    }

    #[test]
    fn user_key_of_rejects_bytes_that_are_not_a_write_record() {
        // A data record, a truncated key and an unterminated one must
        // all report absence rather than a plausible wrong key.
        let mut data = Vec::new();
        data_key(b"a", 1, &mut data);
        assert_eq!(user_key_of(&data), None, "a data record is not a write record");
        assert_eq!(user_key_of(b"W"), None);
        assert_eq!(user_key_of(&[WRITE, b'a', 0, 0, 0, 0, 0, 0][..]), None);
    }

    #[test]
    fn no_composed_key_is_a_prefix_of_another() {
        for a in AWKWARD {
            let mut prefix = Vec::new();
            write_prefix(a, &mut prefix);
            for b in AWKWARD {
                if a == b {
                    continue;
                }
                let mut cb = Vec::new();
                write_key(b, 1, &mut cb);
                assert!(
                    !cb.starts_with(&prefix),
                    "key {b:?} matches key {a:?}'s prefix, so a read of {a:?} \
                     could return {b:?}'s value"
                );
            }
        }
    }

    #[test]
    fn a_key_containing_nul_is_not_confused_with_a_shorter_one() {
        // Keys are raw bytes since the tagged encoding was dropped, so
        // a key may contain the separator byte. `a` and `a\0b` must
        // stay distinguishable, or a read of `a` can return `a\0b`.
        let (mut short, mut long) = (Vec::new(), Vec::new());
        write_key(b"a", 1, &mut short);
        write_key(b"a\x00b", 1, &mut long);
        let mut prefix = Vec::new();
        write_prefix(b"a", &mut prefix);
        assert!(short.starts_with(&prefix));
        assert!(
            !long.starts_with(&prefix),
            "key a\\0b matched key a's prefix: a read of `a` would return `a\\0b`'s value"
        );
    }
}
