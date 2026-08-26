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

/// Bytes a composed key occupies: prefix, key, separator, timestamp.
pub(crate) fn composed_len(key: &[u8]) -> usize {
    2 + key.len() + 8
}

/// Prefix for a write record.
pub(crate) const WRITE: u8 = b'W';
/// Prefix for a data version.
pub(crate) const DATA: u8 = b'D';

/// Separator between the encoded key and the timestamp.
///
/// The encoded key is a tagged UTF-8 string, and `0x00` cannot appear
/// inside one: the text path carries UTF-8, whose only zero byte is a
/// zero codepoint, and the hex path emits only `0-9a-f`. So no key can
/// be a prefix of another once this byte is appended, and a timestamp
/// can never be mistaken for the tail of a longer key.
pub(crate) const SEP: u8 = 0x00;

fn compose(prefix: u8, key: &[u8], ts_bytes: [u8; 8], out: &mut Vec<u8>) {
    out.clear();
    out.reserve(2 + key.len() + 8);
    out.push(prefix);
    out.extend_from_slice(key);
    out.push(SEP);
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
    out.reserve(2 + key.len());
    out.push(WRITE);
    out.extend_from_slice(key);
    out.push(SEP);
}

/// The key a data version is stored at.
pub(crate) fn data_key(key: &[u8], start_ts: u64, out: &mut Vec<u8>) {
    compose(DATA, key, start_ts.to_be_bytes(), out);
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
