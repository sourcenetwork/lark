//! Carrying arbitrary byte keys across a `&str` API.
//!
//! `kovan_mvcc::Storage` and `Txn` take keys as `&str`, and Rust's `str`
//! is guaranteed UTF-8. A key-value store's keys are arbitrary bytes, so
//! something has to give at the boundary.
//!
//! # The encoding
//!
//! One tag byte, then the key:
//!
//! ```text
//! 't' <key bytes>      the key is already valid UTF-8, carried verbatim
//! 'x' <hex of key>     anything else
//! ```
//!
//! The tag is what makes it unambiguous: without it, the hex of one key
//! could equal the text of another, and two distinct keys would share a
//! lock-table entry.
//!
//! # Why adaptive rather than always hex
//!
//! Hex doubles every key. Doubling is only actually necessary for keys
//! that are not text, and in a document store most are: collection
//! names, document ids, index prefixes. Those pay one byte here instead
//! of their own length again. A genuinely binary key still pays 2x,
//! which is the floor for a lossless mapping into UTF-8 with a
//! byte-per-nibble alphabet.
//!
//! # What this deliberately does not promise
//!
//! Order is **not** preserved: `'t'`-tagged keys and `'x'`-tagged keys
//! interleave differently from their raw forms. Nothing needs it to be.
//! The `Storage` trait only ever addresses a key exactly - locks, write
//! records and data versions are all point-addressed, and ordering
//! *within* one key's versions is carried by a separate `u64` timestamp
//! that this module never touches. A future range API would need its own
//! order-preserving path and must not reuse this one.

/// Tag for a key that is already valid UTF-8.
const TAG_TEXT: u8 = b't';
/// Tag for a key that had to be hex-encoded.
const TAG_HEX: u8 = b'x';

const HEX: &[u8; 16] = b"0123456789abcdef";

/// Encode an arbitrary byte key as a `String` kovan-mvcc can carry.
pub(crate) fn encode(key: &[u8]) -> String {
    match std::str::from_utf8(key) {
        Ok(text) => {
            let mut out = String::with_capacity(text.len() + 1);
            out.push(TAG_TEXT as char);
            out.push_str(text);
            out
        }
        Err(_) => {
            let mut out = String::with_capacity(key.len() * 2 + 1);
            out.push(TAG_HEX as char);
            for b in key {
                out.push(HEX[(b >> 4) as usize] as char);
                out.push(HEX[(b & 0x0f) as usize] as char);
            }
            out
        }
    }
}

/// Recover the original key without copying when it was carried as text.
///
/// The text path is the common one and its bytes are already sitting in
/// the string, so it borrows. Only a hex-encoded key allocates, and only
/// because its bytes genuinely are not present in their original form.
pub(crate) fn decode_ref(encoded: &str) -> Option<std::borrow::Cow<'_, [u8]>> {
    let bytes = encoded.as_bytes();
    let (tag, rest) = bytes.split_first()?;
    match *tag {
        TAG_TEXT => Some(std::borrow::Cow::Borrowed(rest)),
        TAG_HEX => decode(encoded).map(std::borrow::Cow::Owned),
        _ => None,
    }
}

/// Encode into a caller-owned buffer, so a hot path can reuse one
/// allocation across operations instead of making a `String` per key.
pub(crate) fn encode_into(key: &[u8], out: &mut String) {
    out.clear();
    match std::str::from_utf8(key) {
        Ok(text) => {
            out.reserve(text.len() + 1);
            out.push(TAG_TEXT as char);
            out.push_str(text);
        }
        Err(_) => {
            out.reserve(key.len() * 2 + 1);
            out.push(TAG_HEX as char);
            for b in key {
                out.push(HEX[(b >> 4) as usize] as char);
                out.push(HEX[(b & 0x0f) as usize] as char);
            }
        }
    }
}

/// Recover the original key.
///
/// Returns `None` for a string this module did not produce, which is a
/// corrupt or foreign lock-table entry rather than something to guess at.
pub(crate) fn decode(encoded: &str) -> Option<Vec<u8>> {
    let bytes = encoded.as_bytes();
    let (tag, rest) = bytes.split_first()?;
    match *tag {
        TAG_TEXT => Some(rest.to_vec()),
        TAG_HEX => {
            if rest.len() % 2 != 0 {
                return None;
            }
            let mut out = Vec::with_capacity(rest.len() / 2);
            for pair in rest.chunks_exact(2) {
                out.push((unhex(pair[0])? << 4) | unhex(pair[1])?);
            }
            Some(out)
        }
        _ => None,
    }
}

fn unhex(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn a_text_key_costs_one_byte() {
        let k = b"collection/users/doc-42";
        let e = encode(k);
        assert_eq!(e.len(), k.len() + 1, "a UTF-8 key must not be expanded");
        assert_eq!(decode(&e).as_deref(), Some(&k[..]));
    }

    #[test]
    fn a_binary_key_round_trips_through_hex() {
        let k = [0x00u8, 0xff, 0x80, 0x41, 0x0a];
        let e = encode(&k);
        assert!(e.starts_with('x'), "a non-UTF-8 key must take the hex path");
        assert_eq!(decode(&e).as_deref(), Some(&k[..]));
    }

    #[test]
    fn the_empty_key_round_trips() {
        assert_eq!(decode(&encode(b"")).as_deref(), Some(&b""[..]));
    }

    /// The tag exists to stop this collision: without it, the text key
    /// "00ff" and the binary key [0x00, 0xff] would encode identically
    /// and share one lock-table entry.
    #[test]
    fn a_text_key_cannot_collide_with_the_hex_of_a_binary_key() {
        let text = b"00ff";
        let binary = [0x00u8, 0xff];
        assert_ne!(encode(text), encode(&binary));
        assert_eq!(decode(&encode(text)).as_deref(), Some(&text[..]));
        assert_eq!(decode(&encode(&binary)).as_deref(), Some(&binary[..]));
    }

    #[test]
    fn a_foreign_string_is_refused_rather_than_guessed_at() {
        assert_eq!(decode(""), None);
        assert_eq!(decode("qwerty"), None, "unknown tag");
        assert_eq!(decode("x0"), None, "odd-length hex");
        assert_eq!(decode("xzz"), None, "not hex");
    }

    #[test]
    fn every_encoding_is_valid_utf8_by_construction() {
        // The whole point: the result has to be a `str`.
        for k in [vec![0u8], vec![0xff; 8], b"plain".to_vec(), vec![]] {
            let e = encode(&k);
            assert!(std::str::from_utf8(e.as_bytes()).is_ok(), "{k:?}");
        }
    }

    #[test]
    fn a_text_key_decodes_without_copying() {
        let e = encode(b"collection/users/doc-42");
        match decode_ref(&e) {
            Some(std::borrow::Cow::Borrowed(b)) => assert_eq!(b, b"collection/users/doc-42"),
            other => panic!("a text key must borrow, got {other:?}"),
        }
    }

    #[test]
    fn encode_into_reuses_the_buffer() {
        let mut buf = String::new();
        encode_into(b"one", &mut buf);
        assert_eq!(buf, "tone");
        let cap = buf.capacity();
        encode_into(b"two", &mut buf);
        assert_eq!(buf, "ttwo");
        assert!(
            buf.capacity() >= cap,
            "the buffer must be reused, not shrunk"
        );
    }

    proptest! {
        /// The borrowing decoder must agree with the owning one on every
        /// input, or the fast path returns something different from the
        /// slow one.
        #[test]
        fn decode_ref_agrees_with_decode(key in proptest::collection::vec(any::<u8>(), 0..128)) {
            let e = encode(&key);
            prop_assert_eq!(decode_ref(&e).map(|c| c.into_owned()), decode(&e));
        }

        /// Any byte string survives the round trip. This is the property
        /// the whole boundary rests on: a key that came back different
        /// would silently address the wrong row.
        #[test]
        fn any_key_round_trips(key in proptest::collection::vec(any::<u8>(), 0..256)) {
            let round = decode(&encode(&key));
            prop_assert_eq!(round.as_deref(), Some(&key[..]));
        }

        /// Distinct keys must stay distinct, or two of them share a lock.
        #[test]
        fn distinct_keys_encode_distinctly(
            a in proptest::collection::vec(any::<u8>(), 0..64),
            b in proptest::collection::vec(any::<u8>(), 0..64),
        ) {
            prop_assume!(a != b);
            prop_assert_ne!(encode(&a), encode(&b));
        }
    }
}
