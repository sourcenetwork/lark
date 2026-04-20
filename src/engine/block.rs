//! SSTable data blocks: prefix-compressed key-value entries with restart points.
//!
//! Block format:
//! ```text
//! [entry_0][entry_1]...[entry_n][restart_0][restart_1]...[restart_m][num_restarts: u32]
//! ```
//!
//! Each entry is: `[shared: varint][unshared: varint][value_len: varint][key_suffix][value]`,
//! where `shared` is the length of the prefix shared with the previous key. Every
//! `RESTART_INTERVAL` entries we reset `shared = 0` and record the entry's byte
//! offset as a restart point, enabling binary search within the block.

use std::io;

use super::internal_key::compare_internal_keys;

/// Entries per restart point. Smaller = faster lookups, larger = better compression.
pub(crate) const RESTART_INTERVAL: usize = 16;

/// Offset and size of a block within an SSTable file.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BlockHandle {
    pub(crate) offset: u64,
    pub(crate) size: u64,
}

/// A decoded data block containing sorted key-value entries.
pub(crate) struct Block {
    data: Vec<u8>,
    restarts: Vec<u32>,
}

impl Block {
    pub(crate) fn decode(data: Vec<u8>) -> io::Result<Self> {
        if data.len() < 4 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "block too small",
            ));
        }

        let num_restarts = u32::from_le_bytes(data[data.len() - 4..].try_into().unwrap()) as usize;
        let restarts_offset = data.len() - 4 - num_restarts * 4;

        let mut restarts = Vec::with_capacity(num_restarts);
        for i in 0..num_restarts {
            let offset = restarts_offset + i * 4;
            restarts.push(u32::from_le_bytes(
                data[offset..offset + 4].try_into().unwrap(),
            ));
        }

        Ok(Self { data, restarts })
    }

    /// Approximate heap bytes held by this block. Used by the
    /// block cache to charge accurate sizes against its capacity
    /// budget. Includes the backing `Vec` allocations plus the
    /// struct itself; excludes any amortized allocator overhead,
    /// which is typically small and not worth modeling.
    pub(crate) fn charge(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.data.capacity()
            + self.restarts.capacity() * std::mem::size_of::<u32>()
    }

    /// The entry region of the block (everything before the restart
    /// array and the trailing `num_restarts` u32).
    pub(crate) fn entry_data(&self) -> &[u8] {
        let data_end = self.data.len() - 4 - self.restarts.len() * 4;
        &self.data[..data_end]
    }

    pub(crate) fn restart_count(&self) -> usize {
        self.restarts.len()
    }

    pub(crate) fn restart_offset(&self, idx: usize) -> usize {
        self.restarts[idx] as usize
    }

    /// Iterate all entries in this block in sorted order.
    pub(crate) fn iter(&self) -> BlockIterator<'_> {
        let data_end = self.data.len() - 4 - self.restarts.len() * 4;
        BlockIterator {
            data: &self.data[..data_end],
            pos: 0,
            current_key: Vec::new(),
        }
    }

    /// Return the first `(key, value)` entry with `key >= target`, if any.
    ///
    /// This is the primitive SSTable readers use for MVCC point lookups: the
    /// caller constructs a search key from `(user_key, snapshot_seq)` and
    /// inspects the returned entry to decide whether it satisfies the query.
    ///
    /// Retained for tests and for the iterator's `seek` path even
    /// though the merge-aware SSTable reader walks blocks manually
    /// to skip past `Merge` entries.
    #[allow(dead_code)]
    pub(crate) fn seek_ge(&self, target: &[u8]) -> Option<(Vec<u8>, Vec<u8>)> {
        let data_end = self.data.len() - 4 - self.restarts.len() * 4;
        let start = self.restart_start_for(target);

        let mut pos = start;
        let mut current_key = Vec::new();
        while pos < data_end {
            let (key, value) = decode_block_entry(&self.data[..data_end], pos, &current_key);
            let entry_size = encoded_entry_size(&self.data[..data_end], pos);
            pos += entry_size;
            current_key = key;

            if compare_internal_keys(&current_key, target).is_ge() {
                return Some((current_key, value));
            }
        }

        None
    }

    /// Binary-search restart points for the first entry whose key could be
    /// `>= target`; returns the byte offset to start the linear walk from.
    fn restart_start_for(&self, target: &[u8]) -> usize {
        let data_end = self.data.len() - 4 - self.restarts.len() * 4;
        let mut left = 0;
        let mut right = self.restarts.len();
        while left < right {
            let mid = left + (right - left) / 2;
            let restart_pos = self.restarts[mid] as usize;
            let (key, _) = decode_block_entry(&self.data[..data_end], restart_pos, &[]);
            if compare_internal_keys(&key, target).is_lt() {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        if left > 0 {
            self.restarts[left - 1] as usize
        } else {
            0
        }
    }
}

pub(crate) struct BlockIterator<'a> {
    data: &'a [u8],
    pos: usize,
    current_key: Vec<u8>,
}

impl<'a> Iterator for BlockIterator<'a> {
    type Item = (Vec<u8>, Vec<u8>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.data.len() {
            return None;
        }

        let (consumed, val_off, val_len) =
            decode_entry_at(self.data, self.pos, &mut self.current_key);
        self.pos += consumed;
        let value = self.data[val_off..val_off + val_len].to_vec();

        Some((self.current_key.clone(), value))
    }
}

/// Block builder: accumulates sorted entries and emits a data block.
pub(crate) struct BlockBuilder {
    buffer: Vec<u8>,
    restarts: Vec<u32>,
    entry_count: usize,
    last_key: Vec<u8>,
    restart_interval: usize,
}

impl BlockBuilder {
    pub(crate) fn new(restart_interval: usize) -> Self {
        Self {
            buffer: Vec::new(),
            restarts: vec![0], // The first entry is always a restart point.
            entry_count: 0,
            last_key: Vec::new(),
            restart_interval,
        }
    }

    pub(crate) fn add(&mut self, key: &[u8], value: &[u8]) {
        let shared = if self.entry_count % self.restart_interval == 0 && self.entry_count > 0 {
            self.restarts.push(self.buffer.len() as u32);
            0 // Restart point: no prefix sharing.
        } else {
            self.last_key
                .iter()
                .zip(key.iter())
                .take_while(|(a, b)| a == b)
                .count()
        };

        let unshared = key.len() - shared;

        encode_varint(&mut self.buffer, shared as u64);
        encode_varint(&mut self.buffer, unshared as u64);
        encode_varint(&mut self.buffer, value.len() as u64);
        self.buffer.extend_from_slice(&key[shared..]);
        self.buffer.extend_from_slice(value);

        self.last_key = key.to_vec();
        self.entry_count += 1;
    }

    pub(crate) fn finish(mut self) -> Vec<u8> {
        for restart in &self.restarts {
            self.buffer.extend_from_slice(&restart.to_le_bytes());
        }
        self.buffer
            .extend_from_slice(&(self.restarts.len() as u32).to_le_bytes());
        self.buffer
    }

    pub(crate) fn estimated_size(&self) -> usize {
        self.buffer.len() + self.restarts.len() * 4 + 4
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.entry_count == 0
    }
}

/// Decode one entry at `pos`. Reconstructs the key in-place into
/// `prev_key` (truncate to shared prefix + extend with unshared).
/// Returns `(bytes_consumed, value_offset_in_data, value_len)`.
/// The value lives at `data[value_offset..value_offset+value_len]`.
pub(crate) fn decode_entry_at(
    data: &[u8],
    pos: usize,
    prev_key: &mut Vec<u8>,
) -> (usize, usize, usize) {
    let mut offset = pos;
    let (shared, n) = decode_varint(&data[offset..]);
    offset += n;
    let (unshared, n) = decode_varint(&data[offset..]);
    offset += n;
    let (value_len, n) = decode_varint(&data[offset..]);
    offset += n;
    let shared = shared as usize;
    let unshared = unshared as usize;
    let value_len = value_len as usize;
    prev_key.truncate(shared);
    prev_key.extend_from_slice(&data[offset..offset + unshared]);
    let value_offset = offset + unshared;
    let consumed = value_offset + value_len - pos;
    (consumed, value_offset, value_len)
}

fn decode_block_entry(data: &[u8], pos: usize, prev_key: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let mut offset = pos;
    let (shared, n) = decode_varint(&data[offset..]);
    offset += n;
    let (unshared, n) = decode_varint(&data[offset..]);
    offset += n;
    let (value_len, n) = decode_varint(&data[offset..]);
    offset += n;

    let shared = shared as usize;
    let unshared = unshared as usize;
    let value_len = value_len as usize;

    let mut key = Vec::with_capacity(shared + unshared);
    key.extend_from_slice(&prev_key[..shared]);
    key.extend_from_slice(&data[offset..offset + unshared]);
    let value = data[offset + unshared..offset + unshared + value_len].to_vec();

    (key, value)
}

pub(crate) fn encoded_entry_size(data: &[u8], pos: usize) -> usize {
    let mut offset = pos;
    let (_, n) = decode_varint(&data[offset..]); // shared
    offset += n;
    let (unshared, n) = decode_varint(&data[offset..]);
    offset += n;
    let (value_len, n) = decode_varint(&data[offset..]);
    offset += n;

    offset - pos + unshared as usize + value_len as usize
}

fn encode_varint(buf: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        buf.push((value as u8) | 0x80);
        value >>= 7;
    }
    buf.push(value as u8);
}

pub(crate) fn decode_varint(data: &[u8]) -> (u64, usize) {
    let mut result: u64 = 0;
    let mut shift = 0;
    for (i, &byte) in data.iter().enumerate() {
        result |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            return (result, i + 1);
        }
        shift += 7;
    }
    (result, data.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_block(pairs: &[(&[u8], &[u8])]) -> Block {
        let mut builder = BlockBuilder::new(4);
        for (k, v) in pairs {
            builder.add(k, v);
        }
        Block::decode(builder.finish()).unwrap()
    }

    // ── varint encoding ──────────────────────────────────────────

    #[test]
    fn test_varint_roundtrip() {
        let test_values = [0u64, 1, 127, 128, 16383, 16384, u64::MAX];
        for val in test_values {
            let mut buf = Vec::new();
            encode_varint(&mut buf, val);
            let (decoded, _) = decode_varint(&buf);
            assert_eq!(val, decoded);
        }
    }

    #[test]
    fn varint_zero_and_127_fit_in_one_byte() {
        for v in [0u64, 42, 127] {
            let mut buf = Vec::new();
            encode_varint(&mut buf, v);
            assert_eq!(buf.len(), 1, "value {v} should encode to 1 byte");
            assert_eq!(decode_varint(&buf), (v, 1));
        }
    }

    #[test]
    fn varint_128_crosses_into_two_bytes() {
        let mut buf = Vec::new();
        encode_varint(&mut buf, 128);
        assert_eq!(buf, vec![0x80, 0x01]);
        assert_eq!(decode_varint(&buf), (128, 2));
    }

    #[test]
    fn varint_u64_max_fits_in_ten_bytes() {
        let mut buf = Vec::new();
        encode_varint(&mut buf, u64::MAX);
        assert_eq!(buf.len(), 10);
        assert_eq!(decode_varint(&buf), (u64::MAX, 10));
    }

    // ── block encode/decode ─────────────────────────────────────

    #[test]
    fn test_block_seek_ge() {
        let block = build_block(&[
            (b"apple", b"red"),
            (b"application", b"software"),
            (b"banana", b"yellow"),
        ]);

        assert_eq!(
            block.seek_ge(b"apple"),
            Some((b"apple".to_vec(), b"red".to_vec()))
        );
        assert_eq!(
            block.seek_ge(b"applicatio"),
            Some((b"application".to_vec(), b"software".to_vec()))
        );
        assert_eq!(
            block.seek_ge(b"b"),
            Some((b"banana".to_vec(), b"yellow".to_vec()))
        );
        assert_eq!(block.seek_ge(b"cherry"), None);
    }

    #[test]
    fn empty_block_iterates_nothing() {
        let builder = BlockBuilder::new(16);
        let block = Block::decode(builder.finish()).unwrap();
        assert_eq!(block.iter().count(), 0);
    }

    #[test]
    fn single_entry_block_round_trips() {
        let block = build_block(&[(b"only", b"one")]);
        let entries: Vec<_> = block.iter().collect();
        assert_eq!(entries, vec![(b"only".to_vec(), b"one".to_vec())]);
    }

    #[test]
    fn iter_preserves_insert_order() {
        let pairs: Vec<(Vec<u8>, Vec<u8>)> = (0..20)
            .map(|i| {
                (
                    format!("key_{:04}", i).into_bytes(),
                    format!("val_{}", i).into_bytes(),
                )
            })
            .collect();
        let mut b = BlockBuilder::new(4);
        for (k, v) in &pairs {
            b.add(k, v);
        }
        let block = Block::decode(b.finish()).unwrap();
        let got: Vec<_> = block.iter().collect();
        assert_eq!(got, pairs);
    }

    #[test]
    fn prefix_compression_round_trips_shared_prefix_keys() {
        let block = build_block(&[(b"zebra_a", b"1"), (b"zebra_b", b"2"), (b"zebra_c", b"3")]);
        let got: Vec<_> = block.iter().collect();
        assert_eq!(
            got,
            vec![
                (b"zebra_a".to_vec(), b"1".to_vec()),
                (b"zebra_b".to_vec(), b"2".to_vec()),
                (b"zebra_c".to_vec(), b"3".to_vec()),
            ]
        );
    }

    #[test]
    fn restart_points_occur_every_interval_entries() {
        let n = 32usize;
        let mut b = BlockBuilder::new(8);
        for i in 0..n {
            let key = format!("k_{:04}", i);
            b.add(key.as_bytes(), b"v");
        }
        let block = Block::decode(b.finish()).unwrap();
        // Entry 0 is always a restart; then 8, 16, 24 → 4 total.
        assert_eq!(block.restart_count(), 4);
        assert_eq!(block.restart_offset(0), 0);
    }

    #[test]
    fn decode_rejects_buffer_smaller_than_footer() {
        assert!(Block::decode(vec![]).is_err());
        assert!(Block::decode(vec![0u8; 3]).is_err());
    }

    // ── seek_ge edge cases ──────────────────────────────────────

    #[test]
    fn seek_ge_before_first_key_returns_first() {
        let block = build_block(&[(b"b", b"2"), (b"c", b"3")]);
        // Empty target sorts before every key — the ordering function
        // short-circuits on suffix-length so any user key > "" wins.
        assert_eq!(block.seek_ge(b""), Some((b"b".to_vec(), b"2".to_vec())));
    }

    #[test]
    fn seek_ge_matches_every_key_after_binary_search() {
        // 100 sorted keys with restart interval 4 exercises the
        // binary-search path over ~25 restart points.
        let keys: Vec<Vec<u8>> = (0..100)
            .map(|i| format!("key_{:05}", i).into_bytes())
            .collect();
        let mut b = BlockBuilder::new(4);
        for k in &keys {
            b.add(k, b"v");
        }
        let block = Block::decode(b.finish()).unwrap();

        for k in &keys {
            assert_eq!(block.seek_ge(k), Some((k.clone(), b"v".to_vec())));
        }
    }

    // ── cache accounting / builder internals ───────────────────

    #[test]
    fn charge_scales_with_data_bytes() {
        let small = build_block(&[(b"k", b"v")]);
        let big_pairs: Vec<(Vec<u8>, Vec<u8>)> = (0..500)
            .map(|i| (format!("key_{:04}", i).into_bytes(), vec![0u8; 128]))
            .collect();
        let mut b = BlockBuilder::new(16);
        for (k, v) in &big_pairs {
            b.add(k, v);
        }
        let big = Block::decode(b.finish()).unwrap();
        assert!(big.charge() > small.charge() * 100);
    }

    #[test]
    fn builder_is_empty_flag_reflects_entry_count() {
        let mut b = BlockBuilder::new(16);
        assert!(b.is_empty());
        b.add(b"k", b"v");
        assert!(!b.is_empty());
    }

    #[test]
    fn builder_estimated_size_grows_monotonically() {
        let mut b = BlockBuilder::new(16);
        let s0 = b.estimated_size();
        b.add(b"a", b"1");
        let s1 = b.estimated_size();
        b.add(b"b", b"2");
        let s2 = b.estimated_size();
        assert!(s1 >= s0);
        assert!(s2 >= s1);
    }

    #[test]
    fn entry_data_length_matches_iter_consumption() {
        let pairs: Vec<(&[u8], &[u8])> = vec![(b"abc", b"1"), (b"abcd", b"22"), (b"xyz", b"333")];
        let mut b = BlockBuilder::new(16);
        for (k, v) in &pairs {
            b.add(k, v);
        }
        let block = Block::decode(b.finish()).unwrap();
        let data_len = block.entry_data().len();

        let mut consumed = 0usize;
        let mut pos = 0usize;
        let data = block.entry_data();
        while pos < data_len {
            let size = encoded_entry_size(data, pos);
            pos += size;
            consumed += size;
        }
        assert_eq!(consumed, data_len);
    }
}
