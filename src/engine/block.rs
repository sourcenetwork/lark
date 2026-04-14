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

            if current_key.as_slice() >= target {
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
            if key.as_slice() < target {
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

        let (key, value) = decode_block_entry(self.data, self.pos, &self.current_key);
        let entry_size = encoded_entry_size(self.data, self.pos);
        self.pos += entry_size;
        self.current_key = key.clone();

        Some((key, value))
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

fn encoded_entry_size(data: &[u8], pos: usize) -> usize {
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

fn decode_varint(data: &[u8]) -> (u64, usize) {
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

    fn build_block(pairs: &[(&[u8], &[u8])]) -> Block {
        let mut builder = BlockBuilder::new(4);
        for (k, v) in pairs {
            builder.add(k, v);
        }
        Block::decode(builder.finish()).unwrap()
    }

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
}
