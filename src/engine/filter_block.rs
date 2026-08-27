//! The filter region of an SSTable: the user-key bloom filter plus the
//! optional prefix bloom filter, decoded as one unit.
//!
//! On-disk format, written by `SsTableWriter::finish`:
//!
//! ```text
//! [prefix_bloom_len: u64 LE][prefix bloom bytes][user-key bloom bytes]
//! ```
//!
//! A `prefix_bloom_len` of `0` means the file was written without a
//! prefix extractor (or the extractor yielded no prefixes). Files that
//! predate prefix blooms wrote the same `0` marker, so the format is
//! backward compatible.
//!
//! The two filters travel together because the footer describes them as
//! one region: one region, one decode, one cache entry, one charge.

use std::io;

use super::bloom::{BloomFilter, decode_bloom_block};

/// Header bytes in front of the prefix bloom.
const PREFIX_LEN_HEADER: usize = 8;

/// Smallest legal encoded bloom block (`decode_bloom_block` reads a
/// 4-byte `num_hashes` header).
const MIN_BLOOM_BLOCK: usize = 4;

/// Both bloom filters of one SSTable.
pub(crate) struct FilterBlock {
    user: BloomFilter,
    prefix: Option<BloomFilter>,
    charge: usize,
}

impl FilterBlock {
    /// Decode the filter region read from an SSTable's bloom range.
    ///
    /// Rejects a region too short to hold the length header, a prefix
    /// length that runs past the region, and either bloom block being
    /// too short to carry its own header.
    pub(crate) fn decode(region: &[u8]) -> io::Result<Self> {
        if region.len() < PREFIX_LEN_HEADER {
            return Err(invalid_data(
                "bloom region too short for prefix-bloom length header",
            ));
        }
        let prefix_len = usize::try_from(u64::from_le_bytes(
            region[0..PREFIX_LEN_HEADER].try_into().unwrap(),
        ))
        .map_err(|_| invalid_data("prefix bloom length is too large to address"))?;
        let user_offset = PREFIX_LEN_HEADER
            .checked_add(prefix_len)
            .ok_or_else(|| invalid_data("prefix bloom length overflows"))?;
        if user_offset > region.len() {
            return Err(invalid_data("prefix bloom length exceeds bloom region"));
        }
        if prefix_len > 0 && prefix_len < MIN_BLOOM_BLOCK {
            return Err(invalid_data("prefix bloom block too short"));
        }
        if region.len() - user_offset < MIN_BLOOM_BLOCK {
            return Err(invalid_data("user bloom block too short"));
        }

        let prefix =
            (prefix_len > 0).then(|| decode_bloom_block(&region[PREFIX_LEN_HEADER..user_offset]));
        let user = decode_bloom_block(&region[user_offset..]);

        Ok(Self {
            user,
            prefix,
            // The decoded filters hold the region's bit arrays minus the
            // two 4-byte `num_hashes` headers and the 8-byte length
            // header, so the region length is the charge to within a
            // handful of bytes and never under-reports.
            charge: std::mem::size_of::<Self>() + region.len(),
        })
    }

    /// Whether this SSTable *might* contain `user_key`. `false` means
    /// definitely not.
    pub(crate) fn may_contain(&self, user_key: &[u8]) -> bool {
        self.user.may_contain(user_key)
    }

    /// Whether this SSTable *might* contain a key with this prefix.
    /// Conservatively `true` when the file carries no prefix bloom,
    /// since there is no negative information to act on.
    pub(crate) fn may_have_prefix(&self, prefix: &[u8]) -> bool {
        match &self.prefix {
            Some(filter) => filter.may_contain(prefix),
            None => true,
        }
    }

    /// Heap bytes held, for block-cache charging.
    pub(crate) fn charge(&self) -> usize {
        self.charge
    }
}

impl std::fmt::Debug for FilterBlock {
    /// Shape only: a bloom filter's bits are not readable and printing
    /// them would leak nothing useful and a great deal of noise.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FilterBlock")
            .field("has_prefix_bloom", &self.prefix.is_some())
            .field("charge", &self.charge)
            .finish()
    }
}

fn invalid_data(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::bloom::{BloomFilterBuilder, encode_bloom_block};

    fn region(prefix_keys: Option<&[&[u8]]>, user_keys: &[&[u8]]) -> Vec<u8> {
        let prefix_bytes = match prefix_keys {
            Some(keys) => {
                let mut builder = BloomFilterBuilder::new(10);
                for key in keys {
                    builder.add_key(key);
                }
                encode_bloom_block(&builder.build())
            }
            None => Vec::new(),
        };
        let mut builder = BloomFilterBuilder::new(10);
        for key in user_keys {
            builder.add_key(key);
        }
        let user_bytes = encode_bloom_block(&builder.build());

        let mut out = Vec::new();
        out.extend_from_slice(&(prefix_bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(&prefix_bytes);
        out.extend_from_slice(&user_bytes);
        out
    }

    #[test]
    fn decodes_region_without_prefix_bloom() {
        let bytes = region(None, &[b"alpha", b"beta"]);
        let filter = FilterBlock::decode(&bytes).unwrap();
        assert!(filter.may_contain(b"alpha"));
        assert!(filter.may_contain(b"beta"));
        assert!(
            filter.may_have_prefix(b"anything"),
            "no prefix bloom means no negative information"
        );
    }

    #[test]
    fn decodes_region_with_prefix_bloom() {
        let bytes = region(Some(&[b"pre"]), &[b"prefix-key"]);
        let filter = FilterBlock::decode(&bytes).unwrap();
        assert!(filter.may_contain(b"prefix-key"));
        assert!(filter.may_have_prefix(b"pre"));
    }

    #[test]
    fn rejects_short_region() {
        for len in 0..PREFIX_LEN_HEADER {
            let err = FilterBlock::decode(&vec![0u8; len]).expect_err("short region");
            assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        }
    }

    #[test]
    fn rejects_prefix_length_past_region() {
        let mut bytes = region(None, &[b"k"]);
        let claimed = bytes.len() as u64 + 1;
        bytes[0..8].copy_from_slice(&claimed.to_le_bytes());
        let err = FilterBlock::decode(&bytes).expect_err("prefix past end");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn rejects_truncated_prefix_bloom() {
        let mut bytes = region(None, &[b"k"]);
        bytes[0..8].copy_from_slice(&2u64.to_le_bytes());
        let err = FilterBlock::decode(&bytes).expect_err("prefix bloom too short");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn rejects_truncated_user_bloom() {
        let bytes = vec![0u8; PREFIX_LEN_HEADER + 3];
        let err = FilterBlock::decode(&bytes).expect_err("user bloom too short");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn charge_covers_the_whole_region() {
        let bytes = region(Some(&[b"pre"]), &[b"a", b"b", b"c"]);
        let filter = FilterBlock::decode(&bytes).unwrap();
        assert!(filter.charge() >= bytes.len());
    }
}
