//! Bloom filter used by SSTables for fast negative point lookups.
//!
//! Implementation: double-hashing with xxh3. The 64-bit xxh3 hash is split
//! into two 32-bit halves (`h1`, `h2`); the `i`-th hash function is
//! `h1 + i * h2`, which approximates independent hash functions closely
//! enough for Bloom-filter purposes (see Kirsch & Mitzenmacher, 2006).

/// On-disk Bloom filter. Stores a packed bit array plus the number of hash
/// functions to apply per key.
pub(crate) struct BloomFilter {
    bits: Vec<u8>,
    num_hashes: u32,
}

impl BloomFilter {
    pub(crate) fn new(bits: Vec<u8>, num_hashes: u32) -> Self {
        Self { bits, num_hashes }
    }

    /// `true` if the key *might* be in the set; `false` means definitely not.
    pub(crate) fn may_contain(&self, key: &[u8]) -> bool {
        if self.bits.is_empty() {
            return true;
        }
        let num_bits = self.bits.len() * 8;
        let h = xxhash_rust::xxh3::xxh3_64(key);
        let h1 = h as u32;
        let h2 = (h >> 32) as u32;

        for i in 0..self.num_hashes {
            let bit_pos = (h1.wrapping_add(h2.wrapping_mul(i))) as usize % num_bits;
            if self.bits[bit_pos / 8] & (1 << (bit_pos % 8)) == 0 {
                return false;
            }
        }
        true
    }
}

/// Accumulates keys and materializes a [`BloomFilter`] sized by `bits_per_key`.
pub(crate) struct BloomFilterBuilder {
    keys: Vec<Vec<u8>>,
    bits_per_key: usize,
}

impl BloomFilterBuilder {
    pub(crate) fn new(bits_per_key: usize) -> Self {
        Self {
            keys: Vec::new(),
            bits_per_key,
        }
    }

    pub(crate) fn add_key(&mut self, key: &[u8]) {
        self.keys.push(key.to_vec());
    }

    pub(crate) fn build(self) -> BloomFilter {
        if self.keys.is_empty() {
            return BloomFilter::new(Vec::new(), 0);
        }

        let num_bits = std::cmp::max(self.keys.len() * self.bits_per_key, 64);
        let num_bytes = num_bits.div_ceil(8);
        let num_bits = num_bytes * 8;
        // Optimal hash count: bits_per_key * ln(2) ≈ bits_per_key * 0.69
        let num_hashes = std::cmp::max((self.bits_per_key as f64 * 0.69) as u32, 1);
        let num_hashes = std::cmp::min(num_hashes, 30);

        let mut bits = vec![0u8; num_bytes];

        for key in &self.keys {
            let h = xxhash_rust::xxh3::xxh3_64(key);
            let h1 = h as u32;
            let h2 = (h >> 32) as u32;

            for i in 0..num_hashes {
                let bit_pos = (h1.wrapping_add(h2.wrapping_mul(i))) as usize % num_bits;
                bits[bit_pos / 8] |= 1 << (bit_pos % 8);
            }
        }

        BloomFilter::new(bits, num_hashes)
    }
}

/// Serialize a bloom filter to a byte buffer (for writing to an SSTable).
pub(crate) fn encode_bloom_block(bloom: &BloomFilter) -> Vec<u8> {
    let mut data = Vec::with_capacity(4 + bloom.bits.len());
    data.extend_from_slice(&bloom.num_hashes.to_le_bytes());
    data.extend_from_slice(&bloom.bits);
    data
}

/// Parse a bloom filter block produced by [`encode_bloom_block`].
pub(crate) fn decode_bloom_block(data: &[u8]) -> BloomFilter {
    if data.len() < 4 {
        return BloomFilter::new(Vec::new(), 0);
    }
    let num_hashes = u32::from_le_bytes(data[0..4].try_into().unwrap());
    let bits = data[4..].to_vec();
    BloomFilter::new(bits, num_hashes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bloom_filter() {
        let mut builder = BloomFilterBuilder::new(10);
        for i in 0..100 {
            builder.add_key(format!("key_{}", i).as_bytes());
        }
        let bloom = builder.build();

        for i in 0..100 {
            assert!(bloom.may_contain(format!("key_{}", i).as_bytes()));
        }

        let mut false_positives = 0;
        for i in 100..200 {
            if bloom.may_contain(format!("key_{}", i).as_bytes()) {
                false_positives += 1;
            }
        }
        // With 10 bits/key, expected false positive rate is ~1%.
        assert!(
            false_positives < 10,
            "too many false positives: {}",
            false_positives
        );
    }
}
