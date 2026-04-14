//! SSTable file format: sorted on-disk tables of MVCC-encoded key-value pairs.
//!
//! Layout:
//! ```text
//! [data block 0][data block 1]...[data block n][bloom block][index block][footer (48 B)]
//! ```
//!
//! Data blocks store **internal keys** — `user_key || !seq || value_type` —
//! sorted so that newer versions of the same user key appear before older
//! ones. Tombstones are first-class entries; reads that land on a tombstone
//! at or before `snapshot_seq` return "deleted" and suppress older versions
//! in lower levels. The bloom filter is keyed on user keys so point lookups
//! can short-circuit regardless of which seq a reader is asking for.

use std::fs::{self, File};
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::block::{Block, BlockBuilder, BlockHandle, RESTART_INTERVAL};
use super::block_cache::BlockCache;
use super::bloom::{decode_bloom_block, encode_bloom_block, BloomFilter, BloomFilterBuilder};
use super::internal_key::{decode_internal_key, lookup_key, user_key_of, VALUE_TYPE_DELETION};

/// SSTable magic number: "LARKSST\0".
const MAGIC: u64 = 0x4C41524B_53535400;

/// Footer size in bytes.
const FOOTER_SIZE: usize = 48;

const COMPRESSION_NONE: u8 = 0x00;
const COMPRESSION_LZ4: u8 = 0x01;

/// SSTable footer (fixed 48 bytes, written at end of file).
#[derive(Debug)]
struct Footer {
    bloom_offset: u64,
    bloom_size: u64,
    index_offset: u64,
    index_size: u64,
    num_entries: u64,
    magic: u64,
}

impl Footer {
    fn encode(&self) -> [u8; FOOTER_SIZE] {
        let mut buf = [0u8; FOOTER_SIZE];
        buf[0..8].copy_from_slice(&self.bloom_offset.to_le_bytes());
        buf[8..16].copy_from_slice(&self.bloom_size.to_le_bytes());
        buf[16..24].copy_from_slice(&self.index_offset.to_le_bytes());
        buf[24..32].copy_from_slice(&self.index_size.to_le_bytes());
        buf[32..40].copy_from_slice(&self.num_entries.to_le_bytes());
        buf[40..48].copy_from_slice(&self.magic.to_le_bytes());
        buf
    }

    fn decode(buf: &[u8; FOOTER_SIZE]) -> io::Result<Self> {
        let magic = u64::from_le_bytes(buf[40..48].try_into().unwrap());
        if magic != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid SSTable magic number",
            ));
        }
        Ok(Self {
            bloom_offset: u64::from_le_bytes(buf[0..8].try_into().unwrap()),
            bloom_size: u64::from_le_bytes(buf[8..16].try_into().unwrap()),
            index_offset: u64::from_le_bytes(buf[16..24].try_into().unwrap()),
            index_size: u64::from_le_bytes(buf[24..32].try_into().unwrap()),
            num_entries: u64::from_le_bytes(buf[32..40].try_into().unwrap()),
            magic,
        })
    }
}

/// Metadata for an SSTable file, tracked by the manifest.
///
/// `smallest_key` and `largest_key` are **user keys**, used by the engine
/// and compaction for overlap checks. The level is tracked positionally
/// in [`super::manifest::Version::levels`], not here.
#[derive(Debug, Clone)]
pub(crate) struct SsTableMeta {
    pub(crate) file_id: u64,
    pub(crate) smallest_key: Vec<u8>,
    pub(crate) largest_key: Vec<u8>,
    pub(crate) file_size: u64,
    pub(crate) num_entries: u64,
}

/// Index entry: the **last internal key** of the data block plus its handle.
#[derive(Debug, Clone)]
struct IndexEntry {
    key: Vec<u8>,
    handle: BlockHandle,
}

/// A user-key / optional-value pair as returned by [`SsTableReader::range_iter`].
/// `None` indicates a tombstone.
pub(crate) type ScanEntry = (Vec<u8>, Option<Vec<u8>>);

/// Result of looking up a user key in a single SSTable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum LookupResult {
    /// No visible version for this user key in this SSTable.
    NotInTable,
    /// Found a value at or before the requested snapshot.
    Found(Vec<u8>),
    /// Found a tombstone at or before the requested snapshot — the caller
    /// must treat the key as deleted and stop searching lower levels.
    FoundTombstone,
}

/// Summary of a finished SSTable, returned by [`SsTableWriter::finish`].
#[derive(Debug)]
pub(crate) struct SsTableWriteSummary {
    pub(crate) smallest_user_key: Vec<u8>,
    pub(crate) largest_user_key: Vec<u8>,
    pub(crate) num_entries: u64,
}

fn encode_index_block(entries: &[(Vec<u8>, BlockHandle)]) -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    for (key, handle) in entries {
        data.extend_from_slice(&(key.len() as u32).to_le_bytes());
        data.extend_from_slice(key);
        data.extend_from_slice(&handle.offset.to_le_bytes());
        data.extend_from_slice(&handle.size.to_le_bytes());
    }
    data
}

fn decode_index_block(data: &[u8]) -> io::Result<Vec<IndexEntry>> {
    if data.len() < 4 {
        return Ok(Vec::new());
    }

    let num_entries = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
    let mut entries = Vec::with_capacity(num_entries);
    let mut pos = 4;

    for _ in 0..num_entries {
        if pos + 4 > data.len() {
            break;
        }
        let key_len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;

        if pos + key_len + 16 > data.len() {
            break;
        }
        let key = data[pos..pos + key_len].to_vec();
        pos += key_len;

        let offset = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
        pos += 8;
        let size = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
        pos += 8;

        entries.push(IndexEntry {
            key,
            handle: BlockHandle { offset, size },
        });
    }

    Ok(entries)
}

// ─── Writer ─────────────────────────────────────────────────────────────────

/// Builds an SSTable file from sorted **internal keys**. Callers are
/// responsible for supplying keys in ascending internal-key order (newer
/// versions of a user key appear before older ones).
pub(crate) struct SsTableWriter {
    writer: BufWriter<File>,
    block_builder: BlockBuilder,
    index_entries: Vec<(Vec<u8>, BlockHandle)>,
    bloom_builder: BloomFilterBuilder,
    block_size: usize,
    current_offset: u64,
    num_entries: u64,
    last_internal_key: Vec<u8>,
    smallest_user_key: Option<Vec<u8>>,
    largest_user_key: Option<Vec<u8>>,
    last_bloom_user_key: Vec<u8>,
    compression: bool,
}

impl SsTableWriter {
    pub(crate) fn new(
        path: &Path,
        block_size: usize,
        bloom_bits_per_key: usize,
        compression: bool,
    ) -> io::Result<Self> {
        let file = File::create(path)?;
        Ok(Self {
            writer: BufWriter::new(file),
            block_builder: BlockBuilder::new(RESTART_INTERVAL),
            index_entries: Vec::new(),
            bloom_builder: BloomFilterBuilder::new(bloom_bits_per_key),
            block_size,
            current_offset: 0,
            num_entries: 0,
            last_internal_key: Vec::new(),
            smallest_user_key: None,
            largest_user_key: None,
            last_bloom_user_key: Vec::new(),
            compression,
        })
    }

    /// Add an `(internal_key, value)` pair. Internal keys must arrive in
    /// sorted order. For tombstones, pass an empty value.
    pub(crate) fn add(&mut self, internal_key: &[u8], value: &[u8]) -> io::Result<()> {
        let user_key = user_key_of(internal_key);

        // Only add each distinct user key to the bloom filter once.
        if user_key != self.last_bloom_user_key.as_slice() {
            self.bloom_builder.add_key(user_key);
            self.last_bloom_user_key = user_key.to_vec();
        }

        self.block_builder.add(internal_key, value);
        self.last_internal_key = internal_key.to_vec();
        self.num_entries += 1;

        if self.smallest_user_key.is_none() {
            self.smallest_user_key = Some(user_key.to_vec());
        }
        match &mut self.largest_user_key {
            Some(existing) if existing.as_slice() == user_key => {}
            slot => *slot = Some(user_key.to_vec()),
        }

        if self.block_builder.estimated_size() >= self.block_size {
            self.flush_block()?;
        }

        Ok(())
    }

    /// Finalize the SSTable. Returns `None` if no entries were added
    /// (caller should delete the empty file).
    pub(crate) fn finish(mut self) -> io::Result<Option<SsTableWriteSummary>> {
        if self.num_entries == 0 {
            return Ok(None);
        }

        if !self.block_builder.is_empty() {
            self.flush_block()?;
        }

        let bloom = self.bloom_builder.build();
        let bloom_data = encode_bloom_block(&bloom);
        let bloom_offset = self.current_offset;
        self.writer.write_all(&bloom_data)?;
        self.current_offset += bloom_data.len() as u64;

        let index_data = encode_index_block(&self.index_entries);
        let index_offset = self.current_offset;
        self.writer.write_all(&index_data)?;
        self.current_offset += index_data.len() as u64;

        let footer = Footer {
            bloom_offset,
            bloom_size: bloom_data.len() as u64,
            index_offset,
            index_size: index_data.len() as u64,
            num_entries: self.num_entries,
            magic: MAGIC,
        };
        self.writer.write_all(&footer.encode())?;
        self.writer.flush()?;

        Ok(Some(SsTableWriteSummary {
            smallest_user_key: self.smallest_user_key.unwrap(),
            largest_user_key: self.largest_user_key.unwrap(),
            num_entries: self.num_entries,
        }))
    }

    fn flush_block(&mut self) -> io::Result<()> {
        let last_internal = self.last_internal_key.clone();
        let block_builder =
            std::mem::replace(&mut self.block_builder, BlockBuilder::new(RESTART_INTERVAL));
        let raw_data = block_builder.finish();

        let block_offset = self.current_offset;

        if self.compression {
            let compressed = lz4_flex::compress_prepend_size(&raw_data);
            let checksum = xxhash_rust::xxh3::xxh3_64(&compressed) as u32;

            self.writer.write_all(&[COMPRESSION_LZ4])?;
            self.writer.write_all(&compressed)?;
            self.writer.write_all(&checksum.to_le_bytes())?;

            let total = 1 + compressed.len() + 4;
            self.current_offset += total as u64;
        } else {
            let checksum = xxhash_rust::xxh3::xxh3_64(&raw_data) as u32;

            self.writer.write_all(&[COMPRESSION_NONE])?;
            self.writer.write_all(&raw_data)?;
            self.writer.write_all(&checksum.to_le_bytes())?;

            let total = 1 + raw_data.len() + 4;
            self.current_offset += total as u64;
        }

        let block_size = self.current_offset - block_offset;
        self.index_entries.push((
            last_internal,
            BlockHandle {
                offset: block_offset,
                size: block_size,
            },
        ));

        Ok(())
    }
}

// ─── Reader ─────────────────────────────────────────────────────────────────

/// Reads an SSTable file. Index and bloom filter are loaded into memory at open.
pub(crate) struct SsTableReader {
    path: PathBuf,
    pub(crate) file_id: u64,
    index: Vec<IndexEntry>,
    bloom: BloomFilter,
}

impl SsTableReader {
    /// Open an SSTable file and load index + bloom into memory.
    pub(crate) fn open(path: &Path, file_id: u64) -> io::Result<Self> {
        let mut file = File::open(path)?;
        let file_size = file.metadata()?.len();

        if file_size < FOOTER_SIZE as u64 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "SSTable file too small",
            ));
        }

        file.seek(SeekFrom::End(-(FOOTER_SIZE as i64)))?;
        let mut footer_buf = [0u8; FOOTER_SIZE];
        file.read_exact(&mut footer_buf)?;
        let footer = Footer::decode(&footer_buf)?;

        file.seek(SeekFrom::Start(footer.bloom_offset))?;
        let mut bloom_data = vec![0u8; footer.bloom_size as usize];
        file.read_exact(&mut bloom_data)?;
        let bloom = decode_bloom_block(&bloom_data);

        file.seek(SeekFrom::Start(footer.index_offset))?;
        let mut index_data = vec![0u8; footer.index_size as usize];
        file.read_exact(&mut index_data)?;
        let index = decode_index_block(&index_data)?;

        Ok(Self {
            path: path.to_path_buf(),
            file_id,
            index,
            bloom,
        })
    }

    /// Point lookup for `user_key` visible at `snapshot_seq`.
    pub(crate) fn get(
        &self,
        user_key: &[u8],
        snapshot_seq: u64,
        cache: &BlockCache,
    ) -> io::Result<LookupResult> {
        if !self.bloom.may_contain(user_key) {
            return Ok(LookupResult::NotInTable);
        }

        let search_key = lookup_key(user_key, snapshot_seq);
        let block_idx = match self
            .index
            .binary_search_by(|e| e.key.as_slice().cmp(&search_key))
        {
            Ok(i) => i,
            Err(i) => {
                if i >= self.index.len() {
                    return Ok(LookupResult::NotInTable);
                }
                i
            }
        };

        let entry = &self.index[block_idx];
        let block = self.read_block(entry.handle, cache)?;
        match block.seek_ge(&search_key) {
            Some((ik, value)) => {
                let (uk, _seq, vt) = decode_internal_key(&ik);
                if uk != user_key {
                    Ok(LookupResult::NotInTable)
                } else if vt == VALUE_TYPE_DELETION {
                    Ok(LookupResult::FoundTombstone)
                } else {
                    Ok(LookupResult::Found(value))
                }
            }
            None => Ok(LookupResult::NotInTable),
        }
    }

    /// Iterate entries visible at `snapshot_seq` in the user-key range
    /// `[start, end)`. Results are deduplicated by user key, keeping the
    /// most recent visible version. Tombstones are reported as `None` so
    /// callers can suppress older versions living in lower levels.
    pub(crate) fn range_iter(
        &self,
        start: Option<&[u8]>,
        end: Option<&[u8]>,
        snapshot_seq: u64,
        cache: &BlockCache,
    ) -> io::Result<Vec<ScanEntry>> {
        let start_block = if let Some(start_user_key) = start {
            // The smallest internal key for `start_user_key` is the one with
            // seq = u64::MAX (because we encode !seq). Binary-searching for
            // the first index entry whose last internal key is >= that puts
            // us on the first block that may contain the range.
            let probe = lookup_key(start_user_key, u64::MAX);
            match self
                .index
                .binary_search_by(|e| e.key.as_slice().cmp(&probe))
            {
                Ok(i) => i,
                Err(i) => {
                    if i >= self.index.len() {
                        return Ok(Vec::new());
                    }
                    i
                }
            }
        } else {
            0
        };

        let mut result: Vec<ScanEntry> = Vec::new();
        let mut last_user_key: Option<Vec<u8>> = None;

        for entry in &self.index[start_block..] {
            let block = self.read_block(entry.handle, cache)?;
            for (ik, value) in block.iter() {
                let (uk, seq, vt) = decode_internal_key(&ik);

                if let Some(e) = end {
                    if uk >= e {
                        return Ok(result);
                    }
                }
                if let Some(s) = start {
                    if uk < s {
                        continue;
                    }
                }
                if seq > snapshot_seq {
                    continue;
                }
                if last_user_key.as_deref() == Some(uk) {
                    continue;
                }
                last_user_key = Some(uk.to_vec());

                if vt == VALUE_TYPE_DELETION {
                    result.push((uk.to_vec(), None));
                } else {
                    result.push((uk.to_vec(), Some(value)));
                }
            }
        }

        Ok(result)
    }

    /// Read every entry in internal-key order with no dedup or filtering.
    /// Used by compaction to merge tables without losing versions.
    pub(crate) fn iter_internal(&self, cache: &BlockCache) -> io::Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let mut result = Vec::new();
        for entry in &self.index {
            let block = self.read_block(entry.handle, cache)?;
            for (ik, value) in block.iter() {
                result.push((ik, value));
            }
        }
        Ok(result)
    }

    fn read_block(&self, handle: BlockHandle, cache: &BlockCache) -> io::Result<Arc<Block>> {
        if let Some(block) = cache.get(self.file_id, handle.offset) {
            return Ok(block);
        }

        let mut file = File::open(&self.path)?;
        file.seek(SeekFrom::Start(handle.offset))?;

        let mut block_data = vec![0u8; handle.size as usize];
        file.read_exact(&mut block_data)?;

        // Frame: [compression_type: u8][payload][checksum: u32]
        let compression_type = block_data[0];
        let checksum_offset = block_data.len() - 4;
        let stored_checksum = u32::from_le_bytes(block_data[checksum_offset..].try_into().unwrap());
        let compressed_data = &block_data[1..checksum_offset];

        let computed_checksum = xxhash_rust::xxh3::xxh3_64(compressed_data) as u32;
        if stored_checksum != computed_checksum {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "block checksum mismatch",
            ));
        }

        let raw_data = match compression_type {
            COMPRESSION_NONE => compressed_data.to_vec(),
            COMPRESSION_LZ4 => lz4_flex::decompress_size_prepended(compressed_data)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?,
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unknown compression type: {}", compression_type),
                ))
            }
        };

        let block = Arc::new(Block::decode(raw_data)?);
        cache.insert(self.file_id, handle.offset, Arc::clone(&block));
        Ok(block)
    }
}

/// Format an SSTable filename from a numeric ID.
pub(crate) fn sst_filename(id: u64) -> String {
    format!("{:06}.sst", id)
}

/// Delete an SSTable file.
pub(crate) fn remove_sst(path: &Path) -> io::Result<()> {
    fs::remove_file(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::internal_key::{encode_internal_key, VALUE_TYPE_VALUE};
    use tempfile::TempDir;

    fn ik(key: &[u8], seq: u64) -> Vec<u8> {
        encode_internal_key(key, seq, VALUE_TYPE_VALUE)
    }

    fn tombstone(key: &[u8], seq: u64) -> Vec<u8> {
        encode_internal_key(key, seq, VALUE_TYPE_DELETION)
    }

    #[test]
    fn test_sstable_write_read() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.sst");
        let cache = BlockCache::new(64 * 1024 * 1024);

        {
            let mut writer = SsTableWriter::new(&path, 4096, 10, true).unwrap();
            // Add 100 distinct user keys at seq=1 in sorted order.
            for i in 0..100 {
                let user_key = format!("key_{:04}", i);
                let value = format!("value_{}", i);
                writer
                    .add(&ik(user_key.as_bytes(), 1), value.as_bytes())
                    .unwrap();
            }
            let summary = writer.finish().unwrap().unwrap();
            assert_eq!(summary.num_entries, 100);
            assert_eq!(summary.smallest_user_key, b"key_0000");
            assert_eq!(summary.largest_user_key, b"key_0099");
        }

        let reader = SsTableReader::open(&path, 1).unwrap();

        // Point lookups at u64::MAX see the latest version.
        assert_eq!(
            reader.get(b"key_0042", u64::MAX, &cache).unwrap(),
            LookupResult::Found(b"value_42".to_vec())
        );
        assert_eq!(
            reader.get(b"nonexistent", u64::MAX, &cache).unwrap(),
            LookupResult::NotInTable
        );
    }

    #[test]
    fn test_sstable_mvcc_visibility() {
        // Two versions of the same key at seq=1 and seq=3, plus a tombstone
        // at seq=5. A snapshot at seq=4 must see the seq=3 version; seq=2
        // must see seq=1; seq=6 must see the tombstone.
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("mvcc.sst");
        let cache = BlockCache::new(1024 * 1024);

        {
            let mut writer = SsTableWriter::new(&path, 4096, 10, false).unwrap();
            // Must be written in internal-key order: newest seq first.
            writer.add(&tombstone(b"k", 5), b"").unwrap();
            writer.add(&ik(b"k", 3), b"v3").unwrap();
            writer.add(&ik(b"k", 1), b"v1").unwrap();
            writer.finish().unwrap().unwrap();
        }

        let reader = SsTableReader::open(&path, 7).unwrap();
        assert_eq!(
            reader.get(b"k", 6, &cache).unwrap(),
            LookupResult::FoundTombstone
        );
        assert_eq!(
            reader.get(b"k", 4, &cache).unwrap(),
            LookupResult::Found(b"v3".to_vec())
        );
        assert_eq!(
            reader.get(b"k", 2, &cache).unwrap(),
            LookupResult::Found(b"v1".to_vec())
        );
        assert_eq!(
            reader.get(b"k", 0, &cache).unwrap(),
            LookupResult::NotInTable
        );
    }

    #[test]
    fn test_sstable_range_iter_dedup_and_tombstones() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("range.sst");
        let cache = BlockCache::new(1024 * 1024);

        {
            let mut writer = SsTableWriter::new(&path, 4096, 10, false).unwrap();
            writer.add(&ik(b"a", 1), b"a1").unwrap();
            writer.add(&tombstone(b"b", 4), b"").unwrap();
            writer.add(&ik(b"b", 2), b"b2").unwrap();
            writer.add(&ik(b"c", 3), b"c3").unwrap();
            writer.finish().unwrap().unwrap();
        }

        let reader = SsTableReader::open(&path, 42).unwrap();
        // At seq=5 we see: a=a1, b=tombstone, c=c3
        let items = reader.range_iter(None, None, 5, &cache).unwrap();
        assert_eq!(items.len(), 3);
        assert_eq!(items[0], (b"a".to_vec(), Some(b"a1".to_vec())));
        assert_eq!(items[1], (b"b".to_vec(), None));
        assert_eq!(items[2], (b"c".to_vec(), Some(b"c3".to_vec())));

        // At seq=3 we see: a=a1, b=b2, c=c3 (tombstone is in the future)
        let items = reader.range_iter(None, None, 3, &cache).unwrap();
        assert_eq!(items.len(), 3);
        assert_eq!(items[1], (b"b".to_vec(), Some(b"b2".to_vec())));
    }

    #[test]
    fn test_sstable_no_compression() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("nocomp.sst");
        let cache = BlockCache::new(1024 * 1024);

        {
            let mut writer = SsTableWriter::new(&path, 4096, 10, false).unwrap();
            writer.add(&ik(b"hello", 1), b"world").unwrap();
            writer.add(&ik(b"test", 1), b"data").unwrap();
            writer.finish().unwrap().unwrap();
        }

        let reader = SsTableReader::open(&path, 2).unwrap();
        assert_eq!(
            reader.get(b"hello", u64::MAX, &cache).unwrap(),
            LookupResult::Found(b"world".to_vec())
        );
        assert_eq!(
            reader.get(b"test", u64::MAX, &cache).unwrap(),
            LookupResult::Found(b"data".to_vec())
        );
    }
}
