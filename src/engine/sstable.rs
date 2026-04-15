//! SSTable file format: sorted on-disk tables of MVCC-encoded key-value pairs.
//!
//! Layout:
//! ```text
//! [data block 0][data block 1]...[data block n][range_tombstone block][bloom region][index block][footer (64 B)]
//! ```
//!
//! The bloom region is two blooms concatenated behind a length header:
//! `[prefix_bloom_len: u64 LE][prefix_bloom_bytes][user_key_bloom_bytes]`.
//! A zero length means the file was written without a prefix extractor.
//!
//! Data blocks store **internal keys** — `user_key || !seq || value_type` —
//! sorted so that newer versions of the same user key appear before older
//! ones. Tombstones are first-class entries; reads that land on a tombstone
//! at or before `snapshot_seq` return "deleted" and suppress older versions
//! in lower levels. The bloom filter is keyed on user keys so point lookups
//! can short-circuit regardless of which seq a reader is asking for.

use std::fs::{self, File};
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::Arc;

use parking_lot::Mutex;

use super::block::{Block, BlockBuilder, BlockHandle, RESTART_INTERVAL};
use super::block_cache::BlockCache;
use super::bloom::{decode_bloom_block, encode_bloom_block, BloomFilter, BloomFilterBuilder};
use super::internal_key::{decode_internal_key, lookup_key, user_key_of, VALUE_TYPE_DELETION};
use super::range_tombstone::{max_covering_seq, RangeTombstone};
use crate::options::{CompressionType, PrefixExtractor};

/// SSTable magic number: "LARKSST\x01" (format version 2 — added the
/// range-tombstone meta block at the bottom of the file).
const MAGIC: u64 = 0x4C41524B_53535401;

/// Footer size in bytes.
const FOOTER_SIZE: usize = 64;

const COMPRESSION_NONE: u8 = 0x00;
const COMPRESSION_LZ4: u8 = 0x01;
const COMPRESSION_SNAPPY: u8 = 0x02;

/// SSTable footer (fixed 64 bytes, written at end of file).
///
/// Layout on disk:
/// ```text
/// [data blocks][range_tombstone_block][bloom_block][index_block][footer]
/// ```
///
/// A `range_tombstone_size` of `0` means the SSTable has no range
/// tombstones; readers skip loading the block in that case.
#[derive(Debug)]
struct Footer {
    range_tombstone_offset: u64,
    range_tombstone_size: u64,
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
        buf[0..8].copy_from_slice(&self.range_tombstone_offset.to_le_bytes());
        buf[8..16].copy_from_slice(&self.range_tombstone_size.to_le_bytes());
        buf[16..24].copy_from_slice(&self.bloom_offset.to_le_bytes());
        buf[24..32].copy_from_slice(&self.bloom_size.to_le_bytes());
        buf[32..40].copy_from_slice(&self.index_offset.to_le_bytes());
        buf[40..48].copy_from_slice(&self.index_size.to_le_bytes());
        buf[48..56].copy_from_slice(&self.num_entries.to_le_bytes());
        buf[56..64].copy_from_slice(&self.magic.to_le_bytes());
        buf
    }

    fn decode(buf: &[u8; FOOTER_SIZE]) -> io::Result<Self> {
        let magic = u64::from_le_bytes(buf[56..64].try_into().unwrap());
        if magic != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid SSTable magic number",
            ));
        }
        Ok(Self {
            range_tombstone_offset: u64::from_le_bytes(buf[0..8].try_into().unwrap()),
            range_tombstone_size: u64::from_le_bytes(buf[8..16].try_into().unwrap()),
            bloom_offset: u64::from_le_bytes(buf[16..24].try_into().unwrap()),
            bloom_size: u64::from_le_bytes(buf[24..32].try_into().unwrap()),
            index_offset: u64::from_le_bytes(buf[32..40].try_into().unwrap()),
            index_size: u64::from_le_bytes(buf[40..48].try_into().unwrap()),
            num_entries: u64::from_le_bytes(buf[48..56].try_into().unwrap()),
            magic,
        })
    }
}

fn encode_range_tombstone_block(tombstones: &[RangeTombstone]) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&(tombstones.len() as u32).to_le_bytes());
    for rt in tombstones {
        buf.extend_from_slice(&(rt.start.len() as u32).to_le_bytes());
        buf.extend_from_slice(&rt.start);
        buf.extend_from_slice(&(rt.end.len() as u32).to_le_bytes());
        buf.extend_from_slice(&rt.end);
        buf.extend_from_slice(&rt.seq.to_le_bytes());
    }
    buf
}

fn decode_range_tombstone_block(data: &[u8]) -> io::Result<Vec<RangeTombstone>> {
    if data.is_empty() {
        return Ok(Vec::new());
    }
    if data.len() < 4 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "range tombstone block too short",
        ));
    }
    let count = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
    let mut out = Vec::with_capacity(count);
    let mut pos = 4;
    for _ in 0..count {
        if pos + 4 > data.len() {
            break;
        }
        let start_len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        if pos + start_len + 4 > data.len() {
            break;
        }
        let start = data[pos..pos + start_len].to_vec();
        pos += start_len;
        let end_len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        if pos + end_len + 8 > data.len() {
            break;
        }
        let end = data[pos..pos + end_len].to_vec();
        pos += end_len;
        let seq = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
        pos += 8;
        out.push(RangeTombstone::new(start, end, seq));
    }
    Ok(out)
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

/// A live SSTable: metadata plus an already-opened reader. Held by
/// [`super::manifest::Version`] so that as long as any version
/// referencing this file is alive, the underlying file descriptor stays
/// open — reads remain valid even after a concurrent compaction unlinks
/// the file from disk (the kernel keeps the inode alive via FD
/// refcounting).
///
/// `Arc<LiveSst>` is shared between versions and between a version and
/// any iterator built from it, so cloning is cheap.
pub(crate) struct LiveSst {
    pub(crate) meta: SsTableMeta,
    pub(crate) reader: Arc<SsTableReader>,
}

impl LiveSst {
    pub(crate) fn new(meta: SsTableMeta, reader: Arc<SsTableReader>) -> Arc<Self> {
        Arc::new(Self { meta, reader })
    }
}

/// Index entry: the **last internal key** of the data block plus its handle.
#[derive(Debug, Clone)]
struct IndexEntry {
    key: Vec<u8>,
    handle: BlockHandle,
}

/// Result of looking up a user key in a single SSTable.
///
/// The `seq` carried by [`LookupResult::Found`] / [`LookupResult::FoundTombstone`]
/// is the sequence number of the winning point entry, which the caller
/// compares against range-tombstone coverage from this and newer sources
/// to decide the final visibility.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum LookupResult {
    /// No visible version for this user key in this SSTable.
    NotInTable,
    /// Found a value at or before the requested snapshot.
    Found { seq: u64, value: Vec<u8> },
    /// Found a tombstone at or before the requested snapshot.
    FoundTombstone { seq: u64 },
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
    prefix_bloom_builder: Option<BloomFilterBuilder>,
    prefix_extractor: Option<Arc<dyn PrefixExtractor>>,
    last_prefix: Option<Vec<u8>>,
    range_tombstones: Vec<RangeTombstone>,
    block_size: usize,
    current_offset: u64,
    num_entries: u64,
    last_internal_key: Vec<u8>,
    smallest_user_key: Option<Vec<u8>>,
    largest_user_key: Option<Vec<u8>>,
    last_bloom_user_key: Vec<u8>,
    compression: CompressionType,
}

impl SsTableWriter {
    pub(crate) fn new(
        path: &Path,
        block_size: usize,
        bloom_bits_per_key: usize,
        compression: CompressionType,
        prefix_extractor: Option<Arc<dyn PrefixExtractor>>,
    ) -> io::Result<Self> {
        let file = File::create(path)?;
        let prefix_bloom_builder = prefix_extractor
            .as_ref()
            .map(|_| BloomFilterBuilder::new(bloom_bits_per_key));
        Ok(Self {
            writer: BufWriter::new(file),
            block_builder: BlockBuilder::new(RESTART_INTERVAL),
            index_entries: Vec::new(),
            bloom_builder: BloomFilterBuilder::new(bloom_bits_per_key),
            prefix_bloom_builder,
            prefix_extractor,
            last_prefix: None,
            range_tombstones: Vec::new(),
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

    /// Attach a range tombstone to the SSTable being built. Range
    /// tombstones are written into a dedicated meta block between the
    /// data blocks and the bloom block, so they're independent of the
    /// point-entry stream and do not affect `add`'s sort-order
    /// invariant.
    pub(crate) fn add_range_tombstone(&mut self, start: &[u8], end: &[u8], seq: u64) {
        self.range_tombstones
            .push(RangeTombstone::new(start.to_vec(), end.to_vec(), seq));
    }

    /// Add an `(internal_key, value)` pair. Internal keys must arrive in
    /// sorted order. For tombstones, pass an empty value.
    pub(crate) fn add(&mut self, internal_key: &[u8], value: &[u8]) -> io::Result<()> {
        let user_key = user_key_of(internal_key);

        // Only add each distinct user key to the bloom filter once.
        if user_key != self.last_bloom_user_key.as_slice() {
            self.bloom_builder.add_key(user_key);
            self.last_bloom_user_key = user_key.to_vec();

            // Prefix bloom: keys arrive in sorted user-key order, so
            // distinct prefixes form contiguous runs. Only hash each
            // distinct prefix once.
            if let (Some(extractor), Some(builder)) = (
                self.prefix_extractor.as_ref(),
                self.prefix_bloom_builder.as_mut(),
            ) {
                if let Some(prefix) = extractor.extract(user_key) {
                    let changed = match &self.last_prefix {
                        Some(p) => p.as_slice() != prefix,
                        None => true,
                    };
                    if changed {
                        builder.add_key(prefix);
                        self.last_prefix = Some(prefix.to_vec());
                    }
                }
            }
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

    /// Finalize the SSTable. Returns `None` only if **nothing** was
    /// added — no point entries and no range tombstones. A file with
    /// only range tombstones (and no point entries) is still a valid
    /// SSTable; its smallest/largest user key range is derived from
    /// the tombstone bounds instead of point-entry bounds.
    pub(crate) fn finish(mut self) -> io::Result<Option<SsTableWriteSummary>> {
        if self.num_entries == 0 && self.range_tombstones.is_empty() {
            return Ok(None);
        }

        if !self.block_builder.is_empty() {
            self.flush_block()?;
        }

        // Range tombstone meta block comes right after the data blocks
        // and before the bloom filter. A size of 0 means no tombstones.
        let range_tombstone_data = if self.range_tombstones.is_empty() {
            Vec::new()
        } else {
            encode_range_tombstone_block(&self.range_tombstones)
        };
        let range_tombstone_offset = self.current_offset;
        self.writer.write_all(&range_tombstone_data)?;
        self.current_offset += range_tombstone_data.len() as u64;

        // Bloom region layout:
        //   [prefix_bloom_len: u64 LE][prefix_bloom_bytes][user_key_bloom_bytes]
        //
        // A `prefix_bloom_len` of 0 means "no prefix bloom" — the file
        // was built without a prefix extractor (or there were zero
        // extractable prefixes). The reader keeps backward compatibility
        // with pre-prefix SSTables via the same 0 marker written
        // unconditionally.
        let prefix_bloom_data = match self.prefix_bloom_builder.take() {
            Some(builder) => encode_bloom_block(&builder.build()),
            None => Vec::new(),
        };
        let user_bloom = self.bloom_builder.build();
        let user_bloom_data = encode_bloom_block(&user_bloom);

        let bloom_offset = self.current_offset;
        let prefix_len_bytes = (prefix_bloom_data.len() as u64).to_le_bytes();
        self.writer.write_all(&prefix_len_bytes)?;
        self.writer.write_all(&prefix_bloom_data)?;
        self.writer.write_all(&user_bloom_data)?;
        let bloom_size = 8 + prefix_bloom_data.len() as u64 + user_bloom_data.len() as u64;
        self.current_offset += bloom_size;

        let index_data = encode_index_block(&self.index_entries);
        let index_offset = self.current_offset;
        self.writer.write_all(&index_data)?;
        self.current_offset += index_data.len() as u64;

        let footer = Footer {
            range_tombstone_offset,
            range_tombstone_size: range_tombstone_data.len() as u64,
            bloom_offset,
            bloom_size,
            index_offset,
            index_size: index_data.len() as u64,
            num_entries: self.num_entries,
            magic: MAGIC,
        };
        self.writer.write_all(&footer.encode())?;
        self.writer.flush()?;

        // Derive the file's declared user-key range. Prefer point
        // entries; fall back to the union of range-tombstone bounds
        // when the file has tombstones but no points.
        let (smallest_user_key, largest_user_key) =
            match (self.smallest_user_key.take(), self.largest_user_key.take()) {
                (Some(s), Some(l)) => (s, l),
                _ => {
                    let mut iter = self.range_tombstones.iter();
                    let first = iter.next().expect("checked non-empty above").clone();
                    let mut smallest = first.start.clone();
                    let mut largest = first.end.clone();
                    for rt in iter {
                        if rt.start < smallest {
                            smallest = rt.start.clone();
                        }
                        if rt.end > largest {
                            largest = rt.end.clone();
                        }
                    }
                    (smallest, largest)
                }
            };

        Ok(Some(SsTableWriteSummary {
            smallest_user_key,
            largest_user_key,
            num_entries: self.num_entries,
        }))
    }

    fn flush_block(&mut self) -> io::Result<()> {
        let last_internal = self.last_internal_key.clone();
        let block_builder =
            std::mem::replace(&mut self.block_builder, BlockBuilder::new(RESTART_INTERVAL));
        let raw_data = block_builder.finish();

        let block_offset = self.current_offset;

        let (codec_byte, payload) = match self.compression {
            CompressionType::None => (COMPRESSION_NONE, raw_data.clone()),
            CompressionType::Lz4 => (COMPRESSION_LZ4, lz4_flex::compress_prepend_size(&raw_data)),
            CompressionType::Snappy => {
                let mut encoder = snap::raw::Encoder::new();
                let compressed = encoder.compress_vec(&raw_data).map_err(|e| {
                    io::Error::new(io::ErrorKind::InvalidData, format!("snappy encode: {e}"))
                })?;
                // Prepend the original length so the reader can pre-size
                // its decode buffer — matches `lz4_flex::compress_prepend_size`.
                let mut framed = Vec::with_capacity(4 + compressed.len());
                framed.extend_from_slice(&(raw_data.len() as u32).to_le_bytes());
                framed.extend_from_slice(&compressed);
                (COMPRESSION_SNAPPY, framed)
            }
        };

        let checksum = xxhash_rust::xxh3::xxh3_64(&payload) as u32;
        self.writer.write_all(&[codec_byte])?;
        self.writer.write_all(&payload)?;
        self.writer.write_all(&checksum.to_le_bytes())?;
        let total = 1 + payload.len() + 4;
        self.current_offset += total as u64;

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

/// Reads an SSTable file. Index and bloom filter are loaded into memory at
/// open; the file handle is held for the reader's lifetime so concurrent
/// compaction unlinking the path cannot corrupt an in-progress read (the OS
/// keeps the bytes alive via file-descriptor refcounting).
pub(crate) struct SsTableReader {
    file: Mutex<File>,
    pub(crate) file_id: u64,
    index: Vec<IndexEntry>,
    bloom: BloomFilter,
    /// Optional prefix bloom filter. `None` when the file was built
    /// without a prefix extractor (or the extractor yielded no prefixes).
    /// A query against a reader without a prefix bloom conservatively
    /// returns `true` — the file might contain the prefix.
    prefix_bloom: Option<BloomFilter>,
    range_tombstones: Vec<RangeTombstone>,
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
        // Peel [prefix_bloom_len: u64 LE][prefix_bytes][user_bytes].
        if bloom_data.len() < 8 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "bloom region too short for prefix-bloom length header",
            ));
        }
        let prefix_len = u64::from_le_bytes(bloom_data[0..8].try_into().unwrap()) as usize;
        if 8 + prefix_len > bloom_data.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "prefix bloom length exceeds bloom region",
            ));
        }
        let prefix_bloom = if prefix_len == 0 {
            None
        } else {
            Some(decode_bloom_block(&bloom_data[8..8 + prefix_len]))
        };
        let bloom = decode_bloom_block(&bloom_data[8 + prefix_len..]);

        file.seek(SeekFrom::Start(footer.index_offset))?;
        let mut index_data = vec![0u8; footer.index_size as usize];
        file.read_exact(&mut index_data)?;
        let index = decode_index_block(&index_data)?;

        let range_tombstones = if footer.range_tombstone_size == 0 {
            Vec::new()
        } else {
            file.seek(SeekFrom::Start(footer.range_tombstone_offset))?;
            let mut rt_data = vec![0u8; footer.range_tombstone_size as usize];
            file.read_exact(&mut rt_data)?;
            decode_range_tombstone_block(&rt_data)?
        };

        Ok(Self {
            file: Mutex::new(file),
            file_id,
            index,
            bloom,
            prefix_bloom,
            range_tombstones,
        })
    }

    /// Whether this SSTable *might* contain a user key whose prefix
    /// equals `prefix`. Returns `true` conservatively when the file
    /// was built without a prefix bloom (no negative information
    /// available). Returns `false` only when the prefix bloom is
    /// present and positively rules the prefix out.
    pub(crate) fn may_have_prefix(&self, prefix: &[u8]) -> bool {
        match &self.prefix_bloom {
            Some(b) => b.may_contain(prefix),
            None => true,
        }
    }

    /// Largest seq of any range tombstone in this SSTable that covers
    /// `user_key` and is visible at `snapshot_seq`. Returns `0` when
    /// nothing covers it — `0` is safe because real seqs start at 1.
    pub(crate) fn covering_range_tombstone_seq(&self, user_key: &[u8], snapshot_seq: u64) -> u64 {
        if self.range_tombstones.is_empty() {
            return 0;
        }
        max_covering_seq(&self.range_tombstones, user_key, snapshot_seq)
    }

    /// Borrow this SSTable's range tombstones. Used by compaction to
    /// merge them into the output file and by the iterator to honor
    /// RT coverage during scans.
    pub(crate) fn range_tombstones(&self) -> &[RangeTombstone] {
        &self.range_tombstones
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
                let (uk, seq, vt) = decode_internal_key(&ik);
                if uk != user_key {
                    Ok(LookupResult::NotInTable)
                } else if vt == VALUE_TYPE_DELETION {
                    Ok(LookupResult::FoundTombstone { seq })
                } else {
                    Ok(LookupResult::Found { seq, value })
                }
            }
            None => Ok(LookupResult::NotInTable),
        }
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

    /// Number of data blocks in this table.
    pub(crate) fn num_blocks(&self) -> usize {
        self.index.len()
    }

    /// Find the first block whose last internal key is `>= target`. Used by
    /// the streaming iterator to seek to a position within this SSTable.
    pub(crate) fn seek_block(&self, target: &[u8]) -> Option<usize> {
        match self
            .index
            .binary_search_by(|e| e.key.as_slice().cmp(target))
        {
            Ok(i) => Some(i),
            Err(i) => {
                if i >= self.index.len() {
                    None
                } else {
                    Some(i)
                }
            }
        }
    }

    /// Materialize every entry in `block_idx` as a vector of `(internal_key,
    /// value)` pairs through the block cache. Used by the streaming iterator
    /// to drive forward through one block at a time.
    pub(crate) fn load_block_entries(
        &self,
        block_idx: usize,
        cache: &BlockCache,
    ) -> io::Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let handle = self.index[block_idx].handle;
        let block = self.read_block(handle, cache)?;
        Ok(block.iter().collect())
    }

    fn read_block(&self, handle: BlockHandle, cache: &BlockCache) -> io::Result<Arc<Block>> {
        if let Some(block) = cache.get(self.file_id, handle.offset) {
            return Ok(block);
        }

        let mut block_data = vec![0u8; handle.size as usize];
        {
            let mut file = self.file.lock();
            file.seek(SeekFrom::Start(handle.offset))?;
            file.read_exact(&mut block_data)?;
        }

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
            COMPRESSION_SNAPPY => {
                if compressed_data.len() < 4 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "snappy block too short",
                    ));
                }
                let raw_len =
                    u32::from_le_bytes(compressed_data[0..4].try_into().unwrap()) as usize;
                let mut decoder = snap::raw::Decoder::new();
                let mut out = vec![0u8; raw_len];
                let n = decoder
                    .decompress(&compressed_data[4..], &mut out)
                    .map_err(|e| {
                        io::Error::new(io::ErrorKind::InvalidData, format!("snappy decode: {e}"))
                    })?;
                out.truncate(n);
                out
            }
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
            let mut writer =
                SsTableWriter::new(&path, 4096, 10, CompressionType::Lz4, None).unwrap();
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
            LookupResult::Found {
                seq: 1,
                value: b"value_42".to_vec()
            }
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
            let mut writer =
                SsTableWriter::new(&path, 4096, 10, CompressionType::None, None).unwrap();
            // Must be written in internal-key order: newest seq first.
            writer.add(&tombstone(b"k", 5), b"").unwrap();
            writer.add(&ik(b"k", 3), b"v3").unwrap();
            writer.add(&ik(b"k", 1), b"v1").unwrap();
            writer.finish().unwrap().unwrap();
        }

        let reader = SsTableReader::open(&path, 7).unwrap();
        assert_eq!(
            reader.get(b"k", 6, &cache).unwrap(),
            LookupResult::FoundTombstone { seq: 5 }
        );
        assert_eq!(
            reader.get(b"k", 4, &cache).unwrap(),
            LookupResult::Found {
                seq: 3,
                value: b"v3".to_vec()
            }
        );
        assert_eq!(
            reader.get(b"k", 2, &cache).unwrap(),
            LookupResult::Found {
                seq: 1,
                value: b"v1".to_vec()
            }
        );
        assert_eq!(
            reader.get(b"k", 0, &cache).unwrap(),
            LookupResult::NotInTable
        );
    }

    #[test]
    fn test_sstable_no_compression() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("nocomp.sst");
        let cache = BlockCache::new(1024 * 1024);

        {
            let mut writer =
                SsTableWriter::new(&path, 4096, 10, CompressionType::None, None).unwrap();
            writer.add(&ik(b"hello", 1), b"world").unwrap();
            writer.add(&ik(b"test", 1), b"data").unwrap();
            writer.finish().unwrap().unwrap();
        }

        let reader = SsTableReader::open(&path, 2).unwrap();
        assert_eq!(
            reader.get(b"hello", u64::MAX, &cache).unwrap(),
            LookupResult::Found {
                seq: 1,
                value: b"world".to_vec()
            }
        );
        assert_eq!(
            reader.get(b"test", u64::MAX, &cache).unwrap(),
            LookupResult::Found {
                seq: 1,
                value: b"data".to_vec()
            }
        );
    }

    #[test]
    fn test_sstable_range_tombstones_roundtrip() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("rt.sst");

        {
            let mut writer =
                SsTableWriter::new(&path, 4096, 10, CompressionType::None, None).unwrap();
            writer.add(&ik(b"a", 1), b"v_a").unwrap();
            writer.add(&ik(b"m", 2), b"v_m").unwrap();
            writer.add(&ik(b"z", 3), b"v_z").unwrap();
            writer.add_range_tombstone(b"b", b"n", 10);
            writer.add_range_tombstone(b"p", b"y", 15);
            let summary = writer.finish().unwrap().unwrap();
            assert_eq!(summary.num_entries, 3);
        }

        let reader = SsTableReader::open(&path, 1).unwrap();
        assert_eq!(reader.range_tombstones().len(), 2);
        assert_eq!(reader.covering_range_tombstone_seq(b"a", 100), 0);
        assert_eq!(reader.covering_range_tombstone_seq(b"b", 100), 10);
        assert_eq!(reader.covering_range_tombstone_seq(b"m", 100), 10);
        assert_eq!(reader.covering_range_tombstone_seq(b"n", 100), 0);
        assert_eq!(reader.covering_range_tombstone_seq(b"p", 100), 15);
        assert_eq!(reader.covering_range_tombstone_seq(b"z", 100), 0);
        // RT at seq 15 is invisible to older snapshots.
        assert_eq!(reader.covering_range_tombstone_seq(b"p", 14), 0);
    }

    #[test]
    fn test_sstable_rt_only_no_points() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("rt_only.sst");

        {
            let mut writer =
                SsTableWriter::new(&path, 4096, 10, CompressionType::None, None).unwrap();
            writer.add_range_tombstone(b"aa", b"kk", 5);
            writer.add_range_tombstone(b"mm", b"zz", 7);
            let summary = writer.finish().unwrap().unwrap();
            assert_eq!(summary.num_entries, 0);
            assert_eq!(summary.smallest_user_key, b"aa");
            assert_eq!(summary.largest_user_key, b"zz");
        }

        let reader = SsTableReader::open(&path, 2).unwrap();
        assert_eq!(reader.range_tombstones().len(), 2);
        assert_eq!(reader.covering_range_tombstone_seq(b"bb", 100), 5);
        assert_eq!(reader.covering_range_tombstone_seq(b"nn", 100), 7);
    }

    #[test]
    fn test_sstable_prefix_bloom_roundtrip() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("prefix.sst");
        let cache = BlockCache::new(1024 * 1024);
        let extractor: Arc<dyn PrefixExtractor> = Arc::new(crate::options::FixedLengthPrefix(4));

        {
            let mut writer = SsTableWriter::new(
                &path,
                4096,
                10,
                CompressionType::None,
                Some(extractor.clone()),
            )
            .unwrap();
            for tenant in &["aaaa", "bbbb", "cccc"] {
                for i in 0..4 {
                    let user_key = format!("{}:key{}", tenant, i);
                    writer.add(&ik(user_key.as_bytes(), 1), b"v").unwrap();
                }
            }
            writer.finish().unwrap().unwrap();
        }

        let reader = SsTableReader::open(&path, 1).unwrap();
        // Present prefixes
        assert!(reader.may_have_prefix(b"aaaa"));
        assert!(reader.may_have_prefix(b"bbbb"));
        assert!(reader.may_have_prefix(b"cccc"));
        // Absent prefixes (with 10 bits/key the FP rate is ~1%; these
        // specific strings should all be rejected).
        let mut false_positives = 0;
        for i in 0..200u32 {
            let p = format!("zz{:02}", i);
            if reader.may_have_prefix(p.as_bytes()) {
                false_positives += 1;
            }
        }
        assert!(
            false_positives < 20,
            "too many prefix bloom false positives: {}",
            false_positives
        );

        // Point lookups still work — user-key bloom is independent.
        assert_eq!(
            reader.get(b"bbbb:key2", u64::MAX, &cache).unwrap(),
            LookupResult::Found {
                seq: 1,
                value: b"v".to_vec()
            }
        );
    }

    #[test]
    fn test_sstable_without_prefix_bloom_is_superset() {
        // A file written without an extractor reports every prefix as
        // possibly present — readers must fall back to conservative
        // behavior, not crash.
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("noprefix.sst");

        {
            let mut writer =
                SsTableWriter::new(&path, 4096, 10, CompressionType::None, None).unwrap();
            writer.add(&ik(b"aaaa:1", 1), b"v").unwrap();
            writer.add(&ik(b"bbbb:1", 1), b"v").unwrap();
            writer.finish().unwrap().unwrap();
        }

        let reader = SsTableReader::open(&path, 1).unwrap();
        // Every query comes back `true` — no negative information.
        assert!(reader.may_have_prefix(b"aaaa"));
        assert!(reader.may_have_prefix(b"zzzz"));
        assert!(reader.may_have_prefix(b"anything"));
    }
}
