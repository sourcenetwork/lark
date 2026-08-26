//! SSTable file format: sorted on-disk tables of MVCC-encoded key-value pairs.
//!
//! Layout:
//! ```text
//! [data block 0][data block 1]...[data block n][range_tombstone block][bloom region][index block][footer]
//! ```
//!
//! The bloom region is two blooms concatenated behind a length header:
//! `[prefix_bloom_len: u64 LE][prefix_bloom_bytes][user_key_bloom_bytes]`.
//! A zero length means the file was written without a prefix extractor.
//!
//! Data blocks store **internal keys** - `user_key || !seq || value_type` -
//! sorted so that newer versions of the same user key appear before older
//! ones. Tombstones are first-class entries; reads that land on a tombstone
//! at or before `snapshot_seq` return "deleted" and suppress older versions
//! in lower levels. The bloom filter is keyed on user keys so point lookups
//! can short-circuit regardless of which seq a reader is asking for.
//!
//! # Format versions and compatibility
//!
//! The trailing 8 bytes of every SSTable are the magic number, so the
//! footer's size is discoverable before anything else is parsed.
//!
//! ```text
//! MAGIC_V1 (version byte 0x01) flat index,        64-byte footer, no metadata checksum
//! MAGIC_V2 (version byte 0x02) partitioned index, 64-byte footer, no metadata checksum
//! MAGIC_V3 (version byte 0x03) flat index,        72-byte footer, metadata checksummed
//! MAGIC_V4 (version byte 0x04) partitioned index, 72-byte footer, metadata checksummed
//! ```
//!
//! lark writes V3 or V4 and reads all four. A table written by an older
//! lark keeps opening and keeps serving; it simply carries no metadata
//! checksum, so bit rot in its footer, index, bloom or range-tombstone
//! block is not detected there. Rewriting such a table (any compaction
//! that touches it) upgrades it to the checksummed form. There is no
//! migration step and no downgrade path: a V3/V4 table is rejected by an
//! older lark with "invalid SSTable magic number", which is the correct
//! loud failure.
//!
//! # What a checksum covers
//!
//! In a V3/V4 table every byte belongs to exactly one checksummed
//! region. A data block is framed `[compression: u8][payload][checksum:
//! u32]`. Each metadata region (range-tombstone block, bloom region,
//! every partitioned index leaf, the index block) carries a 4-byte
//! trailer counted inside the size the footer or the leaf handle records
//! for it. The footer carries an 8-byte checksum over its seven fixed
//! fields and its magic, so a damaged offset, size or entry count is
//! caught before it is trusted.
//!
//! Three holes are deliberate and are not closed here. A V1/V2 table
//! has no metadata checksum at all, which is the price of reading
//! yesterday's files; it is protected only by the region-bound
//! validation against the file size, and the cost of that is measured in
//! `tests/adversarial_sstable_flips.rs` rather than argued away. A
//! checksum binds a region's bytes and its kind, not its position, so
//! two regions of the same kind whose contents a misdirected write
//! exchanged each still verify; data blocks and partitioned index leaves
//! are the kinds a file holds more than one of. And these are xxh3
//! checksums, an accidental-corruption guard and not a MAC: they catch
//! bit rot and torn writes, not an attacker who can rewrite the file and
//! its checksum together.

use std::fs::{self, File};
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};

use parking_lot::Mutex;

use super::block::{Block, BlockBuilder, BlockHandle, RESTART_INTERVAL, decode_entry_at};
use super::block_cache::BlockCache;
use super::bloom::{BloomFilter, BloomFilterBuilder, decode_bloom_block, encode_bloom_block};
use super::checksum;
use super::durability;
use super::internal_key::{
    VALUE_TYPE_DELETION, VALUE_TYPE_MERGE, compare_internal_keys, decode_internal_key, lookup_key,
    user_key_of,
};
use super::range_tombstone::{RangeTombstone, RangeTombstoneSet};
use crate::options::{CompressionType, PrefixExtractor};

/// SSTable magic number: "LARKSST\x01" - flat-index format with the
/// 64-byte footer and no metadata checksums. Legacy: read, never written.
const MAGIC_V1: u64 = 0x4C41524B_53535401;

/// SSTable magic number: "LARKSST\x02" - partitioned-index format with
/// the 64-byte footer and no metadata checksums. The footer's
/// `index_offset/index_size` point to a compact top-level index whose
/// entries each reference a leaf sub-block on disk. Legacy: read, never
/// written.
const MAGIC_V2: u64 = 0x4C41524B_53535402;

/// SSTable magic number: "LARKSST\x03" - flat index, 72-byte footer,
/// metadata regions checksummed. Written by lark today.
const MAGIC_V3: u64 = 0x4C41524B_53535403;

/// SSTable magic number: "LARKSST\x04" - partitioned index, 72-byte
/// footer, metadata regions checksummed. Written by lark today.
const MAGIC_V4: u64 = 0x4C41524B_53535404;

/// Footer size for [`MAGIC_V1`] and [`MAGIC_V2`].
const FOOTER_SIZE_V1: usize = 64;

/// Footer size for [`MAGIC_V3`] and [`MAGIC_V4`]: the V1 fields plus an
/// 8-byte metadata checksum ahead of the magic.
const FOOTER_SIZE_V2: usize = 72;

/// Bytes a checksummed metadata region carries after its payload. The
/// trailer is counted inside the region size the footer records, so a
/// legacy reader's bounds arithmetic needs no special case.
const META_CHECKSUM_LEN: usize = 4;

const COMPRESSION_NONE: u8 = 0x00;
const COMPRESSION_LZ4: u8 = 0x01;
const COMPRESSION_SNAPPY: u8 = 0x02;

/// SSTable footer, written at the end of the file. Its length depends on
/// the format version the magic names: 64 bytes for V1/V2, 72 for V3/V4.
///
/// Layout on disk:
/// ```text
/// [data blocks][range_tombstone_block][bloom_region][index_block][footer]
/// ```
///
/// Footer bytes, all little-endian:
/// ```text
/// [ 0.. 8) range_tombstone_offset
/// [ 8..16) range_tombstone_size   (0 means no range tombstones)
/// [16..24) bloom_offset
/// [24..32) bloom_size
/// [32..40) index_offset
/// [40..48) index_size
/// [48..56) num_entries
/// V1/V2: [56..64) magic
/// V3/V4: [56..64) checksum over bytes 0..56 and the magic
///        [64..72) magic
/// ```
///
/// The seven fixed fields sit at the same offsets from the footer's start
/// in both layouts, and the magic is always the file's last 8 bytes, so
/// the footer's length is discoverable before it is parsed.
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
    /// Whether a table carrying `magic` checksums its metadata regions.
    fn magic_is_checksummed(magic: u64) -> bool {
        magic == MAGIC_V3 || magic == MAGIC_V4
    }

    /// Whether `magic` names the partitioned index layout, where the
    /// footer's index handle points at a top-level index of leaves.
    fn magic_is_partitioned(magic: u64) -> bool {
        magic == MAGIC_V2 || magic == MAGIC_V4
    }

    /// Footer byte length implied by `magic`, or an error naming the
    /// magic when it is not one lark writes or reads.
    fn size_for_magic(magic: u64) -> io::Result<usize> {
        match magic {
            MAGIC_V1 | MAGIC_V2 => Ok(FOOTER_SIZE_V1),
            MAGIC_V3 | MAGIC_V4 => Ok(FOOTER_SIZE_V2),
            other => Err(invalid_data(format!(
                "invalid SSTable magic number: {other:#018x}"
            ))),
        }
    }

    fn checksummed(&self) -> bool {
        Self::magic_is_checksummed(self.magic)
    }

    fn size(&self) -> usize {
        if self.checksummed() {
            FOOTER_SIZE_V2
        } else {
            FOOTER_SIZE_V1
        }
    }

    /// Encode into the layout `self.magic` selects.
    fn encode(&self) -> Vec<u8> {
        let mut buf = vec![0u8; self.size()];
        buf[0..8].copy_from_slice(&self.range_tombstone_offset.to_le_bytes());
        buf[8..16].copy_from_slice(&self.range_tombstone_size.to_le_bytes());
        buf[16..24].copy_from_slice(&self.bloom_offset.to_le_bytes());
        buf[24..32].copy_from_slice(&self.bloom_size.to_le_bytes());
        buf[32..40].copy_from_slice(&self.index_offset.to_le_bytes());
        buf[40..48].copy_from_slice(&self.index_size.to_le_bytes());
        buf[48..56].copy_from_slice(&self.num_entries.to_le_bytes());
        if self.checksummed() {
            let sum = checksum::sst_footer(&buf[0..56], self.magic).to_le_bytes();
            buf[56..64].copy_from_slice(&sum);
            buf[64..72].copy_from_slice(&self.magic.to_le_bytes());
        } else {
            buf[56..64].copy_from_slice(&self.magic.to_le_bytes());
        }
        buf
    }

    /// Decode a footer whose length is exactly what its magic implies.
    ///
    /// The magic is validated first, so a table from a format version
    /// lark does not implement is reported as a magic error rather than
    /// as a checksum error.
    fn decode(buf: &[u8]) -> io::Result<Self> {
        if buf.len() < FOOTER_SIZE_V1 {
            return Err(invalid_data("SSTable footer is truncated"));
        }
        let magic = u64::from_le_bytes(buf[buf.len() - 8..].try_into().unwrap());
        let expected = Self::size_for_magic(magic)?;
        if buf.len() != expected {
            return Err(invalid_data(format!(
                "SSTable footer is {} bytes but its format version needs {expected}",
                buf.len()
            )));
        }
        if Self::magic_is_checksummed(magic) {
            let stored = u64::from_le_bytes(buf[56..64].try_into().unwrap());
            if stored != checksum::sst_footer(&buf[0..56], magic) {
                return Err(invalid_data("SSTable footer checksum mismatch"));
            }
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

/// Read the trailing footer of an SSTable. The magic is the file's last
/// 8 bytes, so the footer's length is known before it is parsed and a
/// legacy table needs no separate code path.
fn read_footer(file: &mut File, file_size: u64) -> io::Result<Footer> {
    if file_size < 8 {
        return Err(invalid_data(
            "SSTable file too small to carry a magic number",
        ));
    }
    file.seek(SeekFrom::End(-8))?;
    let mut magic_buf = [0u8; 8];
    file.read_exact(&mut magic_buf)?;
    let footer_size = Footer::size_for_magic(u64::from_le_bytes(magic_buf))?;
    if file_size < footer_size as u64 {
        return Err(invalid_data("SSTable file too small for its footer"));
    }
    file.seek(SeekFrom::Start(file_size - footer_size as u64))?;
    let mut footer_buf = vec![0u8; footer_size];
    file.read_exact(&mut footer_buf)?;
    Footer::decode(&footer_buf)
}

/// Strip and verify a metadata region's checksum trailer, returning the
/// payload the decoder should parse.
///
/// A `checksummed` of `false` returns the region unchanged, which is how
/// a V1/V2 table keeps reading: those files carry no trailer, so there is
/// nothing to strip and nothing to verify.
fn verify_meta_region<'a>(
    region: &'a [u8],
    kind: u8,
    checksummed: bool,
    name: &'static str,
) -> io::Result<&'a [u8]> {
    if !checksummed {
        return Ok(region);
    }
    if region.len() < META_CHECKSUM_LEN {
        return Err(invalid_data(format!(
            "{name} is too short to carry a checksum"
        )));
    }
    let (payload, trailer) = region.split_at(region.len() - META_CHECKSUM_LEN);
    let stored = u32::from_le_bytes(trailer.try_into().unwrap());
    if stored != checksum::sst_meta(kind, payload) {
        return Err(invalid_data(format!("{name} checksum mismatch")));
    }
    Ok(payload)
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

pub(crate) fn decode_range_tombstone_block(data: &[u8]) -> io::Result<Vec<RangeTombstone>> {
    if data.is_empty() {
        return Ok(Vec::new());
    }
    if data.len() < 4 {
        return Err(invalid_data("range tombstone block too short"));
    }
    let count = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
    let max_records_by_size = (data.len() - 4) / 16;
    let mut out = Vec::with_capacity(count.min(max_records_by_size));
    let mut pos = 4;
    for _ in 0..count {
        let start_len = read_u32(data, &mut pos, "range tombstone start length")? as usize;
        let start = read_bytes(data, &mut pos, start_len, "range tombstone start")?;
        let end_len = read_u32(data, &mut pos, "range tombstone end length")? as usize;
        let end = read_bytes(data, &mut pos, end_len, "range tombstone end")?;
        let seq = read_u64(data, &mut pos, "range tombstone seq")?;
        out.push(RangeTombstone::new(start, end, seq));
    }
    if pos != data.len() {
        return Err(invalid_data("range tombstone block has trailing bytes"));
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
/// open - reads remain valid even after a concurrent compaction unlinks
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

#[derive(Clone)]
pub(crate) enum SsTableBlockCursor {
    Flat(usize),
    Partitioned {
        leaf_idx: usize,
        entry_idx: usize,
        leaf_handles: Arc<Vec<BlockHandle>>,
    },
}

impl SsTableBlockCursor {
    fn handle(&self, flat_index: &[IndexEntry]) -> io::Result<BlockHandle> {
        match self {
            Self::Flat(idx) => flat_index
                .get(*idx)
                .map(|entry| entry.handle)
                .ok_or_else(|| invalid_data("block index out of bounds")),
            Self::Partitioned {
                entry_idx,
                leaf_handles,
                ..
            } => leaf_handles
                .get(*entry_idx)
                .copied()
                .ok_or_else(|| invalid_data("partitioned block index out of bounds")),
        }
    }
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

/// Compute the serialized byte size of an index block without allocating.
fn encoded_index_block_size(entries: &[(Vec<u8>, BlockHandle)]) -> usize {
    // 4 bytes for count, then per entry: 4 (key_len) + key + 8 (offset) + 8 (size)
    4 + entries.iter().map(|(k, _)| 4 + k.len() + 16).sum::<usize>()
}

fn decode_index_block(data: &[u8]) -> io::Result<Vec<IndexEntry>> {
    if data.len() < 4 {
        return Err(invalid_data("index block too short"));
    }

    let num_entries = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
    let max_entries_by_size = (data.len() - 4) / 20;
    let mut entries = Vec::with_capacity(num_entries.min(max_entries_by_size));
    let mut pos = 4;

    for _ in 0..num_entries {
        let key_len = read_u32(data, &mut pos, "index key length")? as usize;
        let key = read_bytes(data, &mut pos, key_len, "index key")?;
        let offset = read_u64(data, &mut pos, "index block offset")?;
        let size = read_u64(data, &mut pos, "index block size")?;

        entries.push(IndexEntry {
            key,
            handle: BlockHandle { offset, size },
        });
    }
    if pos != data.len() {
        return Err(invalid_data("index block has trailing bytes"));
    }

    Ok(entries)
}

/// Append a metadata region followed by its checksum trailer, advancing
/// `offset`. Returns the region's on-disk size, trailer included, which
/// is what the footer or the leaf handle records for it.
fn write_meta_region(
    writer: &mut BufWriter<File>,
    offset: &mut u64,
    payload: &[u8],
    kind: u8,
) -> io::Result<u64> {
    let trailer = checksum::sst_meta(kind, payload).to_le_bytes();
    writer.write_all(payload)?;
    writer.write_all(&trailer)?;
    let size = (payload.len() + trailer.len()) as u64;
    *offset += size;
    Ok(size)
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn approximate_size_from_index(index: &[IndexEntry], lo_probe: &[u8], hi_probe: &[u8]) -> u64 {
    if index.is_empty() {
        return 0;
    }

    let first = index.partition_point(|entry| compare_internal_keys(&entry.key, lo_probe).is_lt());
    let last = index.partition_point(|entry| compare_internal_keys(&entry.key, hi_probe).is_lt());
    let end_idx = last.min(index.len() - 1);
    if first > end_idx {
        return 0;
    }

    index[first..=end_idx]
        .iter()
        .map(|entry| entry.handle.size)
        .sum()
}

fn read_u32(data: &[u8], pos: &mut usize, field: &'static str) -> io::Result<u32> {
    let end = pos
        .checked_add(4)
        .ok_or_else(|| invalid_data(format!("{field} offset overflows")))?;
    if end > data.len() {
        return Err(invalid_data(format!("{field} is truncated")));
    }
    let value = u32::from_le_bytes(data[*pos..end].try_into().unwrap());
    *pos = end;
    Ok(value)
}

fn read_u64(data: &[u8], pos: &mut usize, field: &'static str) -> io::Result<u64> {
    let end = pos
        .checked_add(8)
        .ok_or_else(|| invalid_data(format!("{field} offset overflows")))?;
    if end > data.len() {
        return Err(invalid_data(format!("{field} is truncated")));
    }
    let value = u64::from_le_bytes(data[*pos..end].try_into().unwrap());
    *pos = end;
    Ok(value)
}

fn read_bytes(
    data: &[u8],
    pos: &mut usize,
    len: usize,
    field: &'static str,
) -> io::Result<Vec<u8>> {
    let end = pos
        .checked_add(len)
        .ok_or_else(|| invalid_data(format!("{field} length overflows")))?;
    if end > data.len() {
        return Err(invalid_data(format!("{field} is truncated")));
    }
    let bytes = data[*pos..end].to_vec();
    *pos = end;
    Ok(bytes)
}

/// Check every region the footer points at against `data_end`, the first
/// byte past the region area (the file size less the footer). A
/// checksummed region also has to be long enough to hold its trailer.
fn validate_footer_regions(footer: &Footer, data_end: u64) -> io::Result<()> {
    let trailer = if footer.checksummed() {
        META_CHECKSUM_LEN as u64
    } else {
        0
    };
    if footer.bloom_size < 8 + trailer {
        return Err(invalid_data("bloom region too short"));
    }
    if footer.index_size < 4 + trailer {
        return Err(invalid_data("index block too short"));
    }
    validate_file_region(
        footer.bloom_offset,
        footer.bloom_size,
        data_end,
        "bloom region",
    )?;
    validate_file_region(
        footer.index_offset,
        footer.index_size,
        data_end,
        "index block",
    )?;
    if footer.range_tombstone_size > 0 {
        if footer.range_tombstone_size < 4 + trailer {
            return Err(invalid_data("range tombstone block too short"));
        }
        validate_file_region(
            footer.range_tombstone_offset,
            footer.range_tombstone_size,
            data_end,
            "range tombstone block",
        )?;
    }
    Ok(())
}

fn read_file_region(
    file: &mut File,
    offset: u64,
    size: u64,
    data_end: u64,
    name: &'static str,
) -> io::Result<Vec<u8>> {
    validate_file_region(offset, size, data_end, name)?;
    let len = usize::try_from(size)
        .map_err(|_| invalid_data(format!("{name} is too large to address")))?;
    let mut buf = vec![0u8; len];
    file.seek(SeekFrom::Start(offset))?;
    file.read_exact(&mut buf)?;
    Ok(buf)
}

/// `data_end` is the first byte past the region area: the file size less
/// the footer, whose length depends on the format version.
fn validate_file_region(
    offset: u64,
    size: u64,
    data_end: u64,
    name: &'static str,
) -> io::Result<()> {
    let end = offset
        .checked_add(size)
        .ok_or_else(|| invalid_data(format!("{name} range overflows")))?;
    if end > data_end {
        return Err(invalid_data(format!("{name} extends past SSTable data")));
    }
    if usize::try_from(size).is_err() {
        return Err(invalid_data(format!("{name} is too large to address")));
    }
    Ok(())
}

// ─── Writer ─────────────────────────────────────────────────────────────────

/// Builds an SSTable file from sorted **internal keys**. Callers are
/// responsible for supplying keys in ascending internal-key order (newer
/// versions of a user key appear before older ones).
pub(crate) struct SsTableWriter {
    writer: BufWriter<File>,
    path: PathBuf,
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
    partitioned_index: bool,
    metadata_block_size: usize,
}

impl SsTableWriter {
    pub(crate) fn new(
        path: &Path,
        block_size: usize,
        bloom_bits_per_key: usize,
        compression: CompressionType,
        prefix_extractor: Option<Arc<dyn PrefixExtractor>>,
        partitioned_index: bool,
        metadata_block_size: usize,
    ) -> io::Result<Self> {
        let file = File::create(path)?;
        let prefix_bloom_builder = prefix_extractor
            .as_ref()
            .map(|_| BloomFilterBuilder::new(bloom_bits_per_key));
        Ok(Self {
            writer: BufWriter::new(file),
            path: path.to_path_buf(),
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
            partitioned_index,
            metadata_block_size,
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
            ) && let Some(prefix) = extractor.extract(user_key)
            {
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
    /// added - no point entries and no range tombstones. A file with
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
        let range_tombstone_set =
            RangeTombstoneSet::from_vec(std::mem::take(&mut self.range_tombstones));
        let range_tombstone_data = if range_tombstone_set.is_empty() {
            Vec::new()
        } else {
            encode_range_tombstone_block(range_tombstone_set.as_slice())
        };
        let range_tombstone_offset = self.current_offset;
        let range_tombstone_size = if range_tombstone_data.is_empty() {
            0
        } else {
            write_meta_region(
                &mut self.writer,
                &mut self.current_offset,
                &range_tombstone_data,
                checksum::META_KIND_RANGE_TOMBSTONE,
            )?
        };

        // Bloom region layout:
        //   [prefix_bloom_len: u64 LE][prefix_bloom_bytes][user_key_bloom_bytes]
        //
        // A `prefix_bloom_len` of 0 means "no prefix bloom" - the file
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

        let mut bloom_region =
            Vec::with_capacity(8 + prefix_bloom_data.len() + user_bloom_data.len());
        bloom_region.extend_from_slice(&(prefix_bloom_data.len() as u64).to_le_bytes());
        bloom_region.extend_from_slice(&prefix_bloom_data);
        bloom_region.extend_from_slice(&user_bloom_data);
        let bloom_offset = self.current_offset;
        let bloom_size = write_meta_region(
            &mut self.writer,
            &mut self.current_offset,
            &bloom_region,
            checksum::META_KIND_BLOOM,
        )?;

        let (index_offset, index_size, magic) = if self.partitioned_index {
            // Partition the flat index entries into leaf sub-blocks whose
            // serialized size stays within `metadata_block_size`, write
            // each leaf as a raw encoded index block, and build a
            // top-level index pointing to each leaf.
            let mut top_level: Vec<(Vec<u8>, BlockHandle)> = Vec::new();
            let mut chunk_start = 0usize;
            while chunk_start < self.index_entries.len() {
                let mut chunk_end = chunk_start + 1;
                while chunk_end < self.index_entries.len() {
                    let candidate = &self.index_entries[chunk_start..chunk_end + 1];
                    let size = encoded_index_block_size(candidate);
                    if size > self.metadata_block_size {
                        break;
                    }
                    chunk_end += 1;
                }
                let chunk = &self.index_entries[chunk_start..chunk_end];
                let leaf_data = encode_index_block(chunk);
                let last_key = match chunk.last() {
                    Some((key, _)) => key.clone(),
                    None => return Err(invalid_data("index partitioning produced an empty leaf")),
                };
                let leaf_offset = self.current_offset;
                let leaf_size = write_meta_region(
                    &mut self.writer,
                    &mut self.current_offset,
                    &leaf_data,
                    checksum::META_KIND_INDEX_LEAF,
                )?;

                top_level.push((
                    last_key,
                    BlockHandle {
                        offset: leaf_offset,
                        size: leaf_size,
                    },
                ));
                chunk_start = chunk_end;
            }
            let top_data = encode_index_block(&top_level);
            let top_offset = self.current_offset;
            let top_size = write_meta_region(
                &mut self.writer,
                &mut self.current_offset,
                &top_data,
                checksum::META_KIND_INDEX,
            )?;
            (top_offset, top_size, MAGIC_V4)
        } else {
            let index_data = encode_index_block(&self.index_entries);
            let idx_offset = self.current_offset;
            let idx_size = write_meta_region(
                &mut self.writer,
                &mut self.current_offset,
                &index_data,
                checksum::META_KIND_INDEX,
            )?;
            (idx_offset, idx_size, MAGIC_V3)
        };

        let footer = Footer {
            range_tombstone_offset,
            range_tombstone_size,
            bloom_offset,
            bloom_size,
            index_offset,
            index_size,
            num_entries: self.num_entries,
            magic,
        };
        self.writer.write_all(&footer.encode())?;
        self.writer.flush()?;
        self.writer.get_ref().sync_all()?;
        durability::sync_parent_dir(&self.path)?;

        let mut smallest_user_key = self.smallest_user_key.take();
        let mut largest_user_key = self.largest_user_key.take();
        for rt in range_tombstone_set.iter() {
            if smallest_user_key
                .as_ref()
                .is_none_or(|smallest| rt.start.as_slice() < smallest.as_slice())
            {
                smallest_user_key = Some(rt.start.clone());
            }
            if largest_user_key
                .as_ref()
                .is_none_or(|largest| rt.end.as_slice() > largest.as_slice())
            {
                largest_user_key = Some(rt.end.clone());
            }
        }

        let smallest_user_key = smallest_user_key.expect("checked non-empty above");
        let largest_user_key = largest_user_key.expect("checked non-empty above");

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
                // its decode buffer - matches `lz4_flex::compress_prepend_size`.
                let mut framed = Vec::with_capacity(4 + compressed.len());
                framed.extend_from_slice(&(raw_data.len() as u32).to_le_bytes());
                framed.extend_from_slice(&compressed);
                (COMPRESSION_SNAPPY, framed)
            }
        };

        let checksum = checksum::sst_block(codec_byte, &payload);
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
    /// First byte past the region area: the file size less the footer.
    /// Every region bound is checked against this, so a damaged offset
    /// can never send a read into the footer or past the file.
    data_end: u64,
    pub(crate) file_id: u64,
    index: Vec<IndexEntry>,
    bloom: BloomFilter,
    /// Optional prefix bloom filter. `None` when the file was built
    /// without a prefix extractor (or the extractor yielded no prefixes).
    /// A query against a reader without a prefix bloom conservatively
    /// returns `true` - the file might contain the prefix.
    prefix_bloom: Option<BloomFilter>,
    range_tombstones: RangeTombstoneSet,
    /// `true` when the file was written with `MAGIC_V2` (partitioned
    /// index). `self.index` then holds only the compact top-level
    /// entries; each entry's `handle` points to a leaf sub-block that
    /// must be read via [`SsTableReader::read_index_leaf`].
    partitioned: bool,
    /// Whether this file's metadata regions carry checksum trailers.
    /// False for a legacy V1/V2 table, whose regions have no trailer to
    /// strip and no checksum to verify.
    meta_checksummed: bool,
    #[cfg(test)]
    index_leaf_reads: AtomicUsize,
}

pub(crate) struct SsTableInternalIter<'a> {
    reader: &'a SsTableReader,
    cache: &'a BlockCache,
    next_cursor: Option<SsTableBlockCursor>,
    current_block: Option<Arc<Block>>,
    current_pos: usize,
    current_key: Vec<u8>,
}

impl<'a> SsTableInternalIter<'a> {
    pub(crate) fn next_entry(&mut self) -> io::Result<Option<(Vec<u8>, Vec<u8>)>> {
        loop {
            if let Some(block) = &self.current_block {
                let data = block.entry_data();
                if self.current_pos < data.len() {
                    let (consumed, value_offset, value_len) =
                        decode_entry_at(data, self.current_pos, &mut self.current_key);
                    self.current_pos += consumed;
                    let value = data[value_offset..value_offset + value_len].to_vec();
                    return Ok(Some((self.current_key.clone(), value)));
                }
            }

            let Some(cursor) = self.next_cursor.take() else {
                return Ok(None);
            };
            self.next_cursor = self.reader.next_block_cursor(&cursor)?;

            let block = self.reader.load_block_at_cursor(&cursor, self.cache)?;
            self.current_block = Some(block);
            self.current_pos = 0;
            self.current_key.clear();
        }
    }
}

impl SsTableReader {
    /// Open an SSTable file and load index + bloom into memory.
    pub(crate) fn open(path: &Path, file_id: u64) -> io::Result<Self> {
        let mut file = File::open(path)?;
        let file_size = file.metadata()?.len();

        let footer = read_footer(&mut file, file_size)?;
        let data_end = file_size - footer.size() as u64;
        validate_footer_regions(&footer, data_end)?;
        let meta_checksummed = footer.checksummed();

        let bloom_region = read_file_region(
            &mut file,
            footer.bloom_offset,
            footer.bloom_size,
            data_end,
            "bloom region",
        )?;
        let bloom_data = verify_meta_region(
            &bloom_region,
            checksum::META_KIND_BLOOM,
            meta_checksummed,
            "bloom region",
        )?;
        // Peel [prefix_bloom_len: u64 LE][prefix_bytes][user_bytes].
        if bloom_data.len() < 8 {
            return Err(invalid_data(
                "bloom region too short for prefix-bloom length header",
            ));
        }
        let prefix_len = usize::try_from(u64::from_le_bytes(bloom_data[0..8].try_into().unwrap()))
            .map_err(|_| invalid_data("prefix bloom length is too large to address"))?;
        let user_bloom_offset = 8usize
            .checked_add(prefix_len)
            .ok_or_else(|| invalid_data("prefix bloom length overflows"))?;
        if user_bloom_offset > bloom_data.len() {
            return Err(invalid_data("prefix bloom length exceeds bloom region"));
        }
        if prefix_len > 0 && prefix_len < 4 {
            return Err(invalid_data("prefix bloom block too short"));
        }
        if bloom_data.len() - user_bloom_offset < 4 {
            return Err(invalid_data("user bloom block too short"));
        }
        let prefix_bloom = if prefix_len == 0 {
            None
        } else {
            Some(decode_bloom_block(&bloom_data[8..8 + prefix_len]))
        };
        let bloom = decode_bloom_block(&bloom_data[user_bloom_offset..]);

        let index_region = read_file_region(
            &mut file,
            footer.index_offset,
            footer.index_size,
            data_end,
            "index block",
        )?;
        let index = decode_index_block(verify_meta_region(
            &index_region,
            checksum::META_KIND_INDEX,
            meta_checksummed,
            "index block",
        )?)?;

        let range_tombstones = if footer.range_tombstone_size == 0 {
            Vec::new()
        } else {
            let rt_region = read_file_region(
                &mut file,
                footer.range_tombstone_offset,
                footer.range_tombstone_size,
                data_end,
                "range tombstone block",
            )?;
            decode_range_tombstone_block(verify_meta_region(
                &rt_region,
                checksum::META_KIND_RANGE_TOMBSTONE,
                meta_checksummed,
                "range tombstone block",
            )?)?
        };

        let partitioned = Footer::magic_is_partitioned(footer.magic);

        Ok(Self {
            file: Mutex::new(file),
            data_end,
            file_id,
            index,
            bloom,
            prefix_bloom,
            range_tombstones: RangeTombstoneSet::from_vec(range_tombstones),
            partitioned,
            meta_checksummed,
            #[cfg(test)]
            index_leaf_reads: AtomicUsize::new(0),
        })
    }

    /// Read and decode a leaf index sub-block from disk. Used when
    /// `self.partitioned` is true; the `handle` comes from one of the
    /// top-level index entries in `self.index`. No block cache is
    /// consulted - the OS page cache keeps hot leaves warm.
    fn read_index_leaf(&self, handle: BlockHandle) -> io::Result<Vec<IndexEntry>> {
        #[cfg(test)]
        self.index_leaf_reads.fetch_add(1, Ordering::Relaxed);

        let mut file = self.file.lock();
        let region = read_file_region(
            &mut file,
            handle.offset,
            handle.size,
            self.data_end,
            "partitioned index leaf",
        )?;
        drop(file);
        decode_index_block(verify_meta_region(
            &region,
            checksum::META_KIND_INDEX_LEAF,
            self.meta_checksummed,
            "partitioned index leaf",
        )?)
    }

    #[cfg(test)]
    fn index_leaf_read_count(&self) -> usize {
        self.index_leaf_reads.load(Ordering::Relaxed)
    }

    #[cfg(test)]
    fn reset_index_leaf_read_count(&self) {
        self.index_leaf_reads.store(0, Ordering::Relaxed);
    }

    fn cursor_from_leaf(
        &self,
        leaf_idx: usize,
        entry_idx: usize,
        leaf: Vec<IndexEntry>,
    ) -> Option<SsTableBlockCursor> {
        if entry_idx >= leaf.len() {
            return None;
        }
        let leaf_handles = Arc::new(leaf.into_iter().map(|entry| entry.handle).collect());
        Some(SsTableBlockCursor::Partitioned {
            leaf_idx,
            entry_idx,
            leaf_handles,
        })
    }

    pub(crate) fn first_block_cursor(&self) -> io::Result<Option<SsTableBlockCursor>> {
        if !self.partitioned {
            return Ok((!self.index.is_empty()).then_some(SsTableBlockCursor::Flat(0)));
        }

        for (leaf_idx, top_entry) in self.index.iter().enumerate() {
            let leaf = self.read_index_leaf(top_entry.handle)?;
            if let Some(cursor) = self.cursor_from_leaf(leaf_idx, 0, leaf) {
                return Ok(Some(cursor));
            }
        }
        Ok(None)
    }

    pub(crate) fn last_block_cursor(&self) -> io::Result<Option<SsTableBlockCursor>> {
        if !self.partitioned {
            // `then`, not `then_some`: the argument to `then_some` is
            // evaluated whether or not the condition holds, so the
            // subtraction underflowed on an empty index despite the
            // guard right next to it.
            return Ok(
                (!self.index.is_empty()).then(|| SsTableBlockCursor::Flat(self.index.len() - 1))
            );
        }

        for (leaf_idx, top_entry) in self.index.iter().enumerate().rev() {
            let leaf = self.read_index_leaf(top_entry.handle)?;
            if !leaf.is_empty() {
                let entry_idx = leaf.len() - 1;
                return Ok(self.cursor_from_leaf(leaf_idx, entry_idx, leaf));
            }
        }
        Ok(None)
    }

    pub(crate) fn seek_block_cursor(
        &self,
        target: &[u8],
    ) -> io::Result<Option<SsTableBlockCursor>> {
        if !self.partitioned {
            let idx = match self
                .index
                .binary_search_by(|entry| compare_internal_keys(&entry.key, target))
            {
                Ok(i) => i,
                Err(i) => {
                    if i >= self.index.len() {
                        return Ok(None);
                    }
                    i
                }
            };
            return Ok(Some(SsTableBlockCursor::Flat(idx)));
        }

        let mut leaf_idx = match self
            .index
            .binary_search_by(|entry| compare_internal_keys(&entry.key, target))
        {
            Ok(i) => i,
            Err(i) => {
                if i >= self.index.len() {
                    return Ok(None);
                }
                i
            }
        };

        while leaf_idx < self.index.len() {
            let leaf = self.read_index_leaf(self.index[leaf_idx].handle)?;
            let entry_idx =
                match leaf.binary_search_by(|entry| compare_internal_keys(&entry.key, target)) {
                    Ok(i) => i,
                    Err(i) => i,
                };
            if let Some(cursor) = self.cursor_from_leaf(leaf_idx, entry_idx, leaf) {
                return Ok(Some(cursor));
            }
            leaf_idx += 1;
        }
        Ok(None)
    }

    pub(crate) fn next_block_cursor(
        &self,
        cursor: &SsTableBlockCursor,
    ) -> io::Result<Option<SsTableBlockCursor>> {
        match cursor {
            SsTableBlockCursor::Flat(idx) => {
                let next = idx + 1;
                Ok((next < self.index.len()).then_some(SsTableBlockCursor::Flat(next)))
            }
            SsTableBlockCursor::Partitioned {
                leaf_idx,
                entry_idx,
                leaf_handles,
            } => {
                let next_entry = entry_idx + 1;
                if next_entry < leaf_handles.len() {
                    return Ok(Some(SsTableBlockCursor::Partitioned {
                        leaf_idx: *leaf_idx,
                        entry_idx: next_entry,
                        leaf_handles: Arc::clone(leaf_handles),
                    }));
                }
                for next_leaf_idx in leaf_idx + 1..self.index.len() {
                    let leaf = self.read_index_leaf(self.index[next_leaf_idx].handle)?;
                    if let Some(cursor) = self.cursor_from_leaf(next_leaf_idx, 0, leaf) {
                        return Ok(Some(cursor));
                    }
                }
                Ok(None)
            }
        }
    }

    pub(crate) fn prev_block_cursor(
        &self,
        cursor: &SsTableBlockCursor,
    ) -> io::Result<Option<SsTableBlockCursor>> {
        match cursor {
            SsTableBlockCursor::Flat(idx) => {
                if *idx == 0 {
                    Ok(None)
                } else {
                    Ok(Some(SsTableBlockCursor::Flat(idx - 1)))
                }
            }
            SsTableBlockCursor::Partitioned {
                leaf_idx,
                entry_idx,
                leaf_handles,
            } => {
                if *entry_idx > 0 {
                    return Ok(Some(SsTableBlockCursor::Partitioned {
                        leaf_idx: *leaf_idx,
                        entry_idx: entry_idx - 1,
                        leaf_handles: Arc::clone(leaf_handles),
                    }));
                }
                for prev_leaf_idx in (0..*leaf_idx).rev() {
                    let leaf = self.read_index_leaf(self.index[prev_leaf_idx].handle)?;
                    if !leaf.is_empty() {
                        let entry_idx = leaf.len() - 1;
                        return Ok(self.cursor_from_leaf(prev_leaf_idx, entry_idx, leaf));
                    }
                }
                Ok(None)
            }
        }
    }

    pub(crate) fn load_block_at_cursor(
        &self,
        cursor: &SsTableBlockCursor,
        cache: &BlockCache,
    ) -> io::Result<Arc<Block>> {
        let handle = cursor.handle(&self.index)?;
        self.read_block(handle, cache)
    }

    /// Resolve a lookup key against the (possibly partitioned) index.
    ///
    /// Binary-searches the first data block whose last key is
    /// `>= search_key`. Returns that data block's [`BlockHandle`], or
    /// `None` if every block's last key is strictly less than
    /// `search_key`.
    ///
    /// For non-partitioned (V1) files, this is a single binary
    /// search on the in-memory `self.index`. For partitioned (V2)
    /// files, this is two binary searches: one on the top-level
    /// index, then one on the single leaf that covers `search_key`
    /// (loaded from disk on demand).
    fn find_block_handle(&self, search_key: &[u8]) -> io::Result<Option<BlockHandle>> {
        self.seek_block_cursor(search_key)?
            .map(|cursor| cursor.handle(&self.index))
            .transpose()
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
    /// nothing covers it - `0` is safe because real seqs start at 1.
    pub(crate) fn covering_range_tombstone_seq(&self, user_key: &[u8], snapshot_seq: u64) -> u64 {
        if self.range_tombstones.is_empty() {
            return 0;
        }
        self.range_tombstones
            .max_covering_seq(user_key, snapshot_seq)
    }

    /// Borrow this SSTable's range tombstones. Used by compaction to
    /// merge them into the output file and by the iterator to honor
    /// RT coverage during scans.
    pub(crate) fn range_tombstones(&self) -> &[RangeTombstone] {
        self.range_tombstones.as_slice()
    }

    /// Point lookup for `user_key` visible at `snapshot_seq`.
    ///
    /// Skips past merge operands (they're not valid point-lookup
    /// terminators) and returns the first `Value` / `Deletion` entry
    /// visible at the requested snapshot. A caller that cares about
    /// merge chains uses [`SsTableReader::collect_merge_chain`]
    /// instead.
    pub(crate) fn get(
        &self,
        user_key: &[u8],
        snapshot_seq: u64,
        cache: &BlockCache,
    ) -> io::Result<LookupResult> {
        if !self.bloom.may_contain(user_key) {
            cache.record_bloom_useful();
            return Ok(LookupResult::NotInTable);
        }

        let search_key = lookup_key(user_key, snapshot_seq);
        let handle = match self.find_block_handle(&search_key)? {
            Some(h) => h,
            None => return Ok(LookupResult::NotInTable),
        };

        let block = self.read_block(handle, cache)?;
        for (ik, value) in block.iter_from(&search_key) {
            let (uk, seq, vt) = decode_internal_key(&ik);
            if uk != user_key {
                return Ok(LookupResult::NotInTable);
            }
            match vt {
                VALUE_TYPE_MERGE => continue,
                VALUE_TYPE_DELETION => {
                    cache.record_bloom_full_positive();
                    return Ok(LookupResult::FoundTombstone { seq });
                }
                _ => {
                    cache.record_bloom_full_positive();
                    return Ok(LookupResult::Found { seq, value });
                }
            }
        }
        Ok(LookupResult::NotInTable)
    }

    /// Walk every visible entry for `user_key` at `snapshot_seq` in
    /// newest-seq-first order, appending `(seq, value_type, value)`
    /// tuples onto `out` and stopping at (and including) the first
    /// terminator (`VALUE_TYPE_VALUE` or `VALUE_TYPE_DELETION`).
    /// Returns `true` if a terminator was reached.
    pub(crate) fn collect_merge_chain(
        &self,
        user_key: &[u8],
        snapshot_seq: u64,
        cache: &BlockCache,
        out: &mut Vec<(u64, u8, Vec<u8>)>,
    ) -> io::Result<bool> {
        if !self.bloom.may_contain(user_key) {
            return Ok(false);
        }

        let search_key = lookup_key(user_key, snapshot_seq);
        let handle = match self.find_block_handle(&search_key)? {
            Some(h) => h,
            None => return Ok(false),
        };

        let block = self.read_block(handle, cache)?;
        for (ik, value) in block.iter_from(&search_key) {
            let (uk, seq, vt) = decode_internal_key(&ik);
            if uk != user_key {
                return Ok(false);
            }
            out.push((seq, vt, value));
            if vt != VALUE_TYPE_MERGE {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Read every entry in internal-key order with no dedup or filtering.
    /// Used by compaction to merge tables without losing versions.
    pub(crate) fn iter_internal(&self, cache: &BlockCache) -> io::Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let mut stream = self.iter_internal_stream(cache)?;
        let mut result = Vec::new();
        while let Some(entry) = stream.next_entry()? {
            result.push(entry);
        }
        Ok(result)
    }

    /// Stream every entry in internal-key order with no dedup or
    /// filtering. At most one decoded data block is buffered at a time.
    pub(crate) fn iter_internal_stream<'a>(
        &'a self,
        cache: &'a BlockCache,
    ) -> io::Result<SsTableInternalIter<'a>> {
        Ok(SsTableInternalIter {
            reader: self,
            cache,
            next_cursor: self.first_block_cursor()?,
            current_block: None,
            current_pos: 0,
            current_key: Vec::new(),
        })
    }

    /// Approximate on-disk bytes whose user key falls in
    /// `[start, end)`. Computed from the index alone - no data-block
    /// decompression. Flat-index tables search the in-memory block
    /// index directly; partitioned-index tables only read index leaves
    /// whose top-level key range intersects the requested bounds. The
    /// estimate is accurate to about one data block per partially-covered
    /// range boundary, matching the "within ~block_size" contract in the
    /// `Db::get_approximate_sizes` docs.
    pub(crate) fn approximate_size_in_range(&self, start: &[u8], end: &[u8]) -> u64 {
        if self.index.is_empty() || start >= end {
            return 0;
        }
        let lo_probe = lookup_key(start, u64::MAX);
        let hi_probe = lookup_key(end, u64::MAX);
        if !self.partitioned {
            return approximate_size_from_index(&self.index, &lo_probe, &hi_probe);
        }

        let first_leaf = self
            .index
            .partition_point(|entry| compare_internal_keys(&entry.key, &lo_probe).is_lt());
        let last_leaf = self
            .index
            .partition_point(|entry| compare_internal_keys(&entry.key, &hi_probe).is_lt());
        let end_leaf = last_leaf.min(self.index.len() - 1);
        if first_leaf > end_leaf {
            return 0;
        }

        let mut total = 0;
        for top_entry in &self.index[first_leaf..=end_leaf] {
            let leaf = match self.read_index_leaf(top_entry.handle) {
                Ok(leaf) => leaf,
                Err(_) => return 0,
            };
            total += approximate_size_from_index(&leaf, &lo_probe, &hi_probe);
        }
        total
    }

    fn read_block(&self, handle: BlockHandle, cache: &BlockCache) -> io::Result<Arc<Block>> {
        if let Some(block) = cache.get(self.file_id, handle.offset) {
            return Ok(block);
        }

        if handle.size < 5 {
            return Err(invalid_data("block frame too short"));
        }
        let mut file = self.file.lock();
        let block_data = read_file_region(
            &mut file,
            handle.offset,
            handle.size,
            self.data_end,
            "data block",
        )?;
        drop(file);

        // Frame: [compression_type: u8][payload][checksum: u32].
        // The checksum is an accidental-corruption guard, not a MAC.
        let compression_type = block_data[0];
        let checksum_offset = block_data.len() - 4;
        let stored_checksum = u32::from_le_bytes(block_data[checksum_offset..].try_into().unwrap());
        let compressed_data = &block_data[1..checksum_offset];

        let computed_checksum = checksum::sst_block(compression_type, compressed_data);
        if stored_checksum != computed_checksum
            && stored_checksum != checksum::legacy_payload_u32(compressed_data)
        {
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
                if n != raw_len {
                    return Err(invalid_data("snappy decoded length mismatch"));
                }
                out.truncate(n);
                out
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unknown compression type: {}", compression_type),
                ));
            }
        };

        let block = Arc::new(Block::decode_data_block(raw_data)?);
        cache.insert(self.file_id, handle.offset, Arc::clone(&block));
        Ok(block)
    }
}

/// Whether `path` is a complete SSTable that carries data: at least one
/// point entry, one range tombstone, or one index entry.
///
/// The footer's own `num_entries` is not trusted on its own. It is
/// checksummed only from `MAGIC_V3` on, so a damaged V1 or V2 footer can
/// read back as "holds nothing" over a table that is full, and the
/// caller uses this answer to decide whether opening would discard live
/// data. The index block is the corroborating structure a table with
/// data cannot be missing, and from V3 on it carries its own checksum,
/// so damage to it is an error rather than a wrong answer.
///
/// An `Err` means the file is not a readable SSTable at all, which the
/// caller must not read as "holds nothing": a file whose footer or index
/// will not parse cannot be proved empty.
pub(crate) fn table_carries_data(path: &Path) -> io::Result<bool> {
    let mut file = File::open(path)?;
    let file_size = file.metadata()?.len();
    let footer = read_footer(&mut file, file_size)?;
    let data_end = file_size - footer.size() as u64;
    validate_footer_regions(&footer, data_end)?;
    if footer.num_entries > 0 || footer.range_tombstone_size > 0 {
        return Ok(true);
    }
    let index_region = read_file_region(
        &mut file,
        footer.index_offset,
        footer.index_size,
        data_end,
        "index block",
    )?;
    let index = decode_index_block(verify_meta_region(
        &index_region,
        checksum::META_KIND_INDEX,
        footer.checksummed(),
        "index block",
    )?)?;
    Ok(!index.is_empty())
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
    use crate::engine::internal_key::{VALUE_TYPE_VALUE, encode_internal_key};
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
                SsTableWriter::new(&path, 4096, 10, CompressionType::Lz4, None, false, 4096)
                    .unwrap();
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
                SsTableWriter::new(&path, 4096, 10, CompressionType::None, None, false, 4096)
                    .unwrap();
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
                SsTableWriter::new(&path, 4096, 10, CompressionType::None, None, false, 4096)
                    .unwrap();
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
                SsTableWriter::new(&path, 4096, 10, CompressionType::None, None, false, 4096)
                    .unwrap();
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
                SsTableWriter::new(&path, 4096, 10, CompressionType::None, None, false, 4096)
                    .unwrap();
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
                false,
                4096,
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

        // Point lookups still work - user-key bloom is independent.
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
        // possibly present - readers must fall back to conservative
        // behavior, not crash.
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("noprefix.sst");

        {
            let mut writer =
                SsTableWriter::new(&path, 4096, 10, CompressionType::None, None, false, 4096)
                    .unwrap();
            writer.add(&ik(b"aaaa:1", 1), b"v").unwrap();
            writer.add(&ik(b"bbbb:1", 1), b"v").unwrap();
            writer.finish().unwrap().unwrap();
        }

        let reader = SsTableReader::open(&path, 1).unwrap();
        // Every query comes back `true` - no negative information.
        assert!(reader.may_have_prefix(b"aaaa"));
        assert!(reader.may_have_prefix(b"zzzz"));
        assert!(reader.may_have_prefix(b"anything"));
    }

    // ── filename / misc helpers ─────────────────────────────────

    #[test]
    fn sst_filename_pads_six_digits() {
        assert_eq!(sst_filename(0), "000000.sst");
        assert_eq!(sst_filename(1), "000001.sst");
        assert_eq!(sst_filename(999_999), "999999.sst");
    }

    // ── Footer encode/decode ────────────────────────────────────

    #[test]
    fn footer_round_trip_v1() {
        let f = Footer {
            range_tombstone_offset: 10,
            range_tombstone_size: 20,
            bloom_offset: 30,
            bloom_size: 40,
            index_offset: 70,
            index_size: 80,
            num_entries: 100,
            magic: MAGIC_V1,
        };
        let buf = f.encode();
        let got = Footer::decode(&buf).unwrap();
        assert_eq!(got.range_tombstone_offset, 10);
        assert_eq!(got.range_tombstone_size, 20);
        assert_eq!(got.bloom_offset, 30);
        assert_eq!(got.bloom_size, 40);
        assert_eq!(got.index_offset, 70);
        assert_eq!(got.index_size, 80);
        assert_eq!(got.num_entries, 100);
        assert_eq!(got.magic, MAGIC_V1);
    }

    #[test]
    fn footer_round_trip_v2() {
        let f = Footer {
            range_tombstone_offset: 0,
            range_tombstone_size: 0,
            bloom_offset: 1,
            bloom_size: 2,
            index_offset: 3,
            index_size: 4,
            num_entries: 5,
            magic: MAGIC_V2,
        };
        let got = Footer::decode(&f.encode()).unwrap();
        assert_eq!(got.magic, MAGIC_V2);
    }

    #[test]
    fn footer_round_trip_v3() {
        let f = Footer {
            range_tombstone_offset: 10,
            range_tombstone_size: 20,
            bloom_offset: 30,
            bloom_size: 40,
            index_offset: 70,
            index_size: 80,
            num_entries: 100,
            magic: MAGIC_V3,
        };
        let buf = f.encode();
        assert_eq!(buf.len(), FOOTER_SIZE_V2);
        let got = Footer::decode(&buf).unwrap();
        assert_eq!(got.range_tombstone_offset, 10);
        assert_eq!(got.num_entries, 100);
        assert_eq!(got.magic, MAGIC_V3);
        assert!(got.checksummed());
        assert!(!Footer::magic_is_partitioned(got.magic));
    }

    #[test]
    fn footer_round_trip_v4() {
        let f = Footer {
            range_tombstone_offset: 0,
            range_tombstone_size: 0,
            bloom_offset: 1,
            bloom_size: 2,
            index_offset: 3,
            index_size: 4,
            num_entries: 5,
            magic: MAGIC_V4,
        };
        let got = Footer::decode(&f.encode()).unwrap();
        assert_eq!(got.magic, MAGIC_V4);
        assert!(got.checksummed());
        assert!(Footer::magic_is_partitioned(got.magic));
    }

    #[test]
    fn footer_decode_rejects_bad_magic() {
        let mut buf = [0u8; FOOTER_SIZE_V1];
        // Leave magic at zero - any unknown value should be rejected.
        buf[56..64].copy_from_slice(&0xDEAD_BEEF_u64.to_le_bytes());
        let err = Footer::decode(&buf).expect_err("should reject");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("magic"), "got: {err}");
    }

    #[test]
    fn footer_decode_catches_a_flip_in_every_checksummed_field() {
        let f = Footer {
            range_tombstone_offset: 7,
            range_tombstone_size: 11,
            bloom_offset: 13,
            bloom_size: 17,
            index_offset: 19,
            index_size: 23,
            num_entries: 29,
            magic: MAGIC_V3,
        };
        let clean = f.encode();
        for byte in 0..clean.len() {
            for bit in 0..8u8 {
                let mut damaged = clean.clone();
                damaged[byte] ^= 1 << bit;
                assert!(
                    Footer::decode(&damaged).is_err(),
                    "byte {byte} bit {bit} of a V3 footer was not caught"
                );
            }
        }
    }

    #[test]
    fn a_legacy_footer_carries_no_checksum_and_still_decodes() {
        let f = Footer {
            range_tombstone_offset: 1,
            range_tombstone_size: 0,
            bloom_offset: 2,
            bloom_size: 3,
            index_offset: 4,
            index_size: 5,
            num_entries: 6,
            magic: MAGIC_V1,
        };
        let buf = f.encode();
        assert_eq!(buf.len(), FOOTER_SIZE_V1);
        assert!(!Footer::decode(&buf).unwrap().checksummed());
    }

    #[test]
    fn metadata_checksums_are_domain_separated_by_kind() {
        let payload = b"the same bytes in two different regions";
        let index = checksum::sst_meta(checksum::META_KIND_INDEX, payload);
        assert_ne!(
            index,
            checksum::sst_meta(checksum::META_KIND_BLOOM, payload)
        );
        assert_ne!(
            index,
            checksum::sst_meta(checksum::META_KIND_INDEX_LEAF, payload)
        );
        assert_ne!(
            index,
            checksum::sst_meta(checksum::META_KIND_RANGE_TOMBSTONE, payload)
        );
    }

    #[test]
    fn verify_meta_region_passes_a_legacy_region_through_untouched() {
        let region = b"no trailer here";
        let got = verify_meta_region(region, checksum::META_KIND_INDEX, false, "index block")
            .expect("a legacy region is never verified");
        assert_eq!(got, region);
    }

    /// Write a table in the pre-checksum layout: no trailer on any
    /// metadata region and a 64-byte footer carrying [`MAGIC_V1`] or
    /// [`MAGIC_V2`]. This is the writer lark had before the metadata
    /// checksums landed, kept here so the compatibility contract in the
    /// module docs is tested rather than merely claimed.
    fn write_legacy_table(
        path: &Path,
        entries: &[(Vec<u8>, Vec<u8>)],
        partitioned: bool,
    ) -> io::Result<()> {
        let mut out: Vec<u8> = Vec::new();
        let mut index_entries: Vec<(Vec<u8>, BlockHandle)> = Vec::new();
        let mut bloom_builder = BloomFilterBuilder::new(10);

        // One data block per two entries, so a partitioned layout has
        // several index entries to spread over leaves.
        for pair in entries.chunks(2) {
            let mut block = BlockBuilder::new(RESTART_INTERVAL);
            for (internal_key, value) in pair {
                block.add(internal_key, value);
                bloom_builder.add_key(user_key_of(internal_key));
            }
            let raw = block.finish();
            let offset = out.len() as u64;
            out.push(COMPRESSION_NONE);
            out.extend_from_slice(&raw);
            out.extend_from_slice(&checksum::sst_block(COMPRESSION_NONE, &raw).to_le_bytes());
            let last_key = pair.last().expect("non-empty chunk").0.clone();
            index_entries.push((
                last_key,
                BlockHandle {
                    offset,
                    size: out.len() as u64 - offset,
                },
            ));
        }

        let range_tombstone_offset = out.len() as u64;
        let bloom_offset = out.len() as u64;
        let user_bloom = encode_bloom_block(&bloom_builder.build());
        out.extend_from_slice(&0u64.to_le_bytes());
        out.extend_from_slice(&user_bloom);
        let bloom_size = out.len() as u64 - bloom_offset;

        let (index_offset, index_size, magic) = if partitioned {
            let mut top_level: Vec<(Vec<u8>, BlockHandle)> = Vec::new();
            for chunk in index_entries.chunks(2) {
                let leaf = encode_index_block(chunk);
                let leaf_offset = out.len() as u64;
                out.extend_from_slice(&leaf);
                top_level.push((
                    chunk.last().expect("non-empty chunk").0.clone(),
                    BlockHandle {
                        offset: leaf_offset,
                        size: leaf.len() as u64,
                    },
                ));
            }
            let top = encode_index_block(&top_level);
            let top_offset = out.len() as u64;
            out.extend_from_slice(&top);
            (top_offset, top.len() as u64, MAGIC_V2)
        } else {
            let index = encode_index_block(&index_entries);
            let index_offset = out.len() as u64;
            out.extend_from_slice(&index);
            (index_offset, index.len() as u64, MAGIC_V1)
        };

        let footer = Footer {
            range_tombstone_offset,
            range_tombstone_size: 0,
            bloom_offset,
            bloom_size,
            index_offset,
            index_size,
            num_entries: entries.len() as u64,
            magic,
        };
        let encoded = footer.encode();
        assert_eq!(encoded.len(), FOOTER_SIZE_V1);
        out.extend_from_slice(&encoded);
        fs::write(path, out)
    }

    fn legacy_entries() -> Vec<(Vec<u8>, Vec<u8>)> {
        (0..40)
            .map(|i| {
                (
                    ik(format!("key_{i:04}").as_bytes(), 1),
                    format!("value_{i}").into_bytes(),
                )
            })
            .collect()
    }

    fn assert_legacy_table_reads(path: &Path, partitioned: bool) {
        let on_disk = fs::read(path).unwrap();
        assert_eq!(
            on_disk[on_disk.len() - 8],
            if partitioned { 2 } else { 1 },
            "the fixture must carry a legacy format version byte"
        );

        let reader = SsTableReader::open(path, 7).unwrap();
        assert_eq!(reader.partitioned, partitioned);
        assert!(
            !reader.meta_checksummed,
            "a legacy table has no metadata checksums to verify"
        );

        let cache = BlockCache::new(1024 * 1024);
        for i in 0..40 {
            let key = format!("key_{i:04}");
            assert_eq!(
                reader.get(key.as_bytes(), u64::MAX, &cache).unwrap(),
                LookupResult::Found {
                    seq: 1,
                    value: format!("value_{i}").into_bytes(),
                },
                "legacy table lost {key}"
            );
        }
        assert_eq!(
            reader.get(b"key_9999", u64::MAX, &cache).unwrap(),
            LookupResult::NotInTable
        );
        let scanned = reader.iter_internal(&cache).unwrap();
        assert_eq!(scanned.len(), 40);
    }

    /// A table written before the metadata checksums landed must keep
    /// opening and keep serving every key. A format change that cannot
    /// read yesterday's files is a data-loss bug of its own.
    #[test]
    fn open_accepts_a_legacy_v1_footer() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("legacy-v1.sst");
        write_legacy_table(&path, &legacy_entries(), false).unwrap();
        assert_legacy_table_reads(&path, false);
    }

    /// The same for the legacy partitioned layout, whose index leaves
    /// also carry no checksum trailer.
    #[test]
    fn open_accepts_a_legacy_v2_footer() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("legacy-v2.sst");
        write_legacy_table(&path, &legacy_entries(), true).unwrap();
        assert_legacy_table_reads(&path, true);
    }

    fn write_probe_table(path: &Path, entries: &[(Vec<u8>, Vec<u8>)]) {
        let mut writer = SsTableWriter::new(path, 256, 10, CompressionType::None, None, false, 0)
            .expect("writer");
        for (k, v) in entries {
            writer.add(k, v).expect("add");
        }
        writer.finish().expect("finish");
    }

    /// A complete table whose tail a lost or misdirected write zeroed
    /// must come back as an error, never as "holds nothing".
    ///
    /// The orphan-table guard reads this answer to decide whether
    /// opening would discard live data, and the zeroed shape is
    /// indistinguishable from an unfinished flush output. An `Err` makes
    /// the guard count the file and refuse loudly, which is recoverable;
    /// an `Ok(false)` would make it open silently without whatever the
    /// table held, which is not.
    #[test]
    fn a_table_whose_tail_was_zeroed_cannot_be_proved_empty() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("zeroed-tail.sst");
        write_probe_table(&path, &legacy_entries());
        assert!(table_carries_data(&path).unwrap());

        let bytes = fs::read(&path).unwrap();
        // A sector and a block, the granularities a device actually
        // loses, plus the bare footer length.
        for zeroed in [FOOTER_SIZE_V1, 512, 4096] {
            if zeroed >= bytes.len() {
                continue;
            }
            let victim = dir.path().join(format!("zeroed-{zeroed}.sst"));
            let mut damaged = bytes.clone();
            let from = damaged.len() - zeroed;
            damaged[from..].fill(0);
            fs::write(&victim, &damaged).unwrap();
            assert!(
                table_carries_data(&victim).is_err(),
                "a {zeroed}-byte zeroed tail must not read back as \"holds nothing\"",
            );
        }
    }

    /// A legacy footer carries no checksum, so its `num_entries` can read
    /// back as zero over a table that is full. The index block is the
    /// corroborating structure, and the guard must use it.
    #[test]
    fn a_legacy_footer_that_lies_about_its_entry_count_still_counts_as_carrying_data() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("legacy-liar.sst");
        write_legacy_table(&path, &legacy_entries(), false).unwrap();
        assert!(table_carries_data(&path).unwrap());

        let mut bytes = fs::read(&path).unwrap();
        let footer_start = bytes.len() - FOOTER_SIZE_V1;
        // num_entries lives at [48..56) of the footer, and range
        // tombstones at [8..16). Zero both, the way damage would.
        bytes[footer_start + 48..footer_start + 56].fill(0);
        bytes[footer_start + 8..footer_start + 16].fill(0);
        fs::write(&path, &bytes).unwrap();

        assert!(
            table_carries_data(&path).unwrap(),
            "the index block still holds entries, so the table holds data \
             whatever the unchecksummed footer claims",
        );
    }

    /// The writer emits the checksummed form, and the reader agrees with
    /// the module docs about which magic means what.
    #[test]
    fn a_freshly_written_table_carries_the_checksummed_format() {
        for partitioned in [false, true] {
            let dir = TempDir::new().unwrap();
            let path = dir.path().join("fresh.sst");
            let mut writer = SsTableWriter::new(
                &path,
                256,
                10,
                CompressionType::None,
                None,
                partitioned,
                256,
            )
            .unwrap();
            for (internal_key, value) in legacy_entries() {
                writer.add(&internal_key, &value).unwrap();
            }
            writer.finish().unwrap().unwrap();

            let on_disk = fs::read(&path).unwrap();
            assert_eq!(on_disk[on_disk.len() - 8], if partitioned { 4 } else { 3 });
            let reader = SsTableReader::open(&path, 1).unwrap();
            assert!(reader.meta_checksummed);
            assert_eq!(reader.partitioned, partitioned);
            assert!(table_carries_data(&path).unwrap());
        }
    }

    /// Every byte of a partitioned table's index leaves and of its
    /// range-tombstone block sits inside a checksummed region too. A leaf
    /// is read lazily, so a flip there surfaces on the read that reaches
    /// it rather than at open; either is loud, and neither serves an
    /// answer the caller would believe.
    #[test]
    fn a_flip_in_an_index_leaf_or_a_range_tombstone_block_is_caught() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("partitioned-rt.sst");
        let mut writer =
            SsTableWriter::new(&path, 64, 10, CompressionType::None, None, true, 96).unwrap();
        for i in 0..40 {
            writer
                .add(&ik(format!("key_{i:04}").as_bytes(), 1), b"value")
                .unwrap();
        }
        writer.add_range_tombstone(b"key_0000", b"key_0005", 3);
        writer.finish().unwrap().unwrap();

        let clean = fs::read(&path).unwrap();
        let f = &clean[clean.len() - FOOTER_SIZE_V2..];
        let word = |i: usize| u64::from_le_bytes(f[i * 8..i * 8 + 8].try_into().unwrap());
        let range_tombstones = word(0)..word(0) + word(1);
        // Everything between the end of the bloom region and the
        // top-level index is index leaves.
        let leaves = word(2) + word(3)..word(4);
        assert!(
            !range_tombstones.is_empty(),
            "the fixture has no range tombstones"
        );
        assert!(
            leaves.end - leaves.start > 1,
            "the fixture did not partition its index"
        );

        for offset in range_tombstones.chain(leaves) {
            for bit in [0u8, 5] {
                let mut damaged = clean.clone();
                damaged[offset as usize] ^= 1 << bit;
                fs::write(&path, &damaged).unwrap();
                let cache = BlockCache::new(1024 * 1024);
                let caught = match SsTableReader::open(&path, 1) {
                    Err(_) => true,
                    Ok(reader) => reader.iter_internal(&cache).is_err(),
                };
                assert!(caught, "byte {offset} bit {bit} was not caught");
            }
        }
    }

    fn expect_reader_open_err(path: &Path) -> io::Error {
        match SsTableReader::open(path, 1) {
            Err(e) => e,
            Ok(_) => panic!("expected invalid SSTable"),
        }
    }

    // ── reader error paths ──────────────────────────────────────

    #[test]
    fn open_rejects_file_smaller_than_footer() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("tiny.sst");
        fs::write(&path, vec![0u8; 10]).unwrap();
        let kind = match SsTableReader::open(&path, 1) {
            Err(e) => e.kind(),
            Ok(_) => panic!("expected InvalidData"),
        };
        assert_eq!(kind, io::ErrorKind::InvalidData);
    }

    #[test]
    fn open_rejects_file_with_bogus_magic() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("bogus.sst");
        fs::write(&path, vec![0u8; FOOTER_SIZE_V1]).unwrap();
        let kind = match SsTableReader::open(&path, 1) {
            Err(e) => e.kind(),
            Ok(_) => panic!("expected InvalidData"),
        };
        assert_eq!(kind, io::ErrorKind::InvalidData);
    }

    #[test]
    fn open_rejects_footer_region_past_file_data() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("bad-region.sst");
        let footer = Footer {
            range_tombstone_offset: 0,
            range_tombstone_size: 0,
            bloom_offset: 0,
            bloom_size: 8,
            index_offset: 8,
            index_size: 4,
            num_entries: 0,
            magic: MAGIC_V1,
        };
        fs::write(&path, footer.encode()).unwrap();

        let err = expect_reader_open_err(&path);
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn open_missing_file_returns_not_found() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("never.sst");
        let kind = match SsTableReader::open(&path, 1) {
            Err(e) => e.kind(),
            Ok(_) => panic!("expected NotFound"),
        };
        assert_eq!(kind, io::ErrorKind::NotFound);
    }

    #[test]
    fn read_block_rejects_short_frame() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("short-block.bin");
        fs::write(&path, [0u8; 4]).unwrap();
        let file = File::open(&path).unwrap();
        let reader = SsTableReader {
            file: Mutex::new(file),
            data_end: 4,
            file_id: 1,
            index: Vec::new(),
            bloom: BloomFilter::new(Vec::new(), 0),
            prefix_bloom: None,
            range_tombstones: RangeTombstoneSet::default(),
            partitioned: false,
            meta_checksummed: false,
            index_leaf_reads: AtomicUsize::new(0),
        };
        let cache = BlockCache::new(1024);

        let err = match reader.read_block(BlockHandle { offset: 0, size: 4 }, &cache) {
            Err(e) => e,
            Ok(_) => panic!("expected invalid block frame"),
        };
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn block_checksum_covers_compression_type() {
        let payload = b"payload";
        let baseline = checksum::sst_block(COMPRESSION_NONE, payload);
        assert_ne!(baseline, checksum::sst_block(COMPRESSION_LZ4, payload));
    }

    #[test]
    fn read_block_rejects_compression_header_flip() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("header-flip.sst");
        {
            let mut writer =
                SsTableWriter::new(&path, 4096, 10, CompressionType::None, None, false, 4096)
                    .unwrap();
            writer.add(&ik(b"k", 1), b"v").unwrap();
            writer.finish().unwrap().unwrap();
        }

        let mut bytes = fs::read(&path).unwrap();
        bytes[0] = COMPRESSION_LZ4;
        fs::write(&path, bytes).unwrap();

        let reader = SsTableReader::open(&path, 1).unwrap();
        let cache = BlockCache::new(1024);
        let err = match reader.get(b"k", u64::MAX, &cache) {
            Err(e) => e,
            Ok(v) => panic!("expected checksum error, got {v:?}"),
        };
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    // ── writer empty / summary ──────────────────────────────────

    #[test]
    fn finish_on_empty_writer_returns_none() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("empty.sst");
        let writer =
            SsTableWriter::new(&path, 4096, 10, CompressionType::None, None, false, 4096).unwrap();
        assert!(writer.finish().unwrap().is_none());
    }

    #[test]
    fn finish_with_only_range_tombstones_returns_summary() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("rt_only_summary.sst");
        let mut writer =
            SsTableWriter::new(&path, 4096, 10, CompressionType::None, None, false, 4096).unwrap();
        writer.add_range_tombstone(b"a", b"z", 7);
        let summary = writer.finish().unwrap().expect("should produce summary");
        assert_eq!(summary.num_entries, 0);
        assert_eq!(summary.smallest_user_key, b"a");
        assert_eq!(summary.largest_user_key, b"z");
    }

    #[test]
    fn summary_records_first_and_last_user_keys() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("bounds.sst");
        let mut writer =
            SsTableWriter::new(&path, 4096, 10, CompressionType::None, None, false, 4096).unwrap();
        writer.add(&ik(b"alpha", 1), b"1").unwrap();
        writer.add(&ik(b"bravo", 1), b"2").unwrap();
        writer.add(&ik(b"delta", 1), b"3").unwrap();
        let summary = writer.finish().unwrap().unwrap();
        assert_eq!(summary.smallest_user_key, b"alpha");
        assert_eq!(summary.largest_user_key, b"delta");
        assert_eq!(summary.num_entries, 3);
    }

    #[test]
    fn summary_includes_range_tombstone_bounds_with_point_entries() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("mixed_rt_bounds.sst");
        let mut writer =
            SsTableWriter::new(&path, 4096, 10, CompressionType::None, None, false, 4096).unwrap();
        writer.add(&ik(b"m", 1), b"1").unwrap();
        writer.add(&ik(b"n", 1), b"2").unwrap();
        writer.add_range_tombstone(b"a", b"c", 7);
        writer.add_range_tombstone(b"x", b"z", 8);

        let summary = writer.finish().unwrap().unwrap();

        assert_eq!(summary.smallest_user_key, b"a");
        assert_eq!(summary.largest_user_key, b"z");
        assert_eq!(summary.num_entries, 2);
    }

    // ── multi-block / index ─────────────────────────────────────

    #[test]
    fn multi_block_lookup_walks_index_correctly() {
        // Tiny block size forces each entry into (almost) its own
        // block, giving a tall index and exercising the binary-search
        // path.
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("multi.sst");
        let cache = BlockCache::new(1024 * 1024);

        {
            let mut writer =
                SsTableWriter::new(&path, 64, 10, CompressionType::None, None, false, 4096)
                    .unwrap();
            for i in 0..200 {
                let k = format!("k_{:04}", i);
                let v = format!("v_{}", i);
                writer.add(&ik(k.as_bytes(), 1), v.as_bytes()).unwrap();
            }
            writer.finish().unwrap().unwrap();
        }

        let reader = SsTableReader::open(&path, 1).unwrap();
        for i in 0..200 {
            let k = format!("k_{:04}", i);
            let expected = format!("v_{}", i);
            assert_eq!(
                reader.get(k.as_bytes(), u64::MAX, &cache).unwrap(),
                LookupResult::Found {
                    seq: 1,
                    value: expected.into_bytes()
                },
            );
        }
    }

    #[test]
    fn iter_internal_returns_every_entry_in_order() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("iter.sst");
        let cache = BlockCache::new(1024 * 1024);
        let mut writer =
            SsTableWriter::new(&path, 64, 10, CompressionType::None, None, false, 4096).unwrap();
        writer.add(&ik(b"a", 3), b"a3").unwrap();
        writer.add(&ik(b"a", 1), b"a1").unwrap();
        writer.add(&ik(b"b", 2), b"b2").unwrap();
        writer.finish().unwrap().unwrap();

        let reader = SsTableReader::open(&path, 1).unwrap();
        let pairs = reader.iter_internal(&cache).unwrap();
        assert_eq!(pairs.len(), 3);
        // iter_internal preserves raw internal-key order - no dedup,
        // no tombstone hiding.
        let user_keys: Vec<&[u8]> = pairs.iter().map(|(k, _)| user_key_of(k)).collect();
        assert_eq!(user_keys, vec![&b"a"[..], &b"a"[..], &b"b"[..]]);
    }

    #[test]
    fn iter_internal_stream_matches_vec_iterator() {
        for partitioned in [false, true] {
            let dir = TempDir::new().unwrap();
            let path = dir.path().join("stream.sst");
            let cache = BlockCache::new(1024 * 1024);
            let mut writer =
                SsTableWriter::new(&path, 64, 10, CompressionType::None, None, partitioned, 96)
                    .unwrap();
            for i in 0..128 {
                let key = format!("k{i:04}");
                let value = format!("v{i}");
                writer
                    .add(&ik(key.as_bytes(), 1), value.as_bytes())
                    .unwrap();
            }
            writer.finish().unwrap().unwrap();

            let reader = SsTableReader::open(&path, 1).unwrap();
            let expected = reader.iter_internal(&cache).unwrap();
            let mut stream = reader.iter_internal_stream(&cache).unwrap();
            let mut actual = Vec::new();
            while let Some(entry) = stream.next_entry().unwrap() {
                actual.push(entry);
            }
            assert_eq!(actual, expected);
        }
    }

    // ── approximate size ───────────────────────────────────────

    #[test]
    fn approximate_size_empty_range_is_zero() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("approx.sst");
        let mut writer =
            SsTableWriter::new(&path, 4096, 10, CompressionType::None, None, false, 4096).unwrap();
        writer.add(&ik(b"m", 1), b"v").unwrap();
        writer.finish().unwrap().unwrap();

        let reader = SsTableReader::open(&path, 1).unwrap();
        // start >= end: guaranteed zero.
        assert_eq!(reader.approximate_size_in_range(b"z", b"a"), 0);
        assert_eq!(reader.approximate_size_in_range(b"m", b"m"), 0);
    }

    #[test]
    fn approximate_size_grows_with_range_width() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("approx_grow.sst");
        let mut writer =
            SsTableWriter::new(&path, 128, 10, CompressionType::None, None, false, 4096).unwrap();
        for i in 0..500 {
            writer
                .add(&ik(format!("k_{:04}", i).as_bytes(), 1), b"value")
                .unwrap();
        }
        writer.finish().unwrap().unwrap();

        let reader = SsTableReader::open(&path, 1).unwrap();
        let narrow = reader.approximate_size_in_range(b"k_0000", b"k_0001");
        let wide = reader.approximate_size_in_range(b"k_0000", b"k_0499");
        assert!(
            wide > narrow,
            "wide ({wide}) should exceed narrow ({narrow})"
        );
    }

    // ── compression variants ────────────────────────────────────

    #[test]
    fn lz4_compression_round_trips() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("lz4.sst");
        let cache = BlockCache::new(1024 * 1024);
        {
            let mut writer =
                SsTableWriter::new(&path, 4096, 10, CompressionType::Lz4, None, false, 4096)
                    .unwrap();
            // Repetitive data compresses well - exercises a realistic path.
            for i in 0..50 {
                writer
                    .add(
                        &ik(format!("key_{:03}", i).as_bytes(), 1),
                        b"AAAAAAAAAAAAAAAAAAAAAAAAAA",
                    )
                    .unwrap();
            }
            writer.finish().unwrap().unwrap();
        }
        let reader = SsTableReader::open(&path, 1).unwrap();
        assert_eq!(
            reader.get(b"key_025", u64::MAX, &cache).unwrap(),
            LookupResult::Found {
                seq: 1,
                value: b"AAAAAAAAAAAAAAAAAAAAAAAAAA".to_vec()
            }
        );
    }

    #[test]
    fn snappy_compression_round_trips() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("snappy.sst");
        let cache = BlockCache::new(1024 * 1024);
        {
            let mut writer =
                SsTableWriter::new(&path, 4096, 10, CompressionType::Snappy, None, false, 4096)
                    .unwrap();
            for i in 0..50 {
                writer
                    .add(
                        &ik(format!("key_{:03}", i).as_bytes(), 1),
                        b"BBBBBBBBBBBBBBBBBBBBBBBBBB",
                    )
                    .unwrap();
            }
            writer.finish().unwrap().unwrap();
        }
        let reader = SsTableReader::open(&path, 1).unwrap();
        assert_eq!(
            reader.get(b"key_007", u64::MAX, &cache).unwrap(),
            LookupResult::Found {
                seq: 1,
                value: b"BBBBBBBBBBBBBBBBBBBBBBBBBB".to_vec()
            }
        );
    }

    // ── range tombstone encode/decode ──────────────────────────

    #[test]
    fn range_tombstone_block_round_trip() {
        let input = vec![
            RangeTombstone::new(b"a".to_vec(), b"c".to_vec(), 5),
            RangeTombstone::new(b"f".to_vec(), b"k".to_vec(), 7),
        ];
        let bytes = encode_range_tombstone_block(&input);
        let got = decode_range_tombstone_block(&bytes).unwrap();
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].start, b"a");
        assert_eq!(got[0].end, b"c");
        assert_eq!(got[0].seq, 5);
        assert_eq!(got[1].seq, 7);
    }

    #[test]
    fn range_tombstone_block_empty_input() {
        assert!(decode_range_tombstone_block(&[]).unwrap().is_empty());
    }

    #[test]
    fn range_tombstone_block_rejects_tiny_header() {
        // 1-3 bytes is ambiguous: not empty, not enough for a count.
        let kind = match decode_range_tombstone_block(&[0, 1]) {
            Err(e) => e.kind(),
            Ok(v) => panic!("expected error, got {} tombstones", v.len()),
        };
        assert_eq!(kind, io::ErrorKind::InvalidData);
    }

    #[test]
    fn range_tombstone_block_rejects_truncated_mid_record() {
        // Valid count = 2, but only the first record is complete. The
        // decoder must reject the block rather than silently keeping
        // a prefix of the range-tombstone metadata.
        let mut bytes = vec![];
        bytes.extend_from_slice(&2u32.to_le_bytes());
        // First record: start "aa", end "bb", seq=1 → 4+2+4+2+8 = 20
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(b"aa");
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(b"bb");
        bytes.extend_from_slice(&1u64.to_le_bytes());
        // Second record: deliberately truncated after start_len
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.push(b'c'); // incomplete start

        let err = decode_range_tombstone_block(&bytes).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn range_tombstone_block_rejects_trailing_bytes() {
        let mut bytes =
            encode_range_tombstone_block(&[RangeTombstone::new(b"a".to_vec(), b"b".to_vec(), 1)]);
        bytes.push(0xAA);

        let err = decode_range_tombstone_block(&bytes).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    // ── index block decode ─────────────────────────────────────

    #[test]
    fn index_block_rejects_tiny_header() {
        let err = decode_index_block(&[1, 2]).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn index_block_rejects_truncated_entry() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&4u32.to_le_bytes());
        bytes.extend_from_slice(b"ab");

        let err = decode_index_block(&bytes).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn index_block_rejects_trailing_bytes() {
        let mut bytes = encode_index_block(&[]);
        bytes.push(0xAA);

        let err = decode_index_block(&bytes).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn reader_with_no_range_tombstones_reports_empty_list() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("no_rt.sst");
        {
            let mut writer =
                SsTableWriter::new(&path, 4096, 10, CompressionType::None, None, false, 4096)
                    .unwrap();
            writer.add(&ik(b"k", 1), b"v").unwrap();
            writer.finish().unwrap().unwrap();
        }
        let reader = SsTableReader::open(&path, 1).unwrap();
        assert!(reader.range_tombstones().is_empty());
        assert_eq!(reader.covering_range_tombstone_seq(b"k", u64::MAX), 0);
    }

    // ── bloom short-circuit ─────────────────────────────────────

    #[test]
    fn negative_lookup_on_many_keys_mostly_short_circuits() {
        // Indirect test: with 10 bpk the bloom rejects >99% of absent
        // keys without touching blocks. We can't directly observe
        // block reads but can verify lookup returns NotInTable fast
        // for all absent keys.
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("bloomy.sst");
        let cache = BlockCache::new(1024 * 1024);
        {
            let mut writer =
                SsTableWriter::new(&path, 4096, 10, CompressionType::None, None, false, 4096)
                    .unwrap();
            for i in 0..1000 {
                writer
                    .add(&ik(format!("hit_{:04}", i).as_bytes(), 1), b"v")
                    .unwrap();
            }
            writer.finish().unwrap().unwrap();
        }
        let reader = SsTableReader::open(&path, 1).unwrap();
        for i in 0..5000 {
            let absent = format!("miss_{:05}", i);
            assert_eq!(
                reader.get(absent.as_bytes(), u64::MAX, &cache).unwrap(),
                LookupResult::NotInTable
            );
        }
    }

    // ── partitioned index variant ──────────────────────────────

    fn write_partitioned_fixture(path: &Path) {
        let mut writer =
            SsTableWriter::new(path, 96, 10, CompressionType::None, None, true, 96).unwrap();
        for i in 0..300 {
            writer
                .add(&ik(format!("k_{:04}", i).as_bytes(), 1), b"value")
                .unwrap();
        }
        writer.finish().unwrap().unwrap();
    }

    #[test]
    fn partitioned_index_round_trip() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("partitioned.sst");
        let cache = BlockCache::new(1024 * 1024);
        write_partitioned_fixture(&path);
        let reader = SsTableReader::open(&path, 1).unwrap();
        // Spot-check across the range.
        for i in [0usize, 75, 150, 225, 299] {
            let k = format!("k_{:04}", i);
            assert_eq!(
                reader.get(k.as_bytes(), u64::MAX, &cache).unwrap(),
                LookupResult::Found {
                    seq: 1,
                    value: b"value".to_vec(),
                }
            );
        }
    }

    #[test]
    fn partitioned_point_lookup_reads_one_index_leaf() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("partitioned_point.sst");
        let cache = BlockCache::new(1024 * 1024);
        write_partitioned_fixture(&path);

        let reader = SsTableReader::open(&path, 1).unwrap();
        assert!(
            reader.index.len() > 3,
            "fixture must contain several index leaves"
        );

        reader.reset_index_leaf_read_count();
        assert_eq!(
            reader.get(b"k_0150", u64::MAX, &cache).unwrap(),
            LookupResult::Found {
                seq: 1,
                value: b"value".to_vec(),
            }
        );
        assert_eq!(
            reader.index_leaf_read_count(),
            1,
            "point lookup must load only the selected leaf index"
        );
    }

    #[test]
    fn partitioned_internal_stream_starts_without_expanding_all_leaves() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("partitioned_stream.sst");
        let cache = BlockCache::new(1024 * 1024);
        write_partitioned_fixture(&path);

        let reader = SsTableReader::open(&path, 1).unwrap();
        let leaf_count = reader.index.len();
        assert!(leaf_count > 3, "fixture must contain several index leaves");

        reader.reset_index_leaf_read_count();
        let mut stream = reader.iter_internal_stream(&cache).unwrap();
        assert_eq!(
            reader.index_leaf_read_count(),
            1,
            "stream construction should open only the first leaf"
        );

        let first = stream.next_entry().unwrap().expect("fixture has entries");
        assert_eq!(user_key_of(&first.0), b"k_0000");
        assert!(
            reader.index_leaf_read_count() < leaf_count,
            "reading the first entry must not expand every leaf"
        );
    }

    #[test]
    fn partitioned_approximate_size_reads_only_overlapped_leaves() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("partitioned_approx.sst");
        write_partitioned_fixture(&path);

        let reader = SsTableReader::open(&path, 1).unwrap();
        let leaf_count = reader.index.len();
        assert!(leaf_count > 3, "fixture must contain several index leaves");

        reader.reset_index_leaf_read_count();
        assert!(reader.approximate_size_in_range(b"k_0100", b"k_0101") > 0);
        assert!(
            reader.index_leaf_read_count() <= 2,
            "narrow range should load at most the boundary leaves"
        );
        assert!(
            reader.index_leaf_read_count() < leaf_count,
            "narrow range must not expand every leaf"
        );
    }
}
