//! The write-ahead log: the durable record of every write between one
//! memtable flush and the next.
//!
//! # On-disk format
//!
//! A WAL file is a bare sequence of records with no file header and no
//! framing above the record itself:
//!
//! ```text
//! [len: u32 LE][type: u8][payload: len bytes][checksum: u32 LE]
//! ```
//!
//! The checksum covers the length, the type byte and the payload, so the
//! length field is both what finds the next record and part of what the
//! checksum protects.
//!
//! # How replay tells a crash from corruption
//!
//! Replay runs after something has already gone wrong, so it has to
//! separate the ordinary shape of a crash from real damage. The question
//! it asks of each record is whether the record is *whole*: a record is
//! whole when the file holds its five-byte header, the `len` payload
//! bytes that header promises, and the four checksum bytes after them.
//!
//! * A record the file ends inside is not whole. That is what a process
//!   killed part way through a `write` leaves behind, so replay stops
//!   there, keeps every record before it, and reports the discarded tail
//!   through `tracing`. No acknowledged write is lost by that: under
//!   `DurabilityMode::Immediate` a write returns only once the `fsync` of
//!   its own whole record has returned, and under `Eventual` no
//!   durability was promised in the first place.
//! * A whole record that fails its checksum, carries an unknown type, or
//!   does not parse is corruption, and is an error, with one proven
//!   exception below. Damage that leaves every byte the record claims
//!   present, and wrong, is what bit rot looks like, not what a torn
//!   write looks like.
//! * The exception: a run of zero bytes reaching end-of-file is
//!   unwritten space, not a damaged record. A filesystem that allocates
//!   blocks and does not write them (ext4 with delayed allocation, after
//!   a power cut) reads the tail back as zeros at full length rather
//!   than short, so the tail frames as a whole record: `len` 0, type
//!   `0x00`, checksum 0. lark never writes that shape - it emits no
//!   record of type `0x00`, and the checksum of the empty type-`0x00`
//!   record is not zero - and no bit flip in a written record can
//!   produce it either, because flipping a bit inside real bytes leaves
//!   at least one non-zero byte behind. Replay therefore stops at the
//!   first byte of an all-zero tail and reports it exactly as it reports
//!   a short one. Nothing weaker is accepted: a tail with a single
//!   non-zero byte in it is corruption again.
//! * An incomplete record from which the rest of the file still parses as
//!   whole records, ending exactly at the last byte, is not a torn tail
//!   either. A torn write leaves nothing behind it, so a remainder that
//!   tiles that way means real records lie beyond the damage and the
//!   length field was mangled; stopping there would discard them
//!   silently. Replay refuses, naming the file and both offsets.
//!
//! Two limits of that rule are worth stating, because both are properties
//! of the format rather than of this implementation.
//!
//! The checksum sits after the payload, so the length has to be trusted to
//! find it. Damage to the length field of the *final* record is therefore
//! indistinguishable from a torn tail and is treated as one. The blast
//! radius is that one record, the discard is reported, and no partial or
//! invented data is ever served.
//!
//! Damage in the middle of a log whose final record is *also* torn leaves
//! a remainder that no longer tiles, so it reads as a torn tail starting
//! at the earlier damage and the whole rest of the log is discarded. That
//! needs bit rot and a crash in the same short-lived file, and the discard
//! is reported with the offset and the byte count rather than passing
//! silently.
//!
//! A *partly* zeroed final record - real header, payload zeroed from a
//! block boundary inside it - is neither short nor all-zero, so it still
//! refuses. Under `DurabilityMode::Immediate` that shape is unreachable,
//! because every acknowledged record is fsynced whole before the next one
//! starts, so the zero-fill boundary is always a record boundary. Under
//! `Eventual` it is reachable and costs the open; nothing acknowledged is
//! lost by it, because `Eventual` acknowledges no durability.
//!
//! Closing any of these properly needs a self-checksummed header or
//! fixed-size blocks, which is a format change.
//!
//! # Which file the torn-tail rule may be applied to
//!
//! The rule is only sound for the newest WAL file that contributes to
//! replay. A torn write leaves nothing after it *anywhere*, so an earlier
//! file that ends inside a record while a later file still holds records
//! is damage in the middle of the history, not a crash artifact.
//! [`Wal::replay`] reports its truncation rather than deciding, because
//! only the caller knows the file order; `LarkEngine::open` refuses when
//! a truncated file is followed by one that yielded records.

use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

use super::{checksum, durability};
use crate::WriteBatchOp;

/// Record types in the WAL.
const RECORD_PUT: u8 = 0x01;
const RECORD_DELETE: u8 = 0x02;
const RECORD_DELETE_RANGE: u8 = 0x03;
const RECORD_MERGE: u8 = 0x04;
const RECORD_BATCH: u8 = 0x05;

/// On-disk record header: 4-byte little-endian payload length plus a
/// one-byte record type.
const WAL_HEADER_LEN: usize = 5;
/// Trailing 4-byte little-endian checksum of every record.
const CHECKSUM_LEN: usize = 4;

/// A write-ahead log for crash recovery.
///
/// Records are append-only and carry fast non-cryptographic checksums for
/// torn-write and bit-rot detection. On crash recovery, WAL files are
/// replayed to reconstruct memtable state.
pub(crate) struct Wal {
    writer: BufWriter<File>,
    path: PathBuf,
    parent_synced: bool,
}

/// A replayed WAL entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum WalEntry {
    Put {
        key: Vec<u8>,
        value: Vec<u8>,
        seq: u64,
    },
    Delete {
        key: Vec<u8>,
        seq: u64,
    },
    DeleteRange {
        start: Vec<u8>,
        end: Vec<u8>,
        seq: u64,
    },
    Merge {
        key: Vec<u8>,
        operand: Vec<u8>,
        seq: u64,
    },
}

impl Wal {
    /// Create a new WAL file at the given path.
    pub(crate) fn create(path: &Path) -> io::Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)?;
        Ok(Self {
            writer: BufWriter::new(file),
            path: path.to_path_buf(),
            parent_synced: false,
        })
    }

    /// Append a put record.
    pub(crate) fn append_put(&mut self, key: &[u8], value: &[u8], seq: u64) -> io::Result<()> {
        let data_len = 4 + key.len() + 4 + value.len() + 8;
        let mut data = Vec::with_capacity(data_len);

        encode_put_payload(&mut data, key, value, seq);

        self.write_record(RECORD_PUT, &data)
    }

    /// Append a delete record.
    pub(crate) fn append_delete(&mut self, key: &[u8], seq: u64) -> io::Result<()> {
        let mut data = Vec::with_capacity(4 + key.len() + 8);

        encode_delete_payload(&mut data, key, seq);

        self.write_record(RECORD_DELETE, &data)
    }

    /// Append a merge record - an operand layered on top of any
    /// existing value/merge chain for `key`.
    pub(crate) fn append_merge(&mut self, key: &[u8], operand: &[u8], seq: u64) -> io::Result<()> {
        let mut data = Vec::with_capacity(4 + key.len() + 4 + operand.len() + 8);
        encode_merge_payload(&mut data, key, operand, seq);
        self.write_record(RECORD_MERGE, &data)
    }

    /// Append a range-delete record covering `[start, end)`.
    pub(crate) fn append_delete_range(
        &mut self,
        start: &[u8],
        end: &[u8],
        seq: u64,
    ) -> io::Result<()> {
        let mut data = Vec::with_capacity(4 + start.len() + 4 + end.len() + 8);
        encode_delete_range_payload(&mut data, start, end, seq);

        self.write_record(RECORD_DELETE_RANGE, &data)
    }

    /// Append a batch record from write-batch operations without first
    /// cloning them into replay entries.
    pub(crate) fn append_ops_batch(
        &mut self,
        ops: &[WriteBatchOp],
        base_seq: u64,
    ) -> io::Result<()> {
        if ops.is_empty() {
            return Ok(());
        }

        let mut data = Vec::with_capacity(batch_ops_payload_len(ops));
        data.extend_from_slice(&(ops.len() as u32).to_le_bytes());

        for (i, op) in ops.iter().enumerate() {
            encode_batch_op(&mut data, op, base_seq + i as u64);
        }

        self.write_record(RECORD_BATCH, &data)
    }

    /// Flush and fsync the WAL to disk.
    pub(crate) fn sync(&mut self) -> io::Result<()> {
        self.sync_with_parent_sync(durability::sync_parent_dir)
    }

    fn sync_with_parent_sync(
        &mut self,
        mut sync_parent: impl FnMut(&Path) -> io::Result<()>,
    ) -> io::Result<()> {
        self.writer.flush()?;
        self.writer.get_ref().sync_all()?;
        if !self.parent_synced {
            sync_parent(&self.path)?;
            self.parent_synced = true;
        }
        Ok(())
    }

    /// Flush the buffer (without fsync).
    pub(crate) fn flush(&mut self) -> io::Result<()> {
        self.writer.flush()
    }

    /// Get the path to this WAL file.
    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    fn write_record(&mut self, record_type: u8, data: &[u8]) -> io::Result<()> {
        let len = data.len() as u32;
        let checksum = checksum::wal_record(len, record_type, data);

        self.writer.write_all(&len.to_le_bytes())?;
        self.writer.write_all(&[record_type])?;
        self.writer.write_all(data)?;
        self.writer.write_all(&checksum.to_le_bytes())?;

        Ok(())
    }

    /// Replay a WAL file and return every whole record it holds, plus
    /// what, if anything, was discarded from the tail.
    ///
    /// A record the file ends inside, and a run of zero bytes reaching
    /// end-of-file, are both the ordinary shape of a crash and are
    /// treated as the end of the log: the records before come back and
    /// the discarded tail is reported in the returned [`WalReplay`] and
    /// through `tracing`. Any other whole record that fails its
    /// checksum, carries an unknown type, or does not parse is
    /// corruption and is an error, and so is an incomplete record that
    /// still has whole records after it. The module documentation says
    /// why those cases are told apart this way, and names the cases the
    /// format cannot tell apart.
    ///
    /// The caller owns one further decision this function cannot make:
    /// a discarded tail is only a crash artifact in the newest WAL file
    /// that contributes to replay. See the module docs.
    pub(crate) fn replay(path: &Path) -> io::Result<WalReplay> {
        // Read the file whole: deciding whether an incomplete record is a
        // torn tail or a mangled length field means looking at the bytes
        // after it, and deciding whether a tail was ever written means
        // looking at all of it. A WAL is bounded by `write_buffer_size`
        // plus the one record that crossed it. The decoded entries are
        // already larger than the file, and this holds the file bytes
        // alongside them, so recovery's peak is about twice the WAL
        // rather than about once it: bounded by the same option, at a
        // constant factor.
        let bytes = fs::read(path)?;
        let mut entries = Vec::new();
        let mut truncated_at = None;
        let mut pos = 0usize;

        while pos < bytes.len() {
            let Some(frame) = frame_at(&bytes, pos) else {
                end_of_log_or_corruption(path, &bytes, pos)?;
                truncated_at = Some(pos);
                break;
            };

            let usable_type = matches!(
                frame.record_type,
                RECORD_PUT | RECORD_DELETE | RECORD_DELETE_RANGE | RECORD_MERGE | RECORD_BATCH
            );

            // Hashing the record is the expensive part of replay, so the
            // verdict is computed once here and reused by both the
            // torn-tail check and the corruption check below. Computing it
            // per use hashes every byte of the log twice on the ordinary
            // recovery path, where every record is whole and valid.
            let checksum_ok = frame.checksum_matches();

            if (!usable_type || !checksum_ok) && tail_is_unwritten(&bytes, pos) {
                report_discarded_tail(path, bytes.len() - pos, pos);
                truncated_at = Some(pos);
                break;
            }

            if !checksum_ok {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "WAL checksum mismatch in {} at offset {pos}",
                        path.display()
                    ),
                ));
            }

            match frame.record_type {
                RECORD_PUT => entries.push(parse_put_record(frame.data)?),
                RECORD_DELETE => entries.push(parse_delete_record(frame.data)?),
                RECORD_DELETE_RANGE => entries.push(parse_delete_range_record(frame.data)?),
                RECORD_MERGE => entries.push(parse_merge_record(frame.data)?),
                RECORD_BATCH => entries.extend(parse_batch_record(frame.data)?),
                other => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "unknown WAL record type {other} in {} at offset {pos}",
                            path.display()
                        ),
                    ));
                }
            }

            pos = frame.end;
        }

        let discarded_bytes = truncated_at.map_or(0, |at| (bytes.len() - at) as u64);
        Ok(WalReplay {
            entries,
            truncated_at: truncated_at.map(|at| at as u64),
            discarded_bytes,
        })
    }

    /// Delete a WAL file.
    pub(crate) fn remove(path: &Path) -> io::Result<()> {
        durability::remove_file_and_sync_parent(path)
    }
}

/// What one [`Wal::replay`] produced.
pub(crate) struct WalReplay {
    /// Every whole record decoded, in file order.
    pub(crate) entries: Vec<WalEntry>,
    /// `Some(offset)` when a trailing record was incomplete or unwritten
    /// and the bytes from `offset` to end-of-file were discarded as a
    /// crash artifact. `None` when the file replayed to its last byte.
    pub(crate) truncated_at: Option<u64>,
    /// How many bytes `truncated_at` discarded. Zero when `None`.
    pub(crate) discarded_bytes: u64,
}

/// Format a WAL filename from a numeric ID.
pub(crate) fn wal_filename(id: u64) -> String {
    format!("wal_{:06}.log", id)
}

fn encode_put_payload(out: &mut Vec<u8>, key: &[u8], value: &[u8], seq: u64) {
    out.extend_from_slice(&(key.len() as u32).to_le_bytes());
    out.extend_from_slice(key);
    out.extend_from_slice(&(value.len() as u32).to_le_bytes());
    out.extend_from_slice(value);
    out.extend_from_slice(&seq.to_le_bytes());
}

fn encode_delete_payload(out: &mut Vec<u8>, key: &[u8], seq: u64) {
    out.extend_from_slice(&(key.len() as u32).to_le_bytes());
    out.extend_from_slice(key);
    out.extend_from_slice(&seq.to_le_bytes());
}

fn encode_delete_range_payload(out: &mut Vec<u8>, start: &[u8], end: &[u8], seq: u64) {
    out.extend_from_slice(&(start.len() as u32).to_le_bytes());
    out.extend_from_slice(start);
    out.extend_from_slice(&(end.len() as u32).to_le_bytes());
    out.extend_from_slice(end);
    out.extend_from_slice(&seq.to_le_bytes());
}

fn encode_merge_payload(out: &mut Vec<u8>, key: &[u8], operand: &[u8], seq: u64) {
    out.extend_from_slice(&(key.len() as u32).to_le_bytes());
    out.extend_from_slice(key);
    out.extend_from_slice(&(operand.len() as u32).to_le_bytes());
    out.extend_from_slice(operand);
    out.extend_from_slice(&seq.to_le_bytes());
}

fn put_payload_len(key: &[u8], value: &[u8]) -> usize {
    4 + key.len() + 4 + value.len() + 8
}

fn delete_payload_len(key: &[u8]) -> usize {
    4 + key.len() + 8
}

fn delete_range_payload_len(start: &[u8], end: &[u8]) -> usize {
    4 + start.len() + 4 + end.len() + 8
}

fn merge_payload_len(key: &[u8], operand: &[u8]) -> usize {
    4 + key.len() + 4 + operand.len() + 8
}

fn batch_op_payload_len(op: &WriteBatchOp) -> usize {
    match op {
        WriteBatchOp::Put { key, value } => put_payload_len(key, value),
        WriteBatchOp::Delete { key } => delete_payload_len(key),
        WriteBatchOp::DeleteRange { start, end } => delete_range_payload_len(start, end),
        WriteBatchOp::Merge { key, operand } => merge_payload_len(key, operand),
    }
}

fn batch_payload_len(entry_payload_len: usize) -> usize {
    1 + 4 + entry_payload_len
}

fn batch_ops_payload_len(ops: &[WriteBatchOp]) -> usize {
    4 + ops
        .iter()
        .map(|op| batch_payload_len(batch_op_payload_len(op)))
        .sum::<usize>()
}

fn encode_batch_header(out: &mut Vec<u8>, record_type: u8, payload_len: usize) {
    out.push(record_type);
    out.extend_from_slice(&(payload_len as u32).to_le_bytes());
}

fn encode_batch_op(out: &mut Vec<u8>, op: &WriteBatchOp, seq: u64) {
    match op {
        WriteBatchOp::Put { key, value } => {
            encode_batch_header(out, RECORD_PUT, put_payload_len(key, value));
            encode_put_payload(out, key, value, seq);
        }
        WriteBatchOp::Delete { key } => {
            encode_batch_header(out, RECORD_DELETE, delete_payload_len(key));
            encode_delete_payload(out, key, seq);
        }
        WriteBatchOp::DeleteRange { start, end } => {
            encode_batch_header(
                out,
                RECORD_DELETE_RANGE,
                delete_range_payload_len(start, end),
            );
            encode_delete_range_payload(out, start, end, seq);
        }
        WriteBatchOp::Merge { key, operand } => {
            encode_batch_header(out, RECORD_MERGE, merge_payload_len(key, operand));
            encode_merge_payload(out, key, operand, seq);
        }
    }
}

/// One record framed inside a WAL file.
struct Frame<'a> {
    record_type: u8,
    data: &'a [u8],
    stored_checksum: u32,
    /// Offset one past this record's checksum: where the next record
    /// starts.
    end: usize,
}

// Payload bytes this thread has fed to the record checksum since the
// counter was last taken. Replay's cost is dominated by hashing, and the
// guard against a quadratic resync scan has to be a number that does not
// move with machine load. A thread-local makes the count exact per test
// even though the harness runs tests in parallel, and `#[cfg(test)]`
// keeps every trace of it out of the shipped library.
#[cfg(test)]
thread_local! {
    static CHECKSUM_BYTES: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Take and reset this thread's checksummed-byte count.
#[cfg(test)]
fn take_checksum_bytes() -> u64 {
    CHECKSUM_BYTES.with(|c| c.replace(0))
}

impl Frame<'_> {
    /// Whether the stored checksum matches the bytes of this record.
    /// Logs written before the header joined the checksum's coverage
    /// stored a payload-only checksum, which is still accepted.
    fn checksum_matches(&self) -> bool {
        #[cfg(test)]
        CHECKSUM_BYTES.with(|c| c.set(c.get() + self.data.len() as u64));

        let len = self.data.len() as u32;
        self.stored_checksum == checksum::wal_record(len, self.record_type, self.data)
            || self.stored_checksum == checksum::legacy_payload_u32(self.data)
    }
}

/// Frame the record starting at `offset`, or `None` when the file ends
/// inside it.
///
/// The length field is untrusted input, so nothing is sized from it until
/// the bytes it promises are known to be present: a five-byte header left
/// by a torn write cannot make recovery ask the allocator for 4 GiB.
fn frame_at(bytes: &[u8], offset: usize) -> Option<Frame<'_>> {
    let data_start = offset.checked_add(WAL_HEADER_LEN)?;
    let header = bytes.get(offset..data_start)?;
    let len = u32::from_le_bytes([header[0], header[1], header[2], header[3]]) as usize;
    let data_end = data_start.checked_add(len)?;
    let end = data_end.checked_add(CHECKSUM_LEN)?;
    let data = bytes.get(data_start..data_end)?;
    let stored = bytes.get(data_end..end)?;

    Some(Frame {
        record_type: header[4],
        data,
        stored_checksum: u32::from_le_bytes([stored[0], stored[1], stored[2], stored[3]]),
        end,
    })
}

/// Decide what the incomplete record at `pos` means, and say so.
///
/// `Ok(())` means the tail is a crash artifact and therefore the end of
/// the log; the error means whole records still follow it, which no torn
/// write can produce.
fn end_of_log_or_corruption(path: &Path, bytes: &[u8], pos: usize) -> io::Result<()> {
    let discarded_bytes = bytes.len() - pos;

    if let Some(next) = resync_after(bytes, pos) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "{} is corrupt: the WAL record at offset {pos} runs past the end of the \
                 file, but a whole record follows it at offset {next}, so the \
                 {discarded_bytes} trailing byte(s) are damage rather than a torn write. \
                 Refusing to open rather than discard them.",
                path.display()
            ),
        ));
    }

    report_discarded_tail(path, discarded_bytes, pos);
    Ok(())
}

/// Log one discarded WAL tail. The single place that phrases it, so the
/// short-tail and the zero-tail paths cannot drift apart.
fn report_discarded_tail(path: &Path, discarded_bytes: usize, pos: usize) {
    tracing::warn!(
        path = %path.display(),
        offset = pos,
        discarded_bytes,
        "discarded an incomplete trailing WAL record left by a crash"
    );
}

/// Whether everything from `pos` to end-of-file is zero bytes.
///
/// A zero run reaching end-of-file is space the filesystem allocated and
/// never wrote, which is what ext4 leaves after a power cut when the
/// inode's new length reached the journal and the data blocks did not.
/// It is not a record lark ever emitted: every record type lark writes is
/// non-zero, so a zero run can only frame as `len` 0 with type `0x00`,
/// and no bit flip in a written record can turn its whole tail to zeros.
fn tail_is_unwritten(bytes: &[u8], pos: usize) -> bool {
    bytes[pos..].iter().all(|b| *b == 0)
}

/// Offset of the first whole, checksum-valid, known-type record after the
/// incomplete one at `pos` from which the rest of the file parses as
/// nothing but whole records, ending exactly at the last byte.
///
/// `Some` means real records lie beyond the damage, so the length field of
/// the record at `pos` was mangled rather than its write being torn.
///
/// Requiring the whole remainder to tile is what makes this affordable.
/// Testing each offset on its own would checksum every candidate payload,
/// and a torn tail whose bytes read as plausible lengths (a large value of
/// repeated `0x01` bytes, say) would put gigabytes through the hash per
/// megabyte of tail. The tiling is computed once, backwards, in a single
/// linear pass: each offset is `Some` frame plus one lookup of the answer
/// already computed for the offset the frame ends at. Only the handful of
/// offsets that survive that are ever checksummed. It also sharpens the
/// evidence, because a chance 32-bit checksum match inside a partly
/// written payload now has to land on a tiling as well before it can
/// refuse an open.
fn resync_after(bytes: &[u8], pos: usize) -> Option<usize> {
    let scan_start = pos + 1;
    if scan_start >= bytes.len() {
        return None;
    }

    let mut tiles = vec![false; bytes.len() - scan_start];
    let mut first = None;

    for offset in (scan_start..bytes.len()).rev() {
        let Some(frame) = frame_at(bytes, offset) else {
            continue;
        };
        if frame.end != bytes.len() && !tiles[frame.end - scan_start] {
            continue;
        }
        tiles[offset - scan_start] = true;

        if matches!(
            frame.record_type,
            RECORD_PUT | RECORD_DELETE | RECORD_DELETE_RANGE | RECORD_MERGE | RECORD_BATCH
        ) && frame.checksum_matches()
        {
            first = Some(offset);
        }
    }

    first
}

fn parse_put_record(data: &[u8]) -> io::Result<WalEntry> {
    if data.len() < 16 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "put record too short",
        ));
    }

    let mut pos = 0;
    let key_len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
    pos += 4;

    if pos + key_len + 4 > data.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "put record key overflow",
        ));
    }

    let key = data[pos..pos + key_len].to_vec();
    pos += key_len;

    let value_len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
    pos += 4;

    if pos + value_len + 8 > data.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "put record value overflow",
        ));
    }

    let value = data[pos..pos + value_len].to_vec();
    pos += value_len;

    let seq = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());

    Ok(WalEntry::Put { key, value, seq })
}

fn parse_delete_record(data: &[u8]) -> io::Result<WalEntry> {
    if data.len() < 12 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "delete record too short",
        ));
    }

    let key_len = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
    if 4 + key_len + 8 > data.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "delete record key overflow",
        ));
    }

    let key = data[4..4 + key_len].to_vec();
    let seq = u64::from_le_bytes(data[4 + key_len..4 + key_len + 8].try_into().unwrap());

    Ok(WalEntry::Delete { key, seq })
}

fn parse_delete_range_record(data: &[u8]) -> io::Result<WalEntry> {
    if data.len() < 16 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "delete_range record too short",
        ));
    }

    let mut pos = 0;
    let start_len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
    pos += 4;

    if pos + start_len + 4 > data.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "delete_range record start overflow",
        ));
    }
    let start = data[pos..pos + start_len].to_vec();
    pos += start_len;

    let end_len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
    pos += 4;

    if pos + end_len + 8 > data.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "delete_range record end overflow",
        ));
    }
    let end = data[pos..pos + end_len].to_vec();
    pos += end_len;

    let seq = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());

    Ok(WalEntry::DeleteRange { start, end, seq })
}

fn parse_merge_record(data: &[u8]) -> io::Result<WalEntry> {
    if data.len() < 16 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "merge record too short",
        ));
    }

    let mut pos = 0;
    let key_len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
    pos += 4;

    if pos + key_len + 4 > data.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "merge record key overflow",
        ));
    }

    let key = data[pos..pos + key_len].to_vec();
    pos += key_len;

    let operand_len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
    pos += 4;

    if pos + operand_len + 8 > data.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "merge record operand overflow",
        ));
    }

    let operand = data[pos..pos + operand_len].to_vec();
    pos += operand_len;

    let seq = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());

    Ok(WalEntry::Merge { key, operand, seq })
}

fn parse_batch_record(data: &[u8]) -> io::Result<Vec<WalEntry>> {
    if data.len() < 4 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "batch record too short",
        ));
    }

    let mut pos = 0;
    let count = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
    pos += 4;

    let mut entries = Vec::new();
    for _ in 0..count {
        if pos + 5 > data.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "batch entry header overflow",
            ));
        }

        let record_type = data[pos];
        pos += 1;

        let payload_len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;

        if payload_len > data.len().saturating_sub(pos) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "batch entry payload overflow",
            ));
        }

        let payload = &data[pos..pos + payload_len];
        pos += payload_len;

        let entry = match record_type {
            RECORD_PUT => parse_put_record(payload)?,
            RECORD_DELETE => parse_delete_record(payload)?,
            RECORD_DELETE_RANGE => parse_delete_range_record(payload)?,
            RECORD_MERGE => parse_merge_record(payload)?,
            RECORD_BATCH => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "nested batch records are not supported",
                ));
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unknown WAL batch entry type {record_type}"),
                ));
            }
        };
        entries.push(entry);
    }

    if pos != data.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "batch record has trailing bytes",
        ));
    }

    Ok(entries)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    // ── helpers ──────────────────────────────────────────────────

    fn new_wal(dir: &TempDir) -> (Wal, PathBuf) {
        let path = dir.path().join("test.wal");
        let wal = Wal::create(&path).unwrap();
        (wal, path)
    }

    /// Every whole record a replay produced, with the tail report
    /// dropped. Most tests here assert on the entries alone; the ones
    /// that assert on a discarded tail call [`Wal::replay`] directly.
    fn replay(path: &Path) -> io::Result<Vec<WalEntry>> {
        Ok(Wal::replay(path)?.entries)
    }

    fn flip_byte(path: &Path, offset: usize) {
        let mut bytes = fs::read(path).unwrap();
        bytes[offset] ^= 0xFF;
        fs::write(path, &bytes).unwrap();
    }

    /// Build the *data* payload (everything between record-type and
    /// checksum) for a put record, matching [`Wal::append_put`].
    fn put_data(key: &[u8], value: &[u8], seq: u64) -> Vec<u8> {
        let mut d = Vec::with_capacity(4 + key.len() + 4 + value.len() + 8);
        d.extend_from_slice(&(key.len() as u32).to_le_bytes());
        d.extend_from_slice(key);
        d.extend_from_slice(&(value.len() as u32).to_le_bytes());
        d.extend_from_slice(value);
        d.extend_from_slice(&seq.to_le_bytes());
        d
    }

    /// Append a raw record to an already-opened file. Used to craft
    /// corruption scenarios the public API can't express - unknown
    /// record types, bad-length headers, etc.
    fn append_raw_record(
        f: &mut impl Write,
        record_type: u8,
        data: &[u8],
        checksum_override: Option<u32>,
    ) {
        let len = data.len() as u32;
        let checksum =
            checksum_override.unwrap_or_else(|| checksum::wal_record(len, record_type, data));
        f.write_all(&len.to_le_bytes()).unwrap();
        f.write_all(&[record_type]).unwrap();
        f.write_all(data).unwrap();
        f.write_all(&checksum.to_le_bytes()).unwrap();
    }

    // ── existing sanity tests ───────────────────────────────────

    #[test]
    fn test_wal_write_and_replay() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.wal");

        {
            let mut wal = Wal::create(&path).unwrap();
            wal.append_put(b"key1", b"value1", 1).unwrap();
            wal.append_delete(b"key2", 2).unwrap();
            wal.append_put(b"key3", b"value3", 3).unwrap();
            wal.flush().unwrap();
        }

        let entries = replay(&path).unwrap();
        assert_eq!(entries.len(), 3);

        match &entries[0] {
            WalEntry::Put { key, value, seq } => {
                assert_eq!(key, b"key1");
                assert_eq!(value, b"value1");
                assert_eq!(*seq, 1);
            }
            _ => panic!("expected put"),
        }

        match &entries[1] {
            WalEntry::Delete { key, seq } => {
                assert_eq!(key, b"key2");
                assert_eq!(*seq, 2);
            }
            _ => panic!("expected delete"),
        }
    }

    #[test]
    fn test_wal_filename() {
        assert_eq!(wal_filename(1), "wal_000001.log");
        assert_eq!(wal_filename(42), "wal_000042.log");
    }

    // ── round-trip coverage for every record type ──────────────

    #[test]
    fn put_record_round_trips() {
        let dir = TempDir::new().unwrap();
        let (mut wal, path) = new_wal(&dir);
        wal.append_put(b"k", b"v", 7).unwrap();
        wal.flush().unwrap();
        drop(wal);

        let entries = replay(&path).unwrap();
        match entries.as_slice() {
            [WalEntry::Put { key, value, seq: 7 }] => {
                assert_eq!(key, b"k");
                assert_eq!(value, b"v");
            }
            _ => panic!("expected a single put at seq=7, got {}", entries.len()),
        }
    }

    #[test]
    fn delete_record_round_trips() {
        let dir = TempDir::new().unwrap();
        let (mut wal, path) = new_wal(&dir);
        wal.append_delete(b"gone", 11).unwrap();
        wal.flush().unwrap();
        drop(wal);

        let entries = replay(&path).unwrap();
        match entries.as_slice() {
            [WalEntry::Delete { key, seq: 11 }] => assert_eq!(key, b"gone"),
            _ => panic!("expected a single delete"),
        }
    }

    #[test]
    fn delete_range_record_round_trips() {
        let dir = TempDir::new().unwrap();
        let (mut wal, path) = new_wal(&dir);
        wal.append_delete_range(b"aaa", b"zzz", 5).unwrap();
        wal.flush().unwrap();
        drop(wal);

        let entries = replay(&path).unwrap();
        match entries.as_slice() {
            [WalEntry::DeleteRange { start, end, seq: 5 }] => {
                assert_eq!(start, b"aaa");
                assert_eq!(end, b"zzz");
            }
            _ => panic!("expected a single delete_range"),
        }
    }

    #[test]
    fn merge_record_round_trips() {
        let dir = TempDir::new().unwrap();
        let (mut wal, path) = new_wal(&dir);
        wal.append_merge(b"counter", b"+3", 99).unwrap();
        wal.flush().unwrap();
        drop(wal);

        let entries = replay(&path).unwrap();
        match entries.as_slice() {
            [WalEntry::Merge {
                key,
                operand,
                seq: 99,
            }] => {
                assert_eq!(key, b"counter");
                assert_eq!(operand, b"+3");
            }
            _ => panic!("expected a single merge"),
        }
    }

    #[test]
    fn all_record_types_replay_in_order() {
        let dir = TempDir::new().unwrap();
        let (mut wal, path) = new_wal(&dir);
        wal.append_put(b"p", b"1", 1).unwrap();
        wal.append_delete(b"d", 2).unwrap();
        wal.append_delete_range(b"ra", b"rb", 3).unwrap();
        wal.append_merge(b"m", b"op", 4).unwrap();
        wal.flush().unwrap();
        drop(wal);

        let entries = replay(&path).unwrap();
        assert_eq!(entries.len(), 4);
        assert!(matches!(entries[0], WalEntry::Put { seq: 1, .. }));
        assert!(matches!(entries[1], WalEntry::Delete { seq: 2, .. }));
        assert!(matches!(entries[2], WalEntry::DeleteRange { seq: 3, .. }));
        assert!(matches!(entries[3], WalEntry::Merge { seq: 4, .. }));
    }

    #[test]
    fn batch_record_replays_all_entries_in_order() {
        let dir = TempDir::new().unwrap();
        let (mut wal, path) = new_wal(&dir);
        let ops = vec![
            WriteBatchOp::Put {
                key: b"p".to_vec(),
                value: b"1".to_vec(),
            },
            WriteBatchOp::Delete { key: b"d".to_vec() },
            WriteBatchOp::DeleteRange {
                start: b"ra".to_vec(),
                end: b"rb".to_vec(),
            },
            WriteBatchOp::Merge {
                key: b"m".to_vec(),
                operand: b"op".to_vec(),
            },
        ];
        wal.append_ops_batch(&ops, 10).unwrap();
        wal.flush().unwrap();
        drop(wal);

        assert_eq!(
            replay(&path).unwrap(),
            vec![
                WalEntry::Put {
                    key: b"p".to_vec(),
                    value: b"1".to_vec(),
                    seq: 10,
                },
                WalEntry::Delete {
                    key: b"d".to_vec(),
                    seq: 11,
                },
                WalEntry::DeleteRange {
                    start: b"ra".to_vec(),
                    end: b"rb".to_vec(),
                    seq: 12,
                },
                WalEntry::Merge {
                    key: b"m".to_vec(),
                    operand: b"op".to_vec(),
                    seq: 13,
                },
            ]
        );
    }

    #[test]
    fn empty_ops_batch_append_is_noop() {
        let dir = TempDir::new().unwrap();
        let (mut wal, path) = new_wal(&dir);
        wal.append_ops_batch(&[], 1).unwrap();
        wal.flush().unwrap();
        drop(wal);

        assert!(replay(&path).unwrap().is_empty());
    }

    // ── edge cases ───────────────────────────────────────────────

    #[test]
    fn replay_empty_file_returns_no_entries() {
        let dir = TempDir::new().unwrap();
        let (mut wal, path) = new_wal(&dir);
        wal.flush().unwrap();
        drop(wal);

        assert!(replay(&path).unwrap().is_empty());
    }

    #[test]
    fn round_trip_empty_key_and_value() {
        let dir = TempDir::new().unwrap();
        let (mut wal, path) = new_wal(&dir);
        wal.append_put(b"", b"", 0).unwrap();
        wal.flush().unwrap();
        drop(wal);

        match replay(&path).unwrap().as_slice() {
            [WalEntry::Put { key, value, seq: 0 }] => {
                assert!(key.is_empty());
                assert!(value.is_empty());
            }
            other => panic!("expected empty-key/value put, got {} entries", other.len()),
        }
    }

    #[test]
    fn round_trip_large_value() {
        let dir = TempDir::new().unwrap();
        let (mut wal, path) = new_wal(&dir);
        // 1 MiB value - larger than BufWriter's default 8 KiB buffer,
        // forcing multiple writes to the underlying file.
        let big = vec![0xAB; 1 << 20];
        wal.append_put(b"k", &big, 1).unwrap();
        wal.flush().unwrap();
        drop(wal);

        match replay(&path).unwrap().as_slice() {
            [WalEntry::Put { key, value, seq: 1 }] => {
                assert_eq!(key, b"k");
                assert_eq!(value.len(), 1 << 20);
                assert!(value.iter().all(|&b| b == 0xAB));
            }
            _ => panic!("expected single large put"),
        }
    }

    #[test]
    fn round_trip_boundary_seq_numbers() {
        let dir = TempDir::new().unwrap();
        let (mut wal, path) = new_wal(&dir);
        wal.append_put(b"a", b"1", 0).unwrap();
        wal.append_put(b"b", b"2", u64::MAX).unwrap();
        wal.flush().unwrap();
        drop(wal);

        let entries = replay(&path).unwrap();
        assert_eq!(entries.len(), 2);
        let seqs: Vec<u64> = entries
            .iter()
            .map(|e| match e {
                WalEntry::Put { seq, .. } => *seq,
                _ => panic!("expected put"),
            })
            .collect();
        assert_eq!(seqs, vec![0, u64::MAX]);
    }

    #[test]
    fn replay_many_records() {
        let dir = TempDir::new().unwrap();
        let (mut wal, path) = new_wal(&dir);
        for i in 0..1000u64 {
            wal.append_put(format!("k{:06}", i).as_bytes(), b"v", i)
                .unwrap();
        }
        wal.flush().unwrap();
        drop(wal);

        let entries = replay(&path).unwrap();
        assert_eq!(entries.len(), 1000);
        for (i, entry) in entries.iter().enumerate() {
            match entry {
                WalEntry::Put { key, seq, .. } => {
                    assert_eq!(key, format!("k{:06}", i).as_bytes());
                    assert_eq!(*seq, i as u64);
                }
                _ => panic!("expected put at index {}", i),
            }
        }
    }

    // ── durability ───────────────────────────────────────────────

    #[test]
    fn sync_persists_records_across_drop() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.wal");
        {
            let mut wal = Wal::create(&path).unwrap();
            wal.append_put(b"k", b"v", 1).unwrap();
            wal.sync().unwrap();
        }
        let entries = replay(&path).unwrap();
        assert_eq!(entries.len(), 1);
    }

    #[test]
    fn sync_syncs_parent_dir_once() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.wal");
        let mut wal = Wal::create(&path).unwrap();
        wal.append_put(b"k", b"v", 1).unwrap();

        let mut sync_count = 0;
        wal.sync_with_parent_sync(|sync_path| {
            assert_eq!(sync_path, path.as_path());
            sync_count += 1;
            Ok(())
        })
        .unwrap();
        wal.sync_with_parent_sync(|_| {
            sync_count += 1;
            Ok(())
        })
        .unwrap();

        assert_eq!(sync_count, 1);
    }

    #[test]
    fn sync_retries_parent_dir_sync_after_error() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.wal");
        let mut wal = Wal::create(&path).unwrap();
        wal.append_put(b"k", b"v", 1).unwrap();
        let mut sync_count = 0;

        let err = match wal.sync_with_parent_sync(|_| {
            sync_count += 1;
            Err(io::Error::other("injected parent sync failure"))
        }) {
            Ok(_) => panic!("expected parent sync failure"),
            Err(err) => err,
        };

        assert_eq!(err.kind(), io::ErrorKind::Other);
        assert_eq!(err.to_string(), "injected parent sync failure");

        wal.sync_with_parent_sync(|sync_path| {
            assert_eq!(sync_path, path.as_path());
            sync_count += 1;
            Ok(())
        })
        .unwrap();

        assert_eq!(sync_count, 2);
    }

    #[test]
    fn create_truncates_prior_contents() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.wal");

        {
            let mut wal = Wal::create(&path).unwrap();
            wal.append_put(b"old", b"v", 1).unwrap();
            wal.flush().unwrap();
        }
        {
            let mut wal = Wal::create(&path).unwrap();
            wal.append_put(b"new", b"v", 2).unwrap();
            wal.flush().unwrap();
        }

        match replay(&path).unwrap().as_slice() {
            [WalEntry::Put { key, seq: 2, .. }] => assert_eq!(key, b"new"),
            other => panic!("old contents leaked: got {} entries", other.len()),
        }
    }

    #[test]
    fn remove_deletes_underlying_file() {
        let dir = TempDir::new().unwrap();
        let (mut wal, path) = new_wal(&dir);
        wal.append_put(b"k", b"v", 1).unwrap();
        wal.flush().unwrap();
        drop(wal);

        assert!(path.exists());
        Wal::remove(&path).unwrap();
        assert!(!path.exists());
    }

    #[test]
    fn path_returns_creation_path() {
        let dir = TempDir::new().unwrap();
        let (wal, path) = new_wal(&dir);
        assert_eq!(wal.path(), path);
    }

    // ── corruption / torn tail ──────────────────────────────────

    #[test]
    fn replay_errors_on_trailing_checksum_flip() {
        let dir = TempDir::new().unwrap();
        let (mut wal, path) = new_wal(&dir);
        wal.append_put(b"good", b"v", 1).unwrap();
        wal.append_put(b"torn", b"v", 2).unwrap();
        wal.flush().unwrap();
        drop(wal);

        // Last byte of the file is the high byte of the trailing
        // record's checksum. Replay must fail closed rather than
        // silently keeping only the prefix.
        let len = fs::metadata(&path).unwrap().len() as usize;
        flip_byte(&path, len - 1);

        let kind = match replay(&path) {
            Err(e) => e.kind(),
            Ok(v) => panic!("expected checksum error, got {} entries", v.len()),
        };
        assert_eq!(kind, io::ErrorKind::InvalidData);
    }

    #[test]
    fn replay_errors_on_trailing_data_byte_flip() {
        let dir = TempDir::new().unwrap();
        let (mut wal, path) = new_wal(&dir);
        wal.append_put(b"good", b"v", 1).unwrap();
        wal.append_put(b"torn", b"v", 2).unwrap();
        wal.flush().unwrap();
        drop(wal);

        // Flip a byte deep inside the second record's data (6 bytes
        // before EOF puts us squarely inside the seq-number field).
        let len = fs::metadata(&path).unwrap().len() as usize;
        flip_byte(&path, len - 6);

        let kind = match replay(&path) {
            Err(e) => e.kind(),
            Ok(v) => panic!("expected checksum error, got {} entries", v.len()),
        };
        assert_eq!(kind, io::ErrorKind::InvalidData);
    }

    #[test]
    fn record_checksum_covers_header_fields() {
        let data = put_data(b"k", b"v", 1);
        let len = data.len() as u32;
        let baseline = checksum::wal_record(len, RECORD_PUT, &data);
        assert_ne!(baseline, checksum::wal_record(len + 1, RECORD_PUT, &data));
        assert_ne!(baseline, checksum::wal_record(len, RECORD_DELETE, &data));
    }

    #[test]
    fn replay_errors_on_record_type_header_flip() {
        let dir = TempDir::new().unwrap();
        let (mut wal, path) = new_wal(&dir);
        wal.append_put(b"k", b"v", 1).unwrap();
        wal.flush().unwrap();
        drop(wal);

        flip_byte(&path, 4);

        let kind = match replay(&path) {
            Err(e) => e.kind(),
            Ok(v) => panic!("expected checksum error, got {} entries", v.len()),
        };
        assert_eq!(kind, io::ErrorKind::InvalidData);
    }

    #[test]
    fn replay_accepts_legacy_payload_only_checksum() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("legacy.wal");
        let data = put_data(b"k", b"v", 1);
        let mut f = File::create(&path).unwrap();
        append_raw_record(
            &mut f,
            RECORD_PUT,
            &data,
            Some(checksum::legacy_payload_u32(&data)),
        );
        f.sync_all().unwrap();

        assert_eq!(
            replay(&path).unwrap(),
            vec![WalEntry::Put {
                key: b"k".to_vec(),
                value: b"v".to_vec(),
                seq: 1
            }]
        );
    }

    #[test]
    fn replay_stops_at_a_truncated_trailing_header() {
        let dir = TempDir::new().unwrap();
        let (mut wal, path) = new_wal(&dir);
        wal.append_put(b"good", b"v", 1).unwrap();
        wal.flush().unwrap();
        drop(wal);

        // Simulate a crash in the middle of writing the next record's
        // 5-byte header by appending 2 stray bytes. Nothing acknowledged
        // them, so they are the end of the log and the whole record
        // before them must survive.
        let mut bytes = fs::read(&path).unwrap();
        bytes.extend_from_slice(&[0xFF, 0xFF]);
        fs::write(&path, &bytes).unwrap();

        assert_eq!(
            replay(&path).unwrap(),
            vec![WalEntry::Put {
                key: b"good".to_vec(),
                value: b"v".to_vec(),
                seq: 1,
            }]
        );
    }

    #[test]
    fn replay_treats_a_length_beyond_the_file_as_a_torn_tail() {
        // A single record whose header claims 1000 bytes of payload that
        // the file does not hold, and nothing after it: the signature of
        // a write torn by a crash. Replay yields the records before it,
        // of which there are none, rather than refusing the whole log.
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("bad.wal");
        let mut f = File::create(&path).unwrap();
        f.write_all(&1000u32.to_le_bytes()).unwrap(); // len
        f.write_all(&[RECORD_PUT]).unwrap(); // type
        f.sync_all().unwrap();

        assert!(replay(&path).unwrap().is_empty());
    }

    #[test]
    fn the_empty_type_zero_record_never_checksums_as_valid() {
        // The zero-tail rule leans on this: a run of zeros frames as
        // `len` 0, type 0x00, checksum 0, and that shape must never be
        // mistakable for a record lark wrote.
        assert_ne!(checksum::wal_record(0, 0, &[]), 0);
        assert_ne!(checksum::legacy_payload_u32(&[]), 0);
    }

    #[test]
    fn replay_stops_at_a_zero_filled_tail() {
        // ext4 with delayed allocation: the inode's new length reached
        // the journal, the data blocks did not, so the tail reads back
        // at full length as zeros rather than short.
        let dir = TempDir::new().unwrap();
        let (mut wal, path) = new_wal(&dir);
        wal.append_put(b"good", b"v", 1).unwrap();
        wal.flush().unwrap();
        drop(wal);

        let durable = fs::metadata(&path).unwrap().len() as usize;
        let mut bytes = fs::read(&path).unwrap();
        bytes.resize(durable + 4096, 0);
        fs::write(&path, &bytes).unwrap();

        let report = Wal::replay(&path).unwrap();
        assert_eq!(
            report.entries,
            vec![WalEntry::Put {
                key: b"good".to_vec(),
                value: b"v".to_vec(),
                seq: 1,
            }]
        );
        assert_eq!(report.truncated_at, Some(durable as u64));
        assert_eq!(report.discarded_bytes, 4096);
    }

    #[test]
    fn replay_rejects_a_zero_filled_region_that_still_has_records_after_it() {
        // Zeros in the middle of the log are damage, not unwritten
        // space: whole records follow them.
        let dir = TempDir::new().unwrap();
        let (mut wal, path) = new_wal(&dir);
        wal.append_put(b"good", b"v", 1).unwrap();
        wal.flush().unwrap();
        drop(wal);

        let durable = fs::metadata(&path).unwrap().len() as usize;
        let mut bytes = fs::read(&path).unwrap();
        bytes.resize(durable + 64, 0);
        let mut trailer = Vec::new();
        let pd = put_data(b"after", b"ok", 2);
        append_raw_record(&mut trailer, RECORD_PUT, &pd, None);
        bytes.extend_from_slice(&trailer);
        fs::write(&path, &bytes).unwrap();

        let err = match Wal::replay(&path) {
            Err(e) => e,
            Ok(r) => panic!("expected corruption, got {} entries", r.entries.len()),
        };
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn replay_still_rejects_a_tail_with_one_non_zero_byte_in_it() {
        // The zero-tail rule is exact. One stray non-zero byte and the
        // tail is corruption again, which is what keeps a bit flip in
        // the final record from being discarded as a crash artifact.
        let dir = TempDir::new().unwrap();
        let (mut wal, path) = new_wal(&dir);
        wal.append_put(b"good", b"v", 1).unwrap();
        wal.flush().unwrap();
        drop(wal);

        let durable = fs::metadata(&path).unwrap().len() as usize;
        let mut bytes = fs::read(&path).unwrap();
        bytes.resize(durable + 64, 0);
        bytes[durable + 40] = 0x01;
        fs::write(&path, &bytes).unwrap();

        let err = match Wal::replay(&path) {
            Err(e) => e,
            Ok(r) => panic!("expected corruption, got {} entries", r.entries.len()),
        };
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn replay_reports_the_bytes_a_torn_tail_discarded() {
        let dir = TempDir::new().unwrap();
        let (mut wal, path) = new_wal(&dir);
        wal.append_put(b"good", b"v", 1).unwrap();
        wal.flush().unwrap();
        drop(wal);

        let durable = fs::metadata(&path).unwrap().len() as usize;
        let mut bytes = fs::read(&path).unwrap();
        bytes.extend_from_slice(&[0xFF, 0xFF, 0xFF]);
        fs::write(&path, &bytes).unwrap();

        let report = Wal::replay(&path).unwrap();
        assert_eq!(report.entries.len(), 1);
        assert_eq!(report.truncated_at, Some(durable as u64));
        assert_eq!(report.discarded_bytes, 3);
    }

    #[test]
    fn replay_of_a_whole_log_reports_no_discard() {
        let dir = TempDir::new().unwrap();
        let (mut wal, path) = new_wal(&dir);
        wal.append_put(b"a", b"v", 1).unwrap();
        wal.append_put(b"b", b"v", 2).unwrap();
        wal.flush().unwrap();
        drop(wal);

        let report = Wal::replay(&path).unwrap();
        assert_eq!(report.entries.len(), 2);
        assert_eq!(report.truncated_at, None);
        assert_eq!(report.discarded_bytes, 0);
    }

    #[test]
    fn replay_rejects_a_length_beyond_the_file_when_whole_records_follow() {
        // A mangled length field in the middle of the log is not a torn
        // tail: the records after it are real and stopping there would
        // discard them. Replay must refuse and name where it stopped.
        let dir = TempDir::new().unwrap();
        let (mut wal, path) = new_wal(&dir);
        for i in 0..4u64 {
            wal.append_put(format!("k{i}").as_bytes(), b"v", i).unwrap();
        }
        wal.flush().unwrap();
        drop(wal);

        let mut bytes = fs::read(&path).unwrap();
        let second = frame_at(&bytes, 0).unwrap().end;
        bytes[second..second + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        fs::write(&path, &bytes).unwrap();

        let err = match replay(&path) {
            Err(e) => e,
            Ok(v) => panic!("expected corruption, got {} entries", v.len()),
        };
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        let message = err.to_string();
        assert!(message.contains("test.wal"), "{message}");
        assert!(message.contains(&second.to_string()), "{message}");
    }

    #[test]
    fn replay_hashes_each_whole_record_exactly_once() {
        // Recovery's cost is dominated by hashing the log. Deriving the
        // checksum verdict once and reusing it, rather than recomputing
        // it for the torn-tail check and again for the corruption check,
        // is the difference between one pass over the WAL and two.
        let dir = TempDir::new().unwrap();
        let (mut wal, path) = new_wal(&dir);
        let value = vec![0xCD; 4096];
        for i in 0..16u64 {
            wal.append_put(format!("k{i}").as_bytes(), &value, i)
                .unwrap();
        }
        wal.flush().unwrap();
        drop(wal);

        let payload_bytes: u64 =
            fs::read(&path).unwrap().len() as u64 - 16 * (WAL_HEADER_LEN + CHECKSUM_LEN) as u64;

        take_checksum_bytes();
        assert_eq!(replay(&path).unwrap().len(), 16);
        let hashed = take_checksum_bytes();

        assert_eq!(
            hashed, payload_bytes,
            "a clean replay must hash each record's payload once and no more",
        );
    }

    #[test]
    fn replay_of_a_self_similar_torn_tail_hashes_the_file_about_once() {
        // Every fifth offset of this tail frames as a record claiming a
        // 1 MiB payload that fits, so checking each candidate on its own
        // would put ~600 GiB through the checksum for a 4 MiB tail, and a
        // real 64 MiB write buffer would wedge recovery for hours on a
        // value a caller is free to write. Requiring the remainder to
        // tile collapses it to one pass.
        //
        // The bound is a byte count rather than a duration: it is exactly
        // the quantity the defect blows up, and unlike wall-clock time it
        // does not move with machine load.
        const TAIL: usize = 4 << 20;
        let dir = TempDir::new().unwrap();
        let (mut wal, path) = new_wal(&dir);
        wal.append_put(b"good", b"v", 1).unwrap();
        wal.flush().unwrap();
        drop(wal);

        let mut bytes = fs::read(&path).unwrap();
        bytes.extend_from_slice(&u32::MAX.to_le_bytes());
        bytes.push(RECORD_PUT);
        bytes.extend(
            [0x00, 0x00, 0x10, 0x00, RECORD_PUT]
                .iter()
                .cycle()
                .take(TAIL),
        );
        fs::write(&path, &bytes).unwrap();
        let file_len = bytes.len() as u64;

        take_checksum_bytes();
        let entries = replay(&path).unwrap();
        let hashed = take_checksum_bytes();

        assert_eq!(
            entries,
            vec![WalEntry::Put {
                key: b"good".to_vec(),
                value: b"v".to_vec(),
                seq: 1,
            }]
        );
        assert!(
            hashed <= file_len,
            "replaying a {TAIL}-byte torn tail hashed {hashed} bytes of a {file_len}-byte \
             file, which is not one pass over it",
        );
    }

    #[test]
    fn replay_keeps_every_whole_record_before_a_cut_at_any_offset() {
        // The G25 contract at unit scale: a crash costs at most the
        // record it landed in. Cutting at every byte offset, including
        // inside the first header and at zero, must always yield exactly
        // the records that survived whole.
        let dir = TempDir::new().unwrap();
        let (mut wal, path) = new_wal(&dir);
        for i in 0..4u64 {
            wal.append_put(format!("k{i}").as_bytes(), b"v", i).unwrap();
        }
        wal.flush().unwrap();
        drop(wal);

        let full = fs::read(&path).unwrap();
        let mut boundaries = vec![0usize];
        while let Some(frame) = frame_at(&full, *boundaries.last().unwrap()) {
            boundaries.push(frame.end);
        }
        assert_eq!(boundaries.len(), 5, "four records tile the file");

        for cut in 0..=full.len() {
            fs::write(&path, &full[..cut]).unwrap();
            let whole = boundaries.iter().filter(|b| **b <= cut).count() - 1;
            let entries = replay(&path)
                .unwrap_or_else(|e| panic!("a cut at {cut} is a torn tail, not corruption: {e}"));
            assert_eq!(entries.len(), whole, "cut at {cut}");
        }
    }

    #[test]
    fn replay_errors_on_unknown_record_type() {
        // Unknown record types with valid checksums still indicate a
        // WAL format this reader cannot safely interpret.
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("mixed.wal");
        let mut f = File::create(&path).unwrap();
        let unknown_payload = b"opaque bytes".to_vec();
        append_raw_record(&mut f, 0xEF, &unknown_payload, None);
        let pd = put_data(b"after", b"ok", 42);
        append_raw_record(&mut f, RECORD_PUT, &pd, None);
        f.sync_all().unwrap();

        let kind = match replay(&path) {
            Err(e) => e.kind(),
            Ok(v) => panic!("expected unknown-type error, got {} entries", v.len()),
        };
        assert_eq!(kind, io::ErrorKind::InvalidData);
    }

    #[test]
    fn replay_rejects_malformed_batch_entry_payload() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("bad_batch.wal");
        let mut data = Vec::new();
        data.extend_from_slice(&1u32.to_le_bytes()); // one batch entry
        data.push(RECORD_PUT);
        data.extend_from_slice(&100u32.to_le_bytes()); // impossible payload length
        data.extend_from_slice(&[0xAA, 0xBB]);

        let mut f = File::create(&path).unwrap();
        append_raw_record(&mut f, RECORD_BATCH, &data, None);
        f.sync_all().unwrap();

        let kind = match replay(&path) {
            Err(e) => e.kind(),
            Ok(v) => panic!("expected malformed-batch error, got {} entries", v.len()),
        };
        assert_eq!(kind, io::ErrorKind::InvalidData);
    }

    #[test]
    fn replay_rejects_batch_trailing_bytes() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("trailing_batch.wal");
        let mut data = Vec::new();
        data.extend_from_slice(&0u32.to_le_bytes());
        data.push(0xFF);

        let mut f = File::create(&path).unwrap();
        append_raw_record(&mut f, RECORD_BATCH, &data, None);
        f.sync_all().unwrap();

        let kind = match replay(&path) {
            Err(e) => e.kind(),
            Ok(v) => panic!(
                "expected batch trailing-byte error, got {} entries",
                v.len()
            ),
        };
        assert_eq!(kind, io::ErrorKind::InvalidData);
    }

    #[test]
    fn replay_of_nonexistent_path_errors() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("never_created.wal");
        let kind = match replay(&path) {
            Err(e) => e.kind(),
            Ok(v) => panic!("expected error, got {} entries", v.len()),
        };
        assert_eq!(kind, io::ErrorKind::NotFound);
    }

    // ── parser error paths (exercised directly) ─────────────────

    #[test]
    fn parse_put_rejects_short_data() {
        assert!(parse_put_record(&[0u8; 15]).is_err());
    }

    #[test]
    fn parse_put_rejects_key_len_overflow() {
        // key_len says 100 but only 2 more bytes follow.
        let mut data = Vec::new();
        data.extend_from_slice(&100u32.to_le_bytes());
        data.extend_from_slice(&[0u8; 2]);
        // pad to meet the initial 16-byte short-data bar
        data.resize(16, 0);
        assert!(parse_put_record(&data).is_err());
    }

    #[test]
    fn parse_delete_rejects_short_data() {
        assert!(parse_delete_record(&[0u8; 11]).is_err());
    }

    #[test]
    fn parse_delete_rejects_key_len_overflow() {
        let mut data = Vec::new();
        data.extend_from_slice(&100u32.to_le_bytes());
        data.resize(12, 0);
        assert!(parse_delete_record(&data).is_err());
    }

    #[test]
    fn parse_delete_range_rejects_short_data() {
        assert!(parse_delete_range_record(&[0u8; 15]).is_err());
    }

    #[test]
    fn parse_delete_range_rejects_start_len_overflow() {
        let mut data = Vec::new();
        data.extend_from_slice(&100u32.to_le_bytes());
        data.resize(16, 0);
        assert!(parse_delete_range_record(&data).is_err());
    }

    #[test]
    fn parse_delete_range_rejects_end_len_overflow() {
        // valid start of len 1, then end_len=100 with no trailing bytes.
        let mut data = Vec::new();
        data.extend_from_slice(&1u32.to_le_bytes());
        data.push(b'k');
        data.extend_from_slice(&100u32.to_le_bytes());
        data.resize(16, 0);
        assert!(parse_delete_range_record(&data).is_err());
    }

    #[test]
    fn parse_merge_rejects_short_data() {
        assert!(parse_merge_record(&[0u8; 15]).is_err());
    }

    #[test]
    fn parse_merge_rejects_key_len_overflow() {
        let mut data = Vec::new();
        data.extend_from_slice(&100u32.to_le_bytes());
        data.resize(16, 0);
        assert!(parse_merge_record(&data).is_err());
    }

    #[test]
    fn parse_merge_rejects_operand_len_overflow() {
        let mut data = Vec::new();
        data.extend_from_slice(&1u32.to_le_bytes());
        data.push(b'k');
        data.extend_from_slice(&100u32.to_le_bytes());
        data.resize(16, 0);
        assert!(parse_merge_record(&data).is_err());
    }
}
