//! The write-ahead log: the durable record of every write between one
//! memtable flush and the next.
//!
//! # On-disk format
//!
//! A WAL file opens with a self-checksummed stamp and is followed by a
//! sequence of records:
//!
//! ```text
//! ["REGO"][format: u16 LE][reserved: u16 LE = 0][stamp checksum: u32 LE]
//! [len: u32 LE][type: u8][payload: len bytes][checksum: u32 LE]
//! ...
//! ```
//!
//! # Reading a log written before the stamp existed
//!
//! Logs written before the stamp begin directly with a record, and they
//! still open: a file whose first four bytes are not `REGO` is replayed
//! as a bare record sequence, exactly as it was written. Nothing is
//! rewritten to migrate one. The next log the engine creates carries the
//! stamp, so a database moves forward one WAL at a time simply by being
//! used, and a mixed directory is normal rather than an error state.
//!
//! The two shapes cannot be confused. `REGO` read as the little-endian
//! `len` of a legacy record is 0x4F47_4552, about 1.3 GiB, and
//! [`MAX_RECORD_LEN`] caps a record far below that, so no legacy record
//! can begin with those bytes.
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
//!   does not parse is corruption, and is an error. A torn write cannot
//!   produce it: every byte the record claims is present, and they are
//!   wrong.
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
//! The stamp closes neither of those: it is checksummed for itself, not
//! for the records after it. Closing them needs per-record framing that
//! does not depend on a trusted length, such as fixed-size blocks, and
//! the stamp's `format` field is what lets that arrive without breaking
//! the logs written before it.

use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::sync::Arc;

// The module's own tests craft corrupt WAL files byte by byte, which
// is the one thing that has to bypass the environment.
#[cfg(test)]
use std::fs::File;
#[cfg(test)]
use std::io::Write;

use super::checksum;
use crate::WriteBatchOp;
use crate::env::{Env, WriteFile, WriteMode};

/// Record types in the WAL.
pub(super) const RECORD_PUT: u8 = 0x01;
pub(super) const RECORD_DELETE: u8 = 0x02;
pub(super) const RECORD_DELETE_RANGE: u8 = 0x03;
pub(super) const RECORD_MERGE: u8 = 0x04;
pub(super) const RECORD_BATCH: u8 = 0x05;

/// On-disk record header: 4-byte little-endian payload length plus a
/// one-byte record type.
/// `REGO`, the four bytes every WAL written by this version begins with.
pub(crate) const WAL_MAGIC: [u8; 4] = *b"REGO";

/// Stamp layout: magic, format, reserved, checksum.
pub(crate) const WAL_STAMP_LEN: usize = 12;

/// On-disk WAL format this build writes.
const WAL_FORMAT_V1: u16 = 1;

/// Largest record payload the writer will emit, and the largest a reader
/// will believe. Well under `REGO` read as a little-endian length
/// (0x4F47_4552), which is what keeps a stamped file from ever being
/// mistaken for a legacy record and the reverse.
pub(crate) const MAX_RECORD_LEN: u32 = 1 << 30;

const WAL_HEADER_LEN: usize = 5;
/// Trailing 4-byte little-endian checksum of every record.
const CHECKSUM_LEN: usize = 4;

/// A write-ahead log for crash recovery.
///
/// Records are append-only and carry fast non-cryptographic checksums for
/// torn-write and bit-rot detection. On crash recovery, WAL files are
/// replayed to reconstruct memtable state.
pub(crate) struct Wal {
    /// Through the host environment, so a log is written the same way on
    /// a filesystem, under wasi, and against OPFS in a browser.
    ///
    /// Deliberately unbuffered. Group commit already coalesces every
    /// writer in a group into a single `write_all`, so a buffer in front
    /// of it saves no syscall, and it would change what a crash costs: a
    /// process killed with bytes still in a userspace buffer loses them,
    /// where bytes handed to the host survive in its page cache. That
    /// distinction is the whole basis of `DurabilityMode::Eventual`.
    file: Box<dyn WriteFile>,
    /// Bytes appended so far, tracked in memory rather than queried so
    /// [`Wal::rollback_to`] can discard a failed group without a metadata
    /// syscall on the write path.
    offset: u64,
    path: PathBuf,
    parent_synced: bool,
    env: Arc<dyn Env>,
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
    pub(crate) fn create_in(env: &Arc<dyn Env>, path: &Path) -> io::Result<Self> {
        let mut file = env.open_write(path, WriteMode::Truncate)?;
        // Every log this build creates is stamped. That is what makes the
        // format identifiable and versioned from here on, so a later
        // build can change the framing and still know what it is holding.
        file.write_all(&encode_wal_stamp())?;
        Ok(Self {
            file,
            offset: WAL_STAMP_LEN as u64,
            path: path.to_path_buf(),
            parent_synced: false,
            env: Arc::clone(env),
        })
    }

    /// Create a WAL through the standard environment.
    #[cfg(test)]
    pub(crate) fn create(path: &Path) -> io::Result<Self> {
        Self::create_in(&crate::env::std_env(), path)
    }

    /// Append one fully-formed group of records with a single
    /// `write_all`, and advance the tracked offset.
    ///
    /// `bytes` must already be framed by [`encode_op_record`] or
    /// [`encode_ops_batch_record`]; this function adds no framing of its
    /// own. On failure the tracked offset is left at the pre-call value
    /// so [`Wal::rollback_to`] can discard whatever prefix reached the
    /// file.
    pub(crate) fn append_group(&mut self, bytes: &[u8]) -> io::Result<()> {
        if bytes.is_empty() {
            return Ok(());
        }
        debug_assert!(
            bytes.len() as u64 <= MAX_RECORD_LEN as u64,
            "a group larger than MAX_RECORD_LEN would frame a length that \
             could be mistaken for the REGO stamp"
        );
        self.file.write_all(bytes)?;
        self.offset += bytes.len() as u64;
        Ok(())
    }

    /// Byte offset one past the last successfully appended record.
    pub(crate) fn offset(&self) -> u64 {
        self.offset
    }

    /// Truncate back to `offset` and reposition the write cursor there.
    ///
    /// Called when a group's append or sync failed, so a partially
    /// written group never survives as a torn record that replay would
    /// have to reason about.
    pub(crate) fn rollback_to(&mut self, offset: u64) -> io::Result<()> {
        // The stamp is not a record and is never rolled back over: doing
        // so would leave an unidentifiable file behind.
        debug_assert!(
            offset >= WAL_STAMP_LEN as u64,
            "rollback must not truncate into the WAL stamp"
        );
        // The handle appends, so truncating is enough to move the write
        // position back: there is no cursor to seek.
        self.file.set_len(offset)?;
        self.offset = offset;
        Ok(())
    }

    /// Append a put record.
    pub(crate) fn append_put(&mut self, key: &[u8], value: &[u8], seq: u64) -> io::Result<()> {
        let mut record = Vec::with_capacity(record_len(put_payload_len(key, value)));
        encode_put_record(&mut record, key, value, seq);
        self.append_group(&record)
    }

    /// Append a delete record.
    pub(crate) fn append_delete(&mut self, key: &[u8], seq: u64) -> io::Result<()> {
        let mut record = Vec::with_capacity(record_len(delete_payload_len(key)));
        encode_record(&mut record, RECORD_DELETE, |out| {
            encode_delete_payload(out, key, seq)
        });
        self.append_group(&record)
    }

    /// Append a merge record - an operand layered on top of any
    /// existing value/merge chain for `key`.
    pub(crate) fn append_merge(&mut self, key: &[u8], operand: &[u8], seq: u64) -> io::Result<()> {
        let mut record = Vec::with_capacity(record_len(merge_payload_len(key, operand)));
        encode_record(&mut record, RECORD_MERGE, |out| {
            encode_merge_payload(out, key, operand, seq)
        });
        self.append_group(&record)
    }

    /// Append a range-delete record covering `[start, end)`.
    pub(crate) fn append_delete_range(
        &mut self,
        start: &[u8],
        end: &[u8],
        seq: u64,
    ) -> io::Result<()> {
        let mut record = Vec::with_capacity(record_len(delete_range_payload_len(start, end)));
        encode_record(&mut record, RECORD_DELETE_RANGE, |out| {
            encode_delete_range_payload(out, start, end, seq)
        });
        self.append_group(&record)
    }

    /// Flush the appended bytes to stable storage.
    ///
    /// `sync_data` (`fdatasync`), not `sync_all` (`fsync`): the WAL is
    /// append-only, and `fdatasync` already flushes the metadata a later
    /// read needs, which includes the file size. Inode timestamps are not
    /// needed to replay the log, so flushing them is work no reader will
    /// ever benefit from. How much latency that saves is filesystem
    /// dependent and can be nil.
    ///
    /// The directory entry naming the file is a separate durability
    /// concern that no `fdatasync` on the file itself can cover, so the
    /// parent directory is fsynced once per WAL file on first sync.
    pub(crate) fn sync_data(&mut self) -> io::Result<()> {
        let env = Arc::clone(&self.env);
        self.sync_with_parent_sync(move |p| crate::env::sync_parent_dir(&*env, p))
    }

    fn sync_with_parent_sync(
        &mut self,
        mut sync_parent: impl FnMut(&Path) -> io::Result<()>,
    ) -> io::Result<()> {
        #[cfg(test)]
        if fault::should_fail_sync(&self.path) {
            return Err(io::Error::other("injected WAL sync failure"));
        }
        self.file.sync_data()?;
        if !self.parent_synced {
            sync_parent(&self.path)?;
            self.parent_synced = true;
        }
        Ok(())
    }

    /// Get the path to this WAL file.
    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    /// Replay a WAL file and collect every entry.
    ///
    /// Test-only reference implementation: recovery streams the log
    /// through [`WalReplayIter`] instead, so it never holds more than
    /// one record. This wrapper drains that same iterator, which is
    /// what makes the WAL tests below a check on the streaming reader
    /// rather than on a second, divergent parser.
    ///
    /// [`WalReplayIter`]: super::wal_replay::WalReplayIter
    #[cfg(test)]
    pub(crate) fn replay(path: &Path) -> io::Result<Vec<WalEntry>> {
        let mut iter = super::wal_replay::WalReplayIter::open(
            &crate::env::std_env(),
            path,
            super::wal_replay::WalPosition::Newest,
        )?;
        let mut entries = Vec::new();
        while let Some(entry) = iter.next_entry()? {
            entries.push(entry);
        }
        Ok(entries)
    }

    /// Delete a WAL file from `env`.
    pub(crate) fn remove_in(env: &dyn Env, path: &Path) -> io::Result<()> {
        crate::env::remove_file_and_sync_parent(env, path)
    }

    /// Delete a WAL file through the standard environment.
    #[cfg(test)]
    pub(crate) fn remove(path: &Path) -> io::Result<()> {
        Self::remove_in(&*crate::env::std_env(), path)
    }
}

/// Test-only fault injection for the commit path.
///
/// Scoped to a directory rather than armed globally so two tests running
/// in parallel in one process cannot trip each other's injection.
#[cfg(test)]
pub(crate) mod fault {
    use std::path::{Path, PathBuf};

    use parking_lot::Mutex;

    /// Every directory currently armed. A list rather than a single
    /// slot because tests run in parallel in one process: with one slot,
    /// arming for a second directory silently disarms the first and
    /// disarming from either clears both, which makes both tests flaky.
    static ARMED: Mutex<Vec<PathBuf>> = Mutex::new(Vec::new());

    /// Make every `sync_data` on a WAL under `dir` fail until disarmed.
    pub(crate) fn arm_sync_failure(dir: &Path) {
        ARMED.lock().push(dir.to_path_buf());
    }

    /// Stop failing syncs under `dir`, leaving any other test's arming
    /// in place.
    pub(crate) fn disarm_sync_failure(dir: &Path) {
        let mut armed = ARMED.lock();
        if let Some(at) = armed.iter().rposition(|d| d == dir) {
            armed.remove(at);
        }
    }

    pub(super) fn should_fail_sync(path: &Path) -> bool {
        ARMED.lock().iter().any(|dir| path.starts_with(dir))
    }
}

/// Format a WAL filename from a numeric ID.
pub(crate) fn wal_filename(id: u64) -> String {
    format!("wal_{:06}.log", id)
}

/// Framing overhead every record carries: a 4-byte little-endian payload
/// length, a 1-byte record type, and a trailing 4-byte checksum.
const RECORD_HEADER_LEN: usize = 5;
const RECORD_CHECKSUM_LEN: usize = 4;

/// Total on-disk size of a record with a payload of `payload_len` bytes.
fn record_len(payload_len: usize) -> usize {
    RECORD_HEADER_LEN + payload_len + RECORD_CHECKSUM_LEN
}

/// Frame one record into `out`.
///
/// On-disk record format, unchanged since the first release:
/// `[payload_len u32 LE][record_type u8][payload][checksum u32 LE]`,
/// where the checksum covers the length and type fields as well as the
/// payload (see [`checksum::wal_record`]).
///
/// The length is backfilled after `encode_payload` runs so a payload can
/// be written straight into the group buffer with no intermediate copy.
fn encode_record(out: &mut Vec<u8>, record_type: u8, encode_payload: impl FnOnce(&mut Vec<u8>)) {
    let header = out.len();
    out.extend_from_slice(&[0u8; RECORD_HEADER_LEN]);
    encode_payload(out);

    let payload_start = header + RECORD_HEADER_LEN;
    let len = (out.len() - payload_start) as u32;
    out[header..payload_start - 1].copy_from_slice(&len.to_le_bytes());
    out[payload_start - 1] = record_type;

    let checksum = checksum::wal_record(len, record_type, &out[payload_start..]);
    out.extend_from_slice(&checksum.to_le_bytes());
}

/// On-disk size of the record [`encode_put_record`] emits.
pub(crate) fn put_record_len(key: &[u8], value: &[u8]) -> usize {
    record_len(put_payload_len(key, value))
}

/// Encode a single put as one framed record.
pub(crate) fn encode_put_record(out: &mut Vec<u8>, key: &[u8], value: &[u8], seq: u64) {
    encode_record(out, RECORD_PUT, |o| encode_put_payload(o, key, value, seq));
}

/// Encode one write-batch operation as one framed record at `seq`.
pub(crate) fn encode_op_record(out: &mut Vec<u8>, op: &WriteBatchOp, seq: u64) {
    match op {
        WriteBatchOp::Put { key, value } => encode_put_record(out, key, value, seq),
        WriteBatchOp::Delete { key } => {
            encode_record(out, RECORD_DELETE, |o| encode_delete_payload(o, key, seq));
        }
        WriteBatchOp::DeleteRange { start, end } => {
            encode_record(out, RECORD_DELETE_RANGE, |o| {
                encode_delete_range_payload(o, start, end, seq)
            });
        }
        WriteBatchOp::Merge { key, operand } => {
            encode_record(out, RECORD_MERGE, |o| {
                encode_merge_payload(o, key, operand, seq)
            });
        }
    }
}

/// Encode `ops` as one framed `RECORD_BATCH`, the unit replay restores
/// atomically. `ops` must not be empty.
pub(crate) fn encode_ops_batch_record(out: &mut Vec<u8>, ops: &[WriteBatchOp], base_seq: u64) {
    encode_record(out, RECORD_BATCH, |o| {
        o.extend_from_slice(&(ops.len() as u32).to_le_bytes());
        for (i, op) in ops.iter().enumerate() {
            encode_batch_op(o, op, base_seq + i as u64);
        }
    });
}

/// On-disk size of the record `encode_ops_record` would emit for `ops` at
/// one sequence base: one framed record per op when there is exactly one,
/// a single framed batch record otherwise.
pub(crate) fn ops_record_len(ops: &[WriteBatchOp]) -> usize {
    match ops {
        [] => 0,
        [op] => record_len(batch_op_payload_len(op)),
        _ => record_len(batch_ops_payload_len(ops)),
    }
}

/// Encode `ops` into `out` the way the engine writes them: a lone
/// operation becomes its own record, several become one atomic batch
/// record, so replay restores a multi-op write as a unit.
pub(crate) fn encode_ops_record(out: &mut Vec<u8>, ops: &[WriteBatchOp], base_seq: u64) {
    match ops {
        [] => {}
        [op] => encode_op_record(out, op, base_seq),
        _ => encode_ops_batch_record(out, ops, base_seq),
    }
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

impl Frame<'_> {
    /// Whether the stored checksum matches the bytes of this record.
    /// Logs written before the header joined the checksum's coverage
    /// stored a payload-only checksum, which is still accepted.
    fn checksum_matches(&self) -> bool {
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
/// Decide whether the incomplete record beginning at `record_start` is a
/// torn tail or real damage, for a reader that streams and therefore
/// cannot see the bytes past it.
///
/// Reached at most once per file, and only when the log is already
/// damaged, so the whole-file read it needs is not on any healthy path.
/// A WAL is bounded by `write_buffer_size` plus the one record that
/// crossed it, so this bounds nothing the engine did not already hold.
///
/// `Ok(())` means torn: the records before it stand and the tail is
/// discarded, with the offset and byte count logged. `Err` means whole
/// records follow the damage, so the tail is loss rather than a torn
/// write, and the open is refused.
pub(super) fn classify_incomplete_record(
    env: &dyn crate::env::Env,
    path: &Path,
    record_start: u64,
) -> io::Result<TailVerdict> {
    // Through `Env`, like every other read on the recovery path. A
    // `std::fs::read` here would look on the real filesystem, so a
    // database on any other backend could not reopen the moment its
    // newest log had a partial tail: the ordinary shape of a crash.
    let bytes = env.read(path)?;
    let pos = usize::try_from(record_start)
        .unwrap_or(usize::MAX)
        .min(bytes.len());
    end_of_log_or_corruption(path, &bytes, pos)?;
    Ok(TailVerdict::discarded(&bytes, pos))
}

/// Decide a record that framed cleanly but carries an unusable type or a
/// failing checksum.
///
/// Bytes a crash never wrote read back as zeros, and a power cut can
/// zero a region that was already written. Either way an all-zero tail
/// is the end of the log, not damage: there is nothing after it to lose.
/// A bad record with anything non-zero behind it is real corruption and
/// refuses the open.
pub(super) fn classify_unusable_record(
    env: &dyn crate::env::Env,
    path: &Path,
    record_start: u64,
) -> io::Result<TailVerdict> {
    let bytes = env.read(path)?;
    let pos = usize::try_from(record_start)
        .unwrap_or(usize::MAX)
        .min(bytes.len());
    if !tail_is_unwritten(&bytes, pos) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "WAL checksum mismatch in {} at offset {pos}",
                path.display()
            ),
        ));
    }
    Ok(TailVerdict::discarded(&bytes, pos))
}

/// Where a replay stopped short of the last byte, and how much it threw
/// away. Recovery needs both to tell a torn tail in the newest WAL from
/// damage sitting in the middle of the history.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TailVerdict {
    pub(crate) offset: u64,
    pub(crate) discarded_bytes: u64,
}

impl TailVerdict {
    fn discarded(bytes: &[u8], pos: usize) -> Self {
        let discarded_bytes = (bytes.len() - pos) as u64;
        report_discarded_tail(bytes, discarded_bytes, pos);
        Self {
            offset: pos as u64,
            discarded_bytes,
        }
    }
}

/// Whether every byte from `pos` on is zero, which is how both a crash
/// that never wrote them and a power cut that zeroed them read back.
fn tail_is_unwritten(bytes: &[u8], pos: usize) -> bool {
    bytes[pos..].iter().all(|b| *b == 0)
}

fn report_discarded_tail(_bytes: &[u8], discarded_bytes: u64, pos: usize) {
    tracing::warn!(
        offset = pos,
        discarded_bytes,
        "discarded an incomplete trailing WAL record left by a crash"
    );
}

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

    tracing::warn!(
        path = %path.display(),
        offset = pos,
        discarded_bytes,
        "discarded an incomplete trailing WAL record left by a crash"
    );

    Ok(())
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

/// Encode the stamp a fresh WAL begins with.
fn encode_wal_stamp() -> [u8; WAL_STAMP_LEN] {
    let mut out = [0u8; WAL_STAMP_LEN];
    out[0..4].copy_from_slice(&WAL_MAGIC);
    out[4..6].copy_from_slice(&WAL_FORMAT_V1.to_le_bytes());
    out[6..8].copy_from_slice(&0u16.to_le_bytes());
    let checksum = checksum::wal_stamp(&WAL_MAGIC, WAL_FORMAT_V1, 0);
    out[8..12].copy_from_slice(&checksum.to_le_bytes());
    out
}

/// Validate the stamp at the head of `bytes` and return its length.
///
/// The stamp is mandatory: every log this build creates carries one, so
/// a file that does not is not a log this build wrote. The exception is
/// a log a crash caught before the stamp reached the disk, which reads
/// back as nothing or as zeros; that is an empty log, not damage, and
/// the same rule covers it as covers an unwritten record tail.
pub(super) fn validate_wal_stamp(bytes: &[u8]) -> io::Result<Option<usize>> {
    // Too short to hold a stamp is too short to hold a record, so there
    // is nothing in the file to lose and nothing to misread. That is what
    // a crash during `create` leaves behind.
    if bytes.len() < WAL_STAMP_LEN {
        return Ok(None);
    }
    if bytes[0..4] != WAL_MAGIC {
        // No stamp means no record in this file was ever acknowledged.
        // The stamp is written when the log is created, before any
        // record, so a write that returned `Ok` from this file had a
        // durable stamp behind it. Damage here is therefore a crash
        // during creation, not the loss of an acknowledged write, and
        // the file is discardable whatever the damage looks like:
        // zeros from an unwritten extent, or garbage from a torn one.
        //
        // Refusing instead would turn a recoverable state into an
        // unopenable database, which is the opposite of the durability
        // rule: recovery must reach a valid prefix of the write
        // history, and "everything before this log" is one.
        //
        // The discard is not silent. The caller records it and
        // `reject_tail_discard_before_live_wal` refuses the open if a
        // *later* log still yielded records, because then the missing
        // bytes would be a hole in the middle of the history rather
        // than its end.
        return Ok(None);
    }
    let format = u16::from_le_bytes([bytes[4], bytes[5]]);
    let reserved = u16::from_le_bytes([bytes[6], bytes[7]]);
    let stored = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
    if stored != checksum::wal_stamp(&WAL_MAGIC, format, reserved) {
        return Err(invalid_wal("WAL stamp checksum mismatch"));
    }
    // A newer format is refused rather than guessed at. This is the whole
    // point of the field: an older build must fail loudly on a log a
    // newer one wrote, instead of misreading its framing.
    if format > WAL_FORMAT_V1 {
        return Err(invalid_wal(format!(
            "WAL format {format} was written by a newer lark than this build, \
             which understands up to {WAL_FORMAT_V1}"
        )));
    }
    Ok(Some(WAL_STAMP_LEN))
}

fn invalid_wal(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

pub(super) fn read_wal_header(reader: &mut impl Read) -> io::Result<Option<[u8; 5]>> {
    let mut header = [0u8; 5];
    let mut read = 0;

    while read < header.len() {
        match reader.read(&mut header[read..]) {
            Ok(0) if read == 0 => return Ok(None),
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "truncated WAL record header",
                ));
            }
            Ok(n) => read += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }

    Ok(Some(header))
}

pub(super) fn read_exact_or_truncated(
    reader: &mut impl Read,
    buf: &mut [u8],
    message: &'static str,
) -> io::Result<()> {
    reader.read_exact(buf).map_err(|e| {
        if e.kind() == io::ErrorKind::UnexpectedEof {
            io::Error::new(io::ErrorKind::UnexpectedEof, message)
        } else {
            e
        }
    })
}

pub(super) fn parse_put_record(data: &[u8]) -> io::Result<WalEntry> {
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

pub(super) fn parse_delete_record(data: &[u8]) -> io::Result<WalEntry> {
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

pub(super) fn parse_delete_range_record(data: &[u8]) -> io::Result<WalEntry> {
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

pub(super) fn parse_merge_record(data: &[u8]) -> io::Result<WalEntry> {
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

pub(super) fn parse_batch_record(data: &[u8]) -> io::Result<Vec<WalEntry>> {
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
        }

        let entries = Wal::replay(&path).unwrap();
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
        drop(wal);

        let entries = Wal::replay(&path).unwrap();
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
        drop(wal);

        let entries = Wal::replay(&path).unwrap();
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
        drop(wal);

        let entries = Wal::replay(&path).unwrap();
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
        drop(wal);

        let entries = Wal::replay(&path).unwrap();
        match entries.as_slice() {
            [
                WalEntry::Merge {
                    key,
                    operand,
                    seq: 99,
                },
            ] => {
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
        drop(wal);

        let entries = Wal::replay(&path).unwrap();
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
        let mut record = Vec::new();
        encode_ops_batch_record(&mut record, &ops, 10);
        wal.append_group(&record).unwrap();
        drop(wal);

        assert_eq!(
            Wal::replay(&path).unwrap(),
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
    fn empty_ops_encodes_and_appends_nothing() {
        let dir = TempDir::new().unwrap();
        let (mut wal, path) = new_wal(&dir);
        let mut record = Vec::new();
        encode_ops_record(&mut record, &[], 1);
        assert!(record.is_empty());
        wal.append_group(&record).unwrap();
        assert_eq!(
            wal.offset(),
            WAL_STAMP_LEN as u64,
            "an empty group appends nothing past the stamp"
        );
        drop(wal);

        assert!(Wal::replay(&path).unwrap().is_empty());
    }

    // ── edge cases ───────────────────────────────────────────────

    #[test]
    fn replay_empty_file_returns_no_entries() {
        let dir = TempDir::new().unwrap();
        let (wal, path) = new_wal(&dir);
        drop(wal);

        assert!(Wal::replay(&path).unwrap().is_empty());
    }

    #[test]
    fn round_trip_empty_key_and_value() {
        let dir = TempDir::new().unwrap();
        let (mut wal, path) = new_wal(&dir);
        wal.append_put(b"", b"", 0).unwrap();
        drop(wal);

        match Wal::replay(&path).unwrap().as_slice() {
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
        drop(wal);

        match Wal::replay(&path).unwrap().as_slice() {
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
        drop(wal);

        let entries = Wal::replay(&path).unwrap();
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
        drop(wal);

        let entries = Wal::replay(&path).unwrap();
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
            wal.sync_data().unwrap();
        }
        let entries = Wal::replay(&path).unwrap();
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
        }
        {
            let mut wal = Wal::create(&path).unwrap();
            wal.append_put(b"new", b"v", 2).unwrap();
        }

        match Wal::replay(&path).unwrap().as_slice() {
            [WalEntry::Put { key, seq: 2, .. }] => assert_eq!(key, b"new"),
            other => panic!("old contents leaked: got {} entries", other.len()),
        }
    }

    #[test]
    fn remove_deletes_underlying_file() {
        let dir = TempDir::new().unwrap();
        let (mut wal, path) = new_wal(&dir);
        wal.append_put(b"k", b"v", 1).unwrap();
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

    // ── group append and rollback ───────────────────────────────

    // ── the REGO stamp ──────────────────────────────────────────

    #[test]
    fn a_fresh_log_begins_with_the_rego_stamp() {
        let dir = TempDir::new().unwrap();
        let (mut wal, path) = new_wal(&dir);
        wal.append_put(b"k", b"v", 1).unwrap();
        wal.sync_data().unwrap();
        drop(wal);

        let bytes = fs::read(&path).unwrap();
        assert_eq!(&bytes[0..4], b"REGO", "the log is not stamped");
        assert_eq!(
            u16::from_le_bytes([bytes[4], bytes[5]]),
            WAL_FORMAT_V1,
            "the stamp names the format this build writes"
        );
        assert_eq!(
            validate_wal_stamp(&bytes).unwrap(),
            Some(WAL_STAMP_LEN),
            "the stamp validates against its own checksum"
        );
    }

    #[test]
    fn the_stamp_cannot_be_confused_with_a_record_length() {
        // `REGO` read as a little-endian record length is far above the
        // largest record the writer will emit, so no legacy-shaped record
        // can begin with the magic and no stamp can be read as a record.
        let as_len = u32::from_le_bytes(WAL_MAGIC);
        assert!(
            as_len > MAX_RECORD_LEN,
            "REGO as a length ({as_len}) must exceed MAX_RECORD_LEN ({MAX_RECORD_LEN})"
        );
    }

    #[test]
    fn a_log_from_a_newer_format_is_refused_rather_than_guessed_at() {
        let mut stamp = encode_wal_stamp();
        let future = WAL_FORMAT_V1 + 1;
        stamp[4..6].copy_from_slice(&future.to_le_bytes());
        let checksum = checksum::wal_stamp(&WAL_MAGIC, future, 0);
        stamp[8..12].copy_from_slice(&checksum.to_le_bytes());

        let err = validate_wal_stamp(&stamp).expect_err("a newer format must not be parsed");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(
            err.to_string().contains("newer lark"),
            "the error must say why: {err}"
        );
    }

    #[test]
    fn a_corrupt_stamp_is_damage_not_a_record_stream() {
        let mut stamp = encode_wal_stamp();
        stamp[8] ^= 0xFF;
        let err = validate_wal_stamp(&stamp).expect_err("a bad checksum must not pass");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    /// Bytes that are not a stamped log yield no records rather than an
    /// error, and the difference matters.
    ///
    /// An `fsync` that makes a record durable flushes everything written
    /// before it, and the stamp is written when the log is created. So a
    /// log holding an acknowledged write has a durable stamp behind it,
    /// and damage to the stamp proves nothing in the file was ever
    /// acknowledged. Refusing would turn that into an unopenable
    /// database; discarding reaches "everything before this log", which
    /// is a valid prefix of the write history.
    ///
    /// The discard is not silent, and it is not unconditional: the
    /// caller reports it and `reject_tail_discard_before_live_wal`
    /// refuses the open when a later log still yielded records, because
    /// then the missing bytes are a hole rather than the end. That case
    /// is covered by `a_torn_tail_in_an_earlier_wal_file_is_not_the_end_of_the_log`.
    #[test]
    fn a_file_that_is_not_a_stamped_log_yields_no_records() {
        for bytes in [
            &b"this is not a write-ahead log at all"[..],
            &[0xFFu8; 64][..],
            &[0x00u8; 64][..],
        ] {
            assert_eq!(
                validate_wal_stamp(bytes).expect("an unstamped log is discardable, not an error"),
                None,
                "unstamped bytes must report no stamp so the caller can record the discard",
            );
        }
    }

    /// A stamp that parses but is damaged is still refused. The
    /// checksum and the version field are the parts a later build
    /// relies on to know what framing it is holding, so a wrong value
    /// there is not a crash artifact and must not be guessed at.
    #[test]
    fn a_damaged_but_present_stamp_is_still_refused() {
        let mut stamp = encode_wal_stamp();
        stamp[8] ^= 0xFF;
        let err = validate_wal_stamp(&stamp).expect_err("a corrupt checksum must not pass");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn a_crash_before_the_stamp_reached_disk_reads_as_an_empty_log() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("torn-stamp.wal");
        // Every prefix of a stamp is too short to hold a record, so there
        // is nothing in it to lose and nothing to misread.
        for cut in 0..WAL_STAMP_LEN {
            fs::write(&path, &encode_wal_stamp()[..cut]).unwrap();
            assert!(
                Wal::replay(&path).unwrap().is_empty(),
                "a stamp torn at {cut} must read as an empty log"
            );
        }
    }

    #[test]
    fn offset_tracks_every_appended_byte() {
        let dir = TempDir::new().unwrap();
        let (mut wal, path) = new_wal(&dir);
        assert_eq!(
            wal.offset(),
            WAL_STAMP_LEN as u64,
            "a fresh log holds its stamp"
        );

        wal.append_put(b"k", b"v", 1).unwrap();
        let after_one = wal.offset();
        assert_eq!(
            after_one as usize,
            WAL_STAMP_LEN + put_record_len(b"k", b"v")
        );

        wal.append_put(b"k2", b"v2", 2).unwrap();
        assert_eq!(
            wal.offset() as usize,
            after_one as usize + put_record_len(b"k2", b"v2")
        );

        wal.sync_data().unwrap();
        assert_eq!(fs::metadata(&path).unwrap().len(), wal.offset());
    }

    #[test]
    fn a_group_of_records_replays_exactly_like_individual_appends() {
        let dir = TempDir::new().unwrap();
        let grouped_path = dir.path().join("grouped.wal");
        let individual_path = dir.path().join("individual.wal");

        let ops = vec![
            WriteBatchOp::Put {
                key: b"a".to_vec(),
                value: b"1".to_vec(),
            },
            WriteBatchOp::Delete { key: b"b".to_vec() },
        ];

        {
            let mut wal = Wal::create(&grouped_path).unwrap();
            let mut group = Vec::new();
            encode_put_record(&mut group, b"solo", b"v", 1);
            encode_ops_record(&mut group, &ops, 2);
            wal.append_group(&group).unwrap();
        }
        {
            let mut wal = Wal::create(&individual_path).unwrap();
            wal.append_put(b"solo", b"v", 1).unwrap();
            let mut record = Vec::new();
            encode_ops_batch_record(&mut record, &ops, 2);
            wal.append_group(&record).unwrap();
        }

        assert_eq!(
            fs::read(&grouped_path).unwrap(),
            fs::read(&individual_path).unwrap(),
            "grouping must not change a single on-disk byte"
        );
        assert_eq!(
            Wal::replay(&grouped_path).unwrap(),
            Wal::replay(&individual_path).unwrap()
        );
        assert_eq!(Wal::replay(&grouped_path).unwrap().len(), 3);
    }

    #[test]
    fn rollback_discards_a_group_and_leaves_the_log_replayable() {
        let dir = TempDir::new().unwrap();
        let (mut wal, path) = new_wal(&dir);
        wal.append_put(b"keep", b"v", 1).unwrap();
        let good = wal.offset();

        let mut group = Vec::new();
        encode_put_record(&mut group, b"discard", b"v", 2);
        wal.append_group(&group).unwrap();
        assert!(wal.offset() > good);

        wal.rollback_to(good).unwrap();
        assert_eq!(wal.offset(), good);

        // The cursor moved back with the length, so the next append lands
        // at the boundary rather than past a hole.
        wal.append_put(b"after", b"v", 3).unwrap();
        wal.sync_data().unwrap();
        assert_eq!(fs::metadata(&path).unwrap().len(), wal.offset());

        let entries = Wal::replay(&path).unwrap();
        assert_eq!(
            entries,
            vec![
                WalEntry::Put {
                    key: b"keep".to_vec(),
                    value: b"v".to_vec(),
                    seq: 1,
                },
                WalEntry::Put {
                    key: b"after".to_vec(),
                    value: b"v".to_vec(),
                    seq: 3,
                },
            ]
        );
    }

    #[test]
    fn rollback_of_a_torn_partial_group_leaves_no_record_behind() {
        let dir = TempDir::new().unwrap();
        let (mut wal, path) = new_wal(&dir);
        let good = wal.offset();

        // Simulate a group whose write reached the file only partially.
        let mut group = Vec::new();
        encode_put_record(&mut group, b"torn", b"value", 7);
        wal.append_group(&group[..group.len() - 3]).unwrap();
        // The partial record is a torn tail, so it is discarded rather
        // than replayed: what must never happen is the half-written
        // entry surfacing as a committed one.
        assert!(
            Wal::replay(&path).unwrap().is_empty(),
            "a torn tail must not replay as an entry"
        );

        wal.rollback_to(good).unwrap();
        assert!(Wal::replay(&path).unwrap().is_empty());
    }

    #[test]
    fn record_len_matches_the_bytes_each_encoder_emits() {
        let mut out = Vec::new();
        encode_put_record(&mut out, b"key", b"value", 4);
        assert_eq!(out.len(), put_record_len(b"key", b"value"));

        let ops = vec![
            WriteBatchOp::Merge {
                key: b"m".to_vec(),
                operand: b"o".to_vec(),
            },
            WriteBatchOp::DeleteRange {
                start: b"s".to_vec(),
                end: b"e".to_vec(),
            },
        ];
        let mut out = Vec::new();
        encode_ops_record(&mut out, &ops, 9);
        assert_eq!(out.len(), ops_record_len(&ops));

        let single = vec![WriteBatchOp::Merge {
            key: b"m".to_vec(),
            operand: b"o".to_vec(),
        }];
        let mut out = Vec::new();
        encode_ops_record(&mut out, &single, 9);
        assert_eq!(out.len(), ops_record_len(&single));
    }

    // ── corruption / torn tail ──────────────────────────────────

    #[test]
    fn replay_errors_on_trailing_checksum_flip() {
        let dir = TempDir::new().unwrap();
        let (mut wal, path) = new_wal(&dir);
        wal.append_put(b"good", b"v", 1).unwrap();
        wal.append_put(b"torn", b"v", 2).unwrap();
        drop(wal);

        // Last byte of the file is the high byte of the trailing
        // record's checksum. Replay must fail closed rather than
        // silently keeping only the prefix.
        let len = fs::metadata(&path).unwrap().len() as usize;
        flip_byte(&path, len - 1);

        let kind = match Wal::replay(&path) {
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
        drop(wal);

        // Flip a byte deep inside the second record's data (6 bytes
        // before EOF puts us squarely inside the seq-number field).
        let len = fs::metadata(&path).unwrap().len() as usize;
        flip_byte(&path, len - 6);

        let kind = match Wal::replay(&path) {
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
        drop(wal);

        flip_byte(&path, 4);

        let kind = match Wal::replay(&path) {
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
        f.write_all(&encode_wal_stamp()).unwrap();
        append_raw_record(
            &mut f,
            RECORD_PUT,
            &data,
            Some(checksum::legacy_payload_u32(&data)),
        );
        f.sync_all().unwrap();

        assert_eq!(
            Wal::replay(&path).unwrap(),
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
        drop(wal);

        // Simulate a crash in the middle of writing the next record's
        // 5-byte header by appending 2 stray bytes. Nothing acknowledged
        // them, so they are the end of the log and the whole record
        // before them must survive.
        let mut bytes = fs::read(&path).unwrap();
        bytes.extend_from_slice(&[0xFF, 0xFF]);
        fs::write(&path, &bytes).unwrap();

        assert_eq!(
            Wal::replay(&path).unwrap(),
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

        assert!(Wal::replay(&path).unwrap().is_empty());
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
        wal.sync_data().unwrap();
        drop(wal);

        let mut bytes = fs::read(&path).unwrap();
        let second = frame_at(&bytes, WAL_STAMP_LEN).unwrap().end;
        bytes[second..second + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        fs::write(&path, &bytes).unwrap();

        let err = match Wal::replay(&path) {
            Err(e) => e,
            Ok(v) => panic!("expected corruption, got {} entries", v.len()),
        };
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        let message = err.to_string();
        assert!(message.contains("test.wal"), "{message}");
        assert!(message.contains(&second.to_string()), "{message}");
    }

    #[test]
    fn replay_of_a_self_similar_torn_tail_stays_linear() {
        // Every fifth offset of this tail frames as a record claiming a
        // 1 MiB payload that fits, so checking each candidate on its own
        // would put ~600 GiB through the checksum for a 4 MiB tail, and
        // a real 64 MiB write buffer would wedge recovery for hours on a
        // value a caller is free to write. Requiring the remainder to
        // tile collapses it to one pass. The bound is wall clock because
        // the defect is asymptotic; the margin over the ~50 ms this takes
        // is wide enough that a loaded machine cannot trip it.
        const TAIL: usize = 4 << 20;
        let dir = TempDir::new().unwrap();
        let (mut wal, path) = new_wal(&dir);
        wal.append_put(b"good", b"v", 1).unwrap();
        wal.sync_data().unwrap();
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

        let start = std::time::Instant::now();
        let entries = Wal::replay(&path).unwrap();
        let elapsed = start.elapsed();

        assert_eq!(
            entries,
            vec![WalEntry::Put {
                key: b"good".to_vec(),
                value: b"v".to_vec(),
                seq: 1,
            }]
        );
        assert!(
            elapsed < std::time::Duration::from_secs(20),
            "replaying a {TAIL}-byte torn tail took {elapsed:?}, which is not one pass over it",
        );
    }

    #[test]
    fn replay_keeps_every_whole_record_before_a_cut_at_any_offset() {
        // The the torn-tail rule contract at unit scale: a crash costs at most the
        // record it landed in. Cutting at every byte offset, including
        // inside the first header and at zero, must always yield exactly
        // the records that survived whole.
        let dir = TempDir::new().unwrap();
        let (mut wal, path) = new_wal(&dir);
        for i in 0..4u64 {
            wal.append_put(format!("k{i}").as_bytes(), b"v", i).unwrap();
        }
        wal.sync_data().unwrap();
        drop(wal);

        let full = fs::read(&path).unwrap();
        let mut boundaries = vec![WAL_STAMP_LEN];
        while let Some(frame) = frame_at(&full, *boundaries.last().unwrap()) {
            boundaries.push(frame.end);
        }
        assert_eq!(boundaries.len(), 5, "four records tile the file");

        // Cuts inside the stamp leave a file with no records at all, which
        // the stamp rule already covers; the interesting range starts once
        // a whole stamp is on disk.
        for cut in WAL_STAMP_LEN..=full.len() {
            fs::write(&path, &full[..cut]).unwrap();
            let whole = boundaries.iter().filter(|b| **b <= cut).count() - 1;
            let entries = Wal::replay(&path)
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

        let kind = match Wal::replay(&path) {
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
        // The stamp first: without it the file is an unstamped log and
        // replay reports no records, which would let these probes pass
        // on the wrong reason instead of on the malformed record.
        f.write_all(&encode_wal_stamp()).unwrap();
        append_raw_record(&mut f, RECORD_BATCH, &data, None);
        f.sync_all().unwrap();

        let kind = match Wal::replay(&path) {
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
        // The stamp first: without it the file is an unstamped log and
        // replay reports no records, which would let these probes pass
        // on the wrong reason instead of on the malformed record.
        f.write_all(&encode_wal_stamp()).unwrap();
        append_raw_record(&mut f, RECORD_BATCH, &data, None);
        f.sync_all().unwrap();

        let kind = match Wal::replay(&path) {
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
        let kind = match Wal::replay(&path) {
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
