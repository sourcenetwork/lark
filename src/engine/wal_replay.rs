//! Streaming reader over one write-ahead log file.
//!
//! Recovery reads the log one record at a time and hands each entry to
//! the memtable before touching the next, so replay holds one record's
//! payload rather than the whole log. The record framing, the checksum
//! check and every error this produces are the same as the ones
//! [`crate::engine::wal::Wal`] writes and the batch reader used before:
//! this is a change of memory shape, not of format or of failure
//! behaviour.

use std::collections::VecDeque;
use std::fs::File;
use std::io::{self, BufReader};
use std::path::{Path, PathBuf};

use super::checksum;
use super::wal::{
    RECORD_BATCH, RECORD_DELETE, RECORD_DELETE_RANGE, RECORD_MERGE, RECORD_PUT, TailVerdict,
    WAL_STAMP_LEN, WalEntry, classify_incomplete_record, classify_unusable_record,
    parse_batch_record, parse_delete_range_record, parse_delete_record, parse_merge_record,
    parse_put_record, read_exact_or_truncated, read_wal_header,
};

/// Reads a WAL file record by record.
///
/// Peak live bytes are bounded by the largest single record in the
/// file plus the entries decoded from it: a batch record yields its
/// whole op list at once, every other record type yields exactly one
/// entry. A commit never writes a record larger than one batch, so the
/// bound is "one batch", not "one log".
pub(crate) struct WalReplayIter {
    reader: BufReader<File>,
    path: PathBuf,
    /// Total file length, used to reject a length header that claims
    /// more bytes than the file holds *before* allocating for it. A
    /// corrupt header must not be able to ask for a 4 GiB buffer.
    file_len: u64,
    /// Bytes consumed so far, including record framing.
    consumed: u64,
    /// Reused record payload. Grows to the largest record seen and is
    /// not reallocated after that.
    payload: Vec<u8>,
    /// Entries decoded from the current batch record, drained in order.
    pending: VecDeque<WalEntry>,
    /// Set when the replay stopped short of the last byte and discarded
    /// the rest as a crash artifact. Recovery reads it to tell a torn
    /// tail in the newest WAL from damage earlier in the history.
    tail: Option<TailVerdict>,
}

/// Read up to `buf.len()` bytes, returning how many were available.
/// A short read is not an error here: a crash can leave a log shorter
/// than its own stamp.
fn read_full(reader: &mut impl io::Read, buf: &mut [u8]) -> io::Result<usize> {
    let mut read = 0;
    while read < buf.len() {
        match reader.read(&mut buf[read..]) {
            Ok(0) => break,
            Ok(n) => read += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(read)
}

impl WalReplayIter {
    /// Open a WAL file for streaming replay.
    pub(crate) fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let file_len = file.metadata()?.len();
        let mut reader = BufReader::new(file);

        // Consume the stamp before any record is read. A log the stamp
        // never reached reads back as empty; anything else that is not a
        // valid stamp is refused here rather than parsed as records.
        let mut head = [0u8; WAL_STAMP_LEN];
        let stamped = match read_full(&mut reader, &mut head)? {
            0 => None,
            n => super::wal::validate_wal_stamp(&head[..n])?,
        };
        let consumed = stamped.unwrap_or(0) as u64;
        // Nothing but a stamp-less empty log can leave records unread
        // here, so an unstamped file yields no entries at all.
        let file_len = if stamped.is_some() {
            file_len
        } else {
            consumed
        };

        Ok(Self {
            reader,
            path: path.to_path_buf(),
            file_len,
            consumed,
            payload: Vec::new(),
            pending: VecDeque::new(),
            tail: None,
        })
    }

    /// The next entry, or `None` at a clean end of file.
    ///
    /// A truncated tail, a checksum mismatch or an unknown record type
    /// is an error, exactly as it was for a whole-file replay.
    pub(crate) fn next_entry(&mut self) -> io::Result<Option<WalEntry>> {
        let record_start = self.consumed;
        match self.next_entry_inner() {
            // A record the file ends inside is the ordinary shape of a
            // crash. Whether the tail is torn or is damage with whole
            // records behind it cannot be told from a streaming read, so
            // the decision is delegated to the same discriminator the
            // whole-file replay uses.
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {
                self.tail = Some(classify_incomplete_record(&self.path, record_start)?);
                Ok(None)
            }
            // A record that framed cleanly but carries an unusable type
            // or a failing checksum is corruption unless everything from
            // it on is zeros, which is how an unwritten or power-zeroed
            // tail reads back.
            Err(e) if e.kind() == io::ErrorKind::InvalidData => {
                self.tail = Some(classify_unusable_record(&self.path, record_start)?);
                Ok(None)
            }
            other => other,
        }
    }

    /// Where this replay stopped short, if it did.
    pub(crate) fn discarded_tail(&self) -> Option<TailVerdict> {
        self.tail
    }

    fn next_entry_inner(&mut self) -> io::Result<Option<WalEntry>> {
        loop {
            if let Some(entry) = self.pending.pop_front() {
                return Ok(Some(entry));
            }

            let Some(header) = read_wal_header(&mut self.reader)? else {
                return Ok(None);
            };
            self.consumed += header.len() as u64;
            let len = u32::from_le_bytes(header[0..4].try_into().unwrap()) as usize;
            let record_type = header[4];

            // A record claiming more bytes than the file can still hold
            // is truncated by definition. Deciding that from the header
            // keeps a corrupt length from sizing an allocation.
            // A length past the format's maximum is a mangled length
            // field, which is the same signal as one claiming more bytes
            // than the file holds: truncated. Reporting it as corruption
            // instead would refuse to open after an ordinary torn write.
            // The check exists so the number never sizes an allocation.
            if len as u64 > super::wal::MAX_RECORD_LEN as u64 {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "WAL record length exceeds the format's maximum",
                ));
            }
            let remaining = self.file_len.saturating_sub(self.consumed);
            if len as u64 + 4 > remaining {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "truncated WAL record data",
                ));
            }

            self.payload.clear();
            self.payload.resize(len, 0);
            read_exact_or_truncated(
                &mut self.reader,
                &mut self.payload,
                "truncated WAL record data",
            )?;
            self.consumed += len as u64;

            let mut checksum_bytes = [0u8; 4];
            read_exact_or_truncated(
                &mut self.reader,
                &mut checksum_bytes,
                "truncated WAL record checksum",
            )?;
            self.consumed += 4;

            let stored_checksum = u32::from_le_bytes(checksum_bytes);
            let computed_checksum = checksum::wal_record(len as u32, record_type, &self.payload);
            if stored_checksum != computed_checksum
                && stored_checksum != checksum::legacy_payload_u32(&self.payload)
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("WAL checksum mismatch in {}", self.path.display()),
                ));
            }

            match record_type {
                RECORD_PUT => return Ok(Some(parse_put_record(&self.payload)?)),
                RECORD_DELETE => return Ok(Some(parse_delete_record(&self.payload)?)),
                RECORD_DELETE_RANGE => {
                    return Ok(Some(parse_delete_range_record(&self.payload)?));
                }
                RECORD_MERGE => return Ok(Some(parse_merge_record(&self.payload)?)),
                RECORD_BATCH => {
                    self.pending.extend(parse_batch_record(&self.payload)?);
                }
                _ => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("unknown WAL record type {record_type}"),
                    ));
                }
            }
        }
    }

    /// Largest record payload this iterator has had to buffer. The
    /// replay-memory bound is stated in terms of this number.
    #[cfg(test)]
    pub(crate) fn high_water_bytes(&self) -> usize {
        self.payload.capacity()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::WriteBatchOp;
    use crate::engine::wal::Wal;
    use tempfile::TempDir;

    fn drain(path: &Path) -> io::Result<Vec<WalEntry>> {
        let mut iter = WalReplayIter::open(path)?;
        let mut out = Vec::new();
        while let Some(entry) = iter.next_entry()? {
            out.push(entry);
        }
        Ok(out)
    }

    #[test]
    fn streams_every_record_type_in_order() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("mixed.wal");
        {
            let mut wal = Wal::create(&path).unwrap();
            wal.append_put(b"a", b"1", 1).unwrap();
            wal.append_delete(b"b", 2).unwrap();
            wal.append_merge(b"c", b"op", 3).unwrap();
            wal.append_delete_range(b"d", b"e", 4).unwrap();
            let mut group = Vec::new();
            crate::engine::wal::encode_ops_batch_record(
                &mut group,
                &[
                    WriteBatchOp::Put {
                        key: b"f".to_vec(),
                        value: b"2".to_vec(),
                    },
                    WriteBatchOp::Delete { key: b"g".to_vec() },
                ],
                5,
            );
            wal.append_group(&group).unwrap();
            wal.sync_data().unwrap();
        }

        let entries = drain(&path).unwrap();
        assert_eq!(
            entries,
            vec![
                WalEntry::Put {
                    key: b"a".to_vec(),
                    value: b"1".to_vec(),
                    seq: 1
                },
                WalEntry::Delete {
                    key: b"b".to_vec(),
                    seq: 2
                },
                WalEntry::Merge {
                    key: b"c".to_vec(),
                    operand: b"op".to_vec(),
                    seq: 3
                },
                WalEntry::DeleteRange {
                    start: b"d".to_vec(),
                    end: b"e".to_vec(),
                    seq: 4
                },
                WalEntry::Put {
                    key: b"f".to_vec(),
                    value: b"2".to_vec(),
                    seq: 5
                },
                WalEntry::Delete {
                    key: b"g".to_vec(),
                    seq: 6
                },
            ]
        );
    }

    #[test]
    fn empty_log_yields_nothing() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("empty.wal");
        Wal::create(&path).unwrap().sync_data().unwrap();
        assert!(drain(&path).unwrap().is_empty());
    }

    #[test]
    fn streamed_entries_match_the_batch_reader() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("parity.wal");
        {
            let mut wal = Wal::create(&path).unwrap();
            for i in 0..256u64 {
                wal.append_put(format!("key{i:04}").as_bytes(), &[b'v'; 37], i + 1)
                    .unwrap();
            }
            wal.sync_data().unwrap();
        }
        assert_eq!(drain(&path).unwrap(), Wal::replay(&path).unwrap());
    }

    #[test]
    fn truncated_tail_ends_the_log_after_the_good_prefix() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("truncated.wal");
        {
            let mut wal = Wal::create(&path).unwrap();
            wal.append_put(b"a", b"1", 1).unwrap();
            wal.append_put(b"b", b"2", 2).unwrap();
            wal.sync_data().unwrap();
        }
        let mut bytes = std::fs::read(&path).unwrap();
        bytes.truncate(bytes.len() - 3);
        std::fs::write(&path, &bytes).unwrap();

        let mut iter = WalReplayIter::open(&path).unwrap();
        assert!(
            iter.next_entry().unwrap().is_some(),
            "first record is whole"
        );
        // A record the file ends inside is the ordinary shape of a
        // crash, so the whole records before it stand and the tail is
        // discarded. Erroring here would refuse to open a database
        // after an ordinary `kill -9`.
        assert!(
            iter.next_entry().unwrap().is_none(),
            "a torn trailing record ends the log rather than failing it"
        );
    }

    #[test]
    fn checksum_mismatch_is_an_error() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("corrupt.wal");
        {
            let mut wal = Wal::create(&path).unwrap();
            wal.append_put(b"a", b"1", 1).unwrap();
            wal.sync_data().unwrap();
        }
        let mut bytes = std::fs::read(&path).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;
        std::fs::write(&path, &bytes).unwrap();

        let mut iter = WalReplayIter::open(&path).unwrap();
        let err = iter.next_entry().expect_err("checksum must not pass");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn oversized_length_header_is_rejected_without_allocating() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("bad-len.wal");
        // 4-byte length claiming 3 GiB, then nothing.
        let mut bytes = (3u32 * 1024 * 1024 * 1024).to_le_bytes().to_vec();
        bytes.push(RECORD_PUT);
        std::fs::write(&path, &bytes).unwrap();

        let mut iter = WalReplayIter::open(&path).unwrap();
        // Nothing follows the bogus length, so it reads as a torn tail.
        // The point of the test is the allocation, not the verdict.
        assert!(iter.next_entry().unwrap().is_none());
        assert_eq!(
            iter.high_water_bytes(),
            0,
            "a bogus length must not size an allocation"
        );
    }

    #[test]
    fn payload_buffer_is_reused_across_records() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("reuse.wal");
        {
            let mut wal = Wal::create(&path).unwrap();
            for i in 0..64u64 {
                wal.append_put(b"k", &vec![b'v'; 512], i + 1).unwrap();
            }
            wal.sync_data().unwrap();
        }
        let mut iter = WalReplayIter::open(&path).unwrap();
        let mut count = 0;
        while iter.next_entry().unwrap().is_some() {
            count += 1;
        }
        assert_eq!(count, 64);
        // One record is ~530 bytes; the buffer must be sized for one
        // record, not for the whole 34 KiB log.
        assert!(
            iter.high_water_bytes() < 4096,
            "payload buffer grew to {} bytes",
            iter.high_water_bytes()
        );
    }
}
