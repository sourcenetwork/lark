use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use super::{checksum, durability};

/// Record types in the WAL.
const RECORD_PUT: u8 = 0x01;
const RECORD_DELETE: u8 = 0x02;
const RECORD_DELETE_RANGE: u8 = 0x03;
const RECORD_MERGE: u8 = 0x04;
const RECORD_BATCH: u8 = 0x05;

/// A write-ahead log for crash recovery.
///
/// Records are append-only and carry fast non-cryptographic checksums for
/// torn-write and bit-rot detection. On crash recovery, WAL files are
/// replayed to reconstruct memtable state.
pub(crate) struct Wal {
    writer: BufWriter<File>,
    path: PathBuf,
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

    /// Append a merge record — an operand layered on top of any
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

    /// Append a batch record containing multiple logical WAL entries.
    ///
    /// The batch is one top-level WAL record with one checksum. Replay
    /// expands it only after the whole payload parses successfully, so
    /// a torn or malformed batch cannot recover as a committed prefix.
    pub(crate) fn append_batch(&mut self, entries: &[WalEntry]) -> io::Result<()> {
        if entries.is_empty() {
            return Ok(());
        }

        let mut data = Vec::new();
        data.extend_from_slice(&(entries.len() as u32).to_le_bytes());

        for entry in entries {
            let (record_type, payload) = encode_batch_entry(entry);
            data.push(record_type);
            data.extend_from_slice(&(payload.len() as u32).to_le_bytes());
            data.extend_from_slice(&payload);
        }

        self.write_record(RECORD_BATCH, &data)
    }

    /// Flush and fsync the WAL to disk.
    pub(crate) fn sync(&mut self) -> io::Result<()> {
        self.writer.flush()?;
        self.writer.get_ref().sync_all()
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

    /// Replay a WAL file and return all entries.
    pub(crate) fn replay(path: &Path) -> io::Result<Vec<WalEntry>> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let mut entries = Vec::new();

        // Read record headers: length (4 bytes) + type (1 byte).
        while let Some(header) = read_wal_header(&mut reader)? {
            let len = u32::from_le_bytes(header[0..4].try_into().unwrap()) as usize;
            let record_type = header[4];

            // Read data
            let mut data = vec![0u8; len];
            read_exact_or_truncated(&mut reader, &mut data, "truncated WAL record data")?;

            // Read and verify checksum
            let mut checksum_bytes = [0u8; 4];
            read_exact_or_truncated(
                &mut reader,
                &mut checksum_bytes,
                "truncated WAL record checksum",
            )?;
            let stored_checksum = u32::from_le_bytes(checksum_bytes);
            let computed_checksum = checksum::wal_record(len as u32, record_type, &data);

            if stored_checksum != computed_checksum
                && stored_checksum != checksum::legacy_payload_u32(&data)
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("WAL checksum mismatch in {}", path.display()),
                ));
            }

            match record_type {
                RECORD_PUT => {
                    let entry = parse_put_record(&data)?;
                    entries.push(entry);
                }
                RECORD_DELETE => {
                    let entry = parse_delete_record(&data)?;
                    entries.push(entry);
                }
                RECORD_DELETE_RANGE => {
                    let entry = parse_delete_range_record(&data)?;
                    entries.push(entry);
                }
                RECORD_MERGE => {
                    let entry = parse_merge_record(&data)?;
                    entries.push(entry);
                }
                RECORD_BATCH => {
                    let batch = parse_batch_record(&data)?;
                    entries.extend(batch);
                }
                _ => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("unknown WAL record type {record_type}"),
                    ));
                }
            }
        }

        Ok(entries)
    }

    /// Delete a WAL file.
    pub(crate) fn remove(path: &Path) -> io::Result<()> {
        durability::remove_file_and_sync_parent(path)
    }
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

fn encode_batch_entry(entry: &WalEntry) -> (u8, Vec<u8>) {
    match entry {
        WalEntry::Put { key, value, seq } => {
            let mut payload = Vec::with_capacity(4 + key.len() + 4 + value.len() + 8);
            encode_put_payload(&mut payload, key, value, *seq);
            (RECORD_PUT, payload)
        }
        WalEntry::Delete { key, seq } => {
            let mut payload = Vec::with_capacity(4 + key.len() + 8);
            encode_delete_payload(&mut payload, key, *seq);
            (RECORD_DELETE, payload)
        }
        WalEntry::DeleteRange { start, end, seq } => {
            let mut payload = Vec::with_capacity(4 + start.len() + 4 + end.len() + 8);
            encode_delete_range_payload(&mut payload, start, end, *seq);
            (RECORD_DELETE_RANGE, payload)
        }
        WalEntry::Merge { key, operand, seq } => {
            let mut payload = Vec::with_capacity(4 + key.len() + 4 + operand.len() + 8);
            encode_merge_payload(&mut payload, key, operand, *seq);
            (RECORD_MERGE, payload)
        }
    }
}

fn read_wal_header(reader: &mut impl Read) -> io::Result<Option<[u8; 5]>> {
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

fn read_exact_or_truncated(
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
    /// corruption scenarios the public API can't express — unknown
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
        wal.flush().unwrap();
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
        wal.flush().unwrap();
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
        wal.flush().unwrap();
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
        wal.flush().unwrap();
        drop(wal);

        let entries = Wal::replay(&path).unwrap();
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
        let expected = vec![
            WalEntry::Put {
                key: b"p".to_vec(),
                value: b"1".to_vec(),
                seq: 1,
            },
            WalEntry::Delete {
                key: b"d".to_vec(),
                seq: 2,
            },
            WalEntry::DeleteRange {
                start: b"ra".to_vec(),
                end: b"rb".to_vec(),
                seq: 3,
            },
            WalEntry::Merge {
                key: b"m".to_vec(),
                operand: b"op".to_vec(),
                seq: 4,
            },
        ];
        wal.append_batch(&expected).unwrap();
        wal.flush().unwrap();
        drop(wal);

        assert_eq!(Wal::replay(&path).unwrap(), expected);
    }

    #[test]
    fn empty_batch_append_is_noop() {
        let dir = TempDir::new().unwrap();
        let (mut wal, path) = new_wal(&dir);
        wal.append_batch(&[]).unwrap();
        wal.flush().unwrap();
        drop(wal);

        assert!(Wal::replay(&path).unwrap().is_empty());
    }

    // ── edge cases ───────────────────────────────────────────────

    #[test]
    fn replay_empty_file_returns_no_entries() {
        let dir = TempDir::new().unwrap();
        let (mut wal, path) = new_wal(&dir);
        wal.flush().unwrap();
        drop(wal);

        assert!(Wal::replay(&path).unwrap().is_empty());
    }

    #[test]
    fn round_trip_empty_key_and_value() {
        let dir = TempDir::new().unwrap();
        let (mut wal, path) = new_wal(&dir);
        wal.append_put(b"", b"", 0).unwrap();
        wal.flush().unwrap();
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
        // 1 MiB value — larger than BufWriter's default 8 KiB buffer,
        // forcing multiple writes to the underlying file.
        let big = vec![0xAB; 1 << 20];
        wal.append_put(b"k", &big, 1).unwrap();
        wal.flush().unwrap();
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
        wal.flush().unwrap();
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
        wal.flush().unwrap();
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
            wal.sync().unwrap();
        }
        let entries = Wal::replay(&path).unwrap();
        assert_eq!(entries.len(), 1);
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
        wal.flush().unwrap();
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
        wal.flush().unwrap();
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
    fn replay_errors_on_truncated_trailing_header() {
        let dir = TempDir::new().unwrap();
        let (mut wal, path) = new_wal(&dir);
        wal.append_put(b"good", b"v", 1).unwrap();
        wal.flush().unwrap();
        drop(wal);

        // Simulate a crash in the middle of writing the next record's
        // 5-byte header by appending 2 stray bytes.
        let mut bytes = fs::read(&path).unwrap();
        bytes.extend_from_slice(&[0xFF, 0xFF]);
        fs::write(&path, &bytes).unwrap();

        let kind = match Wal::replay(&path) {
            Err(e) => e.kind(),
            Ok(v) => panic!("expected truncated header error, got {} entries", v.len()),
        };
        assert_eq!(kind, io::ErrorKind::UnexpectedEof);
    }

    #[test]
    fn replay_errors_when_length_header_exceeds_file() {
        // Hand-craft a single record whose header claims 1000 bytes
        // of data but the file actually contains none. Replay should
        // surface an IO error rather than loop or silently truncate.
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("bad.wal");
        let mut f = File::create(&path).unwrap();
        f.write_all(&1000u32.to_le_bytes()).unwrap(); // len
        f.write_all(&[RECORD_PUT]).unwrap(); // type
        f.sync_all().unwrap();

        let kind = match Wal::replay(&path) {
            Err(e) => e.kind(),
            Ok(v) => panic!("expected error, got {} entries", v.len()),
        };
        assert_eq!(kind, io::ErrorKind::UnexpectedEof);
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
