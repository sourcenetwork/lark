use std::fs::{self, File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

/// Record types in the WAL.
const RECORD_PUT: u8 = 0x01;
const RECORD_DELETE: u8 = 0x02;

/// A write-ahead log for crash recovery.
///
/// Records are CRC-protected and append-only. Each memtable gets its own
/// WAL file. On crash recovery, WAL files are replayed to reconstruct
/// memtable state.
pub(crate) struct Wal {
    writer: BufWriter<File>,
    path: PathBuf,
}

/// A replayed WAL entry.
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

        data.extend_from_slice(&(key.len() as u32).to_le_bytes());
        data.extend_from_slice(key);
        data.extend_from_slice(&(value.len() as u32).to_le_bytes());
        data.extend_from_slice(value);
        data.extend_from_slice(&seq.to_le_bytes());

        self.write_record(RECORD_PUT, &data)
    }

    /// Append a delete record.
    pub(crate) fn append_delete(&mut self, key: &[u8], seq: u64) -> io::Result<()> {
        let mut data = Vec::with_capacity(4 + key.len() + 8);

        data.extend_from_slice(&(key.len() as u32).to_le_bytes());
        data.extend_from_slice(key);
        data.extend_from_slice(&seq.to_le_bytes());

        self.write_record(RECORD_DELETE, &data)
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
        let checksum = xxhash_rust::xxh3::xxh3_64(data) as u32;

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

        loop {
            // Read record header: length (4 bytes) + type (1 byte)
            let mut header = [0u8; 5];
            match reader.read_exact(&mut header) {
                Ok(()) => {}
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e),
            }

            let len = u32::from_le_bytes(header[0..4].try_into().unwrap()) as usize;
            let record_type = header[4];

            // Read data
            let mut data = vec![0u8; len];
            reader.read_exact(&mut data)?;

            // Read and verify checksum
            let mut checksum_bytes = [0u8; 4];
            reader.read_exact(&mut checksum_bytes)?;
            let stored_checksum = u32::from_le_bytes(checksum_bytes);
            let computed_checksum = xxhash_rust::xxh3::xxh3_64(&data) as u32;

            if stored_checksum != computed_checksum {
                tracing::warn!(
                    path = %path.display(),
                    "WAL checksum mismatch - truncated record at end of file, stopping replay"
                );
                break;
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
                _ => {
                    tracing::warn!(record_type, "Unknown WAL record type, skipping");
                }
            }
        }

        Ok(entries)
    }

    /// Delete a WAL file.
    pub(crate) fn remove(path: &Path) -> io::Result<()> {
        fs::remove_file(path)
    }
}

/// Format a WAL filename from a numeric ID.
pub(crate) fn wal_filename(id: u64) -> String {
    format!("wal_{:06}.log", id)
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

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

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
}
