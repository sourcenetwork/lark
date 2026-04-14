use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::RwLock;

use super::sstable::SsTableMeta;

/// Maximum number of levels in the LSM tree.
pub(crate) const MAX_LEVELS: usize = 7;

/// A snapshot of which SSTables exist at each level.
#[derive(Clone, Debug)]
pub(crate) struct Version {
    pub(crate) levels: Vec<Vec<SsTableMeta>>,
    pub(crate) next_file_id: u64,
    pub(crate) last_seq: u64,
}

impl Version {
    pub(crate) fn new() -> Self {
        Self {
            levels: (0..MAX_LEVELS).map(|_| Vec::new()).collect(),
            next_file_id: 1,
            last_seq: 0,
        }
    }

    /// Number of SSTables at L0.
    pub(crate) fn l0_count(&self) -> usize {
        self.levels[0].len()
    }

    /// Total size of SSTables at a given level.
    pub(crate) fn level_size(&self, level: usize) -> u64 {
        self.levels[level].iter().map(|m| m.file_size).sum()
    }
}

/// A mutation to the version.
#[derive(Debug, Clone)]
pub(crate) enum VersionEdit {
    AddFile { level: usize, meta: SsTableMeta },
    RemoveFile { level: usize, file_id: u64 },
    SetLastSeq(u64),
    SetNextFileId(u64),
}

/// Tag bytes for serialized version edits.
const TAG_ADD_FILE: u8 = 1;
const TAG_REMOVE_FILE: u8 = 2;
const TAG_LAST_SEQ: u8 = 3;
const TAG_NEXT_FILE_ID: u8 = 4;

impl VersionEdit {
    fn encode(&self, buf: &mut Vec<u8>) {
        match self {
            VersionEdit::AddFile { level, meta } => {
                buf.push(TAG_ADD_FILE);
                buf.extend_from_slice(&(*level as u32).to_le_bytes());
                buf.extend_from_slice(&meta.file_id.to_le_bytes());
                buf.extend_from_slice(&(meta.smallest_key.len() as u32).to_le_bytes());
                buf.extend_from_slice(&meta.smallest_key);
                buf.extend_from_slice(&(meta.largest_key.len() as u32).to_le_bytes());
                buf.extend_from_slice(&meta.largest_key);
                buf.extend_from_slice(&meta.file_size.to_le_bytes());
                buf.extend_from_slice(&meta.num_entries.to_le_bytes());
            }
            VersionEdit::RemoveFile { level, file_id } => {
                buf.push(TAG_REMOVE_FILE);
                buf.extend_from_slice(&(*level as u32).to_le_bytes());
                buf.extend_from_slice(&file_id.to_le_bytes());
            }
            VersionEdit::SetLastSeq(seq) => {
                buf.push(TAG_LAST_SEQ);
                buf.extend_from_slice(&seq.to_le_bytes());
            }
            VersionEdit::SetNextFileId(id) => {
                buf.push(TAG_NEXT_FILE_ID);
                buf.extend_from_slice(&id.to_le_bytes());
            }
        }
    }

    fn decode(data: &[u8], pos: &mut usize) -> io::Result<Option<Self>> {
        if *pos >= data.len() {
            return Ok(None);
        }

        let tag = data[*pos];
        *pos += 1;

        match tag {
            TAG_ADD_FILE => {
                let level = read_u32(data, pos)? as usize;
                let file_id = read_u64(data, pos)?;
                let smallest_key = read_bytes(data, pos)?;
                let largest_key = read_bytes(data, pos)?;
                let file_size = read_u64(data, pos)?;
                let num_entries = read_u64(data, pos)?;

                Ok(Some(VersionEdit::AddFile {
                    level,
                    meta: SsTableMeta {
                        file_id,
                        smallest_key,
                        largest_key,
                        file_size,
                        num_entries,
                    },
                }))
            }
            TAG_REMOVE_FILE => {
                let level = read_u32(data, pos)? as usize;
                let file_id = read_u64(data, pos)?;
                Ok(Some(VersionEdit::RemoveFile { level, file_id }))
            }
            TAG_LAST_SEQ => {
                let seq = read_u64(data, pos)?;
                Ok(Some(VersionEdit::SetLastSeq(seq)))
            }
            TAG_NEXT_FILE_ID => {
                let id = read_u64(data, pos)?;
                Ok(Some(VersionEdit::SetNextFileId(id)))
            }
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unknown version edit tag: {}", tag),
            )),
        }
    }
}

fn read_u32(data: &[u8], pos: &mut usize) -> io::Result<u32> {
    if *pos + 4 > data.len() {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "short read"));
    }
    let val = u32::from_le_bytes(data[*pos..*pos + 4].try_into().unwrap());
    *pos += 4;
    Ok(val)
}

fn read_u64(data: &[u8], pos: &mut usize) -> io::Result<u64> {
    if *pos + 8 > data.len() {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "short read"));
    }
    let val = u64::from_le_bytes(data[*pos..*pos + 8].try_into().unwrap());
    *pos += 8;
    Ok(val)
}

fn read_bytes(data: &[u8], pos: &mut usize) -> io::Result<Vec<u8>> {
    let len = read_u32(data, pos)? as usize;
    if *pos + len > data.len() {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "short read"));
    }
    let bytes = data[*pos..*pos + len].to_vec();
    *pos += len;
    Ok(bytes)
}

/// Manages the current version and persists version edits to a manifest log.
pub(crate) struct VersionSet {
    current: Arc<RwLock<Arc<Version>>>,
    manifest_path: PathBuf,
    manifest_writer: Option<BufWriter<File>>,
}

impl VersionSet {
    /// Create or recover a VersionSet from the given directory.
    pub(crate) fn open(db_dir: &Path) -> io::Result<Self> {
        let manifest_path = db_dir.join("MANIFEST");

        let (version, writer) = if manifest_path.exists() {
            // Recover by replaying the manifest
            let data = fs::read(&manifest_path)?;
            let version = Self::replay_manifest(&data)?;

            let file = OpenOptions::new().append(true).open(&manifest_path)?;
            (version, BufWriter::new(file))
        } else {
            // Create new manifest
            let version = Version::new();
            let file = File::create(&manifest_path)?;
            (version, BufWriter::new(file))
        };

        Ok(Self {
            current: Arc::new(RwLock::new(Arc::new(version))),
            manifest_path,
            manifest_writer: Some(writer),
        })
    }

    /// Get the current version.
    pub(crate) fn current(&self) -> Arc<Version> {
        Arc::clone(&*self.current.read())
    }

    /// Apply a batch of edits atomically: update the in-memory version
    /// and persist to the manifest log.
    pub(crate) fn apply(&mut self, edits: &[VersionEdit]) -> io::Result<()> {
        let mut version = (*self.current()).clone();

        for edit in edits {
            match edit {
                VersionEdit::AddFile { level, meta } => {
                    version.levels[*level].push(meta.clone());
                }
                VersionEdit::RemoveFile { level, file_id } => {
                    version.levels[*level].retain(|m| m.file_id != *file_id);
                }
                VersionEdit::SetLastSeq(seq) => {
                    version.last_seq = *seq;
                }
                VersionEdit::SetNextFileId(id) => {
                    version.next_file_id = *id;
                }
            }
        }

        // Persist to manifest
        let encoded = Self::encode_edits(edits);
        if let Some(writer) = &mut self.manifest_writer {
            writer.write_all(&encoded)?;
            writer.flush()?;
        }

        // Atomically swap version
        *self.current.write() = Arc::new(version);

        Ok(())
    }

    /// Rewrite the manifest from scratch (compaction of manifest itself).
    pub(crate) fn compact_manifest(&mut self) -> io::Result<()> {
        let version = self.current();
        let mut edits = Vec::new();

        edits.push(VersionEdit::SetNextFileId(version.next_file_id));
        edits.push(VersionEdit::SetLastSeq(version.last_seq));

        for (level, files) in version.levels.iter().enumerate() {
            for meta in files {
                edits.push(VersionEdit::AddFile {
                    level,
                    meta: meta.clone(),
                });
            }
        }

        let encoded = Self::encode_edits(&edits);

        // Write to temp file, then rename
        let tmp_path = self.manifest_path.with_extension("tmp");
        {
            let mut file = File::create(&tmp_path)?;
            file.write_all(&encoded)?;
            file.sync_all()?;
        }
        fs::rename(&tmp_path, &self.manifest_path)?;

        // Reopen for append
        let file = OpenOptions::new().append(true).open(&self.manifest_path)?;
        self.manifest_writer = Some(BufWriter::new(file));

        Ok(())
    }

    fn encode_edits(edits: &[VersionEdit]) -> Vec<u8> {
        let mut buf = Vec::new();
        for edit in edits {
            let mut edit_buf = Vec::new();
            edit.encode(&mut edit_buf);

            // Length-prefixed record with checksum
            let len = edit_buf.len() as u32;
            let checksum = xxhash_rust::xxh3::xxh3_64(&edit_buf) as u32;

            buf.extend_from_slice(&len.to_le_bytes());
            buf.extend_from_slice(&edit_buf);
            buf.extend_from_slice(&checksum.to_le_bytes());
        }
        buf
    }

    fn replay_manifest(data: &[u8]) -> io::Result<Version> {
        let mut version = Version::new();
        let mut offset = 0;

        while offset + 4 <= data.len() {
            let len = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
            offset += 4;

            if offset + len + 4 > data.len() {
                tracing::warn!("Truncated manifest record, stopping replay");
                break;
            }

            let edit_data = &data[offset..offset + len];
            offset += len;

            let stored_checksum = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
            offset += 4;

            let computed_checksum = xxhash_rust::xxh3::xxh3_64(edit_data) as u32;
            if stored_checksum != computed_checksum {
                tracing::warn!("Manifest checksum mismatch, stopping replay");
                break;
            }

            let mut pos = 0;
            while let Some(edit) = VersionEdit::decode(edit_data, &mut pos)? {
                match &edit {
                    VersionEdit::AddFile { level, meta } => {
                        version.levels[*level].push(meta.clone());
                    }
                    VersionEdit::RemoveFile { level, file_id } => {
                        version.levels[*level].retain(|m| m.file_id != *file_id);
                    }
                    VersionEdit::SetLastSeq(seq) => {
                        version.last_seq = *seq;
                    }
                    VersionEdit::SetNextFileId(id) => {
                        version.next_file_id = *id;
                    }
                }
            }
        }

        Ok(version)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_version_edit_roundtrip() {
        let edits = vec![
            VersionEdit::AddFile {
                level: 0,
                meta: SsTableMeta {
                    file_id: 1,
                    smallest_key: b"aaa".to_vec(),
                    largest_key: b"zzz".to_vec(),
                    file_size: 1024,
                    num_entries: 100,
                },
            },
            VersionEdit::RemoveFile {
                level: 1,
                file_id: 5,
            },
            VersionEdit::SetLastSeq(42),
            VersionEdit::SetNextFileId(10),
        ];

        let encoded = VersionSet::encode_edits(&edits);
        let version = VersionSet::replay_manifest(&encoded).unwrap();

        assert_eq!(version.levels[0].len(), 1);
        assert_eq!(version.levels[0][0].file_id, 1);
        assert_eq!(version.last_seq, 42);
        assert_eq!(version.next_file_id, 10);
    }

    #[test]
    fn test_version_set_persistence() {
        let dir = TempDir::new().unwrap();

        // Create and apply edits
        {
            let mut vs = VersionSet::open(dir.path()).unwrap();
            vs.apply(&[VersionEdit::AddFile {
                level: 0,
                meta: SsTableMeta {
                    file_id: 1,
                    smallest_key: b"a".to_vec(),
                    largest_key: b"z".to_vec(),
                    file_size: 512,
                    num_entries: 50,
                },
            }])
            .unwrap();
            vs.apply(&[VersionEdit::SetLastSeq(100)]).unwrap();
        }

        // Recover
        let vs = VersionSet::open(dir.path()).unwrap();
        let version = vs.current();
        assert_eq!(version.levels[0].len(), 1);
        assert_eq!(version.last_seq, 100);
    }
}
