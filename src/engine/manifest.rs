use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::RwLock;

use super::sstable::{sst_filename, LiveSst, SsTableMeta, SsTableReader};

/// Maximum number of levels in the LSM tree.
pub(crate) const MAX_LEVELS: usize = 7;

/// A snapshot of which SSTables exist at each level.
///
/// Each level holds `Arc<LiveSst>` — the metadata plus an open reader —
/// so that every file referenced by a live version has a pinned file
/// descriptor. Concurrent compaction can safely `unlink` a file as soon
/// as it's removed from the *current* version because the Arcs in older
/// versions keep the FD alive until those versions are dropped.
#[derive(Clone)]
pub(crate) struct Version {
    pub(crate) levels: Vec<Vec<Arc<LiveSst>>>,
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
        self.levels[level].iter().map(|f| f.meta.file_size).sum()
    }
}

/// A runtime mutation to the version. Carries `Arc<LiveSst>` for
/// `AddFile` so the caller is responsible for opening the reader before
/// the apply, and the manifest machinery never has to touch the
/// filesystem for a runtime edit.
#[derive(Clone)]
pub(crate) enum VersionEdit {
    AddFile { level: usize, file: Arc<LiveSst> },
    RemoveFile { level: usize, file_id: u64 },
    SetLastSeq(u64),
    SetNextFileId(u64),
}

/// Serialized form of a version edit. The manifest on disk is a sequence
/// of these records; runtime edits are converted to records just before
/// being written out.
enum ManifestRecord {
    AddFile { level: usize, meta: SsTableMeta },
    RemoveFile { level: usize, file_id: u64 },
    SetLastSeq(u64),
    SetNextFileId(u64),
}

const TAG_ADD_FILE: u8 = 1;
const TAG_REMOVE_FILE: u8 = 2;
const TAG_LAST_SEQ: u8 = 3;
const TAG_NEXT_FILE_ID: u8 = 4;

impl VersionEdit {
    fn to_record(&self) -> ManifestRecord {
        match self {
            VersionEdit::AddFile { level, file } => ManifestRecord::AddFile {
                level: *level,
                meta: file.meta.clone(),
            },
            VersionEdit::RemoveFile { level, file_id } => ManifestRecord::RemoveFile {
                level: *level,
                file_id: *file_id,
            },
            VersionEdit::SetLastSeq(seq) => ManifestRecord::SetLastSeq(*seq),
            VersionEdit::SetNextFileId(id) => ManifestRecord::SetNextFileId(*id),
        }
    }
}

impl ManifestRecord {
    fn encode(&self, buf: &mut Vec<u8>) {
        match self {
            ManifestRecord::AddFile { level, meta } => {
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
            ManifestRecord::RemoveFile { level, file_id } => {
                buf.push(TAG_REMOVE_FILE);
                buf.extend_from_slice(&(*level as u32).to_le_bytes());
                buf.extend_from_slice(&file_id.to_le_bytes());
            }
            ManifestRecord::SetLastSeq(seq) => {
                buf.push(TAG_LAST_SEQ);
                buf.extend_from_slice(&seq.to_le_bytes());
            }
            ManifestRecord::SetNextFileId(id) => {
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

                Ok(Some(ManifestRecord::AddFile {
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
                Ok(Some(ManifestRecord::RemoveFile { level, file_id }))
            }
            TAG_LAST_SEQ => {
                let seq = read_u64(data, pos)?;
                Ok(Some(ManifestRecord::SetLastSeq(seq)))
            }
            TAG_NEXT_FILE_ID => {
                let id = read_u64(data, pos)?;
                Ok(Some(ManifestRecord::SetNextFileId(id)))
            }
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unknown manifest record tag: {}", tag),
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
    /// Create or recover a VersionSet from the given directory. During
    /// recovery every SSTable referenced by the manifest is opened
    /// eagerly so the returned version is fully populated with live
    /// readers.
    pub(crate) fn open(db_dir: &Path, sst_dir: &Path) -> io::Result<Self> {
        let manifest_path = db_dir.join("MANIFEST");

        let (version, writer) = if manifest_path.exists() {
            let data = fs::read(&manifest_path)?;
            let version = Self::replay_manifest(&data, sst_dir)?;

            let file = OpenOptions::new().append(true).open(&manifest_path)?;
            (version, BufWriter::new(file))
        } else {
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

    /// Path of the manifest file on disk.
    pub(crate) fn manifest_path(&self) -> &Path {
        &self.manifest_path
    }

    /// Apply a batch of edits atomically: update the in-memory version
    /// and persist the serialized records to the manifest log.
    pub(crate) fn apply(&mut self, edits: &[VersionEdit]) -> io::Result<()> {
        let mut version = (*self.current()).clone();

        for edit in edits {
            match edit {
                VersionEdit::AddFile { level, file } => {
                    version.levels[*level].push(Arc::clone(file));
                }
                VersionEdit::RemoveFile { level, file_id } => {
                    version.levels[*level].retain(|f| f.meta.file_id != *file_id);
                }
                VersionEdit::SetLastSeq(seq) => {
                    version.last_seq = *seq;
                }
                VersionEdit::SetNextFileId(id) => {
                    version.next_file_id = *id;
                }
            }
        }

        let records: Vec<ManifestRecord> = edits.iter().map(VersionEdit::to_record).collect();
        let encoded = Self::encode_records(&records);
        if let Some(writer) = &mut self.manifest_writer {
            writer.write_all(&encoded)?;
            writer.flush()?;
        }

        *self.current.write() = Arc::new(version);

        Ok(())
    }

    /// Rewrite the manifest from scratch, emitting the current version as
    /// a single compact sequence of records. Readers in the live
    /// `Version` are preserved — we never close their file descriptors.
    pub(crate) fn compact_manifest(&mut self) -> io::Result<()> {
        let version = self.current();

        let mut records = Vec::new();
        records.push(ManifestRecord::SetNextFileId(version.next_file_id));
        records.push(ManifestRecord::SetLastSeq(version.last_seq));
        for (level, files) in version.levels.iter().enumerate() {
            for file in files {
                records.push(ManifestRecord::AddFile {
                    level,
                    meta: file.meta.clone(),
                });
            }
        }

        let encoded = Self::encode_records(&records);

        let tmp_path = self.manifest_path.with_extension("tmp");
        {
            let mut file = File::create(&tmp_path)?;
            file.write_all(&encoded)?;
            file.sync_all()?;
        }
        fs::rename(&tmp_path, &self.manifest_path)?;

        let file = OpenOptions::new().append(true).open(&self.manifest_path)?;
        self.manifest_writer = Some(BufWriter::new(file));

        Ok(())
    }

    fn encode_records(records: &[ManifestRecord]) -> Vec<u8> {
        let mut buf = Vec::new();
        for record in records {
            let mut record_buf = Vec::new();
            record.encode(&mut record_buf);

            let len = record_buf.len() as u32;
            let checksum = xxhash_rust::xxh3::xxh3_64(&record_buf) as u32;

            buf.extend_from_slice(&len.to_le_bytes());
            buf.extend_from_slice(&record_buf);
            buf.extend_from_slice(&checksum.to_le_bytes());
        }
        buf
    }

    fn replay_manifest(data: &[u8], sst_dir: &Path) -> io::Result<Version> {
        // Two-pass replay. The first pass walks every record and tracks
        // the *logical* state of each level — which file ids are live —
        // without touching the filesystem. Only after replay completes
        // do we open readers for the surviving files.
        //
        // This matters for compaction-heavy histories: when compaction
        // adds an L1 file and removes the L0 inputs, both records land
        // in the manifest, but the inputs' physical files are unlinked
        // from disk by `delete_old_files`. An eager open at AddFile
        // time would fail on the unlinked files even though a later
        // RemoveFile record cancels them out.
        let mut surviving: Vec<Vec<SsTableMeta>> = vec![Vec::new(); MAX_LEVELS];
        let mut last_seq: u64 = 0;
        let mut next_file_id: u64 = 1;
        let mut offset = 0;

        while offset + 4 <= data.len() {
            let len = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
            offset += 4;

            if offset + len + 4 > data.len() {
                tracing::warn!("Truncated manifest record, stopping replay");
                break;
            }

            let record_data = &data[offset..offset + len];
            offset += len;

            let stored_checksum = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
            offset += 4;

            let computed_checksum = xxhash_rust::xxh3::xxh3_64(record_data) as u32;
            if stored_checksum != computed_checksum {
                tracing::warn!("Manifest checksum mismatch, stopping replay");
                break;
            }

            let mut pos = 0;
            while let Some(record) = ManifestRecord::decode(record_data, &mut pos)? {
                match record {
                    ManifestRecord::AddFile { level, meta } => {
                        surviving[level].push(meta);
                    }
                    ManifestRecord::RemoveFile { level, file_id } => {
                        surviving[level].retain(|m| m.file_id != file_id);
                    }
                    ManifestRecord::SetLastSeq(seq) => {
                        last_seq = seq;
                    }
                    ManifestRecord::SetNextFileId(id) => {
                        next_file_id = id;
                    }
                }
            }
        }

        // Second pass: open readers for the survivors.
        let mut version = Version::new();
        version.last_seq = last_seq;
        version.next_file_id = next_file_id;
        for (level, files) in surviving.into_iter().enumerate() {
            for meta in files {
                let path = sst_dir.join(sst_filename(meta.file_id));
                let reader = Arc::new(SsTableReader::open(&path, meta.file_id).map_err(|e| {
                    std::io::Error::new(e.kind(), format!("open {}: {e}", path.display()))
                })?);
                version.levels[level].push(LiveSst::new(meta, reader));
            }
        }

        Ok(version)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// Build a real on-disk SSTable and open a reader for it. Used by
    /// tests that need a non-trivial `LiveSst` instance.
    fn make_live_sst(dir: &Path, file_id: u64, smallest: &[u8], largest: &[u8]) -> Arc<LiveSst> {
        use super::super::internal_key::{encode_internal_key, VALUE_TYPE_VALUE};
        use super::super::sstable::SsTableWriter;
        use crate::options::CompressionType;

        let path = dir.join(sst_filename(file_id));
        let mut writer =
            SsTableWriter::new(&path, 4096, 10, CompressionType::None, None, false, 4096).unwrap();
        writer
            .add(
                &encode_internal_key(smallest, 1, VALUE_TYPE_VALUE),
                b"value",
            )
            .unwrap();
        if smallest != largest {
            writer
                .add(&encode_internal_key(largest, 1, VALUE_TYPE_VALUE), b"value")
                .unwrap();
        }
        let summary = writer.finish().unwrap().unwrap();
        let file_size = std::fs::metadata(&path).unwrap().len();
        let reader = Arc::new(SsTableReader::open(&path, file_id).unwrap());
        LiveSst::new(
            SsTableMeta {
                file_id,
                smallest_key: summary.smallest_user_key,
                largest_key: summary.largest_user_key,
                file_size,
                num_entries: summary.num_entries,
            },
            reader,
        )
    }

    #[test]
    fn test_apply_and_replay_roundtrip() {
        let dir = TempDir::new().unwrap();
        let sst_dir = dir.path().join("sst");
        std::fs::create_dir_all(&sst_dir).unwrap();

        let file1 = make_live_sst(&sst_dir, 1, b"aaa", b"zzz");

        let edits = vec![
            VersionEdit::AddFile {
                level: 0,
                file: Arc::clone(&file1),
            },
            VersionEdit::SetLastSeq(42),
            VersionEdit::SetNextFileId(10),
        ];

        {
            let mut vs = VersionSet::open(dir.path(), &sst_dir).unwrap();
            vs.apply(&edits).unwrap();
            let v = vs.current();
            assert_eq!(v.levels[0].len(), 1);
            assert_eq!(v.levels[0][0].meta.file_id, 1);
            assert_eq!(v.last_seq, 42);
            assert_eq!(v.next_file_id, 10);
        }

        // Recover by replaying the manifest; readers are reopened.
        let vs = VersionSet::open(dir.path(), &sst_dir).unwrap();
        let v = vs.current();
        assert_eq!(v.levels[0].len(), 1);
        assert_eq!(v.levels[0][0].meta.file_id, 1);
        assert_eq!(v.last_seq, 42);
        assert_eq!(v.next_file_id, 10);
    }

    #[test]
    fn test_remove_file_hides_it_from_new_version() {
        let dir = TempDir::new().unwrap();
        let sst_dir = dir.path().join("sst");
        std::fs::create_dir_all(&sst_dir).unwrap();

        let file1 = make_live_sst(&sst_dir, 1, b"a", b"c");
        let file2 = make_live_sst(&sst_dir, 2, b"d", b"f");

        let mut vs = VersionSet::open(dir.path(), &sst_dir).unwrap();
        vs.apply(&[
            VersionEdit::AddFile {
                level: 0,
                file: Arc::clone(&file1),
            },
            VersionEdit::AddFile {
                level: 0,
                file: Arc::clone(&file2),
            },
        ])
        .unwrap();

        // Holding a snapshot of the version *before* removal keeps both
        // files alive — this is the invariant that lets get/iter reads
        // survive concurrent compaction.
        let pinned = vs.current();
        assert_eq!(pinned.levels[0].len(), 2);

        vs.apply(&[VersionEdit::RemoveFile {
            level: 0,
            file_id: 1,
        }])
        .unwrap();

        let v = vs.current();
        assert_eq!(v.levels[0].len(), 1);
        assert_eq!(v.levels[0][0].meta.file_id, 2);
        // Pinned snapshot still sees both files.
        assert_eq!(pinned.levels[0].len(), 2);
    }
}
