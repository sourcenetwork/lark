use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::RwLock;

use super::sstable::{LiveSst, SsTableMeta, SsTableReader, sst_filename, table_carries_data};
use super::{checksum, durability};

/// Maximum number of levels in the LSM tree.
pub(crate) const MAX_LEVELS: usize = 7;

/// A snapshot of which SSTables exist at each level.
///
/// Each level holds `Arc<LiveSst>` - the metadata plus an open reader -
/// so that every file referenced by a live version has a pinned file
/// descriptor. Concurrent compaction can safely `unlink` a file as soon
/// as it's removed from the *current* version because the Arcs in older
/// versions keep the FD alive until those versions are dropped.
#[derive(Clone)]
pub(crate) struct Version {
    pub(crate) levels: Vec<Vec<Arc<LiveSst>>>,
    pub(crate) next_file_id: u64,
    pub(crate) last_seq: u64,
    pub(crate) min_wal_id: u64,
}

impl Version {
    pub(crate) fn new() -> Self {
        Self {
            levels: (0..MAX_LEVELS).map(|_| Vec::new()).collect(),
            next_file_id: 1,
            last_seq: 0,
            min_wal_id: 0,
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
    Reset { next_file_id: u64, min_wal_id: u64 },
}

/// Serialized form of a version edit. The manifest on disk is a sequence
/// of these records; runtime edits are converted to records just before
/// being written out.
enum ManifestRecord {
    AddFile { level: usize, meta: SsTableMeta },
    RemoveFile { level: usize, file_id: u64 },
    SetLastSeq(u64),
    SetNextFileId(u64),
    SetMinWalId(u64),
    Reset { next_file_id: u64, min_wal_id: u64 },
}

const TAG_ADD_FILE: u8 = 1;
const TAG_REMOVE_FILE: u8 = 2;
const TAG_LAST_SEQ: u8 = 3;
const TAG_NEXT_FILE_ID: u8 = 4;
const TAG_MIN_WAL_ID: u8 = 5;
const TAG_RESET: u8 = 6;

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
            VersionEdit::Reset {
                next_file_id,
                min_wal_id,
            } => ManifestRecord::Reset {
                next_file_id: *next_file_id,
                min_wal_id: *min_wal_id,
            },
        }
    }

    fn requires_manifest_sync(&self) -> bool {
        // File-id reservations do not make new data reachable on
        // their own. They are flushed here and become durable with
        // the next synced AddFile/RemoveFile/SetLastSeq edit.
        !matches!(self, VersionEdit::SetNextFileId(_))
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
            ManifestRecord::SetMinWalId(id) => {
                buf.push(TAG_MIN_WAL_ID);
                buf.extend_from_slice(&id.to_le_bytes());
            }
            ManifestRecord::Reset {
                next_file_id,
                min_wal_id,
            } => {
                buf.push(TAG_RESET);
                buf.extend_from_slice(&next_file_id.to_le_bytes());
                buf.extend_from_slice(&min_wal_id.to_le_bytes());
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
                validate_level_index(level)?;
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
                validate_level_index(level)?;
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
            TAG_MIN_WAL_ID => {
                let id = read_u64(data, pos)?;
                Ok(Some(ManifestRecord::SetMinWalId(id)))
            }
            TAG_RESET => {
                let next_file_id = read_u64(data, pos)?;
                let min_wal_id = read_u64(data, pos)?;
                Ok(Some(ManifestRecord::Reset {
                    next_file_id,
                    min_wal_id,
                }))
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

fn validate_level_index(level: usize) -> io::Result<()> {
    if level < MAX_LEVELS {
        return Ok(());
    }

    Err(io::Error::new(
        io::ErrorKind::InvalidData,
        format!("manifest level {level} out of range 0..{MAX_LEVELS}"),
    ))
}

/// Manages the current version and persists version edits to a manifest log.
pub(crate) struct VersionSet {
    current: Arc<RwLock<Arc<Version>>>,
    manifest_path: PathBuf,
    manifest_writer: Option<BufWriter<File>>,
}

struct ManifestReplay {
    version: Version,
    valid_len: usize,
}

/// An unreferenced `*.sst` file that the discarded-table guard could not
/// dismiss as a crash artifact, with the reason it counts.
struct SuspectTable {
    path: PathBuf,
    /// `None` when the file's metadata could not be read.
    len: Option<u64>,
    reason: String,
}

/// How many suspects the guard's error message names before summarising
/// the rest. The cap is reported in the message so a long list never
/// reads as a short one.
const SUSPECTS_NAMED: usize = 8;

fn describe_suspects(suspects: &[SuspectTable]) -> String {
    let mut out = suspects
        .iter()
        .take(SUSPECTS_NAMED)
        .map(|s| match s.len {
            Some(len) => format!("{} ({len} bytes, {})", s.path.display(), s.reason),
            None => format!("{} (size unknown, {})", s.path.display(), s.reason),
        })
        .collect::<Vec<_>>()
        .join(", ");
    if suspects.len() > SUSPECTS_NAMED {
        out.push_str(&format!(
            ", and {} more not named here",
            suspects.len() - SUSPECTS_NAMED
        ));
    }
    out
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
            let replay = Self::replay_manifest(&data, sst_dir)?;
            Self::reject_discarded_tables(&replay, data.len(), sst_dir, &manifest_path)?;

            let file = OpenOptions::new().append(true).open(&manifest_path)?;
            if replay.valid_len < data.len() {
                file.set_len(replay.valid_len as u64)?;
                file.sync_all()?;
            }
            (replay.version, BufWriter::new(file))
        } else {
            let version = Version::new();
            let file = File::create(&manifest_path)?;
            file.sync_all()?;
            durability::sync_parent_dir(&manifest_path)?;
            (version, BufWriter::new(file))
        };

        Ok(Self {
            current: Arc::new(RwLock::new(Arc::new(version))),
            manifest_path,
            manifest_writer: Some(writer),
        })
    }

    /// Recover an existing VersionSet without mutating the manifest.
    ///
    /// This is used by read-only opens: replay still tolerates a
    /// truncated trailing record exactly like the read-write path, but
    /// the file is not repaired in place and no append writer is kept.
    pub(crate) fn open_read_only(db_dir: &Path, sst_dir: &Path) -> io::Result<Self> {
        let manifest_path = db_dir.join("MANIFEST");
        let data = fs::read(&manifest_path)?;
        let replay = Self::replay_manifest(&data, sst_dir)?;
        Self::reject_discarded_tables(&replay, data.len(), sst_dir, &manifest_path)?;

        Ok(Self {
            current: Arc::new(RwLock::new(Arc::new(replay.version))),
            manifest_path,
            manifest_writer: None,
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
        for edit in edits {
            match edit {
                VersionEdit::AddFile { level, .. } | VersionEdit::RemoveFile { level, .. } => {
                    validate_level_index(*level)?;
                }
                VersionEdit::SetLastSeq(_)
                | VersionEdit::SetNextFileId(_)
                | VersionEdit::Reset { .. } => {}
            }
        }

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
                VersionEdit::Reset {
                    next_file_id,
                    min_wal_id,
                } => {
                    version.levels = (0..MAX_LEVELS).map(|_| Vec::new()).collect();
                    version.last_seq = 0;
                    version.next_file_id = *next_file_id;
                    version.min_wal_id = *min_wal_id;
                }
            }
        }

        let records: Vec<ManifestRecord> = edits.iter().map(VersionEdit::to_record).collect();
        let encoded = Self::encode_records(&records);
        let requires_sync = edits.iter().any(VersionEdit::requires_manifest_sync);
        if let Some(writer) = &mut self.manifest_writer {
            writer.write_all(&encoded)?;
            writer.flush()?;
            if requires_sync {
                writer.get_ref().sync_all()?;
            }
        }

        *self.current.write() = Arc::new(version);

        Ok(())
    }

    /// Rewrite the manifest from scratch, emitting the current version as
    /// a single compact sequence of records. Readers in the live
    /// `Version` are preserved - we never close their file descriptors.
    pub(crate) fn compact_manifest(&mut self) -> io::Result<()> {
        let version = self.current();

        let mut records = Vec::new();
        records.push(ManifestRecord::SetNextFileId(version.next_file_id));
        records.push(ManifestRecord::SetLastSeq(version.last_seq));
        records.push(ManifestRecord::SetMinWalId(version.min_wal_id));
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
        durability::sync_parent_dir(&self.manifest_path)?;

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
            let checksum = checksum::manifest_record(len, &record_buf);

            buf.extend_from_slice(&len.to_le_bytes());
            buf.extend_from_slice(&record_buf);
            buf.extend_from_slice(&checksum.to_le_bytes());
        }
        buf
    }

    /// Refuse to open when a manifest that did not replay cleanly ends up
    /// referencing no SSTable at all while the table directory still holds
    /// one that could carry data.
    ///
    /// Replay stops at the first unreadable record and the tail is
    /// discarded, which is correct for a record that a crash left half
    /// written. When the *first* record is unreadable the same rule
    /// silently turns a populated database into an empty one, so that
    /// combination is reported instead of served: the table files are
    /// still on disk and only the manifest needs repairing.
    ///
    /// A crash inside the very first flush leaves the opposite shape: a
    /// table file that holds nothing, next to a WAL that holds every
    /// acknowledged write. Refusing on that file would lose the writes
    /// the WAL still has, so `suspect_tables` rules it out
    /// before the count is taken.
    fn reject_discarded_tables(
        replay: &ManifestReplay,
        manifest_len: usize,
        sst_dir: &Path,
        manifest_path: &Path,
    ) -> io::Result<()> {
        let replayed_cleanly = manifest_len > 0 && replay.valid_len == manifest_len;
        if replayed_cleanly || replay.version.levels.iter().any(|level| !level.is_empty()) {
            return Ok(());
        }
        let suspects = Self::suspect_tables(sst_dir);
        if suspects.is_empty() {
            return Ok(());
        }
        Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "{} is corrupt: it references no SSTable, but {} table file(s) in {} may still hold data. \
                 Opening would discard them, so the database is left untouched. Suspect tables: {}",
                manifest_path.display(),
                suspects.len(),
                sst_dir.display(),
                describe_suspects(&suspects),
            ),
        ))
    }

    /// The unreferenced `*.sst` files that could plausibly hold live data.
    ///
    /// Two shapes are skipped, because neither can be a live table the
    /// manifest is about to discard:
    ///
    /// * a zero-length file: it holds no byte, so it can hold no entry.
    ///   This is what a crash inside the very first flush leaves once
    ///   the directory entry reached the journal and the
    ///   delayed-allocated data blocks did not;
    /// * a file whose footer parses and whose index block is empty: it
    ///   provably holds nothing.
    ///
    /// Everything else counts, including a file whose footer or index
    /// will not parse. Such a file may be a real table with damage in
    /// it, cannot be proved empty, and keeping the database shut
    /// preserves it for repair.
    ///
    /// The line is deliberately drawn at proof rather than at
    /// plausibility. An earlier revision also skipped a file whose last
    /// 64 bytes were all zero, reasoning that a footer ends in a
    /// non-zero magic so a footer's worth of zeros meant the flush never
    /// finished. That inference does not hold: a lost or misdirected
    /// write on an otherwise complete table zeroes a whole sector or
    /// block, which satisfies the same test, and the file's absence from
    /// the manifest cannot be used as corroboration inside the one
    /// branch that runs only when the manifest is already known corrupt.
    /// Skipping such a file opens the database silently without
    /// everything that table held; counting it refuses loudly, names the
    /// file, deletes nothing, and leaves the operator able to move it
    /// aside and recover.
    /// `adv_g28_tears::a_first_flush_cut_that_leaves_an_unprovable_orphan_refuses_and_salvages`
    /// measures the cost of that refusal rather than hiding it: 61
    /// acknowledged writes behind a loud error, all of them recovered
    /// once the orphan is moved aside.
    ///
    /// Nothing is deleted here, so a crash part way through recovery
    /// leaves the directory exactly as this pass found it and the next
    /// open reaches the same verdict.
    fn suspect_tables(sst_dir: &Path) -> Vec<SuspectTable> {
        let entries = match fs::read_dir(sst_dir) {
            Ok(entries) => entries,
            Err(e) => {
                tracing::warn!(
                    dir = %sst_dir.display(),
                    error = %e,
                    "could not list the SSTable directory while checking for discarded tables"
                );
                return Vec::new();
            }
        };

        let mut suspects = Vec::new();
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("sst") {
                continue;
            }
            let len = match entry.metadata() {
                Ok(meta) => Some(meta.len()),
                Err(e) => {
                    suspects.push(SuspectTable {
                        path,
                        len: None,
                        reason: format!("unreadable: {e}"),
                    });
                    continue;
                }
            };
            if len == Some(0) {
                tracing::warn!(
                    path = %path.display(),
                    "ignoring a zero-length orphan SSTable left by a crash inside a flush"
                );
                continue;
            }
            match table_carries_data(&path) {
                Ok(true) => suspects.push(SuspectTable {
                    path,
                    len,
                    reason: "carries data".to_string(),
                }),
                Ok(false) => tracing::warn!(
                    path = %path.display(),
                    "ignoring an orphan SSTable whose footer and index both record nothing"
                ),
                Err(e) => suspects.push(SuspectTable {
                    path,
                    len,
                    reason: format!("unreadable footer: {e}"),
                }),
            }
        }
        suspects.sort_by(|a, b| a.path.cmp(&b.path));
        suspects
    }

    fn replay_manifest(data: &[u8], sst_dir: &Path) -> io::Result<ManifestReplay> {
        // Two-pass replay. The first pass walks every record and tracks
        // the *logical* state of each level - which file ids are live -
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
        let mut min_wal_id: u64 = 0;
        let mut offset = 0;
        let mut valid_len = 0;

        while offset < data.len() {
            if offset + 4 > data.len() {
                tracing::warn!("Truncated manifest record header, stopping replay");
                break;
            }

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

            let computed_checksum = checksum::manifest_record(len as u32, record_data);
            if stored_checksum != computed_checksum
                && stored_checksum != checksum::legacy_payload_u32(record_data)
            {
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
                    ManifestRecord::SetMinWalId(id) => {
                        min_wal_id = id;
                    }
                    ManifestRecord::Reset {
                        next_file_id: reset_next_file_id,
                        min_wal_id: reset_min_wal_id,
                    } => {
                        for level in &mut surviving {
                            level.clear();
                        }
                        last_seq = 0;
                        next_file_id = reset_next_file_id;
                        min_wal_id = reset_min_wal_id;
                    }
                }
            }
            valid_len = offset;
        }

        // Second pass: open readers for the survivors.
        let mut version = Version::new();
        version.last_seq = last_seq;
        version.next_file_id = next_file_id;
        version.min_wal_id = min_wal_id;
        for (level, files) in surviving.into_iter().enumerate() {
            for meta in files {
                let path = sst_dir.join(sst_filename(meta.file_id));
                let reader = Arc::new(SsTableReader::open(&path, meta.file_id).map_err(|e| {
                    std::io::Error::new(e.kind(), format!("open {}: {e}", path.display()))
                })?);
                version.levels[level].push(LiveSst::new(meta, reader));
            }
        }

        Ok(ManifestReplay { version, valid_len })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// Build a real on-disk SSTable and open a reader for it. Used by
    /// tests that need a non-trivial `LiveSst` instance.
    fn make_live_sst(dir: &Path, file_id: u64, smallest: &[u8], largest: &[u8]) -> Arc<LiveSst> {
        use super::super::internal_key::{VALUE_TYPE_VALUE, encode_internal_key};
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

    fn second_record_checksum_offset(path: &Path) -> usize {
        let data = std::fs::read(path).unwrap();
        let first_len = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
        let second_start = 4 + first_len + 4;
        let second_len =
            u32::from_le_bytes(data[second_start..second_start + 4].try_into().unwrap()) as usize;
        second_start + 4 + second_len
    }

    fn test_meta(file_id: u64) -> SsTableMeta {
        SsTableMeta {
            file_id,
            smallest_key: b"a".to_vec(),
            largest_key: b"z".to_vec(),
            file_size: 128,
            num_entries: 2,
        }
    }

    #[test]
    fn manifest_checksum_covers_length_header() {
        let mut record = Vec::new();
        ManifestRecord::SetLastSeq(7).encode(&mut record);
        let len = record.len() as u32;
        let baseline = checksum::manifest_record(len, &record);
        assert_ne!(baseline, checksum::manifest_record(len + 1, &record));
    }

    #[test]
    fn replay_manifest_accepts_legacy_payload_only_checksum() {
        let dir = TempDir::new().unwrap();
        let mut record = Vec::new();
        ManifestRecord::SetLastSeq(7).encode(&mut record);
        let len = record.len() as u32;
        let mut data = Vec::new();
        data.extend_from_slice(&len.to_le_bytes());
        data.extend_from_slice(&record);
        data.extend_from_slice(&checksum::legacy_payload_u32(&record).to_le_bytes());

        let replay = VersionSet::replay_manifest(&data, dir.path()).unwrap();
        assert_eq!(replay.version.last_seq, 7);
        assert_eq!(replay.valid_len, data.len());
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
            assert_eq!(v.min_wal_id, 0);
        }

        // Recover by replaying the manifest; readers are reopened.
        let vs = VersionSet::open(dir.path(), &sst_dir).unwrap();
        let v = vs.current();
        assert_eq!(v.levels[0].len(), 1);
        assert_eq!(v.levels[0][0].meta.file_id, 1);
        assert_eq!(v.last_seq, 42);
        assert_eq!(v.next_file_id, 10);
        assert_eq!(v.min_wal_id, 0);
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
        // files alive - this is the invariant that lets get/iter reads
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

    #[test]
    fn initial_version_has_empty_levels_and_defaults() {
        let v = Version::new();
        assert_eq!(v.levels.len(), MAX_LEVELS);
        assert!(v.levels.iter().all(|l| l.is_empty()));
        assert_eq!(v.next_file_id, 1);
        assert_eq!(v.last_seq, 0);
        assert_eq!(v.min_wal_id, 0);
        assert_eq!(v.l0_count(), 0);
        for level in 0..MAX_LEVELS {
            assert_eq!(v.level_size(level), 0);
        }
    }

    #[test]
    fn manifest_path_returned_as_written() {
        let dir = TempDir::new().unwrap();
        let sst_dir = dir.path().join("sst");
        std::fs::create_dir_all(&sst_dir).unwrap();
        let vs = VersionSet::open(dir.path(), &sst_dir).unwrap();
        assert_eq!(vs.manifest_path(), dir.path().join("MANIFEST"));
    }

    #[test]
    fn manifest_record_encode_decode_round_trip() {
        // Exercise every tag through the encode/decode path used by
        // the replay loop.
        let records = [
            ManifestRecord::AddFile {
                level: 2,
                meta: SsTableMeta {
                    file_id: 99,
                    smallest_key: b"aaa".to_vec(),
                    largest_key: b"zzz".to_vec(),
                    file_size: 4096,
                    num_entries: 128,
                },
            },
            ManifestRecord::RemoveFile {
                level: 1,
                file_id: 7,
            },
            ManifestRecord::SetLastSeq(999),
            ManifestRecord::SetNextFileId(42),
            ManifestRecord::SetMinWalId(11),
            ManifestRecord::Reset {
                next_file_id: 77,
                min_wal_id: 76,
            },
        ];
        for r in &records {
            let mut buf = Vec::new();
            r.encode(&mut buf);
            let mut pos = 0;
            let decoded = match ManifestRecord::decode(&buf, &mut pos) {
                Ok(Some(d)) => d,
                other => panic!("expected decoded record, got {:?}", other.is_ok()),
            };
            // Re-encode and compare - equality via round-trip avoids
            // having to add PartialEq to ManifestRecord.
            let mut rebuf = Vec::new();
            decoded.encode(&mut rebuf);
            assert_eq!(buf, rebuf);
            assert_eq!(pos, buf.len());
        }
    }

    #[test]
    fn manifest_record_decode_rejects_unknown_tag() {
        let data = [0xFFu8];
        let mut pos = 0;
        let kind = match ManifestRecord::decode(&data, &mut pos) {
            Err(e) => e.kind(),
            Ok(_) => panic!("expected error on unknown tag"),
        };
        assert_eq!(kind, io::ErrorKind::InvalidData);
    }

    #[test]
    fn manifest_record_decode_rejects_invalid_level_indexes() {
        let records = [
            ManifestRecord::AddFile {
                level: MAX_LEVELS,
                meta: test_meta(1),
            },
            ManifestRecord::RemoveFile {
                level: MAX_LEVELS,
                file_id: 1,
            },
        ];

        for record in records {
            let mut data = Vec::new();
            record.encode(&mut data);
            let mut pos = 0;
            let kind = match ManifestRecord::decode(&data, &mut pos) {
                Err(e) => e.kind(),
                Ok(_) => panic!("expected invalid level error"),
            };
            assert_eq!(kind, io::ErrorKind::InvalidData);
        }
    }

    #[test]
    fn manifest_record_decode_returns_none_at_eof() {
        let mut pos = 0;
        let got = ManifestRecord::decode(&[], &mut pos).unwrap();
        assert!(got.is_none());
    }

    #[test]
    fn apply_rejects_invalid_level_indexes() {
        let dir = TempDir::new().unwrap();
        let sst_dir = dir.path().join("sst");
        std::fs::create_dir_all(&sst_dir).unwrap();

        let file = make_live_sst(&sst_dir, 1, b"a", b"z");
        let mut vs = VersionSet::open(dir.path(), &sst_dir).unwrap();

        let kind = match vs.apply(&[VersionEdit::AddFile {
            level: MAX_LEVELS,
            file,
        }]) {
            Err(e) => e.kind(),
            Ok(_) => panic!("expected invalid level error"),
        };
        assert_eq!(kind, io::ErrorKind::InvalidData);
        assert_eq!(vs.current().levels.iter().map(Vec::len).sum::<usize>(), 0);

        let kind = match vs.apply(&[VersionEdit::RemoveFile {
            level: MAX_LEVELS,
            file_id: 1,
        }]) {
            Err(e) => e.kind(),
            Ok(_) => panic!("expected invalid level error"),
        };
        assert_eq!(kind, io::ErrorKind::InvalidData);
    }

    #[test]
    fn open_rejects_manifest_with_invalid_level_index() {
        let dir = TempDir::new().unwrap();
        let sst_dir = dir.path().join("sst");
        std::fs::create_dir_all(&sst_dir).unwrap();

        let records = [ManifestRecord::AddFile {
            level: MAX_LEVELS,
            meta: test_meta(1),
        }];
        std::fs::write(
            dir.path().join("MANIFEST"),
            VersionSet::encode_records(&records),
        )
        .unwrap();

        let kind = match VersionSet::open(dir.path(), &sst_dir) {
            Err(e) => e.kind(),
            Ok(_) => panic!("expected invalid level error"),
        };
        assert_eq!(kind, io::ErrorKind::InvalidData);
    }

    #[test]
    fn replay_survives_truncated_trailer() {
        // Write a manifest, then truncate it inside the last record.
        // Replay should stop cleanly and expose the valid prefix.
        let dir = TempDir::new().unwrap();
        let sst_dir = dir.path().join("sst");
        std::fs::create_dir_all(&sst_dir).unwrap();

        let file1 = make_live_sst(&sst_dir, 1, b"a", b"m");
        let mut vs = VersionSet::open(dir.path(), &sst_dir).unwrap();
        vs.apply(&[VersionEdit::AddFile {
            level: 0,
            file: Arc::clone(&file1),
        }])
        .unwrap();
        vs.apply(&[VersionEdit::SetLastSeq(50)]).unwrap();
        drop(vs);

        // Truncate 2 bytes off the end - enough to damage the final
        // record's checksum or tail.
        let path = dir.path().join("MANIFEST");
        let current = std::fs::metadata(&path).unwrap().len();
        OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(current - 2)
            .unwrap();

        let vs = VersionSet::open(dir.path(), &sst_dir).unwrap();
        let v = vs.current();
        // The AddFile should have survived (it was the first record);
        // the SetLastSeq may or may not survive depending on where the
        // truncation landed. Either way, we should NOT panic.
        assert!(v.levels[0].len() <= 1);
    }

    #[test]
    fn reset_record_clears_files_and_sets_wal_floor_atomically() {
        let dir = TempDir::new().unwrap();
        let sst_dir = dir.path().join("sst");
        std::fs::create_dir_all(&sst_dir).unwrap();

        let file1 = make_live_sst(&sst_dir, 1, b"a", b"m");
        let file2 = make_live_sst(&sst_dir, 2, b"n", b"z");
        {
            let mut vs = VersionSet::open(dir.path(), &sst_dir).unwrap();
            vs.apply(&[
                VersionEdit::AddFile {
                    level: 0,
                    file: Arc::clone(&file1),
                },
                VersionEdit::AddFile {
                    level: 1,
                    file: Arc::clone(&file2),
                },
                VersionEdit::SetLastSeq(50),
                VersionEdit::SetNextFileId(9),
            ])
            .unwrap();
            vs.apply(&[VersionEdit::Reset {
                next_file_id: 10,
                min_wal_id: 9,
            }])
            .unwrap();
        }

        let vs = VersionSet::open(dir.path(), &sst_dir).unwrap();
        let v = vs.current();
        assert!(v.levels.iter().all(Vec::is_empty));
        assert_eq!(v.last_seq, 0);
        assert_eq!(v.next_file_id, 10);
        assert_eq!(v.min_wal_id, 9);
    }

    #[test]
    fn open_truncates_truncated_manifest_tail_before_append() {
        let dir = TempDir::new().unwrap();
        let sst_dir = dir.path().join("sst");
        std::fs::create_dir_all(&sst_dir).unwrap();

        {
            let mut vs = VersionSet::open(dir.path(), &sst_dir).unwrap();
            vs.apply(&[VersionEdit::SetLastSeq(7)]).unwrap();
            vs.apply(&[VersionEdit::SetLastSeq(11)]).unwrap();
        }

        let path = dir.path().join("MANIFEST");
        let current = std::fs::metadata(&path).unwrap().len();
        OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(current - 2)
            .unwrap();

        {
            let mut vs = VersionSet::open(dir.path(), &sst_dir).unwrap();
            assert_eq!(vs.current().last_seq, 7);
            vs.apply(&[VersionEdit::SetLastSeq(99)]).unwrap();
        }

        let vs = VersionSet::open(dir.path(), &sst_dir).unwrap();
        assert_eq!(vs.current().last_seq, 99);
    }

    #[test]
    fn open_truncates_corrupt_manifest_tail_before_append() {
        let dir = TempDir::new().unwrap();
        let sst_dir = dir.path().join("sst");
        std::fs::create_dir_all(&sst_dir).unwrap();

        {
            let mut vs = VersionSet::open(dir.path(), &sst_dir).unwrap();
            vs.apply(&[VersionEdit::SetLastSeq(7)]).unwrap();
            vs.apply(&[VersionEdit::SetLastSeq(11)]).unwrap();
        }

        let path = dir.path().join("MANIFEST");
        let checksum_offset = second_record_checksum_offset(&path);
        let mut data = std::fs::read(&path).unwrap();
        data[checksum_offset] ^= 0xFF;
        std::fs::write(&path, data).unwrap();

        {
            let mut vs = VersionSet::open(dir.path(), &sst_dir).unwrap();
            assert_eq!(vs.current().last_seq, 7);
            vs.apply(&[VersionEdit::SetLastSeq(99)]).unwrap();
        }

        let vs = VersionSet::open(dir.path(), &sst_dir).unwrap();
        assert_eq!(vs.current().last_seq, 99);
    }

    #[test]
    fn compact_manifest_rewrites_to_canonical_form() {
        // Apply many edits, then compact. The resulting manifest
        // should replay to the same version.
        let dir = TempDir::new().unwrap();
        let sst_dir = dir.path().join("sst");
        std::fs::create_dir_all(&sst_dir).unwrap();

        let file1 = make_live_sst(&sst_dir, 1, b"a", b"c");
        let file2 = make_live_sst(&sst_dir, 2, b"d", b"f");

        let mut vs = VersionSet::open(dir.path(), &sst_dir).unwrap();
        vs.apply(&[VersionEdit::Reset {
            next_file_id: 7,
            min_wal_id: 7,
        }])
        .unwrap();
        vs.apply(&[VersionEdit::AddFile {
            level: 0,
            file: Arc::clone(&file1),
        }])
        .unwrap();
        vs.apply(&[VersionEdit::AddFile {
            level: 1,
            file: Arc::clone(&file2),
        }])
        .unwrap();
        vs.apply(&[VersionEdit::SetLastSeq(500), VersionEdit::SetNextFileId(99)])
            .unwrap();

        let pre_size = std::fs::metadata(dir.path().join("MANIFEST"))
            .unwrap()
            .len();
        vs.compact_manifest().unwrap();
        let post_size = std::fs::metadata(dir.path().join("MANIFEST"))
            .unwrap()
            .len();
        // Compaction produces a single snapshot, so it is typically
        // not larger than the history it replaced.
        assert!(post_size <= pre_size + 64);

        drop(vs);
        let vs = VersionSet::open(dir.path(), &sst_dir).unwrap();
        let v = vs.current();
        assert_eq!(v.levels[0].len(), 1);
        assert_eq!(v.levels[1].len(), 1);
        assert_eq!(v.last_seq, 500);
        assert_eq!(v.next_file_id, 99);
        assert_eq!(v.min_wal_id, 7);
    }
}
