//! Lifecycle event callbacks for flush, compaction, and ingest.
//!
//! Callers register one or more [`EventListener`] implementations via
//! [`crate::Options::listeners`] to react to engine lifecycle events.
//! Typical uses: metrics pipelines (emit a span per flush), debugging
//! (log which files compaction picked), test harnesses (wait on a
//! callback instead of sleeping), and operational triggers (upload
//! the freshly-flushed SSTable to object storage).
//!
//! # Dispatch
//!
//! Events are dispatched **synchronously** on the thread that
//! triggered them — flush events on the write thread, compaction
//! events on the compaction thread, ingest events on the ingest
//! caller's thread. Listeners **MUST NOT block** or re-enter the
//! database. The contract is "do a cheap thing or spawn a task."
//! Blocking inside a listener stalls the engine and will starve
//! background compaction / flush pipelines.
//!
//! # Non-guarantees
//!
//! - Listener order is unspecified. Multiple listeners registered
//!   in the same `Options::listeners` vector see events in vector
//!   order, but a caller that cares about sequencing should own
//!   its own fan-out.
//! - Per-column-family filtering is out of scope. A listener sees
//!   every event from every CF; callers that only care about a
//!   subset should filter in the callback.
//! - `on_wal_full` is declared for API parity with RocksDB but is
//!   never fired in the current implementation — lark rotates the
//!   WAL alongside every memtable, so there's no separate
//!   "WAL-full" condition.

use std::path::PathBuf;
use std::time::Duration;

use crate::Error;

/// Why the flush / compaction path chose to produce a file, used
/// by [`TableFileCreationInfo::reason`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TableFileCreationReason {
    /// A memtable was flushed to L0.
    Flush,
    /// A compaction merged files at one level into another.
    Compaction,
    /// A file was ingested via [`crate::Db::ingest_external_files`].
    Recovery,
}

/// Information about a flush that just completed.
#[derive(Debug, Clone)]
pub struct FlushJobInfo {
    /// Numeric id of the SSTable that now holds the flushed
    /// memtable's contents.
    pub file_id: u64,
    /// Path of the produced SSTable.
    pub file_path: PathBuf,
    /// File size on disk, in bytes.
    pub file_size: u64,
    /// Number of point entries in the flushed SSTable.
    pub num_entries: u64,
    /// Smallest user key in the file.
    pub smallest_key: Vec<u8>,
    /// Largest user key in the file.
    pub largest_key: Vec<u8>,
    /// Wall-clock duration of the flush, from memtable rotation
    /// through manifest apply.
    pub duration: Duration,
}

/// Information about a compaction job, passed to
/// [`EventListener::on_compaction_begin`] and
/// [`EventListener::on_compaction_completed`].
#[derive(Debug, Clone)]
pub struct CompactionJobInfo {
    /// Source level (the level compaction is reading from).
    pub input_level: usize,
    /// Destination level (one deeper than `input_level`).
    pub output_level: usize,
    /// File ids of the inputs at `input_level` (the "L" files).
    pub input_files_input_level: Vec<u64>,
    /// File ids of the inputs at `output_level` that overlap the
    /// input range (the "L+1" files that get merged in).
    pub input_files_output_level: Vec<u64>,
    /// File ids of the newly-produced output files at
    /// `output_level`. Populated on
    /// [`EventListener::on_compaction_completed`]; empty on
    /// [`EventListener::on_compaction_begin`].
    pub output_files: Vec<u64>,
    /// Wall-clock duration of the compaction. Zero on begin.
    pub duration: Duration,
}

/// Information about a freshly-created SSTable file. Fires for
/// both flush output and compaction output, distinguished by
/// [`TableFileCreationReason`].
#[derive(Debug, Clone)]
pub struct TableFileCreationInfo {
    /// Numeric id of the SSTable.
    pub file_id: u64,
    /// Path of the created file.
    pub file_path: PathBuf,
    /// Level the file was placed at.
    pub level: usize,
    /// Reason the file was produced.
    pub reason: TableFileCreationReason,
    /// File size on disk.
    pub file_size: u64,
    /// Number of point entries in the file.
    pub num_entries: u64,
}

/// Information about an SSTable file that was just unlinked from
/// disk. Fires after compaction has committed the version edit
/// that removed the file from the live set and the physical
/// `unlink(2)` has succeeded.
#[derive(Debug, Clone)]
pub struct TableFileDeletionInfo {
    /// Numeric id of the unlinked file.
    pub file_id: u64,
    /// Path of the unlinked file at the time of deletion.
    pub file_path: PathBuf,
}

/// Information about a file ingested via
/// [`crate::Db::ingest_external_files`].
#[derive(Debug, Clone)]
pub struct ExternalFileIngestionInfo {
    /// Path the caller supplied to `ingest_external_files`.
    pub external_file_path: PathBuf,
    /// Internal file id assigned by the engine.
    pub internal_file_id: u64,
    /// Level the ingested file was placed at.
    pub level: usize,
    /// Total entries in the ingested file (point + range deletes).
    pub num_entries: u64,
    /// Size of the re-emitted file on disk.
    pub file_size: u64,
}

/// Information about a full WAL — declared for API parity with
/// RocksDB but not currently fired by lark. The engine rotates the
/// WAL alongside every memtable, so there is no separate
/// "WAL-full" condition.
#[derive(Debug, Clone)]
pub struct WalFullInfo {
    /// Numeric id of the full WAL file.
    pub wal_id: u64,
    /// Size of the full WAL file in bytes.
    pub size: u64,
}

/// Reason passed to [`EventListener::on_background_error`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackgroundErrorReason {
    /// A background flush (memtable → L0) failed.
    Flush,
    /// A background compaction failed.
    Compaction,
    /// A manifest write or manifest compaction failed.
    Manifest,
    /// A WAL append or WAL sync failed.
    WriteAheadLog,
}

/// Trait implemented by callers that want to react to engine
/// lifecycle events. Registered via [`crate::Options::listeners`].
///
/// Every method has a default empty body, so implementations only
/// override the callbacks they care about.
///
/// **Listeners must not block or re-enter the database.** See the
/// module-level docs for the dispatch contract.
pub trait EventListener: Send + Sync + 'static {
    /// Called after a memtable has been flushed to a new L0
    /// SSTable and the manifest edit has been applied.
    fn on_flush_completed(&self, info: &FlushJobInfo) {
        let _ = info;
    }

    /// Called right before a compaction job starts reading its
    /// input files. The `output_files` field is empty at this
    /// point; it's populated on `on_compaction_completed`.
    fn on_compaction_begin(&self, info: &CompactionJobInfo) {
        let _ = info;
    }

    /// Called after a compaction job has written its output files
    /// and applied the manifest edit.
    fn on_compaction_completed(&self, info: &CompactionJobInfo) {
        let _ = info;
    }

    /// Called when a new SSTable file is created. Fires once per
    /// file for both flush and compaction paths, with
    /// `info.reason` distinguishing the caller.
    fn on_table_file_created(&self, info: &TableFileCreationInfo) {
        let _ = info;
    }

    /// Called after an SSTable file has been physically unlinked
    /// from disk (after the manifest edit that removed it from
    /// the live set has been applied).
    fn on_table_file_deleted(&self, info: &TableFileDeletionInfo) {
        let _ = info;
    }

    /// Called per file inside
    /// [`crate::Db::ingest_external_files`], once the ingested
    /// file has been placed at a level and committed to the
    /// manifest.
    fn on_external_file_ingested(&self, info: &ExternalFileIngestionInfo) {
        let _ = info;
    }

    /// Called when a background flush / compaction / manifest /
    /// WAL operation returns an error. The engine keeps running —
    /// the listener is for observability, not error handling.
    fn on_background_error(&self, reason: BackgroundErrorReason, err: &Error) {
        let _ = (reason, err);
    }

    /// Declared for API parity with RocksDB but not currently
    /// fired — see module-level docs.
    fn on_wal_full(&self, info: &WalFullInfo) {
        let _ = info;
    }
}

/// Dispatch a closure over every listener in a slice. Silences
/// the trivial fan-out boilerplate at every call site. `Arc` is
/// cheap enough to clone through the iterator.
pub(crate) fn dispatch<F>(listeners: &[std::sync::Arc<dyn EventListener>], f: F)
where
    F: Fn(&dyn EventListener),
{
    for l in listeners {
        f(l.as_ref());
    }
}
