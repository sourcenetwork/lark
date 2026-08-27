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
//! triggered them - flush events on the write thread, compaction
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
//! - `on_wal_full` is declared so listener implementations can
//!   target a common shape across storage backends, but regolith
//!   itself never fires it - the WAL is rotated alongside every
//!   memtable, so there's no separate "WAL-full" condition.

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
    /// Zero on a platform whose [`crate::env::Env`] has no
    /// monotonic clock, where nothing was measured.
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
    /// Wall-clock duration of the compaction. Zero on begin, and
    /// zero on a platform whose [`crate::env::Env`] has no monotonic
    /// clock, where nothing was measured.
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

/// Information about a full WAL. The struct is declared so that
/// listener implementations can target a common shape across
/// storage backends; regolith itself never fires this callback,
/// because the engine rotates the WAL alongside every memtable
/// and there is no separate "WAL-full" condition.
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
    /// WAL operation returns an error. The engine keeps running -
    /// the listener is for observability, not error handling.
    fn on_background_error(&self, reason: BackgroundErrorReason, err: &Error) {
        let _ = (reason, err);
    }

    /// Declared so listeners can target a common shape across
    /// storage backends; regolith itself never fires this callback.
    /// See the module-level docs.
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Listener that counts every callback it receives. Used to
    /// verify that `dispatch` reaches every registered listener
    /// and that the default no-op implementations don't panic.
    #[derive(Default)]
    struct CountingListener {
        flushes: AtomicUsize,
        compactions: AtomicUsize,
        files_created: AtomicUsize,
        files_deleted: AtomicUsize,
        ingests: AtomicUsize,
        bg_errors: AtomicUsize,
        wal_fulls: AtomicUsize,
    }

    impl EventListener for CountingListener {
        fn on_flush_completed(&self, _: &FlushJobInfo) {
            self.flushes.fetch_add(1, Ordering::Relaxed);
        }
        fn on_compaction_completed(&self, _: &CompactionJobInfo) {
            self.compactions.fetch_add(1, Ordering::Relaxed);
        }
        fn on_table_file_created(&self, _: &TableFileCreationInfo) {
            self.files_created.fetch_add(1, Ordering::Relaxed);
        }
        fn on_table_file_deleted(&self, _: &TableFileDeletionInfo) {
            self.files_deleted.fetch_add(1, Ordering::Relaxed);
        }
        fn on_external_file_ingested(&self, _: &ExternalFileIngestionInfo) {
            self.ingests.fetch_add(1, Ordering::Relaxed);
        }
        fn on_background_error(&self, _: BackgroundErrorReason, _: &Error) {
            self.bg_errors.fetch_add(1, Ordering::Relaxed);
        }
        fn on_wal_full(&self, _: &WalFullInfo) {
            self.wal_fulls.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn sample_flush() -> FlushJobInfo {
        FlushJobInfo {
            file_id: 1,
            file_path: PathBuf::from("/tmp/x.sst"),
            file_size: 1024,
            num_entries: 10,
            smallest_key: b"a".to_vec(),
            largest_key: b"z".to_vec(),
            duration: Duration::from_millis(5),
        }
    }

    fn sample_compaction() -> CompactionJobInfo {
        CompactionJobInfo {
            input_level: 0,
            output_level: 1,
            input_files_input_level: vec![1, 2],
            input_files_output_level: vec![3],
            output_files: vec![4],
            duration: Duration::from_millis(20),
        }
    }

    #[test]
    fn default_trait_impls_are_noop() {
        struct NoOp;
        impl EventListener for NoOp {}
        let n = NoOp;
        n.on_flush_completed(&sample_flush());
        n.on_compaction_begin(&sample_compaction());
        n.on_compaction_completed(&sample_compaction());
        n.on_wal_full(&WalFullInfo { wal_id: 1, size: 0 });
        // No panic reaching here is the assertion.
    }

    #[test]
    fn dispatch_visits_every_listener() {
        let a = Arc::new(CountingListener::default());
        let b = Arc::new(CountingListener::default());
        let listeners: Vec<Arc<dyn EventListener>> = vec![
            a.clone() as Arc<dyn EventListener>,
            b.clone() as Arc<dyn EventListener>,
        ];

        let info = sample_flush();
        dispatch(&listeners, |l| l.on_flush_completed(&info));
        assert_eq!(a.flushes.load(Ordering::Relaxed), 1);
        assert_eq!(b.flushes.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn dispatch_on_empty_list_is_noop() {
        let empty: Vec<Arc<dyn EventListener>> = Vec::new();
        dispatch(&empty, |_| panic!("should not be called"));
    }

    #[test]
    fn table_file_creation_reason_variants_are_equal_only_to_themselves() {
        assert_eq!(
            TableFileCreationReason::Flush,
            TableFileCreationReason::Flush
        );
        assert_ne!(
            TableFileCreationReason::Flush,
            TableFileCreationReason::Compaction
        );
        assert_ne!(
            TableFileCreationReason::Compaction,
            TableFileCreationReason::Recovery
        );
    }

    #[test]
    fn background_error_reason_copy_and_eq() {
        let r = BackgroundErrorReason::Flush;
        let copy = r; // Copy
        assert_eq!(r, copy);
        assert_ne!(r, BackgroundErrorReason::Compaction);
    }

    #[test]
    fn info_structs_are_cloneable_and_debug_formattable() {
        // Sanity check that every public info struct carries
        // Clone + Debug so telemetry callers can snapshot them.
        let f = sample_flush();
        let _ = format!("{f:?}");
        let _ = f.clone();
        let c = sample_compaction();
        let _ = format!("{c:?}");
        let _ = c.clone();
        let ext = ExternalFileIngestionInfo {
            external_file_path: PathBuf::from("/x"),
            internal_file_id: 1,
            level: 0,
            num_entries: 1,
            file_size: 2,
        };
        let _ = format!("{ext:?}");
        let _ = ext.clone();
    }
}
