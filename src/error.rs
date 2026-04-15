/// Errors returned by lark operations.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// An underlying I/O error from the filesystem layer.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// The engine refused to block the caller and returned early.
    /// Returned when [`crate::WriteOptions::no_slowdown`] is set and
    /// the engine is currently stalling writes (too many L0 files,
    /// too many unflushed memtables, or pending compaction bytes
    /// over the hard limit). The included string names the active
    /// stall condition for diagnostics.
    #[error("engine busy: {0}")]
    Busy(&'static str),
    /// The configured [`crate::MergeOperator`] returned `None` when
    /// asked to combine a value with one or more merge operands,
    /// indicating that the operands were corrupt or the merge
    /// semantics failed. The offending user key is included for
    /// diagnostics.
    #[error("merge operator failed for key {0:?}")]
    MergeFailed(Vec<u8>),
}
