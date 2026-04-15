/// Errors returned by lark operations.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// An underlying I/O error from the filesystem layer.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// The engine refused to block the caller and returned early.
    /// Today this is reserved for future use by `WriteOptions::no_slowdown`
    /// once write-stall / rate-limiter plumbing lands — the flag is
    /// accepted but the engine never actually stalls, so this variant
    /// is currently unreachable via the public API.
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
