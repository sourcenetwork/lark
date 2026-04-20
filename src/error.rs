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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn io_error_converts_via_from() {
        let ioe = std::io::Error::new(std::io::ErrorKind::NotFound, "nope");
        let e: Error = ioe.into();
        assert!(matches!(e, Error::Io(_)));
    }

    #[test]
    fn busy_display_contains_reason() {
        let e = Error::Busy("too many L0 files");
        let msg = format!("{e}");
        assert!(msg.contains("too many L0 files"));
        assert!(msg.contains("busy"));
    }

    #[test]
    fn merge_failed_display_contains_key_bytes() {
        let e = Error::MergeFailed(b"k".to_vec());
        let msg = format!("{e}");
        assert!(msg.contains("merge"));
        // Debug-printed key bytes show up as `[107]`.
        assert!(msg.contains("107"));
    }
}
