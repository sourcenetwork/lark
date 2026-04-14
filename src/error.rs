/// Errors returned by lark operations.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// An underlying I/O error from the filesystem layer.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}
