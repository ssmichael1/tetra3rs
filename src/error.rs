//! Error type for tetra3.

use thiserror::Error;

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;

/// All errors tetra3 can produce on its public API surface.
#[derive(Debug, Error)]
pub enum Error {
    /// File I/O failure (opening or reading a database/catalog file).
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// `postcard` serialization or deserialization failed.
    #[error("postcard error: {0}")]
    Postcard(#[from] postcard::Error),

    /// Catalog file failed to load (wrong magic, unsupported version, etc.).
    #[error("invalid catalog: {0}")]
    InvalidCatalog(String),

    /// Caller-supplied input failed validation.
    #[error("invalid input: {0}")]
    InvalidInput(String),
}
