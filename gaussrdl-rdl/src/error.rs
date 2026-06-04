//! Error types for the GaussRDL v2 engine.

use thiserror::Error;

/// Errors that can occur across the RDL pipeline.
#[derive(Error, Debug)]
pub enum RdlError {
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),

    #[error("data error: {0}")]
    Data(String),

    #[error("schema error: {0}")]
    Schema(String),

    #[error("config error: {0}")]
    Config(String),

    #[error("model error: {0}")]
    Model(String),
}

/// Convenience result alias used throughout the crate.
pub type Result<T> = std::result::Result<T, RdlError>;
