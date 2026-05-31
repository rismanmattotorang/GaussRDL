//! ML models and neural networks for GaussRDL

// Import modules from hyphenated directories
pub mod gnn_models;
pub mod gt_model;

// Re-export main types for easier access
pub use gnn_models::*;
pub use gt_model::*;

// Error types
#[derive(thiserror::Error, Debug)]
pub enum ModelError {
    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("Invalid configuration: {0}")]
    Config(String),
    #[error("Model error: {0}")]
    Model(String),
}

impl From<gaussrdl_core::Error> for ModelError {
    fn from(e: gaussrdl_core::Error) -> Self {
        ModelError::Model(e.to_string())
    }
}

impl From<toml::de::Error> for ModelError {
    fn from(e: toml::de::Error) -> Self {
        ModelError::Model(e.to_string())
    }
}

impl From<toml::ser::Error> for ModelError {
    fn from(e: toml::ser::Error) -> Self {
        ModelError::Model(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, ModelError>;
