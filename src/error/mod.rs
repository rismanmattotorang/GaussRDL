// src/error/mod.rs
use thiserror::Error;

pub type Result<T> = std::result::Result<T, GaussRelgtError>;

#[derive(Error, Debug)]
pub enum GaussRelgtError {
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),
    
    #[error("SurrealDB error: {0}")]
    SurrealDb(#[from] surrealdb::Error),
    
    #[error("Candle ML error: {0}")]
    Candle(#[from] candle_core::Error),
    
    #[error("Serialization error: {0}")]
    Serde(#[from] serde_json::Error),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("TOML parsing error: {0}")]
    Toml(#[from] toml::de::Error),
    
    #[error("Configuration error: {0}")]
    Config(#[from] config::ConfigError),
    
    #[error("Model error: {message}")]
    Model { message: String },
    
    #[error("Training error: {message}")]
    Training { message: String },
    
    #[error("Data processing error: {message}")]
    DataProcessing { message: String },
    
    #[error("Schema validation error: {message}")]
    SchemaValidation { message: String },
    
    #[error("Graph construction error: {message}")]
    GraphConstruction { message: String },
    
    #[error("Temporal constraint violation: {message}")]
    TemporalViolation { message: String },
    
    #[error("Resource exhausted: {resource}")]
    ResourceExhausted { resource: String },
    
    #[error("Invalid configuration: {field} - {reason}")]
    InvalidConfig { field: String, reason: String },
    
    #[error("Not implemented: {feature}")]
    NotImplemented { feature: String },
}

impl GaussRelgtError {
    pub fn model_error(message: impl Into<String>) -> Self {
        Self::Model { message: message.into() }
    }
    
    pub fn training_error(message: impl Into<String>) -> Self {
        Self::Training { message: message.into() }
    }
    
    pub fn data_error(message: impl Into<String>) -> Self {
        Self::DataProcessing { message: message.into() }
    }
    
    pub fn schema_error(message: impl Into<String>) -> Self {
        Self::SchemaValidation { message: message.into() }
    }
    
    pub fn graph_error(message: impl Into<String>) -> Self {
        Self::GraphConstruction { message: message.into() }
    }
    
    pub fn temporal_error(message: impl Into<String>) -> Self {
        Self::TemporalViolation { message: message.into() }
    }
}