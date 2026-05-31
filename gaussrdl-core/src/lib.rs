//! # GaussRDL Core Library
//! 
//! This crate provides the foundational types, traits, and utilities for the GaussRDL
//! (Gaussian Relational Deep Learning) ecosystem. It serves as the core foundation
//! for all other GaussRDL crates.
//! 
//! ## Features
//! 
//! - **Comprehensive Error Handling**: Robust error types with detailed context
//! - **Type-Safe Data Structures**: Strongly typed tables, datasets, and tasks
//! - **Async Support**: Full async/await support for I/O operations
//! - **Serialization**: Built-in serde support for all core types
//! - **Validation**: Comprehensive validation utilities
//! - **Configuration**: Flexible configuration management
//! - **Metrics**: Built-in evaluation metrics
//! 
//! ## Quick Start
//! 
//! ```rust
//! use gaussrdl_core::{
//!     Dataset, Task, Table, DatasetConfig,
//!     Error, Result, ValidationError,
//!     types::{ColumnType, TaskType}
//! };
//! 
//! // Create a dataset configuration
//! let config = DatasetConfig::builder()
//!     .cache_dir("/tmp/gaussrdl")
//!     .download_dir("/tmp/downloads")
//!     .force_download(false)
//!     .build()?;
//! 
//! // Load and validate a dataset
//! let mut dataset = MyDataset::new();
//! dataset.load(&config)?;
//! dataset.validate()?;
//! ```
//! 
//! ## Architecture
//! 
//! The core library is organized into several key modules:
//! 
//! - **error**: Comprehensive error handling with context
//! - **types**: Core data structures and types
//! - **traits**: Abstract interfaces for datasets and tasks
//! - **validation**: Data validation utilities
//! - **config**: Configuration management
//! - **metrics**: Evaluation metrics and utilities
//! - **utils**: Common utilities and helpers

#![warn(missing_docs)]
#![warn(rustdoc::missing_crate_level_docs)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::missing_errors_doc)]

use crate::config::ConfigError;

pub mod error;
pub mod types;
pub mod traits;
pub mod validation;
pub mod config;
pub mod metrics;

// Re-export commonly used types from error module
pub use error::{Error, Result, ValidationError};

// Re-export commonly used types from types module
// Only re-export types that exist in types.rs
pub use types::{
    Table, Database, DatabaseMetadata, ForeignKeyRelation, ColumnType,
    DatasetConfig, TaskType, TaskMetadata, CacheConfig, DownloadConfig,
};

// Re-export commonly used types from traits module
pub use traits::{Dataset, Task};

// Re-export commonly used types from validation module
pub use validation::{Validator, ValidationRule, ValidationContext};

// Re-export commonly used types from config module
pub use config::{Config, ConfigBuilder};

// Re-export commonly used types from metrics module
pub use metrics::{Metric, MetricRegistry, MetricValue};

// Re-export commonly used external types
pub use serde;
pub use async_trait::async_trait;
pub use std::sync::Arc;
pub use std::collections::HashMap;
pub use std::path::{Path, PathBuf};
pub use polars::prelude::*;
pub use chrono::{DateTime, Utc};
pub use half::{f16, bf16};
pub use uuid::Uuid;

// Type aliases for common types
// Removed Float, Double, Int, Long, FloatVec, FloatMatrix, IntVec, IntMatrix, Timestamp, Duration, Interval
// as these are not defined in types.rs and cause conflicts

/// Result type for validation operations
pub type ValidationResult<T> = std::result::Result<T, ValidationError>;

/// Result type for configuration operations
pub type ConfigResult<T> = std::result::Result<T, ConfigError>;

/// Result type for metric operations
pub type MetricResult<T> = std::result::Result<T, Error>;

// Re-export version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const AUTHORS: &str = env!("CARGO_PKG_AUTHORS");
pub const DESCRIPTION: &str = env!("CARGO_PKG_DESCRIPTION");

/// Library version information
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VersionInfo {
    pub version: String,
    pub authors: String,
    pub description: String,
    pub build_timestamp: DateTime<Utc>,
}

impl Default for VersionInfo {
    fn default() -> Self {
        Self {
            version: VERSION.to_string(),
            authors: AUTHORS.to_string(),
            description: DESCRIPTION.to_string(),
            build_timestamp: Utc::now(),
        }
    }
}

/// Get library version information
pub fn version_info() -> VersionInfo {
    VersionInfo::default()
}

/// Initialize the core library
/// 
/// This function should be called once at the start of your application
/// to initialize any global state, logging, or configuration.
pub fn init() -> Result<()> {
    // Initialize logging if not already initialized
    if std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", "info");
    }
    
    // Initialize any other global state here
    Ok(())
}

/// Cleanup the core library
/// 
/// This function should be called when shutting down your application
/// to clean up any global state or resources.
pub fn cleanup() -> Result<()> {
    // Cleanup any global state here
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_info() {
        let info = version_info();
        assert!(!info.version.is_empty());
        assert!(!info.authors.is_empty());
        assert!(!info.description.is_empty());
    }

    #[test]
    fn test_init_cleanup() {
        assert!(init().is_ok());
        assert!(cleanup().is_ok());
    }
} 