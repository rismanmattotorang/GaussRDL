//! Data loading, datasets, and data processing for GaussRDL

pub mod datasets;
pub mod data;
pub mod tasks;
pub mod task;
pub mod dataset;
pub mod metrics;

// Re-export core types for convenience
pub use gaussrdl_core::{Table, DatasetConfig, CacheConfig, DownloadConfig, Error, Result};

// Re-export main functionality
pub use datasets::{get_dataset, DatasetRegistry};
pub use tasks::{get_task, TaskRegistry};
