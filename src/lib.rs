// src/lib.rs
//! RelBench: A Rust implementation of the Relational Deep Learning Benchmark
//! 
//! RelBench is a benchmark designed to facilitate efficient, robust and reproducible research
//! on end-to-end deep learning over relational databases. This crate provides a Rust implementation
//! of the benchmark with enhanced performance and safety features.
//! 
//! # Features
//! 
//! - Comprehensive dataset management
//! - Task specification and evaluation
//! - Graph neural network models
//! - Efficient data loading and caching
//! - Robust error handling
//! - Extensive test coverage
//! 
//! # Example
//! 
//! ```rust,no_run
//! use relbench::{get_dataset, get_task};
//! 
//! #[tokio::main]
//! async fn main() -> relbench::Result<()> {
//!     // Load dataset
//!     let dataset = get_dataset("rel-amazon", true)?;
//!     
//!     // Get task
//!     let task = get_task("rel-amazon", "user-churn", true)?;
//!     
//!     // Get data tables
//!     let train_table = task.get_train_table()?;
//!     let val_table = task.get_val_table()?;
//!     let test_table = task.get_test_table(true)?;
//!     
//!     // Train model and make predictions
//!     let predictions = vec![0.5; test_table.len()];
//!     
//!     // Evaluate predictions
//!     let metrics = task.evaluate(&predictions, None)?;
//!     println!("Metrics: {:?}", metrics);
//!     
//!     Ok(())
//! }
//! ```

// Re-exports
pub use base::{Dataset, Database, Table, Task, EntityTask};
pub use datasets::{get_dataset, DatasetRegistry};
pub use tasks::{get_task, TaskRegistry};
pub use metrics::Metric;
pub use error::{Error, Result};

// Modules
pub mod base;
pub mod datasets;
pub mod tasks;
pub mod metrics;
pub mod utils;
pub mod error;

// Tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_loading() {
        let dataset = get_dataset("rel-amazon", false);
        assert!(dataset.is_ok());
    }

    #[test]
    fn test_task_loading() {
        let task = get_task("rel-amazon", "user-churn", false);
        assert!(task.is_ok());
    }
}