use std::path::Path;
use chrono::{DateTime, Utc};
use crate::error::Result;
use crate::types::{Table, TaskType, DatasetConfig};

/// Dataset trait
pub trait Dataset: Send + Sync + std::fmt::Debug {
    /// Get dataset name
    fn name(&self) -> &str;
    
    /// Get dataset description
    fn description(&self) -> &str;
    
    /// Get dataset version
    fn version(&self) -> &str;
    
    /// Get dataset tables
    fn tables(&self) -> &std::collections::HashMap<String, polars::prelude::DataFrame>;
    
    /// Get validation timestamp
    fn val_timestamp(&self) -> DateTime<Utc>;
    
    /// Get test timestamp
    fn test_timestamp(&self) -> DateTime<Utc>;
    
    /// Load dataset
    fn load(&mut self, config: &DatasetConfig) -> Result<()>;
    
    /// Save dataset
    fn save(&self, path: &Path) -> Result<()>;
    
    /// Load dataset from path
    fn load_from_path(&mut self, path: &Path) -> Result<()>;
    
    /// Validate dataset
    fn validate(&self) -> Result<()>;
}

/// Task trait
pub trait Task: Send + Sync + std::fmt::Debug {
    /// Get task type
    fn task_type(&self) -> TaskType;
    
    /// Get train table
    fn get_train_table(&self) -> Result<Table>;
    
    /// Get validation table
    fn get_val_table(&self) -> Result<Table>;
    
    /// Get test table
    fn get_test_table(&self, _mask_labels: bool) -> Result<Table>;
    
    /// Evaluate predictions
    fn evaluate(&self, _predictions: &[f32], _table: Option<&Table>) -> Result<std::collections::HashMap<String, f32>>;
} 