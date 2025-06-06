use std::path::PathBuf;
use chrono::{DateTime, Utc};
use polars::prelude::*;
use serde::{Serialize, Deserialize};
use crate::error::Result;

/// Core dataset trait
pub trait Dataset: Send + Sync {
    /// Get validation timestamp
    fn val_timestamp(&self) -> DateTime<Utc>;
    
    /// Get test timestamp
    fn test_timestamp(&self) -> DateTime<Utc>;
    
    /// Get dataset name
    fn name(&self) -> &str;
    
    /// Get dataset version
    fn version(&self) -> &str;
    
    /// Get dataset description
    fn description(&self) -> &str;
    
    /// Get dataset tables
    fn tables(&self) -> Result<Vec<Table>>;
    
    /// Get dataset cache directory
    fn cache_dir(&self) -> Option<PathBuf>;
    
    /// Set dataset cache directory
    fn set_cache_dir(&mut self, dir: Option<PathBuf>);
    
    /// Download dataset if needed
    fn download(&self) -> Result<()>;
    
    /// Load dataset into memory
    fn load(&self) -> Result<()>;
    
    /// Preprocess dataset
    fn preprocess(&mut self) -> Result<()>;
    
    /// Validate dataset
    fn validate(&self) -> Result<()>;
}

/// Core table trait
#[derive(Debug, Clone)]
pub struct Table {
    /// Table name
    pub name: String,
    
    /// Table data
    pub data: LazyFrame,
    
    /// Primary key column
    pub primary_key: String,
    
    /// Foreign key columns and their referenced tables
    pub foreign_keys: HashMap<String, String>,
    
    /// Timestamp column if any
    pub timestamp_column: Option<String>,
}

impl Table {
    /// Create a new table
    pub fn new(
        name: String,
        data: LazyFrame,
        primary_key: String,
        foreign_keys: HashMap<String, String>,
        timestamp_column: Option<String>,
    ) -> Self {
        Self {
            name,
            data,
            primary_key,
            foreign_keys,
            timestamp_column,
        }
    }
    
    /// Get number of rows
    pub fn len(&self) -> Result<usize> {
        Ok(self.data.clone().collect()?.height())
    }
    
    /// Check if table is empty
    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }
    
    /// Get column names
    pub fn column_names(&self) -> Result<Vec<String>> {
        Ok(self.data.clone().schema()?.iter_names().map(|s| s.to_string()).collect())
    }
    
    /// Filter table by timestamp
    pub fn filter_by_timestamp(&self, timestamp: DateTime<Utc>) -> Result<Self> {
        if let Some(ts_col) = &self.timestamp_column {
            let filtered = self.data.clone()
                .filter(col(ts_col).lt(lit(timestamp.timestamp())))?;
            
            Ok(Self::new(
                self.name.clone(),
                filtered,
                self.primary_key.clone(),
                self.foreign_keys.clone(),
                self.timestamp_column.clone(),
            ))
        } else {
            Ok(self.clone())
        }
    }
}

/// Core task trait
pub trait Task: Send + Sync {
    /// Get task name
    fn name(&self) -> &str;
    
    /// Get task description 
    fn description(&self) -> &str;
    
    /// Get task type
    fn task_type(&self) -> TaskType;
    
    /// Get task dataset
    fn dataset(&self) -> &dyn Dataset;
    
    /// Get task metrics
    fn metrics(&self) -> Vec<Box<dyn Metric>>;
    
    /// Get training data
    fn get_train_data(&self) -> Result<Table>;
    
    /// Get validation data
    fn get_val_data(&self) -> Result<Table>;
    
    /// Get test data
    fn get_test_data(&self) -> Result<Table>;
    
    /// Evaluate predictions
    fn evaluate(&self, predictions: &[f64], targets: &[f64]) -> Result<Vec<f64>>;
}

/// Task types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskType {
    /// Binary classification
    Binary,
    /// Multi-class classification
    MultiClass,
    /// Regression
    Regression,
    /// Ranking
    Ranking,
}

/// Core metric trait
pub trait Metric: Send + Sync {
    /// Get metric name
    fn name(&self) -> &str;
    
    /// Get metric description
    fn description(&self) -> &str;
    
    /// Compute metric
    fn compute(&self, predictions: &[f64], targets: &[f64]) -> Result<f64>;
}

/// Core model trait
pub trait Model: Send + Sync {
    /// Get model name
    fn name(&self) -> &str;
    
    /// Get model version
    fn version(&self) -> &str;
    
    /// Get model description
    fn description(&self) -> &str;
    
    /// Get model configuration
    fn config(&self) -> &ModelConfig;
    
    /// Train model
    fn train(&mut self, train_data: &Table, val_data: &Table) -> Result<()>;
    
    /// Make predictions
    fn predict(&self, data: &Table) -> Result<Vec<f64>>;
    
    /// Save model
    fn save(&self, path: &Path) -> Result<()>;
    
    /// Load model
    fn load(path: &Path) -> Result<Self> where Self: Sized;
}

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model name
    pub name: String,
    
    /// Model version
    pub version: String,
    
    /// Model architecture
    pub architecture: String,
    
    /// Hidden dimensions
    pub hidden_dims: Vec<usize>,
    
    /// Dropout rate
    pub dropout: f32,
    
    /// Learning rate
    pub learning_rate: f32,
    
    /// Batch size
    pub batch_size: usize,
    
    /// Number of epochs
    pub epochs: usize,
    
    /// Use GPU if available
    pub use_gpu: bool,
    
    /// Random seed
    pub seed: Option<u64>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            version: "1.0.0".to_string(),
            architecture: "mlp".to_string(),
            hidden_dims: vec![256, 128, 64],
            dropout: 0.1,
            learning_rate: 0.001,
            batch_size: 32,
            epochs: 100,
            use_gpu: false,
            seed: None,
        }
    }
} 