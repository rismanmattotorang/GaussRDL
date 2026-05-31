use std::path::PathBuf;
use std::fmt::Debug;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use polars::prelude::*;

/// Represents a table in a relational database
#[derive(Debug, Clone)]
pub struct Table {
    pub data: DataFrame,
}

impl Table {
    pub fn new(data: DataFrame) -> Self {
        Self { data }
    }
}

/// Foreign key relationship between tables
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForeignKeyRelation {
    /// Source table name
    pub source_table: String,
    /// Source column name
    pub source_column: String,
    /// Target table name  
    pub target_table: String,
    /// Target column name
    pub target_column: String,
}

/// Column data types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ColumnType {
    /// Integer type
    Integer,
    /// Float type
    Float,
    /// String type
    String,
    /// Boolean type
    Boolean,
    /// Timestamp type
    Timestamp,
    /// JSON type
    Json,
}

/// Represents a relational database
#[derive(Debug, Clone)]
pub struct Database {
    /// Database name
    pub name: String,
    /// Tables in the database
    pub tables: Vec<Table>,
    /// Database metadata
    pub metadata: DatabaseMetadata,
}

/// Database metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseMetadata {
    /// Database version
    pub version: String,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last modified timestamp
    pub modified_at: DateTime<Utc>,
    /// Description
    pub description: String,
    /// Source URL
    pub source_url: Option<String>,
    /// License
    pub license: Option<String>,
}

/// Dataset configuration
#[derive(Debug, Clone)]
pub struct DatasetConfig {
    pub cache_dir: Option<PathBuf>,
    pub download_dir: Option<PathBuf>,
    pub force_download: bool,
}

impl Default for DatasetConfig {
    fn default() -> Self {
        Self {
            cache_dir: None,
            download_dir: None,
            force_download: false,
        }
    }
}

/// Task types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskType {
    /// Binary classification
    BinaryClassification,
    /// Multi-class classification
    MultiClassification,
    /// Regression
    Regression,
    /// Ranking
    Ranking,
}

/// Task metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskMetadata {
    /// Task name
    pub name: String,
    /// Description
    pub description: String,
    /// Metrics
    pub metrics: Vec<String>,
    /// Paper reference
    pub paper: Option<String>,
}

/// Cache configuration
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Cache directory
    pub cache_dir: PathBuf,
    /// Whether to verify downloads
    pub verify_downloads: bool,
    /// Whether to use cache
    pub use_cache: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            cache_dir: dirs::cache_dir()
                .unwrap_or_else(|| PathBuf::from("/tmp"))
                .join("gaussrdl"),
            verify_downloads: true,
            use_cache: true,
        }
    }
}

/// Download configuration
#[derive(Debug, Clone)]
pub struct DownloadConfig {
    /// Whether to download data
    pub download: bool,
    /// Whether to force download
    pub force_download: bool,
    /// Download timeout in seconds
    pub timeout: u64,
}

impl Default for DownloadConfig {
    fn default() -> Self {
        Self {
            download: true,
            force_download: false,
            timeout: 300,
        }
    }
} 