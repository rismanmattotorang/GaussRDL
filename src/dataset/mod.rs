use std::path::PathBuf;
use std::fs;
use std::io::Write;
use chrono::{DateTime, Utc};
use polars::prelude::*;
use reqwest;
use serde::{Serialize, Deserialize};
use crate::error::{Result, Error};
use crate::core::{Dataset, Table};

/// Dataset metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetMetadata {
    /// Dataset name
    pub name: String,
    
    /// Dataset version
    pub version: String,
    
    /// Dataset description
    pub description: String,
    
    /// Dataset URL
    pub url: String,
    
    /// Dataset size in bytes
    pub size: u64,
    
    /// Dataset hash
    pub hash: String,
    
    /// Dataset tables
    pub tables: Vec<TableMetadata>,
    
    /// Dataset validation timestamp
    pub val_timestamp: DateTime<Utc>,
    
    /// Dataset test timestamp
    pub test_timestamp: DateTime<Utc>,
}

/// Table metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableMetadata {
    /// Table name
    pub name: String,
    
    /// Table schema
    pub schema: Schema,
    
    /// Primary key column
    pub primary_key: String,
    
    /// Foreign key columns and their referenced tables
    pub foreign_keys: std::collections::HashMap<String, String>,
    
    /// Timestamp column if any
    pub timestamp_column: Option<String>,
}

/// Dataset cache manager
pub struct DatasetCache {
    /// Cache directory
    cache_dir: PathBuf,
}

impl DatasetCache {
    /// Create a new dataset cache
    pub fn new(cache_dir: PathBuf) -> Result<Self> {
        fs::create_dir_all(&cache_dir)?;
        Ok(Self { cache_dir })
    }
    
    /// Get dataset cache path
    pub fn get_dataset_path(&self, name: &str, version: &str) -> PathBuf {
        self.cache_dir.join(format!("{}_{}", name, version))
    }
    
    /// Check if dataset is cached
    pub fn is_cached(&self, name: &str, version: &str) -> bool {
        self.get_dataset_path(name, version).exists()
    }
    
    /// Get dataset metadata
    pub fn get_metadata(&self, name: &str, version: &str) -> Result<DatasetMetadata> {
        let path = self.get_dataset_path(name, version).join("metadata.json");
        let metadata = fs::read_to_string(path)?;
        Ok(serde_json::from_str(&metadata)?)
    }
    
    /// Save dataset metadata
    pub fn save_metadata(&self, metadata: &DatasetMetadata) -> Result<()> {
        let path = self.get_dataset_path(&metadata.name, &metadata.version);
        fs::create_dir_all(&path)?;
        
        let metadata_path = path.join("metadata.json");
        let mut file = fs::File::create(metadata_path)?;
        file.write_all(serde_json::to_string_pretty(metadata)?.as_bytes())?;
        
        Ok(())
    }
    
    /// Download dataset
    pub fn download(&self, metadata: &DatasetMetadata) -> Result<()> {
        let path = self.get_dataset_path(&metadata.name, &metadata.version);
        fs::create_dir_all(&path)?;
        
        let client = reqwest::blocking::Client::new();
        let mut response = client.get(&metadata.url)
            .send()
            .map_err(|e| Error::network(e.to_string()))?;
        
        let file_path = path.join("data.zip");
        let mut file = fs::File::create(file_path)?;
        response.copy_to(&mut file)
            .map_err(|e| Error::network(e.to_string()))?;
        
        Ok(())
    }
    
    /// Load dataset table
    pub fn load_table(&self, metadata: &DatasetMetadata, table_name: &str) -> Result<Table> {
        let path = self.get_dataset_path(&metadata.name, &metadata.version);
        let table_path = path.join(format!("{}.parquet", table_name));
        
        let table_metadata = metadata.tables.iter()
            .find(|t| t.name == table_name)
            .ok_or_else(|| Error::dataset(format!("Table {} not found", table_name)))?;
        
        let df = LazyFrame::scan_parquet(&table_path.to_string_lossy(), ScanArgsParquet::default())?;
        
        Ok(Table::new(
            table_name.to_string(),
            df,
            table_metadata.primary_key.clone(),
            table_metadata.foreign_keys.clone(),
            table_metadata.timestamp_column.clone(),
        ))
    }
    
    /// Save dataset table
    pub fn save_table(&self, metadata: &DatasetMetadata, table: &Table) -> Result<()> {
        let path = self.get_dataset_path(&metadata.name, &metadata.version);
        let table_path = path.join(format!("{}.parquet", table.name));
        
        table.data.clone().collect()?.write_parquet(
            &table_path.to_string_lossy(),
            ParquetWriteOptions::default(),
        )?;
        
        Ok(())
    }
}

/// Dataset registry
pub struct DatasetRegistry {
    /// Dataset cache
    cache: DatasetCache,
    
    /// Available datasets
    datasets: std::collections::HashMap<String, DatasetMetadata>,
}

impl DatasetRegistry {
    /// Create a new dataset registry
    pub fn new(cache_dir: PathBuf) -> Result<Self> {
        let cache = DatasetCache::new(cache_dir)?;
        Ok(Self {
            cache,
            datasets: std::collections::HashMap::new(),
        })
    }
    
    /// Register a dataset
    pub fn register(&mut self, metadata: DatasetMetadata) {
        self.datasets.insert(metadata.name.clone(), metadata);
    }
    
    /// Get dataset metadata
    pub fn get_metadata(&self, name: &str) -> Option<&DatasetMetadata> {
        self.datasets.get(name)
    }
    
    /// List available datasets
    pub fn list_datasets(&self) -> Vec<&str> {
        self.datasets.keys().map(|s| s.as_str()).collect()
    }
    
    /// Get dataset
    pub fn get_dataset(&self, name: &str) -> Result<Box<dyn Dataset>> {
        let metadata = self.get_metadata(name)
            .ok_or_else(|| Error::dataset(format!("Dataset {} not found", name)))?;
        
        if !self.cache.is_cached(&metadata.name, &metadata.version) {
            self.cache.download(metadata)?;
        }
        
        Ok(Box::new(CachedDataset {
            metadata: metadata.clone(),
            cache: self.cache.clone(),
        }))
    }
}

/// Cached dataset implementation
#[derive(Clone)]
pub struct CachedDataset {
    /// Dataset metadata
    metadata: DatasetMetadata,
    
    /// Dataset cache
    cache: DatasetCache,
}

impl Dataset for CachedDataset {
    fn val_timestamp(&self) -> DateTime<Utc> {
        self.metadata.val_timestamp
    }
    
    fn test_timestamp(&self) -> DateTime<Utc> {
        self.metadata.test_timestamp
    }
    
    fn name(&self) -> &str {
        &self.metadata.name
    }
    
    fn version(&self) -> &str {
        &self.metadata.version
    }
    
    fn description(&self) -> &str {
        &self.metadata.description
    }
    
    fn tables(&self) -> Result<Vec<Table>> {
        let mut tables = Vec::new();
        for table_metadata in &self.metadata.tables {
            let table = self.cache.load_table(&self.metadata, &table_metadata.name)?;
            tables.push(table);
        }
        Ok(tables)
    }
    
    fn cache_dir(&self) -> Option<PathBuf> {
        Some(self.cache.get_dataset_path(&self.metadata.name, &self.metadata.version))
    }
    
    fn set_cache_dir(&mut self, _dir: Option<PathBuf>) {
        // Cache directory is managed by DatasetCache
    }
    
    fn download(&self) -> Result<()> {
        self.cache.download(&self.metadata)
    }
    
    fn load(&self) -> Result<()> {
        // Tables are loaded on demand
        Ok(())
    }
    
    fn preprocess(&mut self) -> Result<()> {
        // No preprocessing needed for cached dataset
        Ok(())
    }
    
    fn validate(&self) -> Result<()> {
        // Validate all tables
        for table_metadata in &self.metadata.tables {
            let table = self.cache.load_table(&self.metadata, &table_metadata.name)?;
            
            // Check schema
            let schema = table.data.clone().schema()?;
            if schema != table_metadata.schema {
                return Err(Error::validation(format!(
                    "Schema mismatch for table {}: expected {:?}, got {:?}",
                    table_metadata.name, table_metadata.schema, schema
                )));
            }
            
            // Check primary key
            if !schema.get_field(&table_metadata.primary_key).is_some() {
                return Err(Error::validation(format!(
                    "Primary key {} not found in table {}",
                    table_metadata.primary_key, table_metadata.name
                )));
            }
            
            // Check foreign keys
            for (column, _) in &table_metadata.foreign_keys {
                if !schema.get_field(column).is_some() {
                    return Err(Error::validation(format!(
                        "Foreign key {} not found in table {}",
                        column, table_metadata.name
                    )));
                }
            }
            
            // Check timestamp column
            if let Some(column) = &table_metadata.timestamp_column {
                if !schema.get_field(column).is_some() {
                    return Err(Error::validation(format!(
                        "Timestamp column {} not found in table {}",
                        column, table_metadata.name
                    )));
                }
            }
        }
        
        Ok(())
    }
} 