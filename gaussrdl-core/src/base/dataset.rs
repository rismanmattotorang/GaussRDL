use std::path::{Path, PathBuf};
use chrono::{DateTime, Utc};
use crate::error::Result;
use polars::prelude::*;
use std::collections::HashMap;

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

/// Dataset trait
pub trait Dataset: Send + Sync {
    /// Get dataset name
    fn name(&self) -> &str;
    
    /// Get dataset description
    fn description(&self) -> &str;
    
    /// Get dataset version
    fn version(&self) -> &str;
    
    /// Get dataset tables
    fn tables(&self) -> &HashMap<String, DataFrame>;
    
    /// Get dataset validation timestamp
    fn val_timestamp(&self) -> DateTime<Utc>;
    
    /// Get dataset test timestamp
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

/// Base dataset implementation
#[derive(Debug)]
pub struct BaseDataset {
    name: String,
    description: String,
    version: String,
    tables: HashMap<String, DataFrame>,
    val_timestamp: DateTime<Utc>,
    test_timestamp: DateTime<Utc>,
    config: DatasetConfig,
}

impl BaseDataset {
    /// Create new base dataset
    pub fn new(
        name: &str,
        description: &str,
        version: &str,
        val_timestamp: DateTime<Utc>,
        test_timestamp: DateTime<Utc>,
    ) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            version: version.to_string(),
            tables: HashMap::new(),
            val_timestamp,
            test_timestamp,
            config: DatasetConfig::default(),
        }
    }
    
    /// Add table to dataset
    pub fn add_table(&mut self, name: &str, table: DataFrame) {
        self.tables.insert(name.to_string(), table);
    }
    
    /// Get table from dataset
    pub fn get_table(&self, name: &str) -> Option<&DataFrame> {
        self.tables.get(name)
    }
    
    /// Set dataset configuration
    pub fn set_config(&mut self, config: DatasetConfig) {
        self.config = config;
    }
}

impl Dataset for BaseDataset {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn description(&self) -> &str {
        &self.description
    }
    
    fn version(&self) -> &str {
        &self.version
    }
    
    fn tables(&self) -> &HashMap<String, DataFrame> {
        &self.tables
    }
    
    fn val_timestamp(&self) -> DateTime<Utc> {
        self.val_timestamp
    }
    
    fn test_timestamp(&self) -> DateTime<Utc> {
        self.test_timestamp
    }
    
    fn load(&mut self, config: &DatasetConfig) -> Result<()> {
        self.config = config.clone();
        Ok(())
    }
    
    fn save(&self, path: &Path) -> Result<()> {
        // Save dataset metadata
        let metadata = serde_json::json!({
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "val_timestamp": self.val_timestamp.to_rfc3339(),
            "test_timestamp": self.test_timestamp.to_rfc3339(),
        });
        
        std::fs::create_dir_all(path)?;
        let metadata_path = path.join("metadata.json");
        std::fs::write(metadata_path, serde_json::to_string_pretty(&metadata)?)?;
        
        // Save tables
        let tables_dir = path.join("tables");
        std::fs::create_dir_all(&tables_dir)?;
        
        for (name, table) in &self.tables {
            let table_path = tables_dir.join(format!("{}.parquet", name));
            let mut file = std::fs::File::create(table_path)?;
            ParquetWriter::new(&mut file).finish(table)?;
        }
        
        Ok(())
    }
    
    fn load_from_path(&mut self, path: &Path) -> Result<()> {
        // Load dataset metadata
        let metadata_path = path.join("metadata.json");
        let metadata: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(metadata_path)?)?;
        
        self.name = metadata["name"].as_str().unwrap().to_string();
        self.description = metadata["description"].as_str().unwrap().to_string();
        self.version = metadata["version"].as_str().unwrap().to_string();
        self.val_timestamp = DateTime::parse_from_rfc3339(metadata["val_timestamp"].as_str().unwrap())?.with_timezone(&Utc);
        self.test_timestamp = DateTime::parse_from_rfc3339(metadata["test_timestamp"].as_str().unwrap())?.with_timezone(&Utc);
        
        // Load tables
        let tables_dir = path.join("tables");
        for entry in std::fs::read_dir(tables_dir)? {
            let entry = entry?;
            let file_path = entry.path();
            if file_path.extension().unwrap_or_default() == "parquet" {
                let name = file_path.file_stem().unwrap().to_str().unwrap().to_string();
                let file = std::fs::File::open(&file_path)?;
                let table = ParquetReader::new(file).finish()?;
                self.tables.insert(name, table);
            }
        }
        
        Ok(())
    }
    
    fn validate(&self) -> Result<()> {
        // Validate dataset structure
        if self.name.is_empty() {
            return Err(crate::error::Error::validation("Dataset name is empty"));
        }
        
        if self.description.is_empty() {
            return Err(crate::error::Error::validation("Dataset description is empty"));
        }
        
        if self.version.is_empty() {
            return Err(crate::error::Error::validation("Dataset version is empty"));
        }
        
        if self.tables.is_empty() {
            return Err(crate::error::Error::validation("Dataset has no tables"));
        }
        
        // Validate timestamps
        if self.test_timestamp <= self.val_timestamp {
            return Err(crate::error::Error::validation(
                "Test timestamp must be after validation timestamp"
            ));
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;
    use tempfile::tempdir;
    
    #[test]
    fn test_base_dataset() {
        let val_ts = Utc.ymd(2024, 1, 1).and_hms(0, 0, 0);
        let test_ts = Utc.ymd(2024, 2, 1).and_hms(0, 0, 0);
        let temp_dir = tempdir().unwrap();
        
        let dataset = BaseDataset::new(
            "Test Dataset",
            "This is a test dataset",
            "1.0.0",
            val_ts,
            test_ts,
        );
        
        assert_eq!(dataset.name(), "Test Dataset");
        assert_eq!(dataset.description(), "This is a test dataset");
        assert_eq!(dataset.version(), "1.0.0");
        assert_eq!(dataset.val_timestamp(), val_ts);
        assert_eq!(dataset.test_timestamp(), test_ts);
        assert_eq!(dataset.cache_dir().unwrap(), temp_dir.path());
    }
} 