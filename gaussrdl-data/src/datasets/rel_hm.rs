use gaussrdl_core::{Dataset, DatasetConfig};
use gaussrdl_core::Result;
use std::path::{Path, PathBuf};
use std::collections::HashMap;
use polars::prelude::*;
use chrono::{DateTime, Utc, TimeZone};

use reqwest;
use std::io::Write;

const DATASET_URL: &str = "https://relational-datasets.s3.amazonaws.com/hm";
const DATASET_VERSION: &str = "1.0.0";

#[derive(Debug, Clone)]
pub struct RelHMDataset {
    name: String,
    val_timestamp: DateTime<Utc>,
    test_timestamp: DateTime<Utc>,
    tables: HashMap<String, DataFrame>,
    config: DatasetConfig,
}

impl RelHMDataset {
    pub fn new() -> Self {
        let val_ts = Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap();
        let test_ts = Utc.with_ymd_and_hms(2023, 2, 1, 0, 0, 0).unwrap();
        
        Self {
            name: "rel-hm".to_string(),
            val_timestamp: val_ts,
            test_timestamp: test_ts,
            tables: HashMap::new(),
            config: DatasetConfig::default(),
        }
    }
    
    fn download_file(&self, filename: &str) -> Result<PathBuf> {
        let download_dir = self.config.download_dir.as_ref().ok_or_else(|| {
            gaussrdl_core::Error::configuration("Download directory not set")
        })?;
        
        std::fs::create_dir_all(download_dir)?;
        let file_path = download_dir.join(filename);
        
        if !file_path.exists() || self.config.force_download {
            println!("Downloading {}...", filename);
            let url = format!("{}/{}", DATASET_URL, filename);
            let response = reqwest::blocking::get(&url)
                .map_err(|e| gaussrdl_core::Error::data(e.to_string()))?;
            let bytes = response.bytes()
                .map_err(|e| gaussrdl_core::Error::data(e.to_string()))?;
            let mut file = std::fs::File::create(&file_path)
                .map_err(|e| gaussrdl_core::Error::data(e.to_string()))?;
            file.write_all(&bytes)
                .map_err(|e| gaussrdl_core::Error::data(e.to_string()))?;
        }
        
        Ok(file_path)
    }
    
    fn load_items(&mut self) -> Result<()> {
        let file_path = self.download_file("items.parquet")?;
        let file = std::fs::File::open(file_path)?;
        let df = ParquetReader::new(file).finish()?;
        self.tables.insert("items".to_string(), df);
        Ok(())
    }
    
    fn load_stores(&mut self) -> Result<()> {
        let file_path = self.download_file("stores.parquet")?;
        let file = std::fs::File::open(file_path)?;
        let df = ParquetReader::new(file).finish()?;
        self.tables.insert("stores".to_string(), df);
        Ok(())
    }
    
    fn load_sales(&mut self) -> Result<()> {
        let file_path = self.download_file("sales.parquet")?;
        let file = std::fs::File::open(file_path)?;
        let df = ParquetReader::new(file).finish()?;
        self.tables.insert("sales".to_string(), df);
        Ok(())
    }
}

impl Dataset for RelHMDataset {
    fn name(&self) -> &str {
        self.name.as_str()
    }
    
    fn description(&self) -> &str {
        "H&M Fashion Dataset"
    }
    
    fn version(&self) -> &str {
        DATASET_VERSION
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
        
        // Load all tables
        self.load_items()?;
        self.load_stores()?;
        self.load_sales()?;
        
        Ok(())
    }
    
    fn save(&self, _path: &Path) -> Result<()> {
        // TODO: Implement saving
        Ok(())
    }
    
    fn load_from_path(&mut self, _path: &Path) -> Result<()> {
        // TODO: Implement loading from path
        Ok(())
    }
    
    fn validate(&self) -> Result<()> {
        // TODO: Implement validation
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    #[test]
    fn test_dataset_creation() {
        let dataset = RelHMDataset::new();
        assert_eq!(dataset.name(), "rel-hm");
        assert_eq!(dataset.version(), DATASET_VERSION);
    }
    
    #[test]
    fn test_dataset_validation() {
        let dataset = RelHMDataset::new();
        assert!(dataset.validate().is_err()); // Should fail without tables
    }
    
    #[test]
    fn test_dataset_save_load() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let mut dataset = RelHMDataset::new();
        
        // Create some dummy tables
        let df = DataFrame::new(vec![
            Series::new("id", &[1, 2, 3]),
            Series::new("name", &["a", "b", "c"]),
        ])?;
        
        dataset.tables.insert("items".to_string(), df.clone());
        dataset.tables.insert("stores".to_string(), df.clone());
        dataset.tables.insert("sales".to_string(), df);
        
        // Save dataset
        dataset.save(temp_dir.path())?;
        
        // Load dataset
        let mut loaded = RelHMDataset::new();
        loaded.load_from_path(temp_dir.path())?;
        
        assert_eq!(loaded.name(), dataset.name());
        assert_eq!(loaded.version(), dataset.version());
        assert_eq!(loaded.tables().len(), dataset.tables().len());
        
        Ok(())
    }
} 