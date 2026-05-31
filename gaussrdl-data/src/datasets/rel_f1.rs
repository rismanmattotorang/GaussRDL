use std::path::Path;

use chrono::{DateTime, Utc, TimeZone};
use gaussrdl_core::{Dataset, DatasetConfig};
use gaussrdl_core::Result;
use std::collections::HashMap;
use polars::prelude::*;
use reqwest;
use std::io::Write;

const DATASET_URL: &str = "https://relational-datasets.s3.amazonaws.com/f1";
const DATASET_VERSION: &str = "1.0.0";

#[derive(Debug, Clone)]
pub struct RelF1Dataset {
    name: String,
    val_timestamp: DateTime<Utc>,
    test_timestamp: DateTime<Utc>,
    tables: HashMap<String, DataFrame>,
    config: DatasetConfig,
}

impl RelF1Dataset {
    pub fn new() -> Self {
        let val_ts = Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap();
        let test_ts = Utc.with_ymd_and_hms(2023, 2, 1, 0, 0, 0).unwrap();
        
        Self {
            name: "rel-f1".to_string(),
            val_timestamp: val_ts,
            test_timestamp: test_ts,
            tables: HashMap::new(),
            config: DatasetConfig::default(),
        }
    }
    
    fn download_file(&self, filename: &str) -> Result<std::path::PathBuf> {
        let download_dir = self.config.download_dir.as_ref().ok_or_else(|| {
            gaussrdl_core::Error::configuration("Download directory not set")
        })?;
        
        std::fs::create_dir_all(download_dir)?;
        let file_path = download_dir.join(filename);
        
        if !file_path.exists() || self.config.force_download {
            println!("Downloading {}...", filename);
            let url = format!("{}/{}", DATASET_URL, filename);
            let response = reqwest::blocking::get(&url)?;
            let mut file = std::fs::File::create(&file_path)?;
            file.write_all(&response.bytes()?)?;
        }
        
        Ok(file_path)
    }
    
    fn load_drivers(&mut self) -> Result<()> {
        let file_path = self.download_file("drivers.parquet")?;
        let df = LazyFrame::scan_parquet(&file_path, Default::default())?.collect()?;
        self.tables.insert("drivers".to_string(), df);
        Ok(())
    }
    
    fn load_constructors(&mut self) -> Result<()> {
        let file_path = self.download_file("constructors.parquet")?;
        let df = LazyFrame::scan_parquet(&file_path, Default::default())?.collect()?;
        self.tables.insert("constructors".to_string(), df);
        Ok(())
    }
    
    fn load_races(&mut self) -> Result<()> {
        let file_path = self.download_file("races.parquet")?;
        let df = LazyFrame::scan_parquet(&file_path, Default::default())?.collect()?;
        self.tables.insert("races".to_string(), df);
        Ok(())
    }
    
    fn load_results(&mut self) -> Result<()> {
        let file_path = self.download_file("results.parquet")?;
        let df = LazyFrame::scan_parquet(&file_path, Default::default())?.collect()?;
        self.tables.insert("results".to_string(), df);
        Ok(())
    }
}

impl Dataset for RelF1Dataset {
    fn name(&self) -> &str {
        self.name.as_str()
    }
    
    fn description(&self) -> &str {
        "Formula 1 Racing Dataset"
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
        self.load_drivers()?;
        self.load_constructors()?;
        self.load_races()?;
        self.load_results()?;
        
        Ok(())
    }
    
    fn save(&self, path: &Path) -> Result<()> {
        std::fs::create_dir_all(path)?;
        
        // Save tables
        for (name, table) in &self.tables {
            let table_path = path.join(format!("{}.parquet", name));
            let mut file = std::fs::File::create(table_path)?;
            ParquetWriter::new(&mut file).finish(&mut table.clone())?;
        }
        
        Ok(())
    }
    
    fn load_from_path(&mut self, path: &Path) -> Result<()> {
        let required_tables = ["drivers", "constructors", "races", "results"];
        
        for table_name in required_tables.iter() {
            let table_path = path.join(format!("{}.parquet", table_name));
            if table_path.exists() {
                let df = LazyFrame::scan_parquet(table_path, Default::default())?.collect()?;
                self.tables.insert(table_name.to_string(), df);
            }
        }
        
        Ok(())
    }
    
    fn validate(&self) -> Result<()> {
        // Validate required tables
        let required_tables = ["drivers", "constructors", "races", "results"];
        
        for table in required_tables.iter() {
            if !self.tables.contains_key(*table) {
                return Err(gaussrdl_core::Error::validation(
                    format!("Missing required table: {}", table)
                ));
            }
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    #[test]
    fn test_dataset_creation() {
        let dataset = RelF1Dataset::new();
        assert_eq!(dataset.name(), "rel-f1");
        assert_eq!(dataset.version(), DATASET_VERSION);
    }
    
    #[test]
    fn test_dataset_validation() {
        let dataset = RelF1Dataset::new();
        assert!(dataset.validate().is_err()); // Should fail without tables
    }
    
    #[test]
    fn test_dataset_save_load() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let mut dataset = RelF1Dataset::new();
        
        // Create some dummy tables
        let df = DataFrame::new(vec![
            Series::new("id", &[1, 2, 3]),
            Series::new("name", &["a", "b", "c"]),
        ])?;
        
        dataset.tables.insert("drivers".to_string(), df.clone());
        dataset.tables.insert("constructors".to_string(), df.clone());
        dataset.tables.insert("races".to_string(), df.clone());
        dataset.tables.insert("results".to_string(), df);
        
        // Save dataset
        dataset.save(temp_dir.path())?;
        
        // Load dataset
        let mut loaded = RelF1Dataset::new();
        loaded.load_from_path(temp_dir.path())?;
        
        assert_eq!(loaded.name(), dataset.name());
        assert_eq!(loaded.version(), dataset.version());
        assert_eq!(loaded.tables().len(), dataset.tables().len());
        
        Ok(())
    }
} 