use std::path::PathBuf;
use chrono::{DateTime, Utc};
use crate::error::Result;
use crate::base::Database;

/// Dataset trait defining common functionality for all datasets
pub trait Dataset: Send + Sync {
    /// Get the validation timestamp
    fn val_timestamp(&self) -> DateTime<Utc>;
    
    /// Get the test timestamp
    fn test_timestamp(&self) -> DateTime<Utc>;
    
    /// Get the database instance
    fn get_db(&self, upto_test_timestamp: bool) -> Result<Database>;
    
    /// Make the database from raw data
    fn make_db(&self) -> Result<Database>;
    
    /// Validate and correct the database
    fn validate_and_correct_db(&self, db: &mut Database) -> Result<()>;
    
    /// Get the cache directory
    fn cache_dir(&self) -> Option<PathBuf>;
    
    /// Set the cache directory
    fn set_cache_dir(&mut self, dir: Option<PathBuf>);
}

/// Base implementation for datasets
#[derive(Debug)]
pub struct BaseDataset {
    cache_dir: Option<PathBuf>,
    val_timestamp: DateTime<Utc>,
    test_timestamp: DateTime<Utc>,
}

impl BaseDataset {
    /// Create a new dataset instance
    pub fn new(
        val_timestamp: DateTime<Utc>,
        test_timestamp: DateTime<Utc>,
        cache_dir: Option<PathBuf>,
    ) -> Self {
        Self {
            cache_dir,
            val_timestamp,
            test_timestamp,
        }
    }
}

impl Dataset for BaseDataset {
    fn val_timestamp(&self) -> DateTime<Utc> {
        self.val_timestamp
    }
    
    fn test_timestamp(&self) -> DateTime<Utc> {
        self.test_timestamp
    }
    
    fn get_db(&self, upto_test_timestamp: bool) -> Result<Database> {
        let db_path = self.cache_dir.as_ref().map(|dir| dir.join("db"));
        
        let mut db = if let Some(path) = db_path {
            if path.exists() {
                println!("Loading Database from cache...");
                Database::load(&path)?
            } else {
                println!("Making Database from scratch...");
                let mut db = self.make_db()?;
                db.reindex_pkeys_and_fkeys()?;
                
                if let Some(dir) = &self.cache_dir {
                    println!("Caching Database...");
                    db.save(dir)?;
                }
                db
            }
        } else {
            self.make_db()?
        };
        
        if upto_test_timestamp {
            db = db.upto(self.test_timestamp)?;
        }
        
        self.validate_and_correct_db(&mut db)?;
        
        Ok(db)
    }
    
    fn make_db(&self) -> Result<Database> {
        unimplemented!("make_db must be implemented by dataset")
    }
    
    fn validate_and_correct_db(&self, _db: &mut Database) -> Result<()> {
        Ok(()) // Default implementation does nothing
    }
    
    fn cache_dir(&self) -> Option<PathBuf> {
        self.cache_dir.clone()
    }
    
    fn set_cache_dir(&mut self, dir: Option<PathBuf>) {
        self.cache_dir = dir;
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
            val_ts,
            test_ts,
            Some(temp_dir.path().to_path_buf()),
        );
        
        assert_eq!(dataset.val_timestamp(), val_ts);
        assert_eq!(dataset.test_timestamp(), test_ts);
        assert_eq!(dataset.cache_dir().unwrap(), temp_dir.path());
    }
} 