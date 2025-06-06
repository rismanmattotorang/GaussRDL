use std::collections::HashMap;
use std::path::{Path, PathBuf};
use chrono::{DateTime, Utc};
use polars::prelude::*;
use crate::error::Result;
use crate::base::Table;

/// A database is a collection of named tables linked by foreign key - primary key connections
#[derive(Debug, Clone)]
pub struct Database {
    tables: HashMap<String, Table>,
}

impl Database {
    /// Create a new database from a map of tables
    pub fn new(tables: HashMap<String, Table>) -> Self {
        Self { tables }
    }
    
    /// Get a reference to a table by name
    pub fn get_table(&self, name: &str) -> Option<&Table> {
        self.tables.get(name)
    }
    
    /// Get a mutable reference to a table by name
    pub fn get_table_mut(&mut self, name: &str) -> Option<&mut Table> {
        self.tables.get_mut(name)
    }
    
    /// Get all table names
    pub fn table_names(&self) -> Vec<String> {
        self.tables.keys().cloned().collect()
    }
    
    /// Save the database to a directory
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();
        std::fs::create_dir_all(path)?;
        
        for (name, table) in &self.tables {
            let table_path = path.join(format!("{}.parquet", name));
            table.save(&table_path)?;
        }
        Ok(())
    }
    
    /// Load a database from a directory
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let mut tables = HashMap::new();
        
        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            let file_path = entry.path();
            
            if file_path.extension().map_or(false, |ext| ext == "parquet") {
                let name = file_path.file_stem().unwrap().to_string_lossy().into_owned();
                let table = Table::load(&file_path)?;
                tables.insert(name, table);
            }
        }
        
        Ok(Self::new(tables))
    }
    
    /// Get the minimum timestamp across all tables
    pub fn min_timestamp(&self) -> Option<DateTime<Utc>> {
        self.tables
            .values()
            .filter_map(|table| table.min_timestamp())
            .min()
    }
    
    /// Get the maximum timestamp across all tables
    pub fn max_timestamp(&self) -> Option<DateTime<Utc>> {
        self.tables
            .values()
            .filter_map(|table| table.max_timestamp())
            .max()
    }
    
    /// Filter the database to only include rows up to a given timestamp
    pub fn upto(&self, timestamp: DateTime<Utc>) -> Result<Self> {
        let mut filtered_tables = HashMap::new();
        
        for (name, table) in &self.tables {
            filtered_tables.insert(name.clone(), table.upto(timestamp)?);
        }
        
        Ok(Self::new(filtered_tables))
    }
    
    /// Filter the database to only include rows from a given timestamp
    pub fn from(&self, timestamp: DateTime<Utc>) -> Result<Self> {
        let mut filtered_tables = HashMap::new();
        
        for (name, table) in &self.tables {
            filtered_tables.insert(name.clone(), table.from(timestamp)?);
        }
        
        Ok(Self::new(filtered_tables))
    }
    
    /// Reindex primary and foreign keys to ensure consistency
    pub fn reindex_pkeys_and_fkeys(&mut self) -> Result<()> {
        // First pass: reindex primary keys
        for table in self.tables.values_mut() {
            table.reindex_pkey()?;
        }
        
        // Second pass: update foreign keys
        for table in self.tables.values_mut() {
            table.update_fkeys(&self.tables)?;
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
    fn test_database_operations() {
        let mut tables = HashMap::new();
        
        // Create test tables
        let mut users = Table::new(
            DataFrame::new(vec![
                Series::new("id", vec![1, 2, 3]),
                Series::new("name", vec!["Alice", "Bob", "Charlie"]),
            ]).unwrap(),
            "id",
            None,
            HashMap::new(),
        );
        
        let mut posts = Table::new(
            DataFrame::new(vec![
                Series::new("id", vec![1, 2, 3]),
                Series::new("user_id", vec![1, 2, 1]),
                Series::new("title", vec!["Post 1", "Post 2", "Post 3"]),
                Series::new("timestamp", vec![
                    Utc.ymd(2024, 1, 1).and_hms(0, 0, 0),
                    Utc.ymd(2024, 1, 2).and_hms(0, 0, 0),
                    Utc.ymd(2024, 1, 3).and_hms(0, 0, 0),
                ]),
            ]).unwrap(),
            "id",
            Some("timestamp"),
            HashMap::from([("user_id".to_string(), "users".to_string())]),
        );
        
        tables.insert("users".to_string(), users);
        tables.insert("posts".to_string(), posts);
        
        let db = Database::new(tables);
        
        // Test saving and loading
        let temp_dir = tempdir().unwrap();
        db.save(temp_dir.path()).unwrap();
        
        let loaded_db = Database::load(temp_dir.path()).unwrap();
        assert_eq!(loaded_db.table_names().len(), 2);
        
        // Test timestamp operations
        let min_ts = loaded_db.min_timestamp().unwrap();
        let max_ts = loaded_db.max_timestamp().unwrap();
        assert_eq!(min_ts, Utc.ymd(2024, 1, 1).and_hms(0, 0, 0));
        assert_eq!(max_ts, Utc.ymd(2024, 1, 3).and_hms(0, 0, 0));
        
        // Test filtering
        let filtered_db = loaded_db.upto(Utc.ymd(2024, 1, 2).and_hms(0, 0, 0)).unwrap();
        let posts_table = filtered_db.get_table("posts").unwrap();
        assert_eq!(posts_table.len(), 2);
    }
} 