use std::collections::HashMap;
use std::path::{Path, PathBuf};
use chrono::{DateTime, Utc};
use polars::prelude::*;
use serde::{Serialize, Deserialize};
use crate::error::Result;
use crate::base::Table;
use crate::database::{DatabaseConnectionTrait, DatabaseType, TableSchema, ColumnDefinition};

/// Enhanced database abstraction for relational graph learning
#[derive(Debug, Clone)]
pub struct Database {
    tables: HashMap<String, Table>,
    schema: DatabaseSchema,
    metadata: DatabaseMetadata,
}

/// Database schema information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseSchema {
    pub version: String,
    pub tables: HashMap<String, TableSchema>,
    pub foreign_keys: HashMap<String, Vec<ForeignKeyMapping>>,
    pub indexes: HashMap<String, Vec<IndexDefinition>>,
}

/// Foreign key mapping between tables
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForeignKeyMapping {
    pub source_table: String,
    pub source_column: String,
    pub target_table: String,
    pub target_column: String,
    pub constraint_name: String,
}

/// Index definition for performance optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexDefinition {
    pub name: String,
    pub table: String,
    pub columns: Vec<String>,
    pub is_unique: bool,
    pub index_type: IndexType,
}

/// Index types for different use cases
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IndexType {
    BTree,
    Hash,
    GiST,
    GIN,
    BRIN,
}

/// Database metadata for tracking and optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseMetadata {
    pub created_at: DateTime<Utc>,
    pub last_modified: DateTime<Utc>,
    pub total_rows: u64,
    pub total_size_bytes: u64,
    pub table_count: usize,
    pub statistics: DatabaseStatistics,
}

/// Comprehensive database statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseStatistics {
    pub query_performance: HashMap<String, QueryStats>,
    pub table_access_frequency: HashMap<String, u64>,
    pub index_usage: HashMap<String, IndexUsageStats>,
    pub temporal_patterns: TemporalPatterns,
}

/// Query performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryStats {
    pub avg_execution_time_ms: f64,
    pub total_executions: u64,
    pub cache_hit_rate: f64,
}

/// Index usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexUsageStats {
    pub scans: u64,
    pub lookups: u64,
    pub maintenance_cost: f64,
}

/// Temporal access patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalPatterns {
    pub peak_access_hours: Vec<u8>,
    pub seasonal_trends: HashMap<String, f64>,
    pub growth_rate: f64,
}

impl Database {
    /// Create a new enhanced database
    pub fn new(tables: HashMap<String, Table>) -> Self {
        let schema = DatabaseSchema {
            version: "1.0.0".to_string(),
            tables: HashMap::new(),
            foreign_keys: HashMap::new(),
            indexes: HashMap::new(),
        };
        
        let metadata = DatabaseMetadata {
            created_at: Utc::now(),
            last_modified: Utc::now(),
            total_rows: tables.values().map(|t| t.len() as u64).sum(),
            total_size_bytes: 0, // Would calculate actual size
            table_count: tables.len(),
            statistics: DatabaseStatistics {
                query_performance: HashMap::new(),
                table_access_frequency: HashMap::new(),
                index_usage: HashMap::new(),
                temporal_patterns: TemporalPatterns {
                    peak_access_hours: vec![9, 10, 11, 14, 15, 16],
                    seasonal_trends: HashMap::new(),
                    growth_rate: 0.0,
                },
            },
        };
        
        Self { tables, schema, metadata }
    }
    
    /// Create database from external database connection
    pub async fn from_connection<T: DatabaseConnectionTrait>(
        connection: &T,
        table_names: Option<Vec<String>>,
    ) -> Result<Self> {
        let mut tables = HashMap::new();
        
        let names = if let Some(names) = table_names {
            names
        } else {
            connection.get_table_names().await?
        };
        
        for table_name in names {
            // Get table schema
            let schema = connection.get_table_schema(&table_name).await?;
            
            // Execute query to get data
            let query = format!("SELECT * FROM {}", table_name);
            let data = connection.execute_query(&query).await?;
            
            // Convert to Table (this would need proper implementation)
            let table = Table::from_json_data(data, &schema)?;
            tables.insert(table_name, table);
        }
        
        Ok(Self::new(tables))
    }
    
    /// Get a reference to a table by name
    pub fn get_table(&self, name: &str) -> Option<&Table> {
        self.tables.get(name)
    }
    
    /// Get a mutable reference to a table by name
    pub fn get_table_mut(&mut self, name: &str) -> Option<&mut Table> {
        self.update_metadata();
        self.tables.get_mut(name)
    }
    
    /// Get all table names
    pub fn table_names(&self) -> Vec<String> {
        self.tables.keys().cloned().collect()
    }
    
    /// Add a table to the database
    pub fn add_table(&mut self, name: String, table: Table) -> Result<()> {
        self.tables.insert(name.clone(), table);
        self.update_schema_for_table(&name)?;
        self.update_metadata();
        Ok(())
    }
    
    /// Remove a table from the database
    pub fn remove_table(&mut self, name: &str) -> Result<Option<Table>> {
        let table = self.tables.remove(name);
        if table.is_some() {
            self.schema.tables.remove(name);
            self.update_metadata();
        }
        Ok(table)
    }
    
    /// Enhanced save with compression and metadata
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();
        std::fs::create_dir_all(path)?;
        
        // Save tables with compression
        for (name, table) in &self.tables {
            let table_path = path.join(format!("{}.parquet", name));
            table.save_compressed(&table_path)?;
        }
        
        // Save schema and metadata
        let schema_path = path.join("schema.json");
        let schema_json = serde_json::to_string_pretty(&self.schema)?;
        std::fs::write(schema_path, schema_json)?;
        
        let metadata_path = path.join("metadata.json");
        let metadata_json = serde_json::to_string_pretty(&self.metadata)?;
        std::fs::write(metadata_path, metadata_json)?;
        
        Ok(())
    }
    
    /// Enhanced load with validation
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let mut tables = HashMap::new();
        
        // Load schema if available
        let schema_path = path.join("schema.json");
        let schema = if schema_path.exists() {
            let schema_json = std::fs::read_to_string(schema_path)?;
            serde_json::from_str(&schema_json)?
        } else {
            DatabaseSchema {
                version: "1.0.0".to_string(),
                tables: HashMap::new(),
                foreign_keys: HashMap::new(),
                indexes: HashMap::new(),
            }
        };
        
        // Load metadata if available
        let metadata_path = path.join("metadata.json");
        let mut metadata = if metadata_path.exists() {
            let metadata_json = std::fs::read_to_string(metadata_path)?;
            serde_json::from_str(&metadata_json)?
        } else {
            DatabaseMetadata {
                created_at: Utc::now(),
                last_modified: Utc::now(),
                total_rows: 0,
                total_size_bytes: 0,
                table_count: 0,
                statistics: DatabaseStatistics {
                    query_performance: HashMap::new(),
                    table_access_frequency: HashMap::new(),
                    index_usage: HashMap::new(),
                    temporal_patterns: TemporalPatterns {
                        peak_access_hours: vec![9, 10, 11, 14, 15, 16],
                        seasonal_trends: HashMap::new(),
                        growth_rate: 0.0,
                    },
                },
            }
        };
        
        // Load tables
        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            let file_path = entry.path();
            
            if file_path.extension().map_or(false, |ext| ext == "parquet") {
                let name = file_path.file_stem().unwrap().to_string_lossy().into_owned();
                let table = Table::load(&file_path)?;
                tables.insert(name, table);
            }
        }
        
        // Update metadata
        metadata.table_count = tables.len();
        metadata.total_rows = tables.values().map(|t| t.len() as u64).sum();
        metadata.last_modified = Utc::now();
        
        Ok(Self { tables, schema, metadata })
    }
    
    /// Get comprehensive database statistics
    pub fn get_statistics(&self) -> &DatabaseStatistics {
        &self.metadata.statistics
    }
    
    /// Update table access frequency for optimization
    pub fn record_table_access(&mut self, table_name: &str) {
        *self.metadata.statistics.table_access_frequency
            .entry(table_name.to_string())
            .or_insert(0) += 1;
    }
    
    /// Get suggested optimizations based on usage patterns
    pub fn get_optimization_suggestions(&self) -> Vec<OptimizationSuggestion> {
        let mut suggestions = Vec::new();
        
        // Suggest indexes for frequently accessed tables
        for (table_name, access_count) in &self.metadata.statistics.table_access_frequency {
            if *access_count > 1000 && !self.schema.indexes.contains_key(table_name) {
                suggestions.push(OptimizationSuggestion {
                    suggestion_type: OptimizationType::AddIndex,
                    table: table_name.clone(),
                    description: format!("Add index to frequently accessed table: {}", table_name),
                    estimated_benefit: 0.3, // 30% improvement
                });
            }
        }
        
        // Suggest partitioning for large tables
        for (table_name, table) in &self.tables {
            if table.len() > 100_000 {
                suggestions.push(OptimizationSuggestion {
                    suggestion_type: OptimizationType::Partition,
                    table: table_name.clone(),
                    description: format!("Consider partitioning large table: {}", table_name),
                    estimated_benefit: 0.2, // 20% improvement
                });
            }
        }
        
        suggestions
    }
    
    /// Validate database integrity and relationships
    pub fn validate_integrity(&self) -> Result<ValidationReport> {
        let mut report = ValidationReport {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
        };
        
        // Check foreign key constraints
        for (table_name, fkeys) in &self.schema.foreign_keys {
            if let Some(table) = self.get_table(table_name) {
                for fkey in fkeys {
                    if !self.tables.contains_key(&fkey.target_table) {
                        report.errors.push(format!(
                            "Foreign key references non-existent table: {} -> {}",
                            fkey.source_table, fkey.target_table
                        ));
                        report.is_valid = false;
                    }
                }
            }
        }
        
        Ok(report)
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
        
        let mut filtered_db = Self::new(filtered_tables);
        filtered_db.schema = self.schema.clone();
        Ok(filtered_db)
    }
    
    /// Filter the database to only include rows from a given timestamp
    pub fn from(&self, timestamp: DateTime<Utc>) -> Result<Self> {
        let mut filtered_tables = HashMap::new();
        
        for (name, table) in &self.tables {
            filtered_tables.insert(name.clone(), table.from(timestamp)?);
        }
        
        let mut filtered_db = Self::new(filtered_tables);
        filtered_db.schema = self.schema.clone();
        Ok(filtered_db)
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
        
        self.update_metadata();
        Ok(())
    }
    
    /// Update metadata after modifications
    fn update_metadata(&mut self) {
        self.metadata.last_modified = Utc::now();
        self.metadata.table_count = self.tables.len();
        self.metadata.total_rows = self.tables.values().map(|t| t.len() as u64).sum();
    }
    
    /// Update schema information for a specific table
    fn update_schema_for_table(&mut self, table_name: &str) -> Result<()> {
        if let Some(table) = self.get_table(table_name) {
            let table_schema = table.get_schema()?;
            self.schema.tables.insert(table_name.to_string(), table_schema);
        }
        Ok(())
    }
}

/// Optimization suggestion for database performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationSuggestion {
    pub suggestion_type: OptimizationType,
    pub table: String,
    pub description: String,
    pub estimated_benefit: f64, // Percentage improvement
}

/// Types of database optimizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationType {
    AddIndex,
    DropIndex,
    Partition,
    Compress,
    Archive,
    Normalize,
}

/// Database validation report
#[derive(Debug, Clone)]
pub struct ValidationReport {
    pub is_valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;
    use tempfile::tempdir;
    
    #[test]
    fn test_enhanced_database_operations() {
        let mut tables = HashMap::new();
        
        // Create test tables
        let users = Table::new(
            DataFrame::new(vec![
                Series::new("id", vec![1, 2, 3]),
                Series::new("name", vec!["Alice", "Bob", "Charlie"]),
            ]).unwrap(),
            "id",
            None,
            HashMap::new(),
        );
        
        let posts = Table::new(
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
        
        let mut db = Database::new(tables);
        
        // Test optimization suggestions
        db.record_table_access("users");
        for _ in 0..1001 {
            db.record_table_access("users");
        }
        let suggestions = db.get_optimization_suggestions();
        assert!(!suggestions.is_empty());
        
        // Test validation
        let validation_report = db.validate_integrity().unwrap();
        assert!(validation_report.is_valid);
        
        // Test enhanced save/load
        let temp_dir = tempdir().unwrap();
        db.save(temp_dir.path()).unwrap();
        
        let loaded_db = Database::load(temp_dir.path()).unwrap();
        assert_eq!(loaded_db.table_names().len(), 2);
        assert_eq!(loaded_db.metadata.table_count, 2);
    }
} 