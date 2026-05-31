// src/database/mod.rs
use serde::{Deserialize, Serialize};
use std::str::FromStr;
use gaussrdl_core::{Result, Error};
use std::fmt::Debug;

pub mod connection;
pub mod postgres;
pub mod surrealdb;

pub use connection::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DatabaseType {
    Postgres,
    SurrealDb,
}

impl std::fmt::Display for DatabaseType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DatabaseType::Postgres => write!(f, "postgres"),
            DatabaseType::SurrealDb => write!(f, "surrealdb"),
        }
    }
}

impl FromStr for DatabaseType {
    type Err = Error;
    
    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "postgres" | "postgresql" => Ok(Self::Postgres),
            "surrealdb" | "surreal" => Ok(Self::SurrealDb),
            _ => Err(Error::configuration(format!("Unsupported database type: {}", s))),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseInfo {
    pub name: String,
    pub version: String,
    pub table_count: usize,
    pub total_rows: u64,
    pub supports_foreign_keys: bool,
    pub supports_temporal: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseSchema {
    pub database_info: DatabaseInfo,
    pub tables: Vec<TableInfo>,
    pub relationships: Vec<Relationship>,
    pub extracted_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableInfo {
    pub name: String,
    pub row_count: u64,
    pub columns: Vec<ColumnInfo>,
    pub primary_keys: Vec<String>,
    pub foreign_keys: Vec<ForeignKeyInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnInfo {
    pub name: String,
    pub data_type: String,
    pub is_nullable: bool,
    pub is_primary_key: bool,
    pub is_foreign_key: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForeignKeyInfo {
    pub name: String,
    pub columns: Vec<String>,
    pub referenced_table: String,
    pub referenced_columns: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relationship {
    pub id: String,
    pub from_table: String,
    pub from_columns: Vec<String>,
    pub to_table: String,
    pub to_columns: Vec<String>,
}

impl DatabaseSchema {
    pub fn save_to_file(&self, path: &std::path::Path) -> Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }
    
    pub fn load_from_file(path: &std::path::Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let schema = serde_json::from_str(&content)?;
        Ok(schema)
    }
}

pub struct SchemaExtractor {
    connection: DatabaseConnection,
}

impl SchemaExtractor {
    pub fn new(connection: DatabaseConnection) -> Self {
        Self { connection }
    }
    
    pub async fn extract_full_schema(
        &self, 
        _include_samples: bool, 
        _sample_size: usize
    ) -> Result<DatabaseSchema> {
        let info = self.connection.get_info().await?;
        let table_names = self.connection.get_table_names().await?;
        
        let mut tables = Vec::new();
        for table_name in &table_names {
            let table_info = self.extract_table_info(table_name).await?;
            tables.push(table_info);
        }
        
        let relationships = self.extract_relationships().await?;
        
        Ok(DatabaseSchema {
            database_info: info,
            tables,
            relationships,
            extracted_at: chrono::Utc::now(),
        })
    }
    
    async fn extract_table_info(&self, table_name: &str) -> Result<TableInfo> {
        let row_count = self.connection.get_row_count(table_name).await?;
        
        // Simplified implementation - would extract actual column info
        Ok(TableInfo {
            name: table_name.to_string(),
            row_count,
            columns: vec![],
            primary_keys: vec![],
            foreign_keys: vec![],
        })
    }
    
    async fn extract_relationships(&self) -> Result<Vec<Relationship>> {
        // Simplified implementation - would extract actual relationships
        Ok(vec![])
    }
}
