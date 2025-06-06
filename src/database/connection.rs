// src/database/connection.rs
use crate::{Result, database::{DatabaseType, DatabaseInfo}};
use std::sync::Arc;

#[async_trait::async_trait]
pub trait DatabaseConnectionTrait: Send + Sync {
    async fn get_info(&self) -> Result<DatabaseInfo>;
    async fn execute_query(&self, query: &str) -> Result<Vec<serde_json::Value>>;
    async fn get_table_names(&self) -> Result<Vec<String>>;
    async fn get_row_count(&self, table: &str) -> Result<u64>;
    async fn test_connection(&self) -> Result<()>;
    async fn close(&self) -> Result<()>;
}

pub struct DatabaseConnection {
    inner: Arc<dyn DatabaseConnectionTrait>,
    db_type: DatabaseType,
}

impl DatabaseConnection {
    pub async fn new(url: &str, db_type: DatabaseType) -> Result<Self> {
        let inner: Arc<dyn DatabaseConnectionTrait> = match db_type {
            DatabaseType::Postgres => {
                Arc::new(crate::database::postgres::PostgresConnection::new(url).await?)
            }
            DatabaseType::SurrealDb => {
                Arc::new(crate::database::surrealdb::SurrealConnection::new(url).await?)
            }
        };
        
        Ok(Self { inner, db_type })
    }
    
    pub async fn get_info(&self) -> Result<DatabaseInfo> {
        self.inner.get_info().await
    }
    
    pub async fn execute_query(&self, query: &str) -> Result<Vec<serde_json::Value>> {
        self.inner.execute_query(query).await
    }
    
    pub async fn get_table_names(&self) -> Result<Vec<String>> {
        self.inner.get_table_names().await
    }
    
    pub async fn get_row_count(&self, table: &str) -> Result<u64> {
        self.inner.get_row_count(table).await
    }
    
    pub fn get_type(&self) -> DatabaseType {
        self.db_type
    }
}