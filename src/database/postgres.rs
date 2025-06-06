// src/database/postgres.rs
use crate::{Result, database::{DatabaseConnectionTrait, DatabaseInfo}};

pub struct PostgresConnection {
    pool: sqlx::postgres::PgPool,
}

impl PostgresConnection {
    pub async fn new(url: &str) -> Result<Self> {
        let pool = sqlx::postgres::PgPoolOptions::new()
            .max_connections(5)
            .connect(url)
            .await?;
        
        Ok(Self { pool })
    }
}

#[async_trait::async_trait]
impl DatabaseConnectionTrait for PostgresConnection {
    async fn get_info(&self) -> Result<DatabaseInfo> {
        let version_row = sqlx::query("SELECT version()")
            .fetch_one(&self.pool)
            .await?;
        
        let version: String = version_row.get(0);
        
        Ok(DatabaseInfo {
            name: "PostgreSQL".to_string(),
            version,
            table_count: 0,
            total_rows: 0,
            supports_foreign_keys: true,
            supports_temporal: true,
        })
    }
    
    async fn execute_query(&self, _query: &str) -> Result<Vec<serde_json::Value>> {
        Ok(vec![])
    }
    
    async fn get_table_names(&self) -> Result<Vec<String>> {
        let rows = sqlx::query(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
        )
        .fetch_all(&self.pool)
        .await?;
        
        Ok(rows.iter().map(|row| row.get::<String, _>(0)).collect())
    }
    
    async fn get_row_count(&self, table: &str) -> Result<u64> {
        let query = format!("SELECT COUNT(*) FROM {}", table);
        let row = sqlx::query(&query).fetch_one(&self.pool).await?;
        Ok(row.get::<i64, _>(0) as u64)
    }
    
    async fn test_connection(&self) -> Result<()> {
        sqlx::query("SELECT 1").fetch_one(&self.pool).await?;
        Ok(())
    }
    
    async fn close(&self) -> Result<()> {
        self.pool.close().await;
        Ok(())
    }
}