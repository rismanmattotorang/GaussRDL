// src/database/postgres.rs
use sqlx::{postgres::PgPool, Row};
use crate::database::{DatabaseConnectionTrait, DatabaseInfo, TableInfo, ColumnInfo, ForeignKeyInfo};
use crate::database::connection::{ConnectionConfig, TableSchema, ConnectionHealth, DatabaseStats};
use gaussrdl_core::{Result, Error};
use serde_json::Value;


#[derive(Debug)]
pub struct PostgresConnection {
    pool: PgPool,
    database_name: String,
}

impl PostgresConnection {
    pub async fn new(url: &str) -> Result<Self> {
        // Extract database name from URL
        let database_name = Self::extract_database_name(url);
        
        let pool = sqlx::postgres::PgPoolOptions::new()
            .max_connections(10)
            .min_connections(2)
            .idle_timeout(Some(std::time::Duration::from_secs(600)))
            .connect(url)
            .await
            .map_err(|e| Error::database(format!("Failed to connect to PostgreSQL: {}", e)))?;
        
        Ok(Self { pool, database_name })
    }
    
    pub async fn new_with_config(url: &str, config: &ConnectionConfig) -> Result<Self> {
        // Extract database name from URL
        let database_name = Self::extract_database_name(url);
        
        let pool = sqlx::postgres::PgPoolOptions::new()
            .max_connections(config.max_connections as u32)
            .min_connections(1)
            .idle_timeout(Some(config.connection_timeout))
            .acquire_timeout(config.connection_timeout)
            .connect(url)
            .await
            .map_err(|e| Error::database(format!("Failed to connect to PostgreSQL: {}", e)))?;
        
        Ok(Self { pool, database_name })
    }
    
    fn extract_database_name(url: &str) -> String {
        if let Some(start) = url.rfind('/') {
            let after_slash = &url[start + 1..];
            if let Some(end) = after_slash.find('?') {
                after_slash[..end].to_string()
            } else {
                after_slash.to_string()
            }
        } else {
            "postgres".to_string()
        }
    }
    
    async fn get_table_info_detailed(&self, table_name: &str) -> Result<TableInfo> {
        // Get column information
        let column_query = r#"
            SELECT 
                column_name,
                data_type,
                is_nullable::boolean as is_nullable,
                column_default
            FROM information_schema.columns 
            WHERE table_schema = 'public' 
            AND table_name = $1
            ORDER BY ordinal_position
        "#;
        
        let column_rows = sqlx::query(column_query)
            .bind(table_name)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| Error::database(format!("Failed to get column info: {}", e)))?;
        
        let mut columns = Vec::new();
        for row in column_rows {
            let column_name: String = row.get("column_name");
            let data_type: String = row.get("data_type");
            let is_nullable: bool = row.get("is_nullable");
            
            columns.push(ColumnInfo {
                name: column_name.clone(),
                data_type,
                is_nullable,
                is_primary_key: false, // Will be updated below
                is_foreign_key: false, // Will be updated below
            });
        }
        
        // Get primary key information
        let pk_query = r#"
            SELECT column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu 
                ON tc.constraint_name = kcu.constraint_name
                AND tc.table_schema = kcu.table_schema
            WHERE tc.constraint_type = 'PRIMARY KEY'
            AND tc.table_schema = 'public'
            AND tc.table_name = $1
        "#;
        
        let pk_rows = sqlx::query(pk_query)
            .bind(table_name)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| Error::database(format!("Failed to get primary key info: {}", e)))?;
        
        let primary_keys: Vec<String> = pk_rows.iter()
            .map(|row| row.get::<String, _>("column_name"))
            .collect();
        
        // Update primary key flags
        for column in &mut columns {
            if primary_keys.contains(&column.name) {
                column.is_primary_key = true;
            }
        }
        
        // Get foreign key information
        let fk_query = r#"
            SELECT
                tc.constraint_name,
                kcu.column_name,
                ccu.table_name AS foreign_table_name,
                ccu.column_name AS foreign_column_name
            FROM information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
                ON tc.constraint_name = kcu.constraint_name
                AND tc.table_schema = kcu.table_schema
            JOIN information_schema.constraint_column_usage AS ccu
                ON ccu.constraint_name = tc.constraint_name
                AND ccu.table_schema = tc.table_schema
            WHERE tc.constraint_type = 'FOREIGN KEY'
            AND tc.table_schema = 'public'
            AND tc.table_name = $1
        "#;
        
        let fk_rows = sqlx::query(fk_query)
            .bind(table_name)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| Error::database(format!("Failed to get foreign key info: {}", e)))?;
        
        let mut foreign_keys = Vec::new();
        let mut fk_columns = std::collections::HashSet::new();
        
        for row in fk_rows {
            let constraint_name: String = row.get("constraint_name");
            let column_name: String = row.get("column_name");
            let foreign_table: String = row.get("foreign_table_name");
            let foreign_column: String = row.get("foreign_column_name");
            
            fk_columns.insert(column_name.clone());
            
            foreign_keys.push(ForeignKeyInfo {
                name: constraint_name,
                columns: vec![column_name],
                referenced_table: foreign_table,
                referenced_columns: vec![foreign_column],
            });
        }
        
        // Update foreign key flags
        for column in &mut columns {
            if fk_columns.contains(&column.name) {
                column.is_foreign_key = true;
            }
        }
        
        let row_count = self.get_row_count(table_name).await?;
        
        Ok(TableInfo {
            name: table_name.to_string(),
            row_count,
            columns,
            primary_keys,
            foreign_keys,
        })
    }
}

#[async_trait::async_trait]
impl DatabaseConnectionTrait for PostgresConnection {
    async fn get_info(&self) -> Result<DatabaseInfo> {
        let version_row = sqlx::query("SELECT version()")
            .fetch_one(&self.pool)
            .await
            .map_err(|e| Error::database(format!("Failed to get version: {}", e)))?;
        
        let version: String = version_row.get(0);
        
        // Get table count
        let table_count_row = sqlx::query(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public'"
        )
        .fetch_one(&self.pool)
        .await
        .map_err(|e| Error::database(format!("Failed to get table count: {}", e)))?;
        
        let table_count = table_count_row.get::<i64, _>(0) as usize;
        
        // Get total row count across all tables
        let table_names = self.get_table_names().await?;
        let mut total_rows = 0u64;
        
        for table_name in &table_names {
            match self.get_row_count(table_name).await {
                Ok(count) => total_rows += count,
                Err(_) => continue, // Skip tables we can't access
            }
        }
        
        Ok(DatabaseInfo {
            name: format!("PostgreSQL ({})", self.database_name),
            version,
            table_count,
            total_rows,
            supports_foreign_keys: true,
            supports_temporal: true,
        })
    }
    
    async fn execute_query(&self, query: &str) -> Result<Vec<Value>> {
        let rows = sqlx::query(query)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| Error::database(format!("Query execution failed: {}", e)))?;
        
        let mut results = Vec::new();
        for row in rows {
            let mut obj = serde_json::Map::new();
            for i in 0..row.len() {
                if let Ok(column_info) = row.try_get_unchecked::<String, _>(i) {
                    obj.insert(format!("col_{}", i), Value::String(column_info));
                }
            }
            results.push(Value::Object(obj));
        }
        
        Ok(results)
    }
    
    async fn get_table_names(&self) -> Result<Vec<String>> {
        let rows = sqlx::query(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name"
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| Error::database(format!("Failed to get table names: {}", e)))?;
        
        Ok(rows.iter().map(|row| row.get::<String, _>("table_name")).collect())
    }
    
    async fn get_row_count(&self, table: &str) -> Result<u64> {
        // Escape table name to prevent SQL injection
        let escaped_table = format!("\"{}\"", table.replace("\"", "\"\""));
        let query = format!("SELECT COUNT(*) FROM {}", escaped_table);
        
        let row = sqlx::query(&query)
            .fetch_one(&self.pool)
            .await
            .map_err(|e| Error::database(format!("Failed to get row count for table {}: {}", table, e)))?;
        
        Ok(row.get::<i64, _>(0) as u64)
    }
    
    async fn test_connection(&self) -> Result<ConnectionHealth> {
        let start = std::time::Instant::now();
        
        match sqlx::query("SELECT 1 as test").fetch_one(&self.pool).await {
            Ok(_) => {
                let latency = start.elapsed().as_millis() as u64;
                Ok(ConnectionHealth {
                    is_healthy: true,
                    latency_ms: latency,
                    last_check: chrono::Utc::now(),
                    error_message: None,
                    connection_count: 1,
                })
            }
            Err(e) => {
                Ok(ConnectionHealth {
                    is_healthy: false,
                    latency_ms: 0,
                    last_check: chrono::Utc::now(),
                    error_message: Some(e.to_string()),
                    connection_count: 0,
                })
            }
        }
    }
    
    async fn execute_prepared(&self, query: &str, _params: &[serde_json::Value]) -> Result<Vec<serde_json::Value>> {
        // For now, just execute the query without parameters
        // TODO: Implement proper parameter binding
        self.execute_query(query).await
    }

    async fn get_table_schema(&self, table: &str) -> Result<TableSchema> {
        let table_info = self.get_table_info_detailed(table).await?;
        
        let columns = table_info.columns.into_iter().map(|col| {
            crate::database::connection::ColumnDefinition {
                name: col.name,
                data_type: col.data_type,
                is_nullable: col.is_nullable,
                default_value: None,
                is_primary_key: col.is_primary_key,
                is_foreign_key: col.is_foreign_key,
                max_length: None,
            }
        }).collect();

        let foreign_keys = table_info.foreign_keys.into_iter().map(|fk| {
            crate::database::connection::ForeignKeyDefinition {
                column: fk.columns.first().unwrap_or(&String::new()).clone(),
                referenced_table: fk.referenced_table,
                referenced_column: fk.referenced_columns.first().unwrap_or(&String::new()).clone(),
                on_delete: None,
                on_update: None,
            }
        }).collect();

        Ok(TableSchema {
            table_name: table.to_string(),
            columns,
            primary_keys: table_info.primary_keys,
            foreign_keys,
            indexes: Vec::new(), // TODO: Implement index detection
        })
    }

    async fn execute_transaction(&self, queries: Vec<String>) -> Result<Vec<serde_json::Value>> {
        let mut tx = self.pool.begin().await
            .map_err(|e| Error::database(format!("Failed to start transaction: {}", e)))?;
        
        let mut results = Vec::new();
        for query in queries {
            let rows = sqlx::query(&query)
                .fetch_all(&mut *tx)
                .await
                .map_err(|e| Error::database(format!("Transaction query failed: {}", e)))?;
            
            // Convert rows to JSON values
            for row in rows {
                let mut map = serde_json::Map::new();
                for i in 0..row.len() {
                    if let Ok(value) = row.try_get_unchecked::<String, _>(i) {
                        map.insert(format!("col_{}", i), serde_json::Value::String(value));
                    }
                }
                results.push(serde_json::Value::Object(map));
            }
        }
        
        tx.commit().await
            .map_err(|e| Error::database(format!("Failed to commit transaction: {}", e)))?;
        
        Ok(results)
    }

    async fn get_database_stats(&self) -> Result<DatabaseStats> {
        let table_names = self.get_table_names().await?;
        let mut total_rows = 0u64;
        
        for table_name in &table_names {
            if let Ok(count) = self.get_row_count(table_name).await {
                total_rows += count;
            }
        }

        Ok(DatabaseStats {
            total_tables: table_names.len() as u32,
            total_rows,
            database_size_bytes: 0, // TODO: Implement size calculation
            active_connections: 1,
            cache_hit_ratio: 0.95,
            slow_queries: 0,
            last_backup: None,
        })
    }

    async fn create_table_if_not_exists(&self, _table_name: &str, _schema: &TableSchema) -> Result<()> {
        // TODO: Implement table creation
        Ok(())
    }

    async fn close(&self) -> Result<()> {
        self.pool.close().await;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_extract_database_name() {
        assert_eq!(
            PostgresConnection::extract_database_name("postgresql://user:pass@localhost/testdb"),
            "testdb"
        );
        assert_eq!(
            PostgresConnection::extract_database_name("postgresql://user:pass@localhost/testdb?sslmode=disable"),
            "testdb"
        );
        assert_eq!(
            PostgresConnection::extract_database_name("postgresql://localhost/mydb"),
            "mydb"
        );
    }
    
    // Integration tests would require a running PostgreSQL instance
    #[ignore]
    #[tokio::test]
    async fn test_postgres_connection() {
        let url = "postgresql://postgres:password@localhost/test";
        let conn = PostgresConnection::new(url).await;
        assert!(conn.is_ok());
    }
}