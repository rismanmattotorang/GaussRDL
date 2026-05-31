// src/database/connection.rs
use gaussrdl_core::Result;
use crate::database::{DatabaseType, DatabaseInfo};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Enhanced database connection trait with comprehensive functionality
#[async_trait::async_trait]
pub trait DatabaseConnectionTrait: Send + Sync {
    /// Get database information and metadata
    async fn get_info(&self) -> Result<DatabaseInfo>;
    
    /// Execute a query and return results
    async fn execute_query(&self, query: &str) -> Result<Vec<serde_json::Value>>;
    
    /// Execute a prepared statement with parameters (simplified to avoid erased_serde dependency)
    async fn execute_prepared(&self, query: &str, params: &[serde_json::Value]) -> Result<Vec<serde_json::Value>>;
    
    /// Get table names in the database
    async fn get_table_names(&self) -> Result<Vec<String>>;
    
    /// Get row count for a specific table
    async fn get_row_count(&self, table: &str) -> Result<u64>;
    
    /// Get table schema information
    async fn get_table_schema(&self, table: &str) -> Result<TableSchema>;
    
    /// Test database connection health
    async fn test_connection(&self) -> Result<ConnectionHealth>;
    
    /// Execute a simple transaction with a list of queries
    async fn execute_transaction(&self, queries: Vec<String>) -> Result<Vec<serde_json::Value>>;
    
    /// Get database statistics
    async fn get_database_stats(&self) -> Result<DatabaseStats>;
    
    /// Create a table if it doesn't exist
    async fn create_table_if_not_exists(&self, table_name: &str, schema: &TableSchema) -> Result<()>;
    
    /// Close connection gracefully
    async fn close(&self) -> Result<()>;
}

/// Table schema definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableSchema {
    pub table_name: String,
    pub columns: Vec<ColumnDefinition>,
    pub primary_keys: Vec<String>,
    pub foreign_keys: Vec<ForeignKeyDefinition>,
    pub indexes: Vec<IndexDefinition>,
}

/// Column definition for table schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnDefinition {
    pub name: String,
    pub data_type: String,
    pub is_nullable: bool,
    pub default_value: Option<String>,
    pub is_primary_key: bool,
    pub is_foreign_key: bool,
    pub max_length: Option<usize>,
}

/// Foreign key definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForeignKeyDefinition {
    pub column: String,
    pub referenced_table: String,
    pub referenced_column: String,
    pub on_delete: Option<String>,
    pub on_update: Option<String>,
}

/// Index definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexDefinition {
    pub name: String,
    pub columns: Vec<String>,
    pub is_unique: bool,
    pub index_type: String,
}

/// Connection health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionHealth {
    pub is_healthy: bool,
    pub latency_ms: u64,
    pub last_check: chrono::DateTime<chrono::Utc>,
    pub error_message: Option<String>,
    pub connection_count: u32,
}

/// Database statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseStats {
    pub total_tables: u32,
    pub total_rows: u64,
    pub database_size_bytes: u64,
    pub active_connections: u32,
    pub cache_hit_ratio: f64,
    pub slow_queries: u32,
    pub last_backup: Option<chrono::DateTime<chrono::Utc>>,
}

/// Concrete database connection enum instead of trait object
#[derive(Debug)]
pub enum DatabaseConnectionImpl {
    Postgres(crate::database::postgres::PostgresConnection),
    SurrealDb(crate::database::surrealdb::SurrealConnection),
}

impl DatabaseConnectionImpl {
    /// Get database information and metadata
    pub async fn get_info(&self) -> Result<DatabaseInfo> {
        match self {
            Self::Postgres(conn) => conn.get_info().await,
            Self::SurrealDb(conn) => conn.get_info().await,
        }
    }
    
    /// Execute a query and return results
    pub async fn execute_query(&self, query: &str) -> Result<Vec<serde_json::Value>> {
        match self {
            Self::Postgres(conn) => conn.execute_query(query).await,
            Self::SurrealDb(conn) => conn.execute_query(query).await,
        }
    }
    
    /// Execute a prepared statement with parameters
    pub async fn execute_prepared(&self, query: &str, params: &[serde_json::Value]) -> Result<Vec<serde_json::Value>> {
        match self {
            Self::Postgres(conn) => conn.execute_prepared(query, params).await,
            Self::SurrealDb(conn) => conn.execute_prepared(query, params).await,
        }
    }
    
    /// Get table names in the database
    pub async fn get_table_names(&self) -> Result<Vec<String>> {
        match self {
            Self::Postgres(conn) => conn.get_table_names().await,
            Self::SurrealDb(conn) => conn.get_table_names().await,
        }
    }
    
    /// Get row count for a specific table
    pub async fn get_row_count(&self, table: &str) -> Result<u64> {
        match self {
            Self::Postgres(conn) => conn.get_row_count(table).await,
            Self::SurrealDb(conn) => conn.get_row_count(table).await,
        }
    }
    
    /// Get table schema information
    pub async fn get_table_schema(&self, table: &str) -> Result<TableSchema> {
        match self {
            Self::Postgres(conn) => conn.get_table_schema(table).await,
            Self::SurrealDb(conn) => conn.get_table_schema(table).await,
        }
    }
    
    /// Test database connection health
    pub async fn test_connection(&self) -> Result<ConnectionHealth> {
        match self {
            Self::Postgres(conn) => conn.test_connection().await,
            Self::SurrealDb(conn) => conn.test_connection().await,
        }
    }
    
    /// Execute a simple transaction with a list of queries
    pub async fn execute_transaction(&self, queries: Vec<String>) -> Result<Vec<serde_json::Value>> {
        match self {
            Self::Postgres(conn) => conn.execute_transaction(queries).await,
            Self::SurrealDb(conn) => conn.execute_transaction(queries).await,
        }
    }
    
    /// Get database statistics
    pub async fn get_database_stats(&self) -> Result<DatabaseStats> {
        match self {
            Self::Postgres(conn) => conn.get_database_stats().await,
            Self::SurrealDb(conn) => conn.get_database_stats().await,
        }
    }
    
    /// Create a table if it doesn't exist
    pub async fn create_table_if_not_exists(&self, table_name: &str, schema: &TableSchema) -> Result<()> {
        match self {
            Self::Postgres(conn) => conn.create_table_if_not_exists(table_name, schema).await,
            Self::SurrealDb(conn) => conn.create_table_if_not_exists(table_name, schema).await,
        }
    }
    
    /// Close connection gracefully
    pub async fn close(&self) -> Result<()> {
        match self {
            Self::Postgres(conn) => conn.close().await,
            Self::SurrealDb(conn) => conn.close().await,
        }
    }
}

/// Enhanced database connection with pooling and monitoring
pub struct DatabaseConnection {
    inner: Arc<DatabaseConnectionImpl>,
    db_type: DatabaseType,
    connection_pool: Arc<RwLock<ConnectionPool>>,
    health_monitor: Arc<RwLock<HealthMonitor>>,
    config: ConnectionConfig,
}

/// Connection pool for managing multiple database connections
#[derive(Debug)]
pub struct ConnectionPool {
    connections: Vec<Arc<DatabaseConnectionImpl>>,
    max_connections: usize,
    current_index: usize,
    active_connections: usize,
}

/// Health monitoring for database connections
#[derive(Debug)]
pub struct HealthMonitor {
    last_health_check: Instant,
    check_interval: Duration,
    health_history: Vec<ConnectionHealth>,
    max_history: usize,
}

/// Configuration for database connections
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionConfig {
    pub max_connections: usize,
    pub connection_timeout: Duration,
    pub query_timeout: Duration,
    pub health_check_interval: Duration,
    pub retry_attempts: u32,
    pub retry_delay: Duration,
    pub enable_ssl: bool,
    pub ssl_cert_path: Option<String>,
    pub connection_params: HashMap<String, String>,
}

impl Default for ConnectionConfig {
    fn default() -> Self {
        Self {
            max_connections: 10,
            connection_timeout: Duration::from_secs(30),
            query_timeout: Duration::from_secs(60),
            health_check_interval: Duration::from_secs(30),
            retry_attempts: 3,
            retry_delay: Duration::from_secs(1),
            enable_ssl: false,
            ssl_cert_path: None,
            connection_params: HashMap::new(),
        }
    }
}

impl DatabaseConnection {
    /// Create a new database connection with advanced configuration
    pub async fn new(url: &str, db_type: DatabaseType, config: Option<ConnectionConfig>) -> Result<Self> {
        let config = config.unwrap_or_default();
        
        let inner: Arc<DatabaseConnectionImpl> = match db_type {
            DatabaseType::Postgres => {
                let conn = crate::database::postgres::PostgresConnection::new_with_config(url, &config).await?;
                Arc::new(DatabaseConnectionImpl::Postgres(conn))
            }
            DatabaseType::SurrealDb => {
                let conn = crate::database::surrealdb::SurrealConnection::new_with_config(url, &config).await?;
                Arc::new(DatabaseConnectionImpl::SurrealDb(conn))
            }
        };
        
        let connection_pool = Arc::new(RwLock::new(ConnectionPool::new(config.max_connections)));
        let health_monitor = Arc::new(RwLock::new(HealthMonitor::new(config.health_check_interval)));
        
        let conn = Self {
            inner,
            db_type,
            connection_pool,
            health_monitor,
            config,
        };
        
        // Initialize connection pool
        conn.initialize_pool(url).await?;
        
        // Start health monitoring
        conn.start_health_monitoring().await?;
        
        Ok(conn)
    }
    
    /// Initialize connection pool with multiple connections
    async fn initialize_pool(&self, url: &str) -> Result<()> {
        let mut pool = self.connection_pool.write().await;
        
        for _ in 0..self.config.max_connections {
            let connection: Arc<DatabaseConnectionImpl> = match self.db_type {
                DatabaseType::Postgres => {
                    let conn = crate::database::postgres::PostgresConnection::new_with_config(url, &self.config).await?;
                    Arc::new(DatabaseConnectionImpl::Postgres(conn))
                }
                DatabaseType::SurrealDb => {
                    let conn = crate::database::surrealdb::SurrealConnection::new_with_config(url, &self.config).await?;
                    Arc::new(DatabaseConnectionImpl::SurrealDb(conn))
                }
            };
            pool.add_connection(connection);
        }
        
        Ok(())
    }
    
    /// Start health monitoring background task
    async fn start_health_monitoring(&self) -> Result<()> {
        let health_monitor = self.health_monitor.clone();
        let connection = self.inner.clone();
        let interval = self.config.health_check_interval;
        
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);
            loop {
                ticker.tick().await;
                if let Ok(health) = connection.test_connection().await {
                    let mut monitor = health_monitor.write().await;
                    monitor.record_health_check(health);
                }
            }
        });
        
        Ok(())
    }
    
    /// Get a connection from the pool with load balancing
    async fn get_connection(&self) -> Result<Arc<DatabaseConnectionImpl>> {
        let mut pool = self.connection_pool.write().await;
        pool.get_connection()
    }
    
    /// Execute query with automatic retry and connection pooling
    pub async fn execute_query_robust(&self, query: &str) -> Result<Vec<serde_json::Value>> {
        let mut attempts = 0;
        let max_attempts = self.config.retry_attempts;
        
        while attempts < max_attempts {
            match self.get_connection().await {
                Ok(conn) => {
                    match tokio::time::timeout(
                        self.config.query_timeout,
                        conn.execute_query(query)
                    ).await {
                        Ok(Ok(result)) => return Ok(result),
                        Ok(Err(e)) => {
                            attempts += 1;
                            if attempts >= max_attempts {
                                return Err(e);
                            }
                            tokio::time::sleep(self.config.retry_delay).await;
                        }
                        Err(_) => {
                            attempts += 1;
                            if attempts >= max_attempts {
                                return Err(gaussrdl_core::Error::database("Query timeout"));
                            }
                            tokio::time::sleep(self.config.retry_delay).await;
                        }
                    }
                }
                Err(e) => {
                    attempts += 1;
                    if attempts >= max_attempts {
                        return Err(e);
                    }
                    tokio::time::sleep(self.config.retry_delay).await;
                }
            }
        }
        
        Err(gaussrdl_core::Error::database("Failed to execute query after retries"))
    }
    
    /// Get comprehensive database information
    pub async fn get_info(&self) -> Result<DatabaseInfo> {
        self.inner.get_info().await
    }
    
    /// Execute query with basic functionality
    pub async fn execute_query(&self, query: &str) -> Result<Vec<serde_json::Value>> {
        self.inner.execute_query(query).await
    }
    
    /// Get table names
    pub async fn get_table_names(&self) -> Result<Vec<String>> {
        self.inner.get_table_names().await
    }
    
    /// Get row count for table
    pub async fn get_row_count(&self, table: &str) -> Result<u64> {
        self.inner.get_row_count(table).await
    }
    
    /// Get table schema
    pub async fn get_table_schema(&self, table: &str) -> Result<TableSchema> {
        self.inner.get_table_schema(table).await
    }
    
    /// Get current connection health
    pub async fn get_health(&self) -> Result<ConnectionHealth> {
        let monitor = self.health_monitor.read().await;
        monitor.get_current_health()
    }
    
    /// Get database type
    pub fn get_type(&self) -> DatabaseType {
        self.db_type
    }
    
    /// Get connection configuration
    pub fn get_config(&self) -> &ConnectionConfig {
        &self.config
    }
}

impl ConnectionPool {
    fn new(max_connections: usize) -> Self {
        Self {
            connections: Vec::with_capacity(max_connections),
            max_connections,
            current_index: 0,
            active_connections: 0,
        }
    }
    
    fn add_connection(&mut self, connection: Arc<DatabaseConnectionImpl>) {
        if self.connections.len() < self.max_connections {
            self.connections.push(connection);
        }
    }
    
    fn get_connection(&mut self) -> Result<Arc<DatabaseConnectionImpl>> {
        if self.connections.is_empty() {
            return Err(gaussrdl_core::Error::database("No connections available"));
        }
        
        let connection = self.connections[self.current_index].clone();
        self.current_index = (self.current_index + 1) % self.connections.len();
        self.active_connections += 1;
        
        Ok(connection)
    }
}

impl HealthMonitor {
    fn new(check_interval: Duration) -> Self {
        Self {
            last_health_check: Instant::now(),
            check_interval,
            health_history: Vec::new(),
            max_history: 100,
        }
    }
    
    fn record_health_check(&mut self, health: ConnectionHealth) {
        self.last_health_check = Instant::now();
        self.health_history.push(health);
        
        if self.health_history.len() > self.max_history {
            self.health_history.remove(0);
        }
    }
    
    fn get_current_health(&self) -> Result<ConnectionHealth> {
        self.health_history.last()
            .cloned()
            .ok_or_else(|| gaussrdl_core::Error::database("No health information available"))
    }
}