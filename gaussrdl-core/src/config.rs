//! Configuration management for GaussRDL
//! 
//! This module provides comprehensive configuration functionality for managing
//! application settings, dataset configurations, and other configuration needs.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::fs;
use serde::{Deserialize, Serialize};

#[derive(Debug)]
pub enum ConfigError {
    Io(String),
    Serialization(String),
    Deserialization(String),
    Configuration { message: String, field: Option<String> },
    Validation { message: String, field: Option<String> },
    MissingField { field: String },
    InvalidValue { field: String, value: String, reason: String },
}

impl ConfigError {
    pub fn new(message: impl Into<String>) -> Self {
        Self::Configuration { message: message.into(), field: None }
    }
    pub fn with_field(message: impl Into<String>, field: impl Into<String>) -> Self {
        Self::Configuration { message: message.into(), field: Some(field.into()) }
    }
    pub fn validation(message: impl Into<String>) -> Self {
        Self::Validation { message: message.into(), field: None }
    }
    pub fn validation_with_field(message: impl Into<String>, field: impl Into<String>) -> Self {
        Self::Validation { message: message.into(), field: Some(field.into()) }
    }
    pub fn missing_field(field: impl Into<String>) -> Self {
        Self::MissingField { field: field.into() }
    }
    pub fn invalid_value(field: impl Into<String>, value: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::InvalidValue { field: field.into(), value: value.into(), reason: reason.into() }
    }
}

pub type ConfigResult<T> = std::result::Result<T, ConfigError>;

/// Base configuration trait
pub trait Config: Send + Sync + std::fmt::Debug {
    /// Validate the configuration
    fn validate(&self) -> ConfigResult<()>;
    
    /// Get configuration as a string
    fn to_string(&self) -> ConfigResult<String>;
    
    /// Get configuration as JSON
    fn to_json(&self) -> ConfigResult<String>;
    
    /// Get configuration as YAML
    fn to_yaml(&self) -> ConfigResult<String>;
    
    /// Save configuration to file
    fn save_to_file(&self, path: &Path) -> ConfigResult<()>;
    
    /// Load configuration from file
    fn load_from_file(path: &Path) -> ConfigResult<Self>
    where
        Self: Sized;
}

/// Application configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    /// Application name
    pub name: String,
    /// Application version
    pub version: String,
    /// Application description
    pub description: Option<String>,
    /// Logging configuration
    pub logging: LoggingConfig,
    /// Database configuration
    pub database: DatabaseConfig,
    /// Cache configuration
    pub cache: CacheConfig,
    /// Network configuration
    pub network: NetworkConfig,
    /// Security configuration
    pub security: SecurityConfig,
    /// Custom configuration values
    pub custom: HashMap<String, serde_json::Value>,
}

impl AppConfig {
    /// Create a new application configuration
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            description: None,
            logging: LoggingConfig::default(),
            database: DatabaseConfig::default(),
            cache: CacheConfig::default(),
            network: NetworkConfig::default(),
            security: SecurityConfig::default(),
            custom: HashMap::new(),
        }
    }
    
    /// Set description
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
    
    /// Set logging configuration
    pub fn with_logging(mut self, logging: LoggingConfig) -> Self {
        self.logging = logging;
        self
    }
    
    /// Set database configuration
    pub fn with_database(mut self, database: DatabaseConfig) -> Self {
        self.database = database;
        self
    }
    
    /// Set cache configuration
    pub fn with_cache(mut self, cache: CacheConfig) -> Self {
        self.cache = cache;
        self
    }
    
    /// Set network configuration
    pub fn with_network(mut self, network: NetworkConfig) -> Self {
        self.network = network;
        self
    }
    
    /// Set security configuration
    pub fn with_security(mut self, security: SecurityConfig) -> Self {
        self.security = security;
        self
    }
    
    /// Add custom configuration value
    pub fn with_custom(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.custom.insert(key.into(), value);
        self
    }
    
    /// Get custom configuration value
    pub fn get_custom<T>(&self, key: &str) -> ConfigResult<T>
    where
        T: serde::de::DeserializeOwned,
    {
        if let Some(value) = self.custom.get(key) {
            serde_json::from_value(value.clone())
                .map_err(|e| ConfigError::Deserialization(e.to_string()))
        } else {
            Err(ConfigError::missing_field(key))
        }
    }
    
    /// Set custom configuration value
    pub fn set_custom(&mut self, key: impl Into<String>, value: serde_json::Value) {
        self.custom.insert(key.into(), value);
    }
}

impl Config for AppConfig {
    fn validate(&self) -> ConfigResult<()> {
        // Validate required fields
        if self.name.is_empty() {
            return Err(ConfigError::missing_field("name"));
        }
        
        if self.version.is_empty() {
            return Err(ConfigError::missing_field("version"));
        }
        
        // Validate sub-configurations
        self.logging.validate()?;
        self.database.validate()?;
        self.cache.validate()?;
        self.network.validate()?;
        self.security.validate()?;
        
        Ok(())
    }
    
    fn to_string(&self) -> ConfigResult<String> {
        Ok(format!("AppConfig {{ name: {}, version: {} }}", self.name, self.version))
    }
    
    fn to_json(&self) -> ConfigResult<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| ConfigError::Serialization(e.to_string()))
    }
    
    fn to_yaml(&self) -> ConfigResult<String> {
        serde_yaml::to_string(self)
            .map_err(|e| ConfigError::Deserialization(e.to_string()))
    }
    
    fn save_to_file(&self, path: &Path) -> ConfigResult<()> {
        let content = self.to_json()?;
        fs::write(path, content)
            .map_err(|e| ConfigError::Io(e.to_string()))
    }
    
    fn load_from_file(path: &Path) -> ConfigResult<Self> {
        let content = fs::read_to_string(path)
            .map_err(|e| ConfigError::Io(e.to_string()))?;
        
        if path.extension().and_then(|s| s.to_str()) == Some("yaml") {
            serde_yaml::from_str(&content)
                .map_err(|e| ConfigError::Deserialization(e.to_string()))
        } else {
            serde_json::from_str(&content)
                .map_err(|e| ConfigError::Deserialization(e.to_string()))
        }
    }
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level
    pub level: String,
    /// Log file path
    pub file_path: Option<PathBuf>,
    /// Whether to log to console
    pub console: bool,
    /// Log format
    pub format: String,
    /// Maximum log file size in bytes
    pub max_file_size: Option<u64>,
    /// Maximum number of log files to keep
    pub max_files: Option<usize>,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            file_path: None,
            console: true,
            format: "{timestamp} [{level}] {message}".to_string(),
            max_file_size: Some(10 * 1024 * 1024), // 10MB
            max_files: Some(5),
        }
    }
}

impl Config for LoggingConfig {
    fn validate(&self) -> ConfigResult<()> {
        let valid_levels = ["trace", "debug", "info", "warn", "error"];
        if !valid_levels.contains(&self.level.as_str()) {
            return Err(ConfigError::invalid_value(
                "level",
                self.level.clone(),
                format!("Must be one of: {}", valid_levels.join(", "))
            ));
        }
        
        if self.format.is_empty() {
            return Err(ConfigError::missing_field("format"));
        }
        
        Ok(())
    }
    
    fn to_string(&self) -> ConfigResult<String> {
        Ok(format!("LoggingConfig {{ level: {}, console: {} }}", self.level, self.console))
    }
    
    fn to_json(&self) -> ConfigResult<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| ConfigError::Serialization(e.to_string()))
    }
    
    fn to_yaml(&self) -> ConfigResult<String> {
        serde_yaml::to_string(self)
            .map_err(|e| ConfigError::Deserialization(e.to_string()))
    }
    
    fn save_to_file(&self, path: &Path) -> ConfigResult<()> {
        let content = self.to_json()?;
        fs::write(path, content)
            .map_err(|e| ConfigError::Io(e.to_string()))
    }
    
    fn load_from_file(path: &Path) -> ConfigResult<Self> {
        let content = fs::read_to_string(path)
            .map_err(|e| ConfigError::Io(e.to_string()))?;
        
        if path.extension().and_then(|s| s.to_str()) == Some("yaml") {
            serde_yaml::from_str(&content)
                .map_err(|e| ConfigError::Deserialization(e.to_string()))
        } else {
            serde_json::from_str(&content)
                .map_err(|e| ConfigError::Deserialization(e.to_string()))
        }
    }
}

/// Database configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    /// Database type
    pub db_type: String,
    /// Database URL
    pub url: String,
    /// Database name
    pub name: String,
    /// Username
    pub username: Option<String>,
    /// Password
    pub password: Option<String>,
    /// Connection pool size
    pub pool_size: usize,
    /// Connection timeout in seconds
    pub timeout: u64,
    /// Whether to use SSL
    pub ssl: bool,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            db_type: "postgres".to_string(),
            url: "localhost:5432".to_string(),
            name: "gaussrdl".to_string(),
            username: None,
            password: None,
            pool_size: 10,
            timeout: 30,
            ssl: false,
        }
    }
}

impl Config for DatabaseConfig {
    fn validate(&self) -> ConfigResult<()> {
        if self.db_type.is_empty() {
            return Err(ConfigError::missing_field("db_type"));
        }
        
        if self.url.is_empty() {
            return Err(ConfigError::missing_field("url"));
        }
        
        if self.name.is_empty() {
            return Err(ConfigError::missing_field("name"));
        }
        
        if self.pool_size == 0 {
            return Err(ConfigError::invalid_value(
                "pool_size",
                self.pool_size.to_string(),
                "Must be greater than 0".to_string()
            ));
        }
        
        if self.timeout == 0 {
            return Err(ConfigError::invalid_value(
                "timeout",
                self.timeout.to_string(),
                "Must be greater than 0".to_string()
            ));
        }
        
        Ok(())
    }
    
    fn to_string(&self) -> ConfigResult<String> {
        Ok(format!("DatabaseConfig {{ type: {}, url: {}, name: {} }}", self.db_type, self.url, self.name))
    }
    
    fn to_json(&self) -> ConfigResult<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| ConfigError::Serialization(e.to_string()))
    }
    
    fn to_yaml(&self) -> ConfigResult<String> {
        serde_yaml::to_string(self)
            .map_err(|e| ConfigError::Deserialization(e.to_string()))
    }
    
    fn save_to_file(&self, path: &Path) -> ConfigResult<()> {
        let content = self.to_json()?;
        fs::write(path, content)
            .map_err(|e| ConfigError::Io(e.to_string()))
    }
    
    fn load_from_file(path: &Path) -> ConfigResult<Self> {
        let content = fs::read_to_string(path)
            .map_err(|e| ConfigError::Io(e.to_string()))?;
        
        if path.extension().and_then(|s| s.to_str()) == Some("yaml") {
            serde_yaml::from_str(&content)
                .map_err(|e| ConfigError::Deserialization(e.to_string()))
        } else {
            serde_json::from_str(&content)
                .map_err(|e| ConfigError::Deserialization(e.to_string()))
        }
    }
}

/// Cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Cache directory
    pub cache_dir: PathBuf,
    /// Whether to verify downloads
    pub verify_downloads: bool,
    /// Whether to use cache
    pub use_cache: bool,
    /// Maximum cache size in bytes
    pub max_size: Option<u64>,
    /// Cache TTL in seconds
    pub ttl: Option<u64>,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            cache_dir: dirs::cache_dir()
                .unwrap_or_else(|| PathBuf::from("/tmp"))
                .join("gaussrdl"),
            verify_downloads: true,
            use_cache: true,
            max_size: Some(1024 * 1024 * 1024), // 1GB
            ttl: Some(24 * 60 * 60), // 24 hours
        }
    }
}

impl Config for CacheConfig {
    fn validate(&self) -> ConfigResult<()> {
        if self.cache_dir.to_string_lossy().is_empty() {
            return Err(ConfigError::missing_field("cache_dir"));
        }
        
        Ok(())
    }
    
    fn to_string(&self) -> ConfigResult<String> {
        Ok(format!("CacheConfig {{ cache_dir: {:?}, use_cache: {} }}", self.cache_dir, self.use_cache))
    }
    
    fn to_json(&self) -> ConfigResult<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| ConfigError::Serialization(e.to_string()))
    }
    
    fn to_yaml(&self) -> ConfigResult<String> {
        serde_yaml::to_string(self)
            .map_err(|e| ConfigError::Deserialization(e.to_string()))
    }
    
    fn save_to_file(&self, path: &Path) -> ConfigResult<()> {
        let content = self.to_json()?;
        fs::write(path, content)
            .map_err(|e| ConfigError::Io(e.to_string()))
    }
    
    fn load_from_file(path: &Path) -> ConfigResult<Self> {
        let content = fs::read_to_string(path)
            .map_err(|e| ConfigError::Io(e.to_string()))?;
        
        if path.extension().and_then(|s| s.to_str()) == Some("yaml") {
            serde_yaml::from_str(&content)
                .map_err(|e| ConfigError::Deserialization(e.to_string()))
        } else {
            serde_json::from_str(&content)
                .map_err(|e| ConfigError::Deserialization(e.to_string()))
        }
    }
}

/// Network configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// Request timeout in seconds
    pub timeout: u64,
    /// Maximum retries
    pub max_retries: usize,
    /// Retry delay in seconds
    pub retry_delay: u64,
    /// User agent
    pub user_agent: String,
    /// Whether to follow redirects
    pub follow_redirects: bool,
    /// Proxy configuration
    pub proxy: Option<String>,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            timeout: 30,
            max_retries: 3,
            retry_delay: 1,
            user_agent: "GaussRDL/1.0".to_string(),
            follow_redirects: true,
            proxy: None,
        }
    }
}

impl Config for NetworkConfig {
    fn validate(&self) -> ConfigResult<()> {
        if self.timeout == 0 {
            return Err(ConfigError::invalid_value(
                "timeout",
                self.timeout.to_string(),
                "Must be greater than 0".to_string()
            ));
        }
        
        if self.user_agent.is_empty() {
            return Err(ConfigError::missing_field("user_agent"));
        }
        
        Ok(())
    }
    
    fn to_string(&self) -> ConfigResult<String> {
        Ok(format!("NetworkConfig {{ timeout: {}, max_retries: {} }}", self.timeout, self.max_retries))
    }
    
    fn to_json(&self) -> ConfigResult<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| ConfigError::Serialization(e.to_string()))
    }
    
    fn to_yaml(&self) -> ConfigResult<String> {
        serde_yaml::to_string(self)
            .map_err(|e| ConfigError::Deserialization(e.to_string()))
    }
    
    fn save_to_file(&self, path: &Path) -> ConfigResult<()> {
        let content = self.to_json()?;
        fs::write(path, content)
            .map_err(|e| ConfigError::Io(e.to_string()))
    }
    
    fn load_from_file(path: &Path) -> ConfigResult<Self> {
        let content = fs::read_to_string(path)
            .map_err(|e| ConfigError::Io(e.to_string()))?;
        
        if path.extension().and_then(|s| s.to_str()) == Some("yaml") {
            serde_yaml::from_str(&content)
                .map_err(|e| ConfigError::Deserialization(e.to_string()))
        } else {
            serde_json::from_str(&content)
                .map_err(|e| ConfigError::Deserialization(e.to_string()))
        }
    }
}

/// Security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Whether to enable encryption
    pub enable_encryption: bool,
    /// Encryption key
    pub encryption_key: Option<String>,
    /// Whether to enable authentication
    pub enable_auth: bool,
    /// JWT secret
    pub jwt_secret: Option<String>,
    /// JWT expiration in seconds
    pub jwt_expiration: Option<u64>,
    /// Allowed origins for CORS
    pub allowed_origins: Vec<String>,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            enable_encryption: false,
            encryption_key: None,
            enable_auth: false,
            jwt_secret: None,
            jwt_expiration: Some(24 * 60 * 60), // 24 hours
            allowed_origins: vec!["*".to_string()],
        }
    }
}

impl Config for SecurityConfig {
    fn validate(&self) -> ConfigResult<()> {
        if self.enable_encryption && self.encryption_key.is_none() {
            return Err(ConfigError::missing_field("encryption_key"));
        }
        
        if self.enable_auth && self.jwt_secret.is_none() {
            return Err(ConfigError::missing_field("jwt_secret"));
        }
        
        if self.enable_auth && self.jwt_expiration.is_none() {
            return Err(ConfigError::missing_field("jwt_expiration"));
        }
        
        Ok(())
    }
    
    fn to_string(&self) -> ConfigResult<String> {
        Ok(format!("SecurityConfig {{ enable_encryption: {}, enable_auth: {} }}", self.enable_encryption, self.enable_auth))
    }
    
    fn to_json(&self) -> ConfigResult<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| ConfigError::Serialization(e.to_string()))
    }
    
    fn to_yaml(&self) -> ConfigResult<String> {
        serde_yaml::to_string(self)
            .map_err(|e| ConfigError::Deserialization(e.to_string()))
    }
    
    fn save_to_file(&self, path: &Path) -> ConfigResult<()> {
        let content = self.to_json()?;
        fs::write(path, content)
            .map_err(|e| ConfigError::Io(e.to_string()))
    }
    
    fn load_from_file(path: &Path) -> ConfigResult<Self> {
        let content = fs::read_to_string(path)
            .map_err(|e| ConfigError::Io(e.to_string()))?;
        
        if path.extension().and_then(|s| s.to_str()) == Some("yaml") {
            serde_yaml::from_str(&content)
                .map_err(|e| ConfigError::Deserialization(e.to_string()))
        } else {
            serde_json::from_str(&content)
                .map_err(|e| ConfigError::Deserialization(e.to_string()))
        }
    }
}

/// Configuration builder for fluent configuration creation
pub struct ConfigBuilder<T> {
    config: T,
}

impl<T> ConfigBuilder<T> {
    /// Create a new configuration builder
    pub fn new(config: T) -> Self {
        Self { config }
    }
    
    /// Build the configuration
    pub fn build(self) -> ConfigResult<T>
    where
        T: Config,
    {
        self.config.validate()?;
        Ok(self.config)
    }
}

/// Configuration manager for managing multiple configurations
pub struct ConfigManager {
    configs: HashMap<String, Box<dyn Config>>,
}

impl ConfigManager {
    /// Create a new configuration manager
    pub fn new() -> Self {
        Self {
            configs: HashMap::new(),
        }
    }
    
    /// Add a configuration
    pub fn add_config(&mut self, name: impl Into<String>, config: Box<dyn Config>) {
        self.configs.insert(name.into(), config);
    }
    
    /// Get a configuration
    pub fn get_config(&self, name: &str) -> Option<&Box<dyn Config>> {
        self.configs.get(name)
    }
    
    /// Remove a configuration
    pub fn remove_config(&mut self, name: &str) -> Option<Box<dyn Config>> {
        self.configs.remove(name)
    }
    
    /// List all configuration names
    pub fn list_configs(&self) -> Vec<&String> {
        self.configs.keys().collect()
    }
    
    /// Validate all configurations
    pub fn validate_all(&self) -> ConfigResult<()> {
        for (name, config) in &self.configs {
            config.validate()
                .map_err(|e| ConfigError::with_field(format!("{:?}", e), name.clone()))?;
        }
        Ok(())
    }
    
    /// Save all configurations to directory
    pub fn save_all(&self, dir: &Path) -> ConfigResult<()> {
        fs::create_dir_all(dir)
            .map_err(|e| ConfigError::Io(e.to_string()))?;
        
        for (name, config) in &self.configs {
            let file_path = dir.join(format!("{}.json", name));
            config.save_to_file(&file_path)?;
        }
        
        Ok(())
    }
    
    /// Load all configurations from directory
    pub fn load_all(&mut self, dir: &Path) -> ConfigResult<()> {
        if !dir.exists() {
            return Err(ConfigError::new(format!("Directory does not exist: {:?}", dir)));
        }
        
        for entry in fs::read_dir(dir)
            .map_err(|e| ConfigError::Io(e.to_string()))? {
            let entry = entry.map_err(|e| ConfigError::Io(e.to_string()))?;
            let path = entry.path();
            
            if path.is_file() {
                let name = path.file_stem()
                    .and_then(|s| s.to_str())
                    .ok_or_else(|| ConfigError::new("Invalid file name"))?;
                
                // For now, we'll just load as AppConfig
                // In a real implementation, you'd want to determine the type dynamically
                let config: AppConfig = AppConfig::load_from_file(&path)?;
                self.add_config(name.to_string(), Box::new(config));
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
    fn test_app_config_creation() {
        let config = AppConfig::new("test_app", "1.0.0")
            .with_description("Test application")
            .with_logging(LoggingConfig::default())
            .with_database(DatabaseConfig::default());
        
        assert_eq!(config.name, "test_app");
        assert_eq!(config.version, "1.0.0");
        assert!(config.description.is_some());
    }

    #[test]
    fn test_app_config_validation() {
        let config = AppConfig::new("test_app", "1.0.0");
        assert!(config.validate().is_ok());
        
        let invalid_config = AppConfig::new("", "1.0.0");
        assert!(invalid_config.validate().is_err());
    }

    #[test]
    fn test_logging_config_validation() {
        let config = LoggingConfig::default();
        assert!(config.validate().is_ok());
        
        let mut invalid_config = LoggingConfig::default();
        invalid_config.level = "invalid".to_string();
        assert!(invalid_config.validate().is_err());
    }

    #[test]
    fn test_database_config_validation() {
        let config = DatabaseConfig::default();
        assert!(config.validate().is_ok());
        
        let mut invalid_config = DatabaseConfig::default();
        invalid_config.pool_size = 0;
        assert!(invalid_config.validate().is_err());
    }

    #[test]
    fn test_config_serialization() {
        let config = AppConfig::new("test_app", "1.0.0");
        let json = config.to_json().unwrap();
        assert!(json.contains("test_app"));
        assert!(json.contains("1.0.0"));
    }

    #[test]
    fn test_config_save_load() {
        let temp_dir = TempDir::new().unwrap();
        let config = AppConfig::new("test_app", "1.0.0");
        let config_path = temp_dir.path().join("config.json");
        
        config.save_to_file(&config_path).unwrap();
        let loaded_config = AppConfig::load_from_file(&config_path).unwrap();
        
        assert_eq!(config.name, loaded_config.name);
        assert_eq!(config.version, loaded_config.version);
    }

    #[test]
    fn test_config_manager() {
        let mut manager = ConfigManager::new();
        let config = AppConfig::new("test_app", "1.0.0");
        
        manager.add_config("app", Box::new(config));
        assert!(manager.get_config("app").is_some());
        assert_eq!(manager.list_configs(), vec![&"app".to_string()]);
    }

    #[test]
    fn test_config_manager_save_load() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = ConfigManager::new();
        let config = AppConfig::new("test_app", "1.0.0");
        
        manager.add_config("app", Box::new(config));
        manager.save_all(temp_dir.path()).unwrap();
        
        let mut new_manager = ConfigManager::new();
        new_manager.load_all(temp_dir.path()).unwrap();
        
        assert!(new_manager.get_config("app").is_some());
    }
} 