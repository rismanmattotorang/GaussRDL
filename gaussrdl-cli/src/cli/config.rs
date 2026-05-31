use gaussrdl_core::{Result, Error};
use serde::{Serialize, Deserialize};
use std::path::PathBuf;
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaussRelGTConfig {
    pub version: String,
    pub server: ServerConfig,
    pub database: DatabaseConfig,
    pub training: TrainingConfig,
    pub models: ModelsConfig,
    pub monitoring: MonitoringConfig,
    pub logging: LoggingConfig,
    pub security: SecurityConfig,
    pub performance: PerformanceConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub workers: Option<usize>,
    pub max_connections: usize,
    pub timeout_seconds: u64,
    pub enable_cors: bool,
    pub enable_docs: bool,
    pub static_files_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    pub default_type: String,
    pub connections: HashMap<String, DatabaseConnection>,
    pub pool_settings: PoolSettings,
    pub query_timeout_seconds: u64,
    pub enable_query_logging: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConnection {
    pub db_type: String,
    pub url: String,
    pub name: String,
    pub max_connections: Option<usize>,
    pub enable_ssl: bool,
    pub ssl_cert_path: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolSettings {
    pub min_connections: usize,
    pub max_connections: usize,
    pub connection_timeout_seconds: u64,
    pub idle_timeout_seconds: u64,
    pub max_lifetime_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub default_output_dir: PathBuf,
    pub checkpoint_interval: u32,
    pub enable_tensorboard: bool,
    pub tensorboard_port: u16,
    pub distributed: DistributedConfig,
    pub optimization: OptimizationDefaults,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedConfig {
    pub enable: bool,
    pub backend: String,
    pub init_method: String,
    pub world_size: usize,
    pub rank: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationDefaults {
    pub mixed_precision: bool,
    pub gradient_clipping: Option<f64>,
    pub compile_model: bool,
    pub use_checkpoint: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsConfig {
    pub models_dir: PathBuf,
    pub cache_dir: PathBuf,
    pub auto_save_checkpoints: bool,
    pub checkpoint_format: String,
    pub model_registry_path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    pub enable: bool,
    pub metrics_port: u16,
    pub update_interval_seconds: u64,
    pub retention_days: u32,
    pub enable_profiling: bool,
    pub profile_output_dir: PathBuf,
    pub alert_thresholds: AlertThresholds,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    pub cpu_percent: f64,
    pub memory_percent: f64,
    pub disk_percent: f64,
    pub gpu_memory_percent: f64,
    pub training_loss_plateau_epochs: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    pub level: String,
    pub output_dir: PathBuf,
    pub max_file_size_mb: u64,
    pub max_files: u32,
    pub enable_json_format: bool,
    pub enable_console_output: bool,
    pub modules: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    pub enable_auth: bool,
    pub auth_method: String,
    pub jwt_secret: Option<String>,
    pub session_timeout_minutes: u64,
    pub enable_rate_limiting: bool,
    pub rate_limit_requests_per_minute: u32,
    pub enable_tls: bool,
    pub tls_cert_path: Option<PathBuf>,
    pub tls_key_path: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    pub max_cpu_threads: Option<usize>,
    pub memory_limit_gb: Option<f64>,
    pub enable_memory_mapping: bool,
    pub batch_size_limit: usize,
    pub cache_size_mb: usize,
    pub enable_prefetching: bool,
}

impl Default for GaussRelGTConfig {
    fn default() -> Self {
        Self {
            version: "2.0.0".to_string(),
            server: ServerConfig::default(),
            database: DatabaseConfig::default(),
            training: TrainingConfig::default(),
            models: ModelsConfig::default(),
            monitoring: MonitoringConfig::default(),
            logging: LoggingConfig::default(),
            security: SecurityConfig::default(),
            performance: PerformanceConfig::default(),
        }
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8000,
            workers: None,
            max_connections: 1000,
            timeout_seconds: 30,
            enable_cors: true,
            enable_docs: true,
            static_files_dir: None,
        }
    }
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            default_type: "postgres".to_string(),
            connections: HashMap::new(),
            pool_settings: PoolSettings::default(),
            query_timeout_seconds: 60,
            enable_query_logging: false,
        }
    }
}

impl Default for PoolSettings {
    fn default() -> Self {
        Self {
            min_connections: 1,
            max_connections: 10,
            connection_timeout_seconds: 30,
            idle_timeout_seconds: 600,
            max_lifetime_seconds: 3600,
        }
    }
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            default_output_dir: PathBuf::from("./outputs"),
            checkpoint_interval: 10,
            enable_tensorboard: true,
            tensorboard_port: 6006,
            distributed: DistributedConfig::default(),
            optimization: OptimizationDefaults::default(),
        }
    }
}

impl Default for DistributedConfig {
    fn default() -> Self {
        Self {
            enable: false,
            backend: "nccl".to_string(),
            init_method: "env://".to_string(),
            world_size: 1,
            rank: 0,
        }
    }
}

impl Default for OptimizationDefaults {
    fn default() -> Self {
        Self {
            mixed_precision: true,
            gradient_clipping: Some(1.0),
            compile_model: false,
            use_checkpoint: true,
        }
    }
}

impl Default for ModelsConfig {
    fn default() -> Self {
        Self {
            models_dir: PathBuf::from("./models"),
            cache_dir: PathBuf::from("./cache"),
            auto_save_checkpoints: true,
            checkpoint_format: "safetensors".to_string(),
            model_registry_path: PathBuf::from("./models/registry.json"),
        }
    }
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            enable: true,
            metrics_port: 9090,
            update_interval_seconds: 5,
            retention_days: 30,
            enable_profiling: false,
            profile_output_dir: PathBuf::from("./profiles"),
            alert_thresholds: AlertThresholds::default(),
        }
    }
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            cpu_percent: 90.0,
            memory_percent: 90.0,
            disk_percent: 95.0,
            gpu_memory_percent: 95.0,
            training_loss_plateau_epochs: 20,
        }
    }
}

impl Default for LoggingConfig {
    fn default() -> Self {
        let mut modules = HashMap::new();
        modules.insert("gaussrelgt".to_string(), "info".to_string());
        modules.insert("candle_core".to_string(), "warn".to_string());
        modules.insert("tokio".to_string(), "warn".to_string());
        
        Self {
            level: "info".to_string(),
            output_dir: PathBuf::from("./logs"),
            max_file_size_mb: 100,
            max_files: 10,
            enable_json_format: false,
            enable_console_output: true,
            modules,
        }
    }
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            enable_auth: false,
            auth_method: "bearer".to_string(),
            jwt_secret: None,
            session_timeout_minutes: 60,
            enable_rate_limiting: false,
            rate_limit_requests_per_minute: 100,
            enable_tls: false,
            tls_cert_path: None,
            tls_key_path: None,
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            max_cpu_threads: None,
            memory_limit_gb: None,
            enable_memory_mapping: true,
            batch_size_limit: 1024,
            cache_size_mb: 512,
            enable_prefetching: true,
        }
    }
}

pub struct ConfigManager {
    config: GaussRelGTConfig,
    config_path: Option<PathBuf>,
}

impl ConfigManager {
    pub fn new() -> Self {
        Self {
            config: GaussRelGTConfig::default(),
            config_path: None,
        }
    }

    pub fn from_file(path: &PathBuf) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| Error::other(&format!("Failed to read config file: {}", e)))?;
        
        let config = match path.extension().and_then(|s| s.to_str()) {
            Some("yaml") | Some("yml") => {
                serde_yaml::from_str(&content)
                    .map_err(|e| Error::other(&format!("YAML parse error: {}", e)))?
            }
            Some("json") => {
                serde_json::from_str(&content)
                    .map_err(|e| Error::other(&format!("JSON parse error: {}", e)))?
            }
            Some("toml") => {
                toml::from_str(&content)
                    .map_err(|e| Error::other(&format!("TOML parse error: {}", e)))?
            }
            _ => {
                return Err(Error::other("Unsupported config file format"));
            }
        };

        Ok(Self {
            config,
            config_path: Some(path.clone()),
        })
    }

    pub fn get_config(&self) -> &GaussRelGTConfig {
        &self.config
    }

    pub fn get_config_mut(&mut self) -> &mut GaussRelGTConfig {
        &mut self.config
    }

    pub fn update_value(&mut self, key: &str, value: &str) -> Result<()> {
        let keys: Vec<&str> = key.split('.').collect();
        
        match keys.as_slice() {
            ["server", "host"] => self.config.server.host = value.to_string(),
            ["server", "port"] => {
                self.config.server.port = value.parse()
                    .map_err(|_| Error::other("Invalid port number"))?;
            }
            ["database", "default_type"] => self.config.database.default_type = value.to_string(),
            ["logging", "level"] => self.config.logging.level = value.to_string(),
            ["monitoring", "enable"] => {
                self.config.monitoring.enable = value.parse()
                    .map_err(|_| Error::other("Invalid boolean value"))?;
            }
            _ => {
                return Err(Error::other(&format!("Unknown configuration key: {}", key)));
            }
        }

        Ok(())
    }

    pub fn validate(&self) -> Result<Vec<String>> {
        let mut issues = Vec::new();

        // Validate server config
        if self.config.server.port == 0 {
            issues.push("Server port must be greater than 0".to_string());
        }

        if self.config.server.max_connections == 0 {
            issues.push("Max connections must be greater than 0".to_string());
        }

        // Validate database config
        if self.config.database.pool_settings.max_connections < self.config.database.pool_settings.min_connections {
            issues.push("Database max connections must be >= min connections".to_string());
        }

        // Validate training config
        if !self.config.training.default_output_dir.exists() {
            if let Err(_) = std::fs::create_dir_all(&self.config.training.default_output_dir) {
                issues.push("Cannot create training output directory".to_string());
            }
        }

        // Validate models config
        if !self.config.models.models_dir.exists() {
            if let Err(_) = std::fs::create_dir_all(&self.config.models.models_dir) {
                issues.push("Cannot create models directory".to_string());
            }
        }

        // Validate logging config
        if !self.config.logging.output_dir.exists() {
            if let Err(_) = std::fs::create_dir_all(&self.config.logging.output_dir) {
                issues.push("Cannot create logging directory".to_string());
            }
        }

        Ok(issues)
    }

    pub fn save(&self, format: &str) -> Result<()> {
        let path = self.config_path.as_ref()
            .ok_or_else(|| Error::other("No config file path specified"))?;

        let content = match format {
            "yaml" => serde_yaml::to_string(&self.config)
                .map_err(|e| Error::other(&format!("YAML serialization error: {}", e)))?,
            "json" => serde_json::to_string_pretty(&self.config)
                .map_err(|e| Error::other(&format!("JSON serialization error: {}", e)))?,
            "toml" => toml::to_string(&self.config)
                .map_err(|e| Error::other(&format!("TOML serialization error: {}", e)))?,
            _ => return Err(Error::other("Unsupported format")),
        };

        std::fs::write(path, content)
            .map_err(|e| Error::other(&format!("Failed to write config file: {}", e)))?;

        Ok(())
    }

    pub fn export(&self, path: &PathBuf, format: &str, include_defaults: bool) -> Result<()> {
        let config_to_export = if include_defaults {
            &self.config
        } else {
            // TODO: Create a minimal config with only non-default values
            &self.config
        };

        let content = match format {
            "yaml" => serde_yaml::to_string(config_to_export)
                .map_err(|e| Error::other(&format!("YAML serialization error: {}", e)))?,
            "json" => serde_json::to_string_pretty(config_to_export)
                .map_err(|e| Error::other(&format!("JSON serialization error: {}", e)))?,
            "toml" => toml::to_string(config_to_export)
                .map_err(|e| Error::other(&format!("TOML serialization error: {}", e)))?,
            _ => return Err(Error::other("Unsupported format")),
        };

        std::fs::write(path, content)
            .map_err(|e| Error::other(&format!("Failed to write config file: {}", e)))?;

        Ok(())
    }

    pub fn show_section(&self, section: Option<&str>) -> Result<String> {
        match section {
            None => serde_yaml::to_string(&self.config)
                .map_err(|e| Error::other(&format!("Serialization error: {}", e))),
            Some("server") => serde_yaml::to_string(&self.config.server)
                .map_err(|e| Error::other(&format!("Serialization error: {}", e))),
            Some("database") => serde_yaml::to_string(&self.config.database)
                .map_err(|e| Error::other(&format!("Serialization error: {}", e))),
            Some("training") => serde_yaml::to_string(&self.config.training)
                .map_err(|e| Error::other(&format!("Serialization error: {}", e))),
            Some("models") => serde_yaml::to_string(&self.config.models)
                .map_err(|e| Error::other(&format!("Serialization error: {}", e))),
            Some("monitoring") => serde_yaml::to_string(&self.config.monitoring)
                .map_err(|e| Error::other(&format!("Serialization error: {}", e))),
            Some("logging") => serde_yaml::to_string(&self.config.logging)
                .map_err(|e| Error::other(&format!("Serialization error: {}", e))),
            Some("security") => serde_yaml::to_string(&self.config.security)
                .map_err(|e| Error::other(&format!("Serialization error: {}", e))),
            Some("performance") => serde_yaml::to_string(&self.config.performance)
                .map_err(|e| Error::other(&format!("Serialization error: {}", e))),
            Some(section) => Err(Error::other(&format!("Unknown section: {}", section))),
        }
    }
}

impl Default for ConfigManager {
    fn default() -> Self {
        Self::new()
    }
}

pub fn create_default_config(template: &str) -> Result<GaussRelGTConfig> {
    let mut config = GaussRelGTConfig::default();

    match template {
        "development" => {
            config.logging.level = "debug".to_string();
            config.monitoring.enable = true;
            config.monitoring.enable_profiling = true;
            config.server.enable_docs = true;
            config.security.enable_auth = false;
        }
        "production" => {
            config.logging.level = "info".to_string();
            config.logging.enable_console_output = false;
            config.monitoring.enable = true;
            config.monitoring.enable_profiling = false;
            config.server.enable_docs = false;
            config.security.enable_auth = true;
            config.security.enable_tls = true;
            config.security.enable_rate_limiting = true;
        }
        "minimal" => {
            config.monitoring.enable = false;
            config.logging.level = "warn".to_string();
            config.server.enable_docs = false;
            config.security.enable_auth = false;
        }
        "full" => {
            // Full configuration with all features enabled
            config.monitoring.enable = true;
            config.monitoring.enable_profiling = true;
            config.server.enable_docs = true;
            config.security.enable_auth = true;
            config.security.enable_tls = true;
            config.training.distributed.enable = true;
            config.training.enable_tensorboard = true;
        }
        "default" => {
            // Already set to default
        }
        _ => {
            return Err(Error::other(&format!("Unknown template: {}", template)));
        }
    }

    Ok(config)
} 