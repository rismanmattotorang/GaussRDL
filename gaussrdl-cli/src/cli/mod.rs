// src/cli/mod.rs
use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;
use serde::{Serialize, Deserialize};

pub mod handlers;
pub mod monitoring;
pub mod metrics;
pub mod models;
pub mod config;

/// GaussRelGT CLI - Advanced Gaussian Relational Learning Toolkit
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
#[clap(name = "gaussrelgt")]
#[clap(version = "2.0.0")]
pub struct Cli {
    /// Sets the log level
    #[clap(short, long, value_enum, default_value = "info")]
    pub log_level: LogLevel,

    /// Sets the config file path
    #[clap(short, long)]
    pub config: Option<PathBuf>,

    /// Output format for commands
    #[clap(short, long, value_enum, default_value = "table")]
    pub format: OutputFormat,

    /// Enable verbose output
    #[clap(short, long)]
    pub verbose: bool,

    #[clap(subcommand)]
    pub command: Commands,
}

#[derive(ValueEnum, Clone, Debug)]
pub enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

#[derive(ValueEnum, Clone, Debug)]
pub enum OutputFormat {
    Table,
    Json,
    Yaml,
    Csv,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Server management commands
    #[clap(subcommand)]
    Server(ServerCommands),

    /// Database management commands  
    #[clap(subcommand)]
    Database(DatabaseCommands),

    /// Dataset management commands
    #[clap(subcommand)]
    Dataset(DatasetCommands),

    /// Model management commands
    #[clap(subcommand)]
    Model(ModelCommands),

    /// Training management commands
    #[clap(subcommand)]
    Training(TrainingCommands),

    /// Monitoring and metrics commands
    #[clap(subcommand)]
    Monitor(MonitorCommands),

    /// System diagnostics and health checks
    #[clap(subcommand)]
    System(SystemCommands),
}

#[derive(Subcommand, Debug)]
pub enum ServerCommands {
    /// Start the server
    Start {
        /// Host address
        #[clap(long, default_value = "127.0.0.1")]
        host: String,
        
        /// Port number
        #[clap(long, default_value = "8000")]
        port: u16,
        
        /// Enable API documentation
        #[clap(long)]
        docs: bool,

        /// Enable HTTPS
        #[clap(long)]
        https: bool,

        /// Maximum concurrent connections
        #[clap(long, default_value = "1000")]
        max_connections: usize,

        /// Worker threads
        #[clap(long)]
        workers: Option<usize>,
    },

    /// Stop the server
    Stop {
        /// Server port to stop
        #[clap(long, default_value = "8000")]
        port: u16,

        /// Force stop
        #[clap(long)]
        force: bool,
    },

    /// Check server status
    Status {
        /// Server endpoint
        #[clap(long, default_value = "http://127.0.0.1:8000")]
        endpoint: String,
    },
}

#[derive(Subcommand, Debug)]
pub enum DatabaseCommands {
    /// Test database connection
    Test {
        /// Database type
        #[clap(long, value_enum)]
        db_type: DatabaseType,
        
        /// Connection URL
        #[clap(long)]
        url: String,

        /// Connection timeout
        #[clap(long, default_value = "30")]
        timeout: u64,
    },

    /// Get database schema
    Schema {
        /// Database type
        #[clap(long, value_enum)]
        db_type: DatabaseType,
        
        /// Connection URL
        #[clap(long)]
        url: String,
        
        /// Specific table name
        #[clap(long)]
        table: Option<String>,
    },

    /// List database tables
    Tables {
        /// Database type
        #[clap(long, value_enum)]
        db_type: DatabaseType,
        
        /// Connection URL
        #[clap(long)]
        url: String,

        /// Show table sizes
        #[clap(long)]
        show_sizes: bool,
    },

    /// Execute SQL query
    Query {
        /// Database type
        #[clap(long, value_enum)]
        db_type: DatabaseType,
        
        /// Connection URL
        #[clap(long)]
        url: String,

        /// SQL query string
        #[clap(long)]
        sql: Option<String>,

        /// SQL file path
        #[clap(long)]
        file: Option<PathBuf>,
    },

    /// Monitor database performance
    Monitor {
        /// Database type
        #[clap(long, value_enum)]
        db_type: DatabaseType,
        
        /// Connection URL
        #[clap(long)]
        url: String,

        /// Monitoring duration in seconds
        #[clap(long, default_value = "60")]
        duration: u64,
    },
}

#[derive(ValueEnum, Clone, Debug)]
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

#[derive(Subcommand, Debug)]
pub enum DatasetCommands {
    /// List available datasets
    List {
        /// Filter by dataset name pattern
        #[clap(long)]
        pattern: Option<String>,
        
        /// Show dataset details
        #[clap(long)]
        details: bool,

        /// Show cached datasets only
        #[clap(long)]
        cached_only: bool,
    },

    /// Download datasets
    Download {
        /// Dataset name
        dataset: String,
        
        /// Force re-download
        #[clap(long)]
        force: bool,

        /// Download to specific directory
        #[clap(long)]
        output_dir: Option<PathBuf>,

        /// Verify download integrity
        #[clap(long)]
        verify: bool,
    },

    /// Dataset information and statistics
    Info {
        /// Dataset name
        dataset: String,

        /// Show detailed statistics
        #[clap(long)]
        detailed: bool,

        /// Generate data profile
        #[clap(long)]
        profile: bool,
    },

    /// Validate dataset integrity
    Validate {
        /// Dataset name
        dataset: String,

        /// Fix issues automatically
        #[clap(long)]
        fix: bool,
    },
}

#[derive(Subcommand, Debug)]
pub enum ModelCommands {
    /// List available models
    List {
        /// Filter by model type
        #[clap(long)]
        model_type: Option<String>,

        /// Show trained models only
        #[clap(long)]
        trained_only: bool,

        /// Include model metrics
        #[clap(long)]
        include_metrics: bool,
    },

    /// Create a new model
    Create {
        /// Model name
        name: String,

        /// Model type
        #[clap(long, value_enum)]
        model_type: ModelType,

        /// Model configuration file
        #[clap(long)]
        config: PathBuf,
    },

    /// Model information
    Info {
        /// Model name or path
        model: String,

        /// Show detailed architecture
        #[clap(long)]
        architecture: bool,

        /// Show parameter details
        #[clap(long)]
        parameters: bool,
    },

    /// Validate model configuration
    Validate {
        /// Model configuration file
        config: PathBuf,

        /// Check compatibility with dataset
        #[clap(long)]
        dataset: Option<String>,
    },

    /// Export model
    Export {
        /// Model name or path
        model: String,

        /// Export format
        #[clap(long, value_enum, default_value = "onnx")]
        format: ExportFormat,

        /// Output path
        #[clap(long)]
        output: PathBuf,
    },
}

#[derive(ValueEnum, Clone, Debug)]
pub enum ModelType {
    BaseGnn,
    Gat,
    Rgcn,
    LightRdl,
    StageGnn,
}

#[derive(ValueEnum, Clone, Debug)]
pub enum ExportFormat {
    Onnx,
    Json,
}

#[derive(Subcommand, Debug)]
pub enum TrainingCommands {
    /// Start training
    Start {
        /// Model configuration
        #[clap(long)]
        model_config: PathBuf,

        /// Training configuration
        #[clap(long)]
        train_config: PathBuf,

        /// Dataset name
        #[clap(long)]
        dataset: String,

        /// Output directory
        #[clap(long)]
        output_dir: PathBuf,

        /// Resume from checkpoint
        #[clap(long)]
        resume: Option<PathBuf>,

        /// Enable distributed training
        #[clap(long)]
        distributed: bool,
    },

    /// Monitor training progress
    Monitor {
        /// Training job ID
        job_id: String,

        /// Refresh interval in seconds
        #[clap(long, default_value = "5")]
        interval: u64,
    },

    /// Stop training
    Stop {
        /// Training job ID
        job_id: String,

        /// Force stop
        #[clap(long)]
        force: bool,
    },

    /// List training jobs
    Jobs {
        /// Show all jobs (including completed)
        #[clap(long)]
        all: bool,

        /// Filter by status
        #[clap(long)]
        status: Option<String>,
    },
}

#[derive(Subcommand, Debug)]
pub enum MonitorCommands {
    /// System monitoring
    System {
        /// Monitoring duration in seconds
        #[clap(long, default_value = "60")]
        duration: u64,

        /// Refresh interval in seconds
        #[clap(long, default_value = "1")]
        interval: u64,

        /// Include GPU metrics
        #[clap(long)]
        gpu: bool,
    },

    /// Model performance monitoring
    Model {
        /// Model name or path
        model: String,

        /// Dataset for evaluation
        #[clap(long)]
        dataset: String,

        /// Monitoring duration
        #[clap(long, default_value = "300")]
        duration: u64,
    },

    /// Training monitoring dashboard
    Training {
        /// Training job ID
        job_id: String,

        /// Enable real-time updates
        #[clap(long)]
        realtime: bool,

        /// Dashboard port
        #[clap(long, default_value = "8080")]
        port: u16,
    },
}

#[derive(Subcommand, Debug)]
pub enum SystemCommands {
    /// System health check
    Health {
        /// Comprehensive check
        #[clap(long)]
        comprehensive: bool,

        /// Check external dependencies
        #[clap(long)]
        external: bool,
    },

    /// System diagnostics
    Diagnostics {
        /// Include system logs
        #[clap(long)]
        logs: bool,

        /// Include performance metrics
        #[clap(long)]
        performance: bool,
    },

    /// Resource usage analysis
    Resources {
        /// Analysis duration in seconds
        #[clap(long, default_value = "60")]
        duration: u64,

        /// Include process details
        #[clap(long)]
        processes: bool,
    },
}

/// Enhanced CLI configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CliConfig {
    pub log_level: String,
    pub config_path: Option<PathBuf>,
    pub output_format: String,
    pub verbose: bool,
}

impl Default for CliConfig {
    fn default() -> Self {
        Self {
            log_level: "info".to_string(),
            config_path: None,
            output_format: "table".to_string(),
            verbose: false,
        }
    }
}

impl From<&Cli> for CliConfig {
    fn from(cli: &Cli) -> Self {
        Self {
            log_level: format!("{:?}", cli.log_level).to_lowercase(),
            config_path: cli.config.clone(),
            output_format: format!("{:?}", cli.format).to_lowercase(),
            verbose: cli.verbose,
        }
    }
}