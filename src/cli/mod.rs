// src/cli/mod.rs
use clap::{Parser, Subcommand};
use std::path::PathBuf;

/// RelBench CLI - A Relational Deep Learning Benchmark Tool
#[derive(Parser, Debug)]
#[clap(author, version, about)]
pub struct Cli {
    /// Sets the log level (error, warn, info, debug, trace)
    #[clap(short, long, default_value = "info")]
    pub log_level: String,

    /// Sets the config file path
    #[clap(short, long)]
    pub config: Option<PathBuf>,

    /// Sets the cache directory
    #[clap(long)]
    pub cache_dir: Option<PathBuf>,

    #[clap(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// List available datasets and tasks
    List {
        /// Filter by dataset name
        #[clap(long)]
        dataset: Option<String>,
        
        /// Filter by task type
        #[clap(long)]
        task_type: Option<String>,
    },

    /// Download a dataset
    Download {
        /// Dataset name
        #[clap(required = true)]
        dataset: String,
        
        /// Force re-download even if cached
        #[clap(long)]
        force: bool,
    },

    /// Run a task
    Run {
        /// Dataset name
        #[clap(required = true)]
        dataset: String,
        
        /// Task name
        #[clap(required = true)]
        task: String,
        
        /// Model configuration file
        #[clap(long)]
        model_config: Option<PathBuf>,
        
        /// Output directory for results
        #[clap(long)]
        output_dir: Option<PathBuf>,
        
        /// Number of epochs
        #[clap(long, default_value = "100")]
        epochs: usize,
        
        /// Batch size
        #[clap(long, default_value = "32")]
        batch_size: usize,
        
        /// Learning rate
        #[clap(long, default_value = "0.001")]
        learning_rate: f32,
        
        /// Number of worker threads
        #[clap(long)]
        workers: Option<usize>,
        
        /// Use GPU if available
        #[clap(long)]
        gpu: bool,
    },

    /// Evaluate a trained model
    Evaluate {
        /// Dataset name
        #[clap(required = true)]
        dataset: String,
        
        /// Task name
        #[clap(required = true)]
        task: String,
        
        /// Model checkpoint path
        #[clap(required = true)]
        checkpoint: PathBuf,
        
        /// Output file for metrics
        #[clap(long)]
        output: Option<PathBuf>,
    },

    /// Start the server mode
    Serve {
        /// Host address
        #[clap(long, default_value = "127.0.0.1")]
        host: String,
        
        /// Port number
        #[clap(long, default_value = "8000")]
        port: u16,
        
        /// Enable API documentation
        #[clap(long)]
        docs: bool,
    },
}

/// CLI configuration
#[derive(Debug, Clone)]
pub struct CliConfig {
    pub log_level: String,
    pub config_path: Option<PathBuf>,
    pub cache_dir: Option<PathBuf>,
}

impl From<&Cli> for CliConfig {
    fn from(cli: &Cli) -> Self {
        Self {
            log_level: cli.log_level.clone(),
            config_path: cli.config.clone(),
            cache_dir: cli.cache_dir.clone(),
        }
    }
}