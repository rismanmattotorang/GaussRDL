// src/main.rs
use clap::Parser;
use env_logger::Env;

mod cli;
mod server;
mod base;
mod tasks;
mod error;
mod metrics;
mod models;
mod training;

use cli::{Cli, CliConfig, Commands};
use error::Result;

#[tokio::main]
async fn main() -> Result<()> {
    // Parse command line arguments
    let cli = Cli::parse();
    
    // Initialize logging
    env_logger::Builder::from_env(Env::default().default_filter_or(&cli.log_level))
        .init();
        
    // Create config
    let config = CliConfig::from(&cli);
    
    // Create command handler
    let handler = cli::handlers::CommandHandler::new(config)?;
    
    // Handle command
    handler.handle_command(&cli.command)?;
    
    Ok(())
}

pub use base::{Dataset, Database, Table};
pub use tasks::{Task, TaskProvider, TaskType};
pub use error::Error;

// Re-export commonly used functions
pub use tasks::get_task;
pub use base::get_dataset;
