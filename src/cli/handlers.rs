use std::path::PathBuf;
use tokio::runtime::Runtime;
use crate::cli::{Commands, CliConfig};
use crate::error::Result;
use crate::{get_dataset, get_task};
use crate::server::Server;

/// Command handler for the CLI
pub struct CommandHandler {
    config: CliConfig,
    runtime: Runtime,
}

impl CommandHandler {
    /// Create a new command handler
    pub fn new(config: CliConfig) -> Result<Self> {
        let runtime = Runtime::new()?;
        Ok(Self { config, runtime })
    }
    
    /// Handle a CLI command
    pub fn handle_command(&self, command: &Commands) -> Result<()> {
        match command {
            Commands::List { dataset, task_type } => {
                self.handle_list(dataset.as_deref(), task_type.as_deref())
            }
            Commands::Download { dataset, force } => {
                self.handle_download(dataset, *force)
            }
            Commands::Run {
                dataset,
                task,
                model_config,
                output_dir,
                epochs,
                batch_size,
                learning_rate,
                workers,
                gpu,
            } => {
                self.handle_run(
                    dataset,
                    task,
                    model_config.as_deref(),
                    output_dir.as_deref(),
                    *epochs,
                    *batch_size,
                    *learning_rate,
                    *workers,
                    *gpu,
                )
            }
            Commands::Evaluate {
                dataset,
                task,
                checkpoint,
                output,
            } => {
                self.handle_evaluate(dataset, task, checkpoint, output.as_deref())
            }
            Commands::Serve { host, port, docs } => {
                self.handle_serve(host, *port, *docs)
            }
        }
    }
    
    /// Handle the list command
    fn handle_list(&self, dataset: Option<&str>, task_type: Option<&str>) -> Result<()> {
        let registry = crate::tasks::TaskRegistry::new();
        let tasks = registry.list();
        
        let filtered_tasks: Vec<_> = tasks.into_iter()
            .filter(|task| {
                dataset.map_or(true, |d| task.starts_with(d)) &&
                task_type.map_or(true, |t| {
                    let task = registry.get(task).unwrap();
                    format!("{:?}", task.task_type()).to_lowercase() == t.to_lowercase()
                })
            })
            .collect();
            
        for task in filtered_tasks {
            println!("{}", task);
        }
        
        Ok(())
    }
    
    /// Handle the download command
    fn handle_download(&self, dataset: &str, force: bool) -> Result<()> {
        self.runtime.block_on(async {
            let dataset = get_dataset(dataset, true)?;
            if force {
                dataset.set_cache_dir(None);
            }
            dataset.get_db(true)?;
            Ok(())
        })
    }
    
    /// Handle the run command
    fn handle_run(
        &self,
        dataset: &str,
        task: &str,
        model_config: Option<&PathBuf>,
        output_dir: Option<&PathBuf>,
        epochs: usize,
        batch_size: usize,
        learning_rate: f32,
        workers: Option<usize>,
        gpu: bool,
    ) -> Result<()> {
        self.runtime.block_on(async {
            let task = get_task(dataset, task, true)?;
            
            // Load model config
            let model_config = if let Some(path) = model_config {
                serde_yaml::from_reader(std::fs::File::open(path)?)?
            } else {
                Default::default()
            };
            
            // Configure training
            let train_config = crate::training::TrainConfig {
                epochs,
                batch_size,
                learning_rate,
                workers,
                gpu,
                output_dir: output_dir.map(PathBuf::from),
                ..Default::default()
            };
            
            // Train model
            let trainer = crate::training::Trainer::new(task, model_config, train_config)?;
            trainer.train()?;
            
            Ok(())
        })
    }
    
    /// Handle the evaluate command
    fn handle_evaluate(
        &self,
        dataset: &str,
        task: &str,
        checkpoint: &PathBuf,
        output: Option<&PathBuf>,
    ) -> Result<()> {
        self.runtime.block_on(async {
            let task = get_task(dataset, task, false)?;
            
            // Load model
            let model = crate::models::Model::load(checkpoint)?;
            
            // Get test data
            let test_table = task.get_test_table()?;
            
            // Make predictions
            let predictions = model.predict(&test_table)?;
            
            // Evaluate
            let metrics = task.evaluate(&predictions, None)?;
            
            // Save or print results
            if let Some(path) = output {
                serde_json::to_writer_pretty(
                    std::fs::File::create(path)?,
                    &metrics,
                )?;
            } else {
                println!("Metrics: {:#?}", metrics);
            }
            
            Ok(())
        })
    }
    
    /// Handle the serve command
    fn handle_serve(&self, host: &str, port: u16, docs: bool) -> Result<()> {
        self.runtime.block_on(async {
            let server = Server::new(host, port, docs)?;
            server.run().await
        })
    }
} 