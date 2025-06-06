// src/cli/commands.rs
use clap::Parser;
use std::path::PathBuf;
use crate::{Result, database::DatabaseType};
use console::style;
use tracing::{info, error};

#[derive(Parser)]
pub struct ConnectCommand {
    /// Database connection string
    #[arg(short, long)]
    pub url: String,
    
    /// Database type (postgres, surrealdb)
    #[arg(short, long, default_value = "postgres")]
    pub db_type: DatabaseType,
    
    /// Test connection timeout in seconds
    #[arg(long, default_value = "30")]
    pub timeout: u64,
}

impl ConnectCommand {
    pub async fn execute(&self) -> Result<()> {
        use crate::database::DatabaseConnection;
        
        println!("{}", style("Testing database connection...").cyan());
        
        let conn = DatabaseConnection::new(&self.url, self.db_type).await?;
        let info = conn.get_info().await?;
        
        println!("{}", style("✓ Connection successful!").green());
        println!("Database: {} v{}", info.name, info.version);
        println!("Tables: {}", info.table_count);
        println!("Total rows: {}", info.total_rows);
        
        Ok(())
    }
}

#[derive(Parser)]
pub struct SchemaCommand {
    /// Database connection string
    #[arg(short, long)]
    pub url: String,
    
    /// Database type
    #[arg(short, long, default_value = "postgres")]
    pub db_type: DatabaseType,
    
    /// Output file for schema
    #[arg(short, long)]
    pub output: Option<PathBuf>,
    
    /// Include sample data
    #[arg(long)]
    pub include_samples: bool,
    
    /// Maximum sample size per table
    #[arg(long, default_value = "1000")]
    pub sample_size: usize,
}

impl SchemaCommand {
    pub async fn execute(&self) -> Result<()> {
        use crate::database::{DatabaseConnection, SchemaExtractor};
        
        println!("{}", style("Extracting database schema...").cyan());
        
        let conn = DatabaseConnection::new(&self.url, self.db_type).await?;
        let extractor = SchemaExtractor::new(conn);
        let schema = extractor.extract_full_schema(self.include_samples, self.sample_size).await?;
        
        if let Some(output_path) = &self.output {
            schema.save_to_file(output_path)?;
            println!("{} Schema saved to: {}", style("✓").green(), output_path.display());
        } else {
            println!("{}", serde_json::to_string_pretty(&schema)?);
        }
        
        println!("Tables analyzed: {}", schema.tables.len());
        println!("Foreign key relationships: {}", schema.relationships.len());
        
        Ok(())
    }
}

#[derive(Parser)]
pub struct PrepareCommand {
    /// Database connection string
    #[arg(short, long)]
    pub url: String,
    
    /// Database type
    #[arg(short, long, default_value = "postgres")]
    pub db_type: DatabaseType,
    
    /// Schema file (if not provided, will extract from DB)
    #[arg(long)]
    pub schema: Option<PathBuf>,
    
    /// Output directory for prepared data
    #[arg(short, long, default_value = "./data")]
    pub output_dir: PathBuf,
    
    /// Target table for prediction
    #[arg(short, long)]
    pub target_table: String,
    
    /// Target column for prediction
    #[arg(long)]
    pub target_column: String,
    
    /// Task type (classification, regression)
    #[arg(long, default_value = "classification")]
    pub task_type: String,
    
    /// Train/validation/test split ratios
    #[arg(long, default_value = "0.7,0.15,0.15")]
    pub split_ratios: String,
    
    /// Maximum graph size (number of nodes)
    #[arg(long, default_value = "1000000")]
    pub max_graph_size: usize,
    
    /// Enable temporal constraints
    #[arg(long)]
    pub temporal: bool,
    
    /// Timestamp column name
    #[arg(long)]
    pub timestamp_column: Option<String>,
}

impl PrepareCommand {
    pub async fn execute(&self) -> Result<()> {
        use crate::{
            database::{DatabaseConnection, SchemaExtractor},
            graph::GraphBuilder,
            data::DataPipeline,
        };
        use indicatif::{ProgressBar, ProgressStyle};
        
        println!("{}", style("Preparing data for RELGT training...").cyan());
        
        // Load or extract schema
        let schema = if let Some(schema_path) = &self.schema {
            crate::database::DatabaseSchema::load_from_file(schema_path)?
        } else {
            let conn = DatabaseConnection::new(&self.url, self.db_type).await?;
            let extractor = SchemaExtractor::new(conn);
            extractor.extract_full_schema(true, 10000).await?
        };
        
        // Build relational entity graph
        let pb = ProgressBar::new_spinner();
        pb.set_style(ProgressStyle::default_spinner().template("{spinner:.green} {msg}").unwrap());
        pb.set_message("Building relational entity graph...");
        
        let mut builder = GraphBuilder::new(schema);
        builder.set_max_nodes(self.max_graph_size);
        
        if self.temporal {
            if let Some(ts_col) = &self.timestamp_column {
                builder.enable_temporal_constraints(ts_col);
            }
        }
        
        let conn = DatabaseConnection::new(&self.url, self.db_type).await?;
        let graph = builder.build_from_database(&conn).await?;
        
        pb.set_message("Preparing training data...");
        
        // Create data pipeline
        let split_ratios: Vec<f64> = self.split_ratios
            .split(',')
            .map(|s| s.parse().unwrap())
            .collect();
        
        let pipeline = DataPipeline::new()
            .with_target(&self.target_table, &self.target_column)
            .with_task_type(&self.task_type)
            .with_split_ratios(&split_ratios)
            .with_output_dir(&self.output_dir);
        
        let prepared_data = pipeline.prepare(&graph).await?;
        
        pb.finish_with_message("Data preparation completed!");
        
        println!("{} Data prepared successfully!", style("✓").green());
        println!("Training samples: {}", prepared_data.train_size);
        println!("Validation samples: {}", prepared_data.val_size);
        println!("Test samples: {}", prepared_data.test_size);
        println!("Output directory: {}", self.output_dir.display());
        
        Ok(())
    }
}

#[derive(Parser)]
pub struct TrainCommand {
    /// Prepared data directory
    #[arg(short, long, default_value = "./data")]
    pub data_dir: PathBuf,
    
    /// Model configuration file
    #[arg(short, long)]
    pub model_config: Option<PathBuf>,
    
    /// Output directory for model checkpoints
    #[arg(short, long, default_value = "./checkpoints")]
    pub output_dir: PathBuf,
    
    /// Number of training epochs
    #[arg(long, default_value = "100")]
    pub epochs: usize,
    
    /// Batch size
    #[arg(long, default_value = "256")]
    pub batch_size: usize,
    
    /// Learning rate
    #[arg(long, default_value = "1e-4")]
    pub learning_rate: f64,
    
    /// Number of warmup steps
    #[arg(long, default_value = "1000")]
    pub warmup_steps: usize,
    
    /// Gradient clipping threshold
    #[arg(long, default_value = "1.0")]
    pub grad_clip: f64,
    
    /// Early stopping patience
    #[arg(long, default_value = "10")]
    pub patience: usize,
    
    /// Validation frequency (epochs)
    #[arg(long, default_value = "1")]
    pub val_freq: usize,
    
    /// Save checkpoint frequency (epochs)
    #[arg(long, default_value = "5")]
    pub save_freq: usize,
    
    /// Resume from checkpoint
    #[arg(long)]
    pub resume: Option<PathBuf>,
    
    /// Enable mixed precision training
    #[arg(long)]
    pub mixed_precision: bool,
    
    /// Enable distributed training
    #[arg(long)]
    pub distributed: bool,
}

impl TrainCommand {
    pub async fn execute(&self) -> Result<()> {
        use crate::{
            model::{RelgtModel, RelgtConfig},
            training::{RelgtTrainer, TrainingConfig},
            data::DataLoader,
        };
        
        println!("{}", style("Starting RELGT training...").cyan());
        
        // Load prepared data
        let data_loader = DataLoader::from_directory(&self.data_dir)?;
        let (train_data, val_data, _test_data) = data_loader.load_splits()?;
        
        // Load or create model configuration
        let model_config = if let Some(config_path) = &self.model_config {
            RelgtConfig::load_from_file(config_path)?
        } else {
            RelgtConfig::default_for_data(&train_data)?
        };
        
        // Create training configuration
        let training_config = TrainingConfig {
            epochs: self.epochs,
            batch_size: self.batch_size,
            learning_rate: self.learning_rate,
            warmup_steps: self.warmup_steps,
            grad_clip: self.grad_clip,
            patience: self.patience,
            val_freq: self.val_freq,
            save_freq: self.save_freq,
            mixed_precision: self.mixed_precision,
            distributed: self.distributed,
            output_dir: self.output_dir.clone(),
            resume_checkpoint: self.resume.clone(),
            ..Default::default()
        };
        
        // Initialize model and trainer
        let device = crate::utils::get_device()?;
        let model = RelgtModel::new(model_config, device.clone())?;
        let mut trainer = RelgtTrainer::new(model, training_config, device)?;
        
        // Start training
        let training_result = trainer.train(train_data, val_data).await?;
        
        println!("{} Training completed!", style("✓").green());
        println!("Best validation score: {:.4}", training_result.best_val_score);
        println!("Total training time: {:?}", training_result.total_time);
        println!("Model saved to: {}", self.output_dir.display());
        
        Ok(())
    }
}

#[derive(Parser)]
pub struct EvaluateCommand {
    /// Model checkpoint path
    #[arg(short, long)]
    pub model_path: PathBuf,
    
    /// Test data directory
    #[arg(short, long, default_value = "./data")]
    pub data_dir: PathBuf,
    
    /// Output file for evaluation results
    #[arg(short, long)]
    pub output: Option<PathBuf>,
    
    /// Batch size for evaluation
    #[arg(long, default_value = "512")]
    pub batch_size: usize,
    
    /// Enable detailed analysis
    #[arg(long)]
    pub detailed: bool,
    
    /// Generate visualizations
    #[arg(long)]
    pub visualize: bool,
    
    /// Export predictions
    #[arg(long)]
    pub export_predictions: bool,
}

impl EvaluateCommand {
    pub async fn execute(&self) -> Result<()> {
        use crate::{
            model::RelgtModel,
            evaluation::{RelgtEvaluator, EvaluationConfig},
            data::DataLoader,
        };
        
        println!("{}", style("Evaluating RELGT model...").cyan());
        
        // Load model and data
        let device = crate::utils::get_device()?;
        let model = RelgtModel::load_from_checkpoint(&self.model_path, device.clone())?;
        let data_loader = DataLoader::from_directory(&self.data_dir)?;
        let (_train_data, _val_data, test_data) = data_loader.load_splits()?;
        
        // Create evaluation configuration
        let eval_config = EvaluationConfig {
            batch_size: self.batch_size,
            detailed_analysis: self.detailed,
            generate_visualizations: self.visualize,
            export_predictions: self.export_predictions,
            output_path: self.output.clone(),
            ..Default::default()
        };
        
        // Run evaluation
        let evaluator = RelgtEvaluator::new(model, eval_config, device)?;
        let results = evaluator.evaluate(test_data).await?;
        
        // Display and save results
        results.print_summary();
        
        if let Some(output_path) = &self.output {
            results.save_to_file(output_path)?;
            println!("Results saved to: {}", output_path.display());
        }
        
        Ok(())
    }
}

#[derive(Parser)]
pub struct PredictCommand {
    /// Model checkpoint path
    #[arg(short, long)]
    pub model_path: PathBuf,
    
    /// Input data (JSON, CSV, or database URL)
    #[arg(short, long)]
    pub input: String,
    
    /// Output file for predictions
    #[arg(short, long)]
    pub output: PathBuf,
    
    /// Input format (json, csv, database)
    #[arg(long, default_value = "json")]
    pub input_format: String,
    
    /// Batch size for inference
    #[arg(long, default_value = "1024")]
    pub batch_size: usize,
    
    /// Include prediction confidence scores
    #[arg(long)]
    pub include_confidence: bool,
    
    /// Include attention weights
    #[arg(long)]
    pub include_attention: bool,
}

impl PredictCommand {
    pub async fn execute(&self) -> Result<()> {
        use crate::{
            model::RelgtModel,
            data::PredictionPipeline,
        };
        
        println!("{}", style("Making predictions with RELGT model...").cyan());
        
        // Load model and create prediction pipeline
        let device = crate::utils::get_device()?;
        let model = RelgtModel::load_from_checkpoint(&self.model_path, device.clone())?;
        
        let pipeline = PredictionPipeline::new(model, device)
            .with_batch_size(self.batch_size)
            .with_confidence_scores(self.include_confidence)
            .with_attention_weights(self.include_attention);
        
        // Make predictions based on input format
        let predictions = match self.input_format.as_str() {
            "json" => pipeline.predict_from_json(&self.input).await?,
            "csv" => pipeline.predict_from_csv(&self.input).await?,
            "database" => pipeline.predict_from_database(&self.input).await?,
            _ => return Err(crate::GaussRelgtError::InvalidConfig {
                field: "input_format".to_string(),
                reason: "Must be one of: json, csv, database".to_string(),
            }),
        };
        
        // Save predictions
        predictions.save_to_file(&self.output)?;
        
        println!("{} Predictions completed!", style("✓").green());
        println!("Processed {} samples", predictions.len());
        println!("Results saved to: {}", self.output.display());
        
        Ok(())
    }
}

#[derive(Parser)]
pub struct ExportCommand {
    /// Model checkpoint path
    #[arg(short, long)]
    pub model_path: PathBuf,
    
    /// Export format (onnx, torchscript, safetensors)
    #[arg(short, long, default_value = "onnx")]
    pub format: String,
    
    /// Output file path
    #[arg(short, long)]
    pub output: PathBuf,
    
    /// Optimize for inference
    #[arg(long)]
    pub optimize: bool,
    
    /// Target device for optimization (cpu, gpu)
    #[arg(long, default_value = "cpu")]
    pub target_device: String,
}

impl ExportCommand {
    pub async fn execute(&self) -> Result<()> {
        use crate::model::{RelgtModel, ModelExporter};
        
        println!("{}", style("Exporting RELGT model...").cyan());
        
        // Load model and create exporter
        let device = crate::utils::get_device()?;
        let model = RelgtModel::load_from_checkpoint(&self.model_path, device)?;
        
        let exporter = ModelExporter::new(model)
            .with_optimization(self.optimize)
            .with_target_device(&self.target_device);
        
        // Export model
        match self.format.as_str() {
            "onnx" => exporter.export_onnx(&self.output).await?,
            "torchscript" => exporter.export_torchscript(&self.output).await?,
            "safetensors" => exporter.export_safetensors(&self.output).await?,
            _ => return Err(crate::GaussRelgtError::InvalidConfig {
                field: "format".to_string(),
                reason: "Must be one of: onnx, torchscript, safetensors".to_string(),
            }),
        }
        
        println!("{} Model exported successfully!", style("✓").green());
        println!("Format: {}", self.format);
        println!("Output: {}", self.output.display());
        
        Ok(())
    }
}

#[derive(Parser)]
pub struct BenchmarkCommand {
    /// Benchmark suite to run
    #[arg(short, long, default_value = "all")]
    pub suite: String,
    
    /// Number of benchmark iterations
    #[arg(long, default_value = "10")]
    pub iterations: usize,
    
    /// Warmup iterations
    #[arg(long, default_value = "3")]
    pub warmup: usize,
    
    /// Output file for benchmark results
    #[arg(short, long)]
    pub output: Option<PathBuf>,
    
    /// Compare with baseline models
    #[arg(long)]
    pub compare_baselines: bool,
    
    /// Generate performance profile
    #[arg(long)]
    pub profile: bool,
}

impl BenchmarkCommand {
    pub async fn execute(&self) -> Result<()> {
        use crate::evaluation::BenchmarkSuite;
        
        println!("{}", style("Running RELGT benchmarks...").cyan());
        
        let suite = BenchmarkSuite::new()
            .with_iterations(self.iterations)
            .with_warmup(self.warmup)
            .with_profiling(self.profile)
            .with_baseline_comparison(self.compare_baselines);
        
        let results = match self.suite.as_str() {
            "all" => suite.run_all_benchmarks().await?,
            "training" => suite.run_training_benchmarks().await?,
            "inference" => suite.run_inference_benchmarks().await?,
            "memory" => suite.run_memory_benchmarks().await?,
            _ => return Err(crate::GaussRelgtError::InvalidConfig {
                field: "suite".to_string(),
                reason: "Must be one of: all, training, inference, memory".to_string(),
            }),
        };
        
        // Display and save results
        results.print_summary();
        
        if let Some(output_path) = &self.output {
            results.save_to_file(output_path)?;
            println!("Benchmark results saved to: {}", output_path.display());
        }
        
        Ok(())
    }
}
