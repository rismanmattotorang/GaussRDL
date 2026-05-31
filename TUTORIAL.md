# GaussRDL Tutorial

## Overview

This tutorial provides step-by-step examples for using GaussRDL, from basic setup to advanced features. Follow along to learn how to work with relational graph neural networks effectively.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Basic Training](#basic-training)
3. [Advanced Training](#advanced-training)
4. [Model Evaluation](#model-evaluation)
5. [Inference and Deployment](#inference-and-deployment)
6. [Custom Models](#custom-models)
7. [Performance Optimization](#performance-optimization)
8. [Real-world Examples](#real-world-examples)

## Getting Started

### Prerequisites

Before starting this tutorial, ensure you have:

- Rust 1.70+ installed
- 8GB+ RAM available
- CUDA 11.8+ (optional, for GPU acceleration)
- Basic understanding of graph neural networks

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/gaussrdl.git
cd gaussrdl

# Build the project
cargo build --release

# Verify installation
cargo run --bin gaussrdl -- --version
```

### First Steps

Let's start with a simple example to verify everything is working:

```rust
// examples/first_steps.rs
use gaussrdl::*;

fn main() -> Result<()> {
    println!("Welcome to GaussRDL!");
    
    // Check available datasets
    let datasets = list_available_datasets();
    println!("Available datasets: {:?}", datasets);
    
    // Check available models
    let models = list_available_models();
    println!("Available models: {:?}", models);
    
    // Check available tasks
    let tasks = list_available_tasks();
    println!("Available tasks: {:?}", tasks);
    
    Ok(())
}
```

Run this example:

```bash
cargo run --example first_steps
```

## Basic Training

### Tutorial 1: Training a Simple RGCN Model

In this tutorial, we'll train a Relational Graph Convolutional Network (RGCN) on the Amazon dataset for user churn prediction.

#### Step 1: Load Data

```rust
// examples/tutorial_1_basic_training.rs
use gaussrdl::*;

fn main() -> Result<()> {
    println!("Tutorial 1: Basic RGCN Training");
    
    // Load the Amazon dataset
    println!("Loading Amazon dataset...");
    let dataset = get_dataset("rel-amazon", true)?;
    println!("Dataset loaded: {}", dataset.name());
    
    // Get the user churn task
    println!("Loading user churn task...");
    let task = get_task("rel-amazon", "user-churn", true)?;
    println!("Task loaded: {}", task.name());
    
    // Display dataset information
    println!("Dataset splits: {:?}", dataset.splits());
    println!("Task metrics: {:?}", task.metrics());
    
    Ok(())
}
```

#### Step 2: Create Model Configuration

```rust
// Continue in the same file
fn create_rgcn_config() -> UnifiedModelConfig {
    let mut config = UnifiedModelConfig::default();
    
    // Model architecture
    config.model_type = ModelType::RGCN;
    config.hidden_dim = 128;
    config.num_layers = 3;
    config.num_relations = 5;
    config.num_bases = 4;
    config.dropout = 0.1;
    config.use_layer_norm = true;
    config.use_residual = true;
    
    // Training parameters
    config.learning_rate = 0.001;
    config.batch_size = 32;
    config.epochs = 50;
    config.early_stopping = 10;
    config.gradient_clipping = Some(1.0);
    
    // Device configuration
    config.device.device_type = DeviceType::Auto;
    
    config
}
```

#### Step 3: Train the Model

```rust
// Continue in the same file
fn train_model(dataset: &dyn Dataset, task: &dyn Task) -> Result<()> {
    println!("Creating RGCN model...");
    let config = create_rgcn_config();
    let model = create_model(config.clone())?;
    
    println!("Model created with {} parameters", model.parameter_count());
    
    // Create trainer
    println!("Starting training...");
    let trainer = Trainer::new(config);
    
    // Train the model
    let metrics = trainer.train(&model, dataset, task)?;
    
    // Print results
    println!("Training completed!");
    println!("Final metrics:");
    for (metric, value) in metrics {
        println!("  {}: {:.4}", metric, value);
    }
    
    // Save the model
    println!("Saving model...");
    model.save("tutorial_1_rgcn_model.pt")?;
    println!("Model saved to tutorial_1_rgcn_model.pt");
    
    Ok(())
}
```

#### Step 4: Complete Example

```rust
fn main() -> Result<()> {
    println!("Tutorial 1: Basic RGCN Training");
    
    // Load data
    let dataset = get_dataset("rel-amazon", true)?;
    let task = get_task("rel-amazon", "user-churn", true)?;
    
    // Train model
    train_model(&dataset, &task)?;
    
    println!("Tutorial 1 completed successfully!");
    Ok(())
}
```

Run the tutorial:

```bash
cargo run --example tutorial_1_basic_training
```

### Tutorial 2: Training with Different Models

Let's compare different model architectures on the same task.

#### Step 1: Model Comparison Function

```rust
// examples/tutorial_2_model_comparison.rs
use gaussrdl::*;
use std::collections::HashMap;

fn compare_models(dataset: &dyn Dataset, task: &dyn Task) -> Result<HashMap<String, f64>> {
    let models = vec![
        ("RGCN", create_rgcn_config()),
        ("GAT", create_gat_config()),
        ("LightRDL", create_lightrdl_config()),
    ];
    
    let mut results = HashMap::new();
    
    for (model_name, config) in models {
        println!("Training {} model...", model_name);
        
        let model = create_model(config.clone())?;
        let trainer = Trainer::new(config);
        let metrics = trainer.train(&model, dataset, task)?;
        
        // Store the main metric (e.g., accuracy)
        if let Some(accuracy) = metrics.get("accuracy") {
            results.insert(model_name.to_string(), *accuracy);
        }
        
        // Save model
        model.save(&format!("tutorial_2_{}_model.pt", model_name.to_lowercase()))?;
    }
    
    Ok(results)
}

fn create_rgcn_config() -> UnifiedModelConfig {
    let mut config = UnifiedModelConfig::default();
    config.model_type = ModelType::RGCN;
    config.hidden_dim = 128;
    config.num_layers = 3;
    config.num_relations = 5;
    config.num_bases = 4;
    config.learning_rate = 0.001;
    config.epochs = 30;
    config
}

fn create_gat_config() -> UnifiedModelConfig {
    let mut config = UnifiedModelConfig::default();
    config.model_type = ModelType::GAT;
    config.hidden_dim = 128;
    config.num_layers = 3;
    config.num_heads = Some(8);
    config.learning_rate = 0.001;
    config.epochs = 30;
    config
}

fn create_lightrdl_config() -> UnifiedModelConfig {
    let mut config = UnifiedModelConfig::default();
    config.model_type = ModelType::LightRDL;
    config.hidden_dim = 96;
    config.num_layers = 3;
    config.num_relations = 5;
    config.learning_rate = 0.001;
    config.epochs = 30;
    config
}
```

#### Step 2: Run Comparison

```rust
fn main() -> Result<()> {
    println!("Tutorial 2: Model Comparison");
    
    // Load data
    let dataset = get_dataset("rel-amazon", true)?;
    let task = get_task("rel-amazon", "user-churn", true)?;
    
    // Compare models
    let results = compare_models(&dataset, &task)?;
    
    // Print results
    println!("\nModel Comparison Results:");
    println!("=========================");
    for (model, accuracy) in results {
        println!("{}: {:.4}", model, accuracy);
    }
    
    // Find best model
    let best_model = results.iter()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    println!("\nBest model: {} with accuracy {:.4}", best_model.0, best_model.1);
    
    Ok(())
}
```

## Advanced Training

### Tutorial 3: Hyperparameter Optimization

Learn how to optimize hyperparameters automatically.

#### Step 1: Define Search Space

```rust
// examples/tutorial_3_hyperparameter_optimization.rs
use gaussrdl::*;

fn create_search_space() -> HyperparameterSpace {
    HyperparameterSpace {
        hidden_dim: vec![64, 128, 256],
        num_layers: vec![2, 3, 4],
        learning_rate: vec![0.0001, 0.001, 0.01],
        dropout: vec![0.1, 0.2, 0.3],
        num_bases: vec![2, 4, 8],
    }
}
```

#### Step 2: Optimization Function

```rust
fn optimize_hyperparameters(dataset: &dyn Dataset, task: &dyn Task) -> Result<UnifiedModelConfig> {
    println!("Starting hyperparameter optimization...");
    
    let search_space = create_search_space();
    let optimizer = HyperparameterOptimizer::new(search_space);
    
    // Run optimization
    let best_config = optimizer.optimize(dataset, task)?;
    
    println!("Best configuration found:");
    println!("  Hidden dim: {}", best_config.hidden_dim);
    println!("  Num layers: {}", best_config.num_layers);
    println!("  Learning rate: {}", best_config.learning_rate);
    println!("  Dropout: {}", best_config.dropout);
    println!("  Num bases: {}", best_config.num_bases);
    
    Ok(best_config)
}
```

#### Step 3: Train with Best Configuration

```rust
fn main() -> Result<()> {
    println!("Tutorial 3: Hyperparameter Optimization");
    
    // Load data
    let dataset = get_dataset("rel-amazon", true)?;
    let task = get_task("rel-amazon", "user-churn", true)?;
    
    // Optimize hyperparameters
    let best_config = optimize_hyperparameters(&dataset, &task)?;
    
    // Train with best configuration
    println!("Training with optimized configuration...");
    let model = create_model(best_config.clone())?;
    let trainer = Trainer::new(best_config);
    let metrics = trainer.train(&model, &dataset, &task)?;
    
    println!("Final results with optimized config:");
    for (metric, value) in metrics {
        println!("  {}: {:.4}", metric, value);
    }
    
    // Save optimized model
    model.save("tutorial_3_optimized_model.pt")?;
    
    Ok(())
}
```

### Tutorial 4: Distributed Training

Learn how to scale training across multiple GPUs.

#### Step 1: Distributed Configuration

```rust
// examples/tutorial_4_distributed_training.rs
use gaussrdl::*;

fn setup_distributed_training() -> Result<DistributedConfig> {
    let config = DistributedConfig {
        world_size: 2,  // Number of GPUs
        rank: 0,        // Current GPU rank
        backend: "nccl".to_string(),
        init_method: "env://".to_string(),
    };
    
    Ok(config)
}
```

#### Step 2: Distributed Training

```rust
fn train_distributed(dataset: &dyn Dataset, task: &dyn Task) -> Result<()> {
    println!("Setting up distributed training...");
    
    let dist_config = setup_distributed_training()?;
    let model_config = create_rgcn_config();
    
    // Create distributed trainer
    let trainer = DistributedTrainer::new(dist_config);
    
    // Train model
    let model = create_model(model_config.clone())?;
    let metrics = trainer.train(&model, dataset, task)?;
    
    println!("Distributed training completed!");
    for (metric, value) in metrics {
        println!("  {}: {:.4}", metric, value);
    }
    
    Ok(())
}
```

## Model Evaluation

### Tutorial 5: Comprehensive Evaluation

Learn how to evaluate models thoroughly.

#### Step 1: Evaluation Metrics

```rust
// examples/tutorial_5_evaluation.rs
use gaussrdl::*;

fn evaluate_model(model: &dyn RelGTModel, dataset: &dyn Dataset, task: &dyn Task) -> Result<()> {
    println!("Evaluating model...");
    
    // Get evaluation splits
    let splits = vec!["train", "val", "test"];
    
    for split in splits {
        println!("Evaluating on {} split...", split);
        
        // Get data for this split
        let table = dataset.get_table(split)?;
        
        // Make predictions
        let predictions = model.predict(&ModelInput::from_table(&table)?)?;
        
        // Evaluate
        let metrics = task.evaluate(&predictions, Some(&table))?;
        
        println!("{} split results:", split);
        for (metric, value) in metrics {
            println!("  {}: {:.4}", metric, value);
        }
    }
    
    Ok(())
}
```

#### Step 2: Model Analysis

```rust
fn analyze_model(model: &dyn RelGTModel) -> Result<()> {
    println!("Model Analysis:");
    println!("  Parameters: {}", model.parameter_count());
    println!("  Model type: {:?}", model.model_type());
    
    // Get model summary
    let summary = model.summary()?;
    println!("  Architecture: {}", summary.architecture);
    println!("  Input shape: {:?}", summary.input_shape);
    println!("  Output shape: {:?}", summary.output_shape);
    
    Ok(())
}
```

## Inference and Deployment

### Tutorial 6: Model Deployment

Learn how to deploy trained models for inference.

#### Step 1: Load and Test Model

```rust
// examples/tutorial_6_deployment.rs
use gaussrdl::*;

fn load_and_test_model() -> Result<()> {
    println!("Loading trained model...");
    
    // Load model
    let model = load_model("tutorial_1_rgcn_model.pt")?;
    
    // Test with sample input
    let sample_input = create_sample_input()?;
    let prediction = model.predict(&sample_input)?;
    
    println!("Sample prediction: {:?}", prediction);
    
    Ok(())
}

fn create_sample_input() -> Result<ModelInput> {
    // Create a sample input for testing
    let node_features = Tensor::randn(0.0, 1.0, &[10, 128], &Device::Cpu)?;
    let edge_index = Tensor::new(&[[0, 1, 2, 3], [1, 2, 3, 4]], &Device::Cpu)?;
    
    Ok(ModelInput {
        node_features,
        edge_index: Some(edge_index),
        edge_features: None,
        node_labels: None,
    })
}
```

#### Step 2: Start Inference Server

```rust
fn start_inference_server() -> Result<()> {
    println!("Starting inference server...");
    
    // Create server configuration
    let server_config = ServerConfig {
        host: "0.0.0.0".to_string(),
        port: 8080,
        models: vec![
            ModelConfig {
                name: "rgcn".to_string(),
                path: "tutorial_1_rgcn_model.pt".to_string(),
                config: None,
            }
        ],
    };
    
    // Start server
    let server = InferenceServer::new(server_config)?;
    server.start()?;
    
    println!("Server started on http://localhost:8080");
    println!("Press Ctrl+C to stop");
    
    // Keep server running
    std::thread::park();
    
    Ok(())
}
```

#### Step 3: Client Example

```rust
fn test_client() -> Result<()> {
    println!("Testing client...");
    
    let client = InferenceClient::new("http://localhost:8080")?;
    
    // Test health check
    let health = client.health()?;
    println!("Server health: {:?}", health);
    
    // Test prediction
    let input = create_sample_input()?;
    let prediction = client.predict("rgcn", &input)?;
    
    println!("Client prediction: {:?}", prediction);
    
    Ok(())
}
```

## Custom Models

### Tutorial 7: Building Custom Models

Learn how to create custom model architectures.

#### Step 1: Custom Model Definition

```rust
// examples/tutorial_7_custom_model.rs
use gaussrdl::*;
use candle_core::{Device, Tensor, DType};
use candle_nn::{VarBuilder, Module, Linear};

#[derive(Debug, Clone)]
struct CustomGNNConfig {
    hidden_dim: usize,
    num_layers: usize,
    dropout: f64,
}

#[derive(Debug)]
struct CustomGNN {
    config: CustomGNNConfig,
    layers: Vec<Linear>,
    output_layer: Linear,
    device: Device,
}

impl CustomGNN {
    fn new(config: CustomGNNConfig, vb: VarBuilder) -> Result<Self> {
        let device = vb.device();
        let mut layers = Vec::new();
        
        // Create hidden layers
        for i in 0..config.num_layers {
            let layer = linear(config.hidden_dim, config.hidden_dim, vb.pp(&format!("layer_{}", i)))?;
            layers.push(layer);
        }
        
        // Create output layer
        let output_layer = linear(config.hidden_dim, 1, vb.pp("output"))?;
        
        Ok(Self {
            config,
            layers,
            output_layer,
            device,
        })
    }
}
```

#### Step 2: Implement Model Trait

```rust
impl RelGTModel for CustomGNN {
    type Config = CustomGNNConfig;

    fn new(config: Self::Config, vb: VarBuilder) -> Result<Self> {
        Self::new(config, vb)
    }

    fn forward(&self, input: &ModelInput) -> Result<ModelOutput> {
        let mut x = input.node_features.clone();
        
        // Apply hidden layers
        for layer in &self.layers {
            x = layer.forward(&x)?;
            x = x.relu()?;
        }
        
        // Apply output layer
        let output = self.output_layer.forward(&x)?;
        
        Ok(ModelOutput::simple(output))
    }
    
    fn parameter_count(&self) -> usize {
        let mut count = 0;
        for layer in &self.layers {
            count += layer.weight().shape().elem_count();
            if let Some(bias) = layer.bias() {
                count += bias.shape().elem_count();
            }
        }
        count += self.output_layer.weight().shape().elem_count();
        if let Some(bias) = self.output_layer.bias() {
            count += bias.shape().elem_count();
        }
        count
    }
    
    fn model_type(&self) -> ModelType {
        ModelType::Custom("CustomGNN".to_string())
    }
    
    fn summary(&self) -> Result<ModelSummary> {
        Ok(ModelSummary {
            architecture: "CustomGNN".to_string(),
            input_shape: vec![self.config.hidden_dim],
            output_shape: vec![1],
            parameter_count: self.parameter_count(),
            layers: vec![
                LayerInfo {
                    name: "hidden_layers".to_string(),
                    input_shape: vec![self.config.hidden_dim],
                    output_shape: vec![self.config.hidden_dim],
                    parameters: self.layers.len() * self.config.hidden_dim * self.config.hidden_dim,
                },
                LayerInfo {
                    name: "output_layer".to_string(),
                    input_shape: vec![self.config.hidden_dim],
                    output_shape: vec![1],
                    parameters: self.config.hidden_dim,
                }
            ],
        })
    }
}
```

#### Step 3: Train Custom Model

```rust
fn train_custom_model(dataset: &dyn Dataset, task: &dyn Task) -> Result<()> {
    println!("Training custom model...");
    
    // Create custom configuration
    let config = CustomGNNConfig {
        hidden_dim: 128,
        num_layers: 3,
        dropout: 0.1,
    };
    
    // Create model
    let vb = VarBuilder::new("custom_model", Device::Cpu);
    let model = CustomGNN::new(config.clone(), vb)?;
    
    // Create training configuration
    let mut train_config = UnifiedModelConfig::default();
    train_config.model_type = ModelType::Custom("CustomGNN".to_string());
    train_config.hidden_dim = config.hidden_dim;
    train_config.num_layers = config.num_layers;
    train_config.dropout = config.dropout;
    train_config.learning_rate = 0.001;
    train_config.epochs = 30;
    
    // Train model
    let trainer = Trainer::new(train_config);
    let metrics = trainer.train(&model, dataset, task)?;
    
    println!("Custom model training completed!");
    for (metric, value) in metrics {
        println!("  {}: {:.4}", metric, value);
    }
    
    Ok(())
}
```

## Performance Optimization

### Tutorial 8: Performance Tuning

Learn how to optimize model performance.

#### Step 1: Memory Optimization

```rust
// examples/tutorial_8_performance.rs
use gaussrdl::*;

fn optimize_memory_usage() -> Result<()> {
    println!("Optimizing memory usage...");
    
    let mut config = UnifiedModelConfig::rgcn(128, 5, 4);
    
    // Memory optimization settings
    config.batch_size = 16;  // Smaller batch size
    config.gradient_accumulation_steps = Some(4);  // Gradient accumulation
    config.use_mixed_precision = true;  // Mixed precision
    config.gradient_checkpointing = true;  // Gradient checkpointing
    
    println!("Memory-optimized configuration:");
    println!("  Batch size: {}", config.batch_size);
    println!("  Gradient accumulation steps: {:?}", config.gradient_accumulation_steps);
    println!("  Mixed precision: {}", config.use_mixed_precision);
    println!("  Gradient checkpointing: {}", config.gradient_checkpointing);
    
    Ok(())
}
```

#### Step 2: Speed Optimization

```rust
fn optimize_speed() -> Result<()> {
    println!("Optimizing for speed...");
    
    let mut config = UnifiedModelConfig::rgcn(128, 5, 4);
    
    // Speed optimization settings
    config.device.device_type = DeviceType::CUDA;  // Use GPU
    config.use_mixed_precision = true;  // Mixed precision
    config.num_workers = Some(4);  // Multiple workers
    config.pin_memory = true;  // Pin memory
    
    println!("Speed-optimized configuration:");
    println!("  Device: {:?}", config.device.device_type);
    println!("  Mixed precision: {}", config.use_mixed_precision);
    println!("  Num workers: {:?}", config.num_workers);
    println!("  Pin memory: {}", config.pin_memory);
    
    Ok(())
}
```

#### Step 3: Benchmarking

```rust
fn benchmark_model(dataset: &dyn Dataset, task: &dyn Task) -> Result<()> {
    println!("Benchmarking model performance...");
    
    let config = UnifiedModelConfig::rgcn(128, 5, 4);
    let model = create_model(config.clone())?;
    
    // Benchmark training
    let start = std::time::Instant::now();
    let trainer = Trainer::new(config);
    let _metrics = trainer.train(&model, dataset, task)?;
    let training_time = start.elapsed();
    
    println!("Training time: {:?}", training_time);
    
    // Benchmark inference
    let table = dataset.get_table("test")?;
    let input = ModelInput::from_table(&table)?;
    
    let start = std::time::Instant::now();
    let _predictions = model.predict(&input)?;
    let inference_time = start.elapsed();
    
    println!("Inference time: {:?}", inference_time);
    
    Ok(())
}
```

## Real-world Examples

### Tutorial 9: Amazon Recommendation System

Build a recommendation system using the Amazon dataset.

#### Step 1: Data Analysis

```rust
// examples/tutorial_9_amazon_recommendation.rs
use gaussrdl::*;

fn analyze_amazon_data() -> Result<()> {
    println!("Analyzing Amazon dataset...");
    
    let dataset = get_dataset("rel-amazon", true)?;
    
    // Get dataset statistics
    let train_table = dataset.get_table("train")?;
    let val_table = dataset.get_table("val")?;
    let test_table = dataset.get_table("test")?;
    
    println!("Dataset statistics:");
    println!("  Train samples: {}", train_table.num_rows());
    println!("  Validation samples: {}", val_table.num_rows());
    println!("  Test samples: {}", test_table.num_rows());
    
    // Analyze features
    if let Some(features) = &train_table.features {
        println!("  Features: {:?}", features.keys().collect::<Vec<_>>());
    }
    
    Ok(())
}
```

#### Step 2: Recommendation Model

```rust
fn train_recommendation_model() -> Result<()> {
    println!("Training recommendation model...");
    
    let dataset = get_dataset("rel-amazon", true)?;
    let task = get_task("rel-amazon", "user-churn", true)?;
    
    // Use RGCN for recommendation
    let mut config = UnifiedModelConfig::rgcn(256, 5, 4);
    config.hidden_dim = 256;  // Larger model for better recommendations
    config.num_layers = 4;
    config.learning_rate = 0.001;
    config.epochs = 100;
    config.early_stopping = 15;
    
    let model = create_model(config.clone())?;
    let trainer = Trainer::new(config);
    let metrics = trainer.train(&model, &dataset, &task)?;
    
    println!("Recommendation model training completed!");
    for (metric, value) in metrics {
        println!("  {}: {:.4}", metric, value);
    }
    
    // Save model
    model.save("amazon_recommendation_model.pt")?;
    
    Ok(())
}
```

#### Step 3: Generate Recommendations

```rust
fn generate_recommendations() -> Result<()> {
    println!("Generating recommendations...");
    
    // Load trained model
    let model = load_model("amazon_recommendation_model.pt")?;
    
    // Load test data
    let dataset = get_dataset("rel-amazon", false)?;
    let test_table = dataset.get_table("test")?;
    
    // Generate predictions
    let input = ModelInput::from_table(&test_table)?;
    let predictions = model.predict(&input)?;
    
    // Process recommendations
    let recommendations = process_recommendations(&predictions, &test_table)?;
    
    println!("Generated {} recommendations", recommendations.len());
    
    // Save recommendations
    save_recommendations(&recommendations, "amazon_recommendations.json")?;
    
    Ok(())
}

fn process_recommendations(predictions: &ModelOutput, table: &Table) -> Result<Vec<Recommendation>> {
    // Process predictions into recommendations
    let mut recommendations = Vec::new();
    
    // Implementation depends on the specific format of predictions and table
    // This is a simplified example
    
    Ok(recommendations)
}

#[derive(Debug, serde::Serialize)]
struct Recommendation {
    user_id: String,
    item_id: String,
    score: f64,
    confidence: f64,
}
```

### Tutorial 10: F1 Racing Prediction

Build a prediction system for Formula 1 racing using the F1 dataset.

#### Step 1: F1 Data Analysis

```rust
// examples/tutorial_10_f1_prediction.rs
use gaussrdl::*;

fn analyze_f1_data() -> Result<()> {
    println!("Analyzing F1 dataset...");
    
    let dataset = get_dataset("rel-f1", true)?;
    
    // Get dataset information
    let train_table = dataset.get_table("train")?;
    println!("F1 dataset loaded with {} training samples", train_table.num_rows());
    
    // Analyze relationships
    if let Some(edges) = &train_table.edges {
        println!("Edge types: {:?}", edges.keys().collect::<Vec<_>>());
    }
    
    Ok(())
}
```

#### Step 2: F1 Prediction Model

```rust
fn train_f1_prediction_model() -> Result<()> {
    println!("Training F1 prediction model...");
    
    let dataset = get_dataset("rel-f1", true)?;
    let task = get_task("rel-f1", "driver-position", true)?;
    
    // Use GAT for F1 prediction (attention helps with complex relationships)
    let mut config = UnifiedModelConfig::gat(128, 8);
    config.hidden_dim = 128;
    config.num_layers = 3;
    config.num_heads = Some(8);
    config.learning_rate = 0.001;
    config.epochs = 80;
    config.early_stopping = 10;
    
    let model = create_model(config.clone())?;
    let trainer = Trainer::new(config);
    let metrics = trainer.train(&model, &dataset, &task)?;
    
    println!("F1 prediction model training completed!");
    for (metric, value) in metrics {
        println!("  {}: {:.4}", metric, value);
    }
    
    // Save model
    model.save("f1_prediction_model.pt")?;
    
    Ok(())
}
```

## Conclusion

This tutorial has covered the essential aspects of using GaussRDL:

1. **Basic Training**: How to train simple models
2. **Model Comparison**: Comparing different architectures
3. **Hyperparameter Optimization**: Finding optimal configurations
4. **Distributed Training**: Scaling across multiple devices
5. **Model Evaluation**: Comprehensive evaluation techniques
6. **Deployment**: Deploying models for inference
7. **Custom Models**: Building custom architectures
8. **Performance Optimization**: Optimizing for speed and memory
9. **Real-world Examples**: Practical applications

### Next Steps

1. **Experiment**: Try different datasets and model configurations
2. **Optimize**: Apply performance optimization techniques
3. **Deploy**: Deploy models in production environments
4. **Contribute**: Contribute to the GaussRDL project
5. **Learn More**: Explore advanced features and research papers

### Resources

- [User Guide](USERGUIDE.md) - Comprehensive usage guide
- [Developer Guide](DEVELOPERGUIDE.md) - Development and contribution guide
- [Models Guide](MODELS.md) - Detailed model documentation
- [API Documentation](https://docs.rs/gaussrdl) - Complete API reference

Happy learning and building with GaussRDL! 🚀