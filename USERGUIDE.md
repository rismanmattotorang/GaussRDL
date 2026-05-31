# GaussRDL User Guide

## Overview

GaussRDL is a high-performance, modular Rust implementation of Relational Deep Learning for temporal knowledge graphs. This guide provides comprehensive information for users to get started with GaussRDL and use its advanced features effectively.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Core Concepts](#core-concepts)
4. [Data Loading](#data-loading)
5. [Model Training](#model-training)
6. [Inference](#inference)
7. [CLI Usage](#cli-usage)
8. [Server API](#server-api)
9. [Configuration](#configuration)
10. [Advanced Features](#advanced-features)
11. [Troubleshooting](#troubleshooting)

## Installation

### System Requirements

- **Operating System**: Linux, macOS, or Windows
- **Rust**: 1.70 or later
- **Memory**: 8GB+ RAM recommended
- **Storage**: 10GB+ free space
- **GPU**: CUDA 11.8+ (optional, for acceleration)

### Installation Methods

#### From Source (Recommended)

```bash
# Clone repository
git clone https://github.com/your-org/gaussrdl.git
cd gaussrdl

# Build in release mode
cargo build --release

# Install globally
cargo install --path .
```

#### Using Cargo

```bash
# Install from crates.io
cargo install gaussrdl
```

#### Docker

```bash
# Pull Docker image
docker pull gaussrdl/gaussrdl:latest

# Run container
docker run -it --rm gaussrdl/gaussrdl:latest
```

### Verification

```bash
# Check installation
gaussrdl --version

# List available commands
gaussrdl --help

# Check system compatibility
gaussrdl system check
```

## Quick Start

### Basic Training Workflow

```rust
use gaussrdl::*;

fn main() -> Result<()> {
    // 1. Load dataset
    let dataset = get_dataset("rel-amazon", true)?;
    
    // 2. Get task
    let task = get_task("rel-amazon", "user-churn", true)?;
    
    // 3. Create model configuration
    let config = UnifiedModelConfig::rgcn(128, 5, 4);
    
    // 4. Create model
    let model = create_model(config)?;
    
    // 5. Train model
    let trainer = Trainer::new(config);
    let metrics = trainer.train(&model, &dataset, &task)?;
    
    // 6. Print results
    println!("Training completed!");
    for (metric, value) in metrics {
        println!("{}: {:.4}", metric, value);
    }
    
    Ok(())
}
```

### Using the CLI

```bash
# Train a model
gaussrdl train \
    --model rgcn \
    --dataset amazon \
    --task user-churn \
    --config config.yaml \
    --output results.json

# Start inference server
gaussrdl server --port 8080

# Make predictions
gaussrdl predict \
    --model-path model.pt \
    --input input.json \
    --output predictions.json
```

## Core Concepts

### Datasets

GaussRDL supports multiple relational datasets:

- **Amazon**: Product recommendation dataset
- **F1**: Formula 1 racing dataset
- **H&M**: Fashion recommendation dataset
- **Avito**: Classified ads dataset
- **Trial**: Synthetic trial dataset

### Tasks

Common tasks include:

- **User Churn**: Predict user churn behavior
- **Link Prediction**: Predict missing links in graphs
- **Node Classification**: Classify nodes in graphs
- **Graph Classification**: Classify entire graphs

### Models

Available model architectures:

- **RGCN**: Relational Graph Convolutional Network
- **GAT**: Graph Attention Network
- **LightRDL**: Lightweight Relational Deep Learning
- **StageGNN**: Staged Graph Neural Network
- **RelGT**: Relational Graph Transformer

## Data Loading

### Loading Datasets

```rust
use gaussrdl::*;

// Load dataset with automatic download
let dataset = get_dataset("rel-amazon", true)?;

// Load dataset without download
let dataset = get_dataset("rel-amazon", false)?;

// Get dataset information
println!("Dataset: {}", dataset.name());
println!("Splits: {:?}", dataset.splits());
```

### Dataset Configuration

```rust
use gaussrdl_core::DownloadConfig;

let config = DownloadConfig {
    cache_dir: Some(PathBuf::from("./cache")),
    download_dir: Some(PathBuf::from("./downloads")),
    force_download: false,
};

let dataset = get_dataset_with_config("rel-amazon", &config)?;
```

### Custom Data Loading

```rust
use gaussrdl_core::{Dataset, Table};

struct CustomDataset {
    name: String,
    data: HashMap<String, Table>,
}

impl Dataset for CustomDataset {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn splits(&self) -> Vec<String> {
        vec!["train".to_string(), "val".to_string(), "test".to_string()]
    }
    
    fn get_table(&self, split: &str) -> Result<Table> {
        self.data.get(split)
            .cloned()
            .ok_or_else(|| Error::Dataset("Split not found".to_string()))
    }
}
```

## Model Training

### Basic Training

```rust
use gaussrdl::*;

// Create training configuration
let config = UnifiedModelConfig::rgcn(128, 5, 4);

// Create model
let model = create_model(config)?;

// Create trainer
let trainer = Trainer::new(config);

// Train model
let metrics = trainer.train(&model, &dataset, &task)?;

// Save model
model.save("model.pt")?;
```

### Advanced Training Configuration

```rust
use gaussrdl::*;

let mut config = UnifiedModelConfig::default();
config.model_type = ModelType::RGCN;
config.hidden_dim = 256;
config.num_layers = 3;
config.num_relations = 5;
config.num_bases = 4;
config.dropout = 0.1;
config.learning_rate = 0.001;
config.batch_size = 32;
config.epochs = 100;
config.early_stopping = 10;
config.gradient_clipping = Some(1.0);

// Device configuration
config.device.device_type = DeviceType::Auto;
config.device.memory_limit = Some(8192);
config.device.use_mixed_precision = false;
```

### Training with Callbacks

```rust
use gaussrdl::*;

struct CustomCallback {
    best_metric: f64,
}

impl TrainingCallback for CustomCallback {
    fn on_epoch_end(&mut self, epoch: usize, metrics: &HashMap<String, f64>) {
        if let Some(accuracy) = metrics.get("accuracy") {
            if *accuracy > self.best_metric {
                self.best_metric = *accuracy;
                println!("New best accuracy: {:.4}", accuracy);
            }
        }
    }
}

let mut callback = CustomCallback { best_metric: 0.0 };
let trainer = Trainer::with_callback(config, callback);
trainer.train(&model, &dataset, &task)?;
```

### Distributed Training

```rust
use gaussrdl::*;

// Configure distributed training
let config = DistributedConfig {
    world_size: 4,
    rank: 0,
    backend: "nccl".to_string(),
    init_method: "env://".to_string(),
};

let trainer = DistributedTrainer::new(config);
trainer.train(&model, &dataset, &task)?;
```

## Inference

### Basic Inference

```rust
use gaussrdl::*;

// Load trained model
let model = load_model("model.pt")?;

// Load input data
let input = load_input("input.json")?;

// Make predictions
let predictions = model.predict(&input)?;

// Save predictions
save_predictions(&predictions, "predictions.json")?;
```

### Batch Inference

```rust
use gaussrdl::*;

let model = load_model("model.pt")?;
let inputs = load_batch_inputs("batch_inputs.json")?;

let predictions = model.predict_batch(&inputs)?;

for (i, pred) in predictions.iter().enumerate() {
    println!("Input {}: {:?}", i, pred);
}
```

### Real-time Inference

```rust
use gaussrdl::*;

// Start inference server
let server = InferenceServer::new("model.pt")?;
server.start("0.0.0.0:8080")?;

// Client code
let client = InferenceClient::new("http://localhost:8080")?;
let prediction = client.predict(&input)?;
```

## CLI Usage

### Training Commands

```bash
# Basic training
gaussrdl train --model rgcn --dataset amazon --task user-churn

# Training with configuration file
gaussrdl train --config config.yaml

# Training with specific parameters
gaussrdl train \
    --model rgcn \
    --dataset amazon \
    --task user-churn \
    --hidden-dim 256 \
    --num-layers 3 \
    --learning-rate 0.001 \
    --epochs 100

# Distributed training
gaussrdl train --distributed --world-size 4 --rank 0
```

### Server Commands

```bash
# Start inference server
gaussrdl server --port 8080 --model-path model.pt

# Start server with multiple models
gaussrdl server \
    --port 8080 \
    --models rgcn:model_rgcn.pt,gat:model_gat.pt

# Start server with configuration
gaussrdl server --config server_config.yaml
```

### Utility Commands

```bash
# Download dataset
gaussrdl download --dataset amazon

# List available datasets
gaussrdl list datasets

# List available models
gaussrdl list models

# List available tasks
gaussrdl list tasks

# Check system compatibility
gaussrdl system check

# Profile model performance
gaussrdl profile --model rgcn --dataset amazon
```

### Monitoring Commands

```bash
# Monitor training progress
gaussrdl monitor --job-id train_001

# Monitor system resources
gaussrdl monitor system

# Monitor model performance
gaussrdl monitor model --model-path model.pt
```

## Server API

### REST API Endpoints

#### Health Check
```bash
curl http://localhost:8080/health
```

#### Model Information
```bash
curl http://localhost:8080/models
curl http://localhost:8080/models/rgcn
```

#### Predictions
```bash
# Single prediction
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rgcn",
    "input": {
      "nodes": [...],
      "edges": [...],
      "features": [...]
    }
  }'

# Batch prediction
curl -X POST http://localhost:8080/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rgcn",
    "inputs": [...]
  }'
```

#### Model Management
```bash
# Load model
curl -X POST http://localhost:8080/models/load \
  -H "Content-Type: application/json" \
  -d '{"name": "my_model", "path": "model.pt"}'

# Unload model
curl -X DELETE http://localhost:8080/models/my_model
```

### WebSocket API

```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8080/ws');

// Send prediction request
ws.send(JSON.stringify({
    type: 'predict',
    model: 'rgcn',
    input: {...}
}));

// Receive prediction
ws.onmessage = function(event) {
    const response = JSON.parse(event.data);
    console.log('Prediction:', response.prediction);
};
```

## Configuration

### Configuration Files

#### Training Configuration (config.yaml)
```yaml
model:
  type: rgcn
  hidden_dim: 256
  num_layers: 3
  num_relations: 5
  num_bases: 4
  dropout: 0.1
  use_layer_norm: true
  use_residual: true
  use_edge_features: false

training:
  epochs: 100
  learning_rate: 0.001
  batch_size: 32
  early_stopping: 10
  gradient_clipping: 1.0
  optimizer: adam
  weight_decay: 0.0001

dataset:
  name: rel_amazon
  split_ratio: [0.8, 0.1, 0.1]
  cache_dir: ./cache
  force_download: false

device:
  type: auto
  memory_limit: 8192
  use_mixed_precision: false

logging:
  level: info
  file: training.log
  tensorboard: true
```

#### Server Configuration (server_config.yaml)
```yaml
server:
  host: 0.0.0.0
  port: 8080
  workers: 4
  max_connections: 1000
  timeout: 30

models:
  - name: rgcn
    path: models/rgcn.pt
    config: models/rgcn_config.yaml
  - name: gat
    path: models/gat.pt
    config: models/gat_config.yaml

security:
  enable_auth: false
  api_key: null
  cors_origins: ["*"]

monitoring:
  enable_metrics: true
  metrics_port: 9090
  health_check_interval: 30
```

### Environment Variables

```bash
# Dataset configuration
export GAUSSRDL_CACHE_DIR=./cache
export GAUSSRDL_DOWNLOAD_DIR=./downloads

# Model configuration
export GAUSSRDL_MODEL_DIR=./models
export GAUSSRDL_CONFIG_DIR=./configs

# Server configuration
export GAUSSRDL_SERVER_HOST=0.0.0.0
export GAUSSRDL_SERVER_PORT=8080

# Logging configuration
export RUST_LOG=info
export GAUSSRDL_LOG_LEVEL=info
export GAUSSRDL_LOG_FILE=./logs/gaussrdl.log
```

## Advanced Features

### Custom Models

```rust
use gaussrdl::*;

#[derive(Debug, Clone)]
struct CustomModel {
    config: CustomConfig,
    layers: Vec<Linear>,
}

impl RelGTModel for CustomModel {
    type Config = CustomConfig;

    fn new(config: Self::Config, vb: VarBuilder) -> Result<Self> {
        let mut layers = Vec::new();
        for i in 0..config.num_layers {
            let layer = linear(config.hidden_dim, config.hidden_dim, vb.pp(&format!("layer_{}", i)))?;
            layers.push(layer);
        }
        
        Ok(Self { config, layers })
    }

    fn forward(&self, input: &ModelInput) -> Result<ModelOutput> {
        let mut x = input.node_features.clone();
        
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }
        
        Ok(ModelOutput::simple(x))
    }
    
    // Implement other required methods...
}
```

### Custom Metrics

```rust
use gaussrdl_metrics::*;

#[derive(Debug)]
struct CustomMetric {
    name: String,
}

impl Metric for CustomMetric {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn compute(&self, predictions: &[f64], targets: &[f64]) -> Result<f64> {
        // Implement custom metric computation
        let mut sum = 0.0;
        for (pred, target) in predictions.iter().zip(targets.iter()) {
            sum += (pred - target).abs();
        }
        Ok(sum / predictions.len() as f64)
    }
    
    fn higher_is_better(&self) -> bool {
        false
    }
}
```

### Custom Datasets

```rust
use gaussrdl_core::*;

struct CustomDataset {
    name: String,
    data: HashMap<String, Table>,
}

impl Dataset for CustomDataset {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn splits(&self) -> Vec<String> {
        vec!["train".to_string(), "val".to_string(), "test".to_string()]
    }
    
    fn get_table(&self, split: &str) -> Result<Table> {
        self.data.get(split)
            .cloned()
            .ok_or_else(|| Error::Dataset("Split not found".to_string()))
    }
    
    fn load(&mut self, config: &DownloadConfig) -> Result<()> {
        // Implement data loading logic
        Ok(())
    }
}
```

### Model Ensembling

```rust
use gaussrdl::*;

// Create ensemble of models
let models = vec![
    create_model(UnifiedModelConfig::rgcn(128, 5, 4))?,
    create_model(UnifiedModelConfig::gat(128, 8))?,
    create_model(UnifiedModelConfig::lightrdl(96, 5))?,
];

// Create ensemble predictor
let ensemble = EnsemblePredictor::new(models);

// Make ensemble predictions
let predictions = ensemble.predict(&input)?;
```

### Hyperparameter Optimization

```rust
use gaussrdl::*;

// Define hyperparameter search space
let search_space = HyperparameterSpace {
    hidden_dim: vec![64, 128, 256],
    num_layers: vec![2, 3, 4],
    learning_rate: vec![0.001, 0.01, 0.1],
    dropout: vec![0.1, 0.2, 0.3],
};

// Create optimizer
let optimizer = HyperparameterOptimizer::new(search_space);

// Run optimization
let best_config = optimizer.optimize(&dataset, &task)?;
```

## Troubleshooting

### Common Issues

#### Installation Problems

**Issue**: Rust version too old
```bash
# Solution: Update Rust
rustup update
rustup default stable
```

**Issue**: CUDA not found
```bash
# Solution: Install CUDA or disable GPU
cargo build --no-default-features
```

#### Training Problems

**Issue**: Out of memory
```bash
# Solution: Reduce batch size
gaussrdl train --batch-size 16

# Solution: Use gradient accumulation
gaussrdl train --gradient-accumulation-steps 4
```

**Issue**: Slow training
```bash
# Solution: Enable GPU acceleration
gaussrdl train --device cuda

# Solution: Use mixed precision
gaussrdl train --mixed-precision
```

#### Model Problems

**Issue**: Model not converging
```bash
# Solution: Adjust learning rate
gaussrdl train --learning-rate 0.0001

# Solution: Add regularization
gaussrdl train --weight-decay 0.001
```

**Issue**: Overfitting
```bash
# Solution: Increase dropout
gaussrdl train --dropout 0.3

# Solution: Use early stopping
gaussrdl train --early-stopping 5
```

### Performance Optimization

#### Memory Optimization
```bash
# Use memory-efficient settings
gaussrdl train \
    --batch-size 16 \
    --gradient-accumulation-steps 4 \
    --mixed-precision \
    --gradient-checkpointing
```

#### Speed Optimization
```bash
# Use GPU acceleration
gaussrdl train --device cuda

# Use multiple GPUs
gaussrdl train --distributed --world-size 2

# Use optimized compilation
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

### Debugging

#### Enable Debug Logging
```bash
# Set debug log level
export RUST_LOG=debug
gaussrdl train --model rgcn --dataset amazon
```

#### Profile Performance
```bash
# Profile CPU usage
cargo install flamegraph
cargo flamegraph --bin gaussrdl -- train --model rgcn

# Profile memory usage
cargo install heim
cargo heim --bin gaussrdl -- train --model rgcn
```

#### Check System Resources
```bash
# Check GPU usage
nvidia-smi

# Check memory usage
free -h

# Check disk space
df -h
```

### Getting Help

1. **Check Documentation**: Review this guide and other documentation
2. **Search Issues**: Look for similar issues on GitHub
3. **Ask Questions**: Use GitHub Discussions
4. **Report Bugs**: Create detailed bug reports
5. **Join Community**: Participate in community channels

## Conclusion

This user guide provides comprehensive information for using GaussRDL effectively. Start with the quick start examples and gradually explore advanced features as needed.

Remember:
- Always check system requirements before installation
- Use appropriate model configurations for your use case
- Monitor training progress and system resources
- Optimize performance based on your hardware
- Keep backups of trained models and configurations

Happy modeling! 🚀 