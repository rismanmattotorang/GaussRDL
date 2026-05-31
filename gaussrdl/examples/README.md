# GaussRelGT Examples

This directory contains comprehensive examples demonstrating how to use GaussRelGT, similar to the RelBench Python examples.

## Available Examples

### 1. Dataset Loading (`dataset_loading.rs`)
Demonstrates how to load and explore different datasets:
- Loading rel-amazon, rel-f1, and rel-hm datasets
- Dataset validation and information retrieval
- Dataset comparison and performance analysis

```bash
cargo run --example dataset_loading
```

### 2. Task Evaluation (`task_evaluation.rs`)
Shows how to evaluate different prediction tasks:
- User churn prediction
- Item churn prediction  
- Driver position prediction
- Item sales prediction
- Performance benchmarking

```bash
cargo run --example task_evaluation
```

### 3. Metrics Evaluation (`metrics_evaluation.rs`)
Comprehensive demonstration of evaluation metrics:
- Classification metrics (AUROC, Accuracy, Precision, Recall, F1)
- Regression metrics (RMSE, MAE, R², MAPE)
- Ranking metrics (MAP, MRR, NDCG, Hits)
- Link prediction metrics
- Performance comparison

```bash
cargo run --example metrics_evaluation
```

### 4. Distributed Training (`distributed_training.rs`)
Advanced distributed training setup:
- Creating distributed nodes (master, worker, parameter server)
- Distributed training simulation
- Fault tolerance demonstration
- Performance monitoring

```bash
cargo run --example distributed_training
```

### 5. Model-Specific Examples
- `rgcn_example.rs` - Relational Graph Convolutional Network
- `gat_example.rs` - Graph Attention Network
- `lightrdl_example.rs` - Lightweight Relational Deep Learning
- `stagegnn_example.rs` - Staged Graph Neural Network
- `relgt_example.rs` - Relational Graph Transformer

### 6. Existing Examples
- `basic_usage.rs` - Basic library usage
- `custom_model.rs` - Custom model implementation
- `memory_optimization.rs` - Memory optimization techniques
- `parallel_processing.rs` - Parallel processing examples

## Running Examples

To run any example:

```bash
# Run a specific example
cargo run --example dataset_loading

# Run with verbose output
RUST_LOG=debug cargo run --example task_evaluation

# Run with release optimizations
cargo run --release --example metrics_evaluation
```

## Example Features

### Dataset Loading Example
- ✅ Load multiple dataset types
- ✅ Validate dataset integrity
- ✅ Compare dataset properties
- ✅ Performance benchmarking
- ✅ Error handling demonstration

### Task Evaluation Example
- ✅ Load different task types (classification, regression)
- ✅ Generate mock predictions
- ✅ Evaluate model performance
- ✅ Benchmark evaluation speed
- ✅ Compare task metrics

### Metrics Evaluation Example
- ✅ Classification metrics with perfect/worst cases
- ✅ Regression metrics with known values
- ✅ Ranking metrics with different k values
- ✅ Performance testing with large datasets
- ✅ Metric relationship validation

### Distributed Training Example
- ✅ Multi-node setup (async)
- ✅ Parameter server architecture
- ✅ Fault tolerance testing
- ✅ Real-time performance monitoring
- ✅ Scalability demonstration

## Integration with Tests

These examples complement the comprehensive test suite in the `tests/` directory:
- `test_datasets.rs` - Dataset functionality tests
- `test_tasks.rs` - Task evaluation tests  
- `test_metrics.rs` - Metrics calculation tests
- `test_integration.rs` - End-to-end integration tests

## Common Usage Patterns

### Basic Workflow
```rust
use gaussrelgt::tasks::*;

// Load a task
let task = get_task("rel-amazon", "user-churn", false)?;

// Get training data
let train_table = task.get_train_table()?;

// Generate predictions (your model here)
let predictions = vec![0.8, 0.3, 0.9, 0.1, 0.7];

// Evaluate
let results = task.evaluate(&predictions, None)?;
println!("Results: {:?}", results);
```

### Working with Datasets
```rust
use gaussrelgt::datasets::*;
use gaussrelgt::base::{Dataset, DatasetConfig};

let mut dataset = RelAmazonDataset::new();
let config = DatasetConfig::default();
dataset.load(&config)?;

println!("Dataset: {}", dataset.name());
println!("Tables: {:?}", dataset.tables().keys().collect::<Vec<_>>());
```

### Using Metrics
```rust
use gaussrelgt::metrics::*;

let auroc = AUROC::new();
let predictions = vec![0.9, 0.2, 0.8, 0.1];
let targets = vec![1.0, 0.0, 1.0, 0.0];
// Note: metrics implementation may vary
```

## Performance Notes

- Examples are designed to run quickly in development
- Use `--release` flag for performance benchmarking
- Large dataset operations may require more memory
- Distributed examples use localhost for simplicity

## Error Handling

All examples include comprehensive error handling:
- Graceful degradation when datasets aren't available
- Clear error messages for debugging
- Fallback behaviors for test environments

## Contributing

When adding new examples:
1. Follow the existing naming convention
2. Include comprehensive documentation
3. Add error handling and logging
4. Include performance benchmarks where relevant
5. Update this README

## Related Documentation

- [Main README](../README.md) - Project overview
- [Tests README](../tests/README.md) - Testing documentation
- [User Guide](../USERGUIDE.md) - Detailed usage guide
- [Developer Guide](../DEVELOPERGUIDE.md) - Development instructions 