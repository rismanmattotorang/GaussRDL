# RelBench

A high-performance Rust implementation of the Relational Deep Learning Benchmark.

## Features

- **High Performance**
  - Native Rust implementation with zero-cost abstractions
  - Efficient parallel processing with Rayon
  - Memory-mapped data loading with polars
  - Optimized tensor operations

- **Memory Safety**
  - Leverages Rust's ownership system
  - No memory leaks or data races
  - Efficient memory management
  - Zero-copy operations where possible

- **Type Safety**
  - Strong type system prevents runtime errors
  - Compile-time guarantees
  - Clear error handling with Result types

- **Modern Architecture**
  - Clean, modular design
  - Clear separation of concerns
  - Extensive test coverage
  - Comprehensive documentation

## Installation

### From crates.io

Add RelBench to your `Cargo.toml`:

```toml
[dependencies]
relbench = "1.0.0"
```

### From source

```bash
git clone https://github.com/yourusername/relbench.git
cd relbench
cargo build --release
```

## Quick Start

### CLI Usage

1. List available tasks:
```bash
relbench list
```

2. Download a dataset:
```bash
relbench download rel-amazon
```

3. Run a task:
```bash
relbench run rel-amazon user-churn \
  --epochs 100 \
  --batch-size 32 \
  --learning-rate 0.001 \
  --output-dir ./output
```

4. Evaluate a model:
```bash
relbench evaluate rel-amazon user-churn \
  --checkpoint ./output/model_best.json \
  --output metrics.json
```

5. Start the server:
```bash
relbench serve --host 127.0.0.1 --port 8000 --docs
```

### API Usage

```rust
use relbench::{get_dataset, get_task};

#[tokio::main]
async fn main() -> relbench::Result<()> {
    // Load dataset
    let dataset = get_dataset("rel-amazon", true)?;
    
    // Get task
    let task = get_task("rel-amazon", "user-churn", true)?;
    
    // Get data tables
    let train_table = task.get_train_table()?;
    let val_table = task.get_val_table()?;
    let test_table = task.get_test_table(true)?;
    
    // Train model and make predictions
    let predictions = vec![0.5; test_table.len()];
    
    // Evaluate predictions
    let metrics = task.evaluate(&predictions, None)?;
    println!("Metrics: {:?}", metrics);
    
    Ok(())
}
```

## Datasets

RelBench supports multiple datasets out of the box:

- **Amazon**: E-commerce dataset with user-item interactions
- **Formula 1**: Racing dataset with temporal relationships
- **H&M**: Fashion retail dataset with customer behavior

## Tasks

### Entity Prediction
- User churn prediction
- Item churn prediction
- Driver position prediction
- Constructor position prediction

### Recommendation
- Item recommendation
- User-item interaction prediction
- Next purchase prediction

## Models

RelBench implements state-of-the-art graph neural networks:

### Base GNN
```rust
let config = BaseGNNConfig {
    hidden_dim: 256,
    num_layers: 3,
    dropout: 0.1,
    use_layer_norm: true,
    use_residual: true,
    activation: "relu".to_string(),
};

let model = BaseGNN::new(config, vb)?;
```

### RGCN
```rust
let config = RGCNConfig {
    hidden_dim: 256,
    num_relations: 5,
    num_bases: 3,
    num_layers: 3,
    dropout: 0.1,
    use_layer_norm: true,
    use_residual: true,
    activation: "relu".to_string(),
};

let model = RGCN::new(config, vb)?;
```

### GAT
```rust
let config = GATConfig {
    hidden_dim: 256,
    num_heads: 8,
    num_layers: 3,
    dropout: 0.1,
    attention_dropout: 0.1,
    use_layer_norm: true,
    use_residual: true,
    activation: "relu".to_string(),
};

let model = GAT::new(config, vb)?;
```

## API Documentation

The API documentation is available at:
- [docs.rs/relbench](https://docs.rs/relbench)
- Local server: http://localhost:8000/docs (when running in server mode)

## Performance Optimization

### Memory Management
```rust
use relbench::utils::memory::MemoryConfig;

let config = MemoryConfig {
    use_mmap: true,
    cache_size: 1024 * 1024 * 1024, // 1GB
    prefetch: true,
};
```

### Parallel Processing
```rust
use relbench::utils::parallel::ParallelConfig;

let config = ParallelConfig {
    num_threads: 8,
    chunk_size: 1000,
    use_rayon: true,
};
```

## Development

### Prerequisites
- Rust 1.75 or later
- Cargo
- Optional: CUDA toolkit for GPU support

### Building
```bash
# Debug build
cargo build

# Release build
cargo build --release

# Build with GPU support
cargo build --release --features cuda
```

### Testing
```bash
# Run all tests
cargo test

# Run specific test
cargo test test_name

# Run benchmarks
cargo bench
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## Roadmap

See [TODO.md](TODO.md) for planned improvements and features.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.

## License

RelBench is licensed under the MIT License. See [LICENSE](LICENSE) for details.