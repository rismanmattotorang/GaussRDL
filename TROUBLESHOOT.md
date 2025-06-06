# Troubleshooting Guide

This document provides solutions to common issues you might encounter while using RelBench.

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Runtime Issues](#runtime-issues)
3. [Dataset Issues](#dataset-issues)
4. [GPU Issues](#gpu-issues)
5. [Memory Issues](#memory-issues)
6. [Performance Issues](#performance-issues)

## Installation Issues

### Cargo Build Fails

**Problem**: `cargo build` fails with compilation errors.

**Solutions**:
1. Ensure you have the latest stable Rust:
```bash
rustup update stable
```

2. Check system dependencies:
```bash
# Ubuntu/Debian
sudo apt-get install build-essential pkg-config libssl-dev

# macOS
brew install openssl pkg-config

# Windows
# Install Visual Studio Build Tools
```

3. Clean and rebuild:
```bash
cargo clean
cargo build
```

### GPU Support Not Working

**Problem**: GPU features not available after installation.

**Solutions**:
1. Enable GPU features in Cargo.toml:
```toml
[dependencies]
relbench = { version = "0.1.0", features = ["cuda"] }
```

2. Install CUDA toolkit for CUDA support:
```bash
# Check CUDA installation
nvcc --version
```

3. Install Metal SDK for Metal support (macOS only).

## Runtime Issues

### Dataset Download Fails

**Problem**: Unable to download datasets.

**Solutions**:
1. Check internet connection
2. Verify disk space
3. Check permissions:
```bash
# Set correct permissions
chmod -R u+w ~/.cache/relbench
```

4. Use alternative download location:
```rust
let config = DownloadConfig {
    cache_dir: PathBuf::from("./data"),
    ..Default::default()
};
```

### Task Evaluation Errors

**Problem**: Task evaluation fails with errors.

**Solutions**:
1. Verify input dimensions match
2. Check for NaN values:
```rust
if predictions.iter().any(|x| x.is_nan()) {
    // Handle NaN values
}
```

3. Ensure correct data types

## Dataset Issues

### Missing Data

**Problem**: Dataset missing expected tables or columns.

**Solutions**:
1. Verify dataset integrity:
```rust
dataset.validate()?;
```

2. Re-download with force option:
```rust
let config = DownloadConfig {
    force_download: true,
    ..Default::default()
};
```

### Data Format Issues

**Problem**: Data loading fails due to format issues.

**Solutions**:
1. Check schema compatibility
2. Validate data types
3. Use data conversion utilities:
```rust
use relbench::utils::convert_data_type;
```

## GPU Issues

### CUDA Out of Memory

**Problem**: GPU memory exhausted during training.

**Solutions**:
1. Reduce batch size
2. Enable gradient checkpointing
3. Use memory-efficient attention
4. Monitor GPU memory:
```rust
use relbench::utils::gpu_memory_info;
```

### Metal Performance

**Problem**: Poor performance with Metal backend.

**Solutions**:
1. Update Metal drivers
2. Optimize tensor operations
3. Use appropriate batch sizes
4. Enable performance logging:
```rust
use relbench::utils::enable_metal_logging;
```

## Memory Issues

### Out of Memory

**Problem**: Process runs out of memory.

**Solutions**:
1. Enable memory mapping:
```rust
use relbench::utils::enable_memory_mapping;
```

2. Reduce batch size
3. Use streaming data loading
4. Monitor memory usage:
```rust
use relbench::utils::memory_usage;
```

### Memory Leaks

**Problem**: Memory usage grows over time.

**Solutions**:
1. Use memory profiling tools
2. Implement proper cleanup
3. Monitor allocations:
```rust
use relbench::utils::track_allocations;
```

## Performance Issues

### Slow Data Loading

**Problem**: Data loading takes too long.

**Solutions**:
1. Enable caching:
```rust
let config = CacheConfig {
    use_cache: true,
    ..Default::default()
};
```

2. Use parallel loading:
```rust
use relbench::utils::parallel_load;
```

3. Optimize disk I/O

### Slow Model Training

**Problem**: Model training is slower than expected.

**Solutions**:
1. Profile code:
```rust
use relbench::utils::profile_section;
```

2. Enable optimizations:
```toml
[profile.release]
lto = true
codegen-units = 1
```

3. Use parallel processing:
```rust
use rayon::prelude::*;
```

## Getting Help

If you're still experiencing issues:

1. Check the [GitHub Issues](https://github.com/yourusername/relbench/issues)
2. Join our [Discord Community](https://discord.gg/yourdiscord)
3. Create a new issue with:
   - Rust version (`rustc --version`)
   - OS details
   - Error messages
   - Minimal reproducible example

## Contributing Solutions

Found a solution not listed here? Please contribute:

1. Fork the repository
2. Add your solution
3. Submit a pull request

## Updates

This guide is regularly updated. Check the latest version on GitHub.