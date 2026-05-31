# Model Architecture Documentation

## Overview

The GaussRelGT project implements a unified, modular architecture for relational graph neural networks. The system provides a consistent interface across multiple model types while maintaining flexibility and extensibility.

## Unified Architecture

### Core Components

#### 1. RelGTModel Trait (`src/models/mod.rs`)

The `RelGTModel` trait serves as the unified interface for all model implementations:

```rust
pub trait RelGTModel: Send + Sync {
    type Config: Clone + Serialize + for<'de> Deserialize<'de>;
    
    // Core functionality
    fn forward(&self, inputs: &ModelInput) -> Result<ModelOutput>;
    fn predict(&self, input: &Table) -> Result<Vec<f64>>;
    fn predict_with_uncertainty(&self, input: &Table) -> Result<(Vec<f64>, Vec<f64>)>;
    
    // Training and validation
    fn train_step(&mut self, input: &ModelInput, targets: &Tensor) -> Result<f64>;
    fn validate(&self, input: &ModelInput, targets: &Tensor) -> Result<HashMap<String, f64>>;
    
    // Model management
    fn save(&self, path: &str) -> Result<()>;
    fn load(&mut self, path: &str) -> Result<()>;
    fn to_device(&mut self, device: &Device) -> Result<()>;
    
    // Introspection
    fn parameter_count(&self) -> usize;
    fn memory_usage(&self) -> usize;
    fn summary(&self) -> ModelSummary;
}
```

#### 2. Unified Configuration (`UnifiedModelConfig`)

All models use a single configuration structure that adapts to different model types:

```rust
pub struct UnifiedModelConfig {
    pub model_type: ModelType,
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub output_dim: usize,
    pub num_layers: usize,
    pub dropout: f64,
    pub activation: ActivationType,
    pub device: DeviceConfig,
    pub model_specific: HashMap<String, serde_json::Value>,
}
```

#### 3. Model Input/Output Structures

**ModelInput**: Unified input structure supporting both homogeneous and heterogeneous graphs:
```rust
pub struct ModelInput {
    pub node_features: Tensor,
    pub edge_index: Tensor,
    pub edge_features: Option<Tensor>,
    pub edge_types: Option<Tensor>,
    pub node_types: Option<Tensor>,
    pub batch: Option<Tensor>,
    pub metadata: HashMap<String, Tensor>,
}
```

**ModelOutput**: Comprehensive output with interpretability features:
```rust
pub struct ModelOutput {
    pub output: Tensor,
    pub attention_weights: Option<Tensor>,
    pub hidden_states: Option<Vec<Tensor>>,
    pub auxiliary: HashMap<String, Tensor>,
}
```

## Model Implementations

### 1. BaseGNN (`src/models/base_gnn.rs`)
- **Purpose**: Fundamental graph neural network with basic message passing
- **Features**: Configurable aggregation, activation functions, layer normalization
- **Use Case**: Baseline model and foundation for other implementations

### 2. Graph Attention Network (GAT) (`src/models/gat.rs`)
- **Purpose**: Attention-based graph neural network
- **Features**: Multi-head attention, attention weight visualization, interpretable representations
- **Mathematical Foundation**: 
  ```
  α_ij = softmax(LeakyReLU(a^T [W h_i || W h_j]))
  h_i' = σ(Σ_j α_ij W h_j)
  ```

### 3. Relational Graph Convolutional Network (RGCN) (`src/models/rgcn.rs`)
- **Purpose**: Multi-relational graph processing
- **Features**: Relation-specific transformations, basis decomposition, scalable to many relations
- **Mathematical Foundation**:
  ```
  h_i^(l+1) = σ(Σ_r Σ_j∈N_r(i) 1/c_i,r W_r^(l) h_j^(l) + W_0^(l) h_i^(l))
  ```

### 4. LightRDL (`src/models/lightrdl.rs`)
- **Purpose**: Lightweight relational deep learning
- **Features**: Memory-efficient design, fast inference, relation embeddings
- **Optimizations**: Reduced parameter count, efficient convolutions

### 5. StageGNN (`src/models/stagegnn.rs`)
- **Purpose**: Multi-stage graph processing
- **Features**: Edge-aware convolutions, staged message passing, advanced aggregation
- **Architecture**: Sequential processing stages with residual connections

## Support Modules

### 1. Candle Compatibility (`src/models/candle_utils.rs`)
Provides compatibility wrappers for the Candle framework:
- Safe tensor operations
- Layer creation helpers
- Device management utilities
- Activation function implementations

### 2. Checkpoint Management (`src/models/checkpoint.rs`)
Comprehensive model persistence system:
- Version tracking with git integration
- Optimizer state preservation
- Training metrics storage
- Compatibility validation

### 3. Embeddings (`src/models/embeddings.rs`)
Unified embedding system for relational data:
- Node type embeddings
- Relation embeddings
- Positional embeddings
- Temporal embeddings
- Categorical feature embeddings

## Factory Pattern

The `create_model()` function provides a unified entry point:

```rust
pub fn create_model(config: UnifiedModelConfig) -> Result<Box<dyn RelGTModel<Config = UnifiedModelConfig>>> {
    config.validate()?;
    
    let device = match config.device.device_type {
        DeviceType::CPU => Device::Cpu,
        DeviceType::CUDA(id) => Device::new_cuda(id)?,
        DeviceType::Metal => Device::new_metal(0)?,
        DeviceType::Auto => Device::cuda_if_available(0)?,
    };
    
    match config.model_type {
        ModelType::GAT => Ok(Box::new(GAT::new(config, vb)?)),
        ModelType::RGCN => Ok(Box::new(RGCN::new(config, vb)?)),
        // ... other models
    }
}
```

## Advanced Features

### 1. Uncertainty Quantification
All models support uncertainty estimation through:
- Monte Carlo dropout
- Ensemble predictions
- Confidence intervals

### 2. Model Interpretability
- Attention weight visualization
- Feature importance scoring
- Layer-wise representation analysis
- Gradient-based explanations

### 3. Batch Processing
Efficient handling of multiple inputs:
- Dynamic batching
- Memory optimization
- Parallel processing

### 4. Device Management
Automatic device selection and transfer:
- CPU/GPU detection
- Memory management
- Mixed precision support

## Performance Optimizations

### 1. Memory Efficiency
- Gradient checkpointing
- In-place operations where possible
- Efficient tensor storage

### 2. Computational Efficiency
- Vectorized operations
- Sparse tensor support
- Optimized message passing

### 3. Scalability
- Distributed training support
- Large graph handling
- Incremental learning

## Configuration Examples

### GAT Configuration
```rust
let config = UnifiedModelConfig::gat(256, 8, 3)
    .with_dropout(0.1)
    .with_attention_dropout(0.1)
    .with_device(DeviceType::Auto);
```

### RGCN Configuration
```rust
let config = UnifiedModelConfig::rgcn(128, 10, 2)
    .with_basis_decomposition(true)
    .with_regularization(0.01);
```

## Migration Guide

### From Legacy Models
1. **Configuration**: Convert old config to `UnifiedModelConfig`
2. **Interface**: Update method calls to use `RelGTModel` trait
3. **Input/Output**: Adapt to new `ModelInput`/`ModelOutput` structures
4. **Checkpoints**: Use new checkpoint system for model persistence

### Best Practices
1. **Always validate configurations** before model creation
2. **Use factory function** for model instantiation
3. **Leverage unified interface** for consistent behavior
4. **Monitor memory usage** with built-in utilities
5. **Save checkpoints regularly** during training

## Future Extensibility

The architecture supports easy addition of new models:
1. Implement `RelGTModel` trait
2. Add model type to `ModelType` enum
3. Update factory function
4. Add model-specific configuration options

## Testing Strategy

Each model includes comprehensive tests:
- Unit tests for individual components
- Integration tests for full workflows
- Performance benchmarks
- Memory usage validation

## Error Handling

Robust error handling throughout:
- Configuration validation
- Runtime error recovery
- Detailed error messages
- Graceful degradation

This unified architecture provides a solid foundation for relational graph neural networks while maintaining flexibility for future enhancements and research directions. 