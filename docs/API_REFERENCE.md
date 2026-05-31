# API Reference Documentation

## Overview

This document provides comprehensive API documentation for the GaussRelGT unified model architecture. All models implement the `RelGTModel` trait, providing a consistent interface across different graph neural network architectures.

## Core Traits and Interfaces

### RelGTModel Trait

The central trait that all models implement:

```rust
pub trait RelGTModel: Send + Sync {
    type Config: Clone + Serialize + for<'de> Deserialize<'de>;
    
    // Core functionality
    fn forward(&self, inputs: &ModelInput) -> Result<ModelOutput>;
    fn predict(&self, input: &Table) -> Result<Vec<f64>>;
    fn predict_with_uncertainty(&self, input: &Table) -> Result<(Vec<f64>, Vec<f64>)>;
    fn predict_batch(&self, inputs: &[Table]) -> Result<Vec<Vec<f64>>>;
    
    // Model explanation and interpretability
    fn explain(&self, input: &Table) -> Result<HashMap<String, f64>>;
    
    // Training and validation
    fn train_step(&mut self, input: &ModelInput, targets: &Tensor) -> Result<f64>;
    fn validate(&self, input: &ModelInput, targets: &Tensor) -> Result<HashMap<String, f64>>;
    
    // Model persistence
    fn save(&self, path: &str) -> Result<()>;
    fn load(&mut self, path: &str) -> Result<()>;
    
    // Model introspection
    fn parameter_count(&self) -> usize;
    fn memory_usage(&self) -> usize;
    fn summary(&self) -> ModelSummary;
    
    // Device management
    fn to_device(&mut self, device: &Device) -> Result<()>;
    fn set_training(&mut self, training: bool);
    
    // Configuration access
    fn config(&self) -> &Self::Config;
}
```

#### Method Details

##### `forward(inputs: &ModelInput) -> Result<ModelOutput>`
Performs a forward pass through the model.

**Parameters:**
- `inputs`: Unified input structure containing graph data

**Returns:**
- `ModelOutput`: Comprehensive output with primary results and auxiliary information

**Example:**
```rust
let input = ModelInput::homogeneous(node_features, edge_index);
let output = model.forward(&input)?;
let predictions = output.output;
```

##### `predict(input: &Table) -> Result<Vec<f64>>`
High-level prediction interface for tabular data.

**Parameters:**
- `input`: Table containing relational data

**Returns:**
- `Vec<f64>`: Prediction values

**Example:**
```rust
let predictions = model.predict(&table)?;
```

##### `predict_with_uncertainty(input: &Table) -> Result<(Vec<f64>, Vec<f64>)>`
Prediction with uncertainty quantification.

**Parameters:**
- `input`: Table containing relational data

**Returns:**
- `(Vec<f64>, Vec<f64>)`: Tuple of (predictions, uncertainties)

**Example:**
```rust
let (predictions, uncertainties) = model.predict_with_uncertainty(&table)?;
```

##### `explain(input: &Table) -> Result<HashMap<String, f64>>`
Generate explanations for model predictions.

**Parameters:**
- `input`: Table containing relational data

**Returns:**
- `HashMap<String, f64>`: Feature importance scores

**Example:**
```rust
let explanations = model.explain(&table)?;
for (feature, importance) in explanations {
    println!("{}: {:.4}", feature, importance);
}
```

## Data Structures

### ModelInput

Unified input structure supporting both homogeneous and heterogeneous graphs:

```rust
pub struct ModelInput {
    pub node_features: Tensor,              // Node feature matrix [N, F]
    pub edge_index: Tensor,                 // Edge connectivity [2, E]
    pub edge_features: Option<Tensor>,      // Edge features [E, D] (optional)
    pub edge_types: Option<Tensor>,         // Edge type labels [E] (optional)
    pub node_types: Option<Tensor>,         // Node type labels [N] (optional)
    pub batch: Option<Tensor>,              // Batch indices [N] (optional)
    pub metadata: HashMap<String, Tensor>,  // Additional tensors
}
```

#### Constructor Methods

##### `homogeneous(node_features: Tensor, edge_index: Tensor) -> Self`
Create input for homogeneous graphs.

**Example:**
```rust
let input = ModelInput::homogeneous(node_features, edge_index);
```

##### `heterogeneous(node_features: Tensor, edge_index: Tensor, edge_types: Tensor, node_types: Option<Tensor>) -> Self`
Create input for heterogeneous graphs.

**Example:**
```rust
let input = ModelInput::heterogeneous(
    node_features, 
    edge_index, 
    edge_types, 
    Some(node_types)
);
```

#### Builder Methods

##### `with_edge_features(self, edge_features: Tensor) -> Self`
Add edge features to the input.

##### `with_batch(self, batch: Tensor) -> Self`
Add batch information for batched graphs.

##### `with_metadata(self, key: String, value: Tensor) -> Self`
Add custom metadata tensors.

**Example:**
```rust
let input = ModelInput::homogeneous(node_features, edge_index)
    .with_edge_features(edge_features)
    .with_batch(batch_indices)
    .with_metadata("custom_data".to_string(), custom_tensor);
```

### ModelOutput

Comprehensive output structure with interpretability features:

```rust
pub struct ModelOutput {
    pub output: Tensor,                     // Primary output [N, H]
    pub attention_weights: Option<Tensor>,  // Attention weights (for GAT)
    pub hidden_states: Option<Vec<Tensor>>, // Layer-wise representations
    pub auxiliary: HashMap<String, Tensor>, // Additional outputs
}
```

#### Constructor Methods

##### `simple(output: Tensor) -> Self`
Create simple output with just the primary result.

##### `with_attention(self, attention_weights: Tensor) -> Self`
Add attention weights for interpretability.

##### `with_hidden_states(self, hidden_states: Vec<Tensor>) -> Self`
Add layer-wise hidden representations.

##### `with_auxiliary(self, key: String, value: Tensor) -> Self`
Add auxiliary outputs.

## Configuration System

### UnifiedModelConfig

Central configuration structure for all models:

```rust
pub struct UnifiedModelConfig {
    // Architecture parameters
    pub model_type: ModelType,
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub output_dim: usize,
    pub num_layers: usize,
    
    // Neural network parameters
    pub dropout: f64,
    pub activation: ActivationType,
    pub use_layer_norm: bool,
    pub use_batch_norm: bool,
    pub use_residual: bool,
    
    // Training parameters
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub gradient_clipping: Option<f64>,
    
    // Graph-specific parameters
    pub edge_dim: Option<usize>,
    pub num_relations: Option<usize>,
    pub num_heads: Option<usize>,
    pub use_edge_features: bool,
    
    // Advanced features
    pub use_attention: bool,
    pub attention_dropout: Option<f64>,
    pub use_gating: bool,
    pub use_skip_connections: bool,
    
    // Device and performance
    pub device: DeviceConfig,
    pub mixed_precision: bool,
    pub gradient_checkpointing: bool,
    
    // Model-specific configurations
    pub model_specific: HashMap<String, serde_json::Value>,
}
```

#### Factory Methods

##### `gat(hidden_dim: usize, num_heads: usize, num_layers: usize) -> Self`
Create GAT-specific configuration.

**Example:**
```rust
let config = UnifiedModelConfig::gat(256, 8, 3);
```

##### `rgcn(hidden_dim: usize, num_relations: usize, num_layers: usize) -> Self`
Create RGCN-specific configuration.

**Example:**
```rust
let config = UnifiedModelConfig::rgcn(256, 10, 3);
```

##### `lightrdl(hidden_dim: usize, num_layers: usize) -> Self`
Create LightRDL-specific configuration.

##### `stagegnn(hidden_dim: usize, edge_dim: usize, num_layers: usize) -> Self`
Create StageGNN-specific configuration.

#### Validation

##### `validate(&self) -> Result<()>`
Validate configuration parameters.

**Example:**
```rust
config.validate()?; // Returns error if invalid
```

## Model Factory

### create_model Function

Central factory function for creating models:

```rust
pub fn create_model(config: UnifiedModelConfig) -> Result<Box<dyn RelGTModel<Config = UnifiedModelConfig>>>
```

**Parameters:**
- `config`: Unified configuration specifying model type and parameters

**Returns:**
- `Box<dyn RelGTModel>`: Trait object implementing the unified interface

**Example:**
```rust
let config = UnifiedModelConfig::gat(256, 8, 3);
let model = create_model(config)?;
```

## Model Implementations

### BaseGNN

Basic graph neural network implementation.

**Configuration:**
```rust
let config = UnifiedModelConfig {
    model_type: ModelType::BaseGNN,
    hidden_dim: 256,
    num_layers: 3,
    activation: ActivationType::ReLU,
    use_layer_norm: true,
    ..Default::default()
};
```

### GAT (Graph Attention Networks)

Multi-head attention-based graph neural network.

**Special Features:**
- Returns attention weights in `ModelOutput.attention_weights`
- Supports multi-head attention
- Interpretable attention patterns

**Configuration:**
```rust
let config = UnifiedModelConfig::gat(256, 8, 3)
    .with_attention_dropout(0.1);
```

### RGCN (Relational Graph Convolutional Networks)

Multi-relational graph processing.

**Special Features:**
- Handles multiple edge types
- Basis decomposition for efficiency
- Relation-specific transformations

**Configuration:**
```rust
let config = UnifiedModelConfig::rgcn(256, 10, 3)
    .with_basis_decomposition(true);
```

### LightRDL

Lightweight relational deep learning.

**Special Features:**
- Memory-efficient design
- Fast inference
- Relation embeddings

### StageGNN

Multi-stage graph processing with edge awareness.

**Special Features:**
- Edge-aware convolutions
- Multi-stage processing
- Advanced aggregation

## Support Modules

### CandleHelper

Compatibility utilities for the Candle framework:

```rust
impl CandleHelper {
    pub fn linear(vb: VarBuilder, in_dim: usize, out_dim: usize, name: &str) -> Result<Linear>;
    pub fn layer_norm(vb: VarBuilder, normalized_shape: usize, name: &str, eps: f64) -> Result<LayerNorm>;
    pub fn dropout(tensor: &Tensor, dropout_prob: f64, training: bool) -> CandleResult<Tensor>;
    pub fn index_select(tensor: &Tensor, indices: &Tensor, dim: usize) -> CandleResult<Tensor>;
    pub fn scatter_add(tensor: &Tensor, indices: &Tensor, src: &Tensor, dim: usize) -> CandleResult<Tensor>;
    pub fn zeros(shape: &[usize], dtype: DType, device: &Device) -> CandleResult<Tensor>;
    pub fn ones(shape: &[usize], dtype: DType, device: &Device) -> CandleResult<Tensor>;
}
```

### Checkpoint Management

#### ModelCheckpoint

```rust
pub struct ModelCheckpoint {
    pub version: ModelVersion,
    pub config: UnifiedModelConfig,
    pub model_summary: ModelSummary,
    pub state_dict: HashMap<String, Vec<f32>>,
    pub optimizer_state: Option<OptimizerState>,
    pub training_metrics: TrainingMetrics,
}
```

**Methods:**
- `new<M: RelGTModel>(model: &M, optimizer_state: Option<OptimizerState>, metrics: TrainingMetrics) -> Result<Self>`
- `save(&self, path: &Path) -> Result<()>`
- `load(path: &Path) -> Result<Self>`
- `validate_compatibility(&self, current_config: &UnifiedModelConfig) -> Result<()>`

#### CheckpointManager

```rust
pub struct CheckpointManager {
    // Automatic checkpoint management
}
```

**Methods:**
- `new(checkpoint_dir: &Path, max_checkpoints: usize, save_optimizer_state: bool, save_best_only: bool) -> Result<Self>`
- `save_checkpoint<M: RelGTModel>(&self, model: &M, optimizer_state: Option<OptimizerState>, metrics: TrainingMetrics) -> Result<PathBuf>`
- `load_latest_checkpoint(&self) -> Result<Option<ModelCheckpoint>>`
- `load_best_checkpoint(&self) -> Result<Option<ModelCheckpoint>>`
- `list_checkpoints(&self) -> Result<Vec<ModelCheckpoint>>`

### Embeddings

#### RelationalEmbeddings

Unified embedding system for relational data:

```rust
pub struct RelationalEmbeddings {
    // Embedding components
}
```

**Methods:**
- `new(config: &RelationalEmbeddingConfig, device: Device, vb: VarBuilder) -> Result<Self>`
- `embed_node_types(&self, node_types: &[String], indices: &Tensor) -> CandleResult<Option<Tensor>>`
- `embed_relations(&self, relation_indices: &Tensor) -> CandleResult<Option<Tensor>>`
- `embed_positions(&self, positions: &Tensor) -> CandleResult<Option<Tensor>>`
- `embed_temporal(&self, day: &Tensor, month: &Tensor, year: &Tensor, hour: &Tensor) -> CandleResult<Option<Tensor>>`
- `embed_categorical(&self, feature_name: &str, indices: &Tensor) -> CandleResult<Option<Tensor>>`

## Error Handling

All functions return `Result<T>` types with comprehensive error information:

```rust
pub type Result<T> = std::result::Result<T, crate::error::Error>;
```

### Error Types

- **Configuration Errors**: Invalid model configurations
- **Runtime Errors**: Tensor operation failures
- **Device Errors**: GPU/CPU device issues
- **IO Errors**: File system operations
- **Model Errors**: Model-specific failures

### Error Handling Patterns

```rust
// Basic error handling
match model.forward(&input) {
    Ok(output) => println!("Success: {:?}", output.output.shape()),
    Err(e) => eprintln!("Error: {}", e),
}

// Using ? operator
let output = model.forward(&input)?;
let predictions = model.predict(&table)?;
```

## Device Management

### DeviceConfig

```rust
pub struct DeviceConfig {
    pub device_type: DeviceType,
    pub mixed_precision: bool,
    pub memory_limit: Option<usize>,
}

pub enum DeviceType {
    CPU,
    CUDA(usize),  // GPU ID
    Metal,
    Auto,         // Automatic selection
}
```

### Device Operations

```rust
// Automatic device selection
let config = UnifiedModelConfig {
    device: DeviceConfig {
        device_type: DeviceType::Auto,
        mixed_precision: true,
        memory_limit: Some(8192), // 8GB
    },
    ..Default::default()
};

// Manual device transfer
model.to_device(&Device::new_cuda(0)?)?;
```

## Performance Considerations

### Memory Management

- Use `gradient_checkpointing: true` for memory-constrained environments
- Enable `mixed_precision: true` for faster training
- Monitor memory usage with `model.memory_usage()`

### Batch Processing

```rust
// Efficient batch prediction
let batch_predictions = model.predict_batch(&tables)?;

// Manual batching for large datasets
for chunk in tables.chunks(batch_size) {
    let predictions = model.predict_batch(chunk)?;
    // Process predictions
}
```

### Device Optimization

```rust
// Automatic device selection
let config = UnifiedModelConfig {
    device: DeviceConfig {
        device_type: DeviceType::Auto,
        ..Default::default()
    },
    ..Default::default()
};
```

## Usage Examples

### Complete Workflow

```rust
use gaussrelgt::models::{create_model, UnifiedModelConfig, ModelType, ModelInput};

// 1. Create configuration
let config = UnifiedModelConfig::gat(256, 8, 3);

// 2. Create model
let mut model = create_model(config)?;

// 3. Prepare input
let input = ModelInput::homogeneous(node_features, edge_index)
    .with_edge_features(edge_features);

// 4. Forward pass
let output = model.forward(&input)?;

// 5. Get predictions
let predictions = model.predict(&table)?;

// 6. Get explanations
let explanations = model.explain(&table)?;

// 7. Save model
model.save("model.safetensors")?;
```

### Model Comparison

```rust
// Create different models with same configuration base
let base_config = UnifiedModelConfig {
    hidden_dim: 256,
    num_layers: 3,
    dropout: 0.1,
    ..Default::default()
};

let gat_config = UnifiedModelConfig { 
    model_type: ModelType::GAT,
    num_heads: Some(8),
    ..base_config.clone()
};

let rgcn_config = UnifiedModelConfig {
    model_type: ModelType::RGCN,
    num_relations: Some(10),
    ..base_config
};

let gat_model = create_model(gat_config)?;
let rgcn_model = create_model(rgcn_config)?;

// Both models have the same interface
let gat_predictions = gat_model.predict(&table)?;
let rgcn_predictions = rgcn_model.predict(&table)?;
```

This API reference provides comprehensive documentation for the unified GaussRelGT model architecture. All models implement the same interface, ensuring consistency and ease of use across different graph neural network architectures. 