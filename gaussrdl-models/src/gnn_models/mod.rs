use serde::{Serialize, Deserialize};
use crate::Result;
use gaussrdl_core::Table;
use std::collections::HashMap;
use candle_core::{Device, Tensor, DType};
use candle_nn::VarBuilder;
use candle_core::error::Error;
use candle_nn::VarMap;

pub mod base_gnn;
pub mod gat;
pub mod rgcn;
pub mod lightrdl;
pub mod stagegnn;
pub mod candle_utils;
pub mod checkpoint;
pub mod embeddings;

// Re-export model implementations
pub use base_gnn::{BaseGNN, BaseGNNConfig};
pub use gat::{GAT, GATConfig};
pub use rgcn::{RGCN, RGCNConfig};
pub use lightrdl::{LightRDL, LightRDLConfig};
pub use stagegnn::{StageGNN, StageGNNConfig};
pub use candle_utils::CandleHelper;
pub use checkpoint::{ModelCheckpoint, CheckpointManager, TrainingMetrics, OptimizerState};
pub use embeddings::{RelationalEmbeddings, RelationalEmbeddingConfig};

/// Device types for model execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeviceType {
    CPU,
    CUDA(usize),
    Metal,
    Auto,
}

/// Device configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceConfig {
    pub device_type: DeviceType,
    pub memory_limit: Option<usize>,
    pub use_mixed_precision: bool,
}

impl Default for DeviceConfig {
    fn default() -> Self {
        Self {
            device_type: DeviceType::Auto,
            memory_limit: None,
            use_mixed_precision: false,
        }
    }
}

/// Model types supported by the system
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelType {
    RGCN,
    StageGNN,
    LightRDL,
    GAT,
    BaseGNN,
    Custom(String),
}

impl std::fmt::Display for ModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelType::RGCN => write!(f, "RGCN"),
            ModelType::StageGNN => write!(f, "StageGNN"),
            ModelType::LightRDL => write!(f, "LightRDL"),
            ModelType::GAT => write!(f, "GAT"),
            ModelType::BaseGNN => write!(f, "BaseGNN"),
            ModelType::Custom(name) => write!(f, "{}", name),
        }
    }
}

/// Activation function types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActivationType {
    ReLU,
    GELU,
    Tanh,
    Sigmoid,
    LeakyReLU,
    Swish,
}

impl std::fmt::Display for ActivationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ActivationType::ReLU => write!(f, "relu"),
            ActivationType::GELU => write!(f, "gelu"),
            ActivationType::Tanh => write!(f, "tanh"),
            ActivationType::Sigmoid => write!(f, "sigmoid"),
            ActivationType::LeakyReLU => write!(f, "leaky_relu"),
            ActivationType::Swish => write!(f, "swish"),
        }
    }
}

/// Input to a model forward pass
#[derive(Debug, Clone)]
pub struct ModelInput {
    pub node_features: Tensor,
    pub edge_index: Tensor,
    pub edge_types: Option<Tensor>,
    pub edge_features: Option<Tensor>,
    pub node_types: Option<Tensor>,
    pub batch: Option<Tensor>,
}

impl ModelInput {
    /// Create homogeneous graph input
    pub fn homogeneous(node_features: Tensor, edge_index: Tensor) -> Self {
        Self {
            node_features,
            edge_index,
            edge_types: None,
            edge_features: None,
            node_types: None,
            batch: None,
        }
    }

    /// Create heterogeneous graph input
    pub fn heterogeneous(
        node_features: Tensor, 
        edge_index: Tensor, 
        edge_types: Tensor, 
        node_types: Option<Tensor>
    ) -> Self {
        Self {
            node_features,
            edge_index,
            edge_types: Some(edge_types),
            edge_features: None,
            node_types,
            batch: None,
        }
    }

    /// Add edge types
    pub fn with_edge_types(mut self, edge_types: Tensor) -> Self {
        self.edge_types = Some(edge_types);
        self
    }

    /// Add edge features
    pub fn with_edge_features(mut self, edge_features: Tensor) -> Self {
        self.edge_features = Some(edge_features);
        self
    }

    /// Add node types
    pub fn with_node_types(mut self, node_types: Tensor) -> Self {
        self.node_types = Some(node_types);
        self
    }

    /// Add batch information
    pub fn with_batch(mut self, batch: Tensor) -> Self {
        self.batch = Some(batch);
        self
    }

    /// Get number of nodes - simple implementation
    pub fn num_nodes(&self) -> Result<usize> {
        self.node_features.dim(0)
            .map_err(|e| crate::ModelError::Candle(e))
    }

    /// Get number of edges - simple implementation  
    pub fn num_edges(&self) -> Result<usize> {
        self.edge_index.dim(1)
            .map_err(|e| crate::ModelError::Candle(e))
    }

    /// Get feature dimension - simple implementation
    pub fn feature_dim(&self) -> Result<usize> {
        self.node_features.dim(1)
            .map_err(|e| crate::ModelError::Candle(e))
    }

    /// Add metadata (placeholder implementation)
    pub fn with_metadata(self, _key: String, _value: Tensor) -> Self {
        // Placeholder - metadata could be stored in a HashMap if needed
        self
    }
}

/// Output from a model forward pass
#[derive(Debug, Clone)]
pub struct ModelOutput {
    pub output: Tensor,
    pub hidden_states: Option<Vec<Tensor>>,
    pub attention_weights: Option<Tensor>,
    pub graph_embedding: Option<Tensor>,
}

impl ModelOutput {
    /// Create simple output with just the main tensor
    pub fn simple(output: Tensor) -> Self {
        Self {
            output,
            hidden_states: None,
            attention_weights: None,
            graph_embedding: None,
        }
    }

    /// Add hidden states
    pub fn with_hidden_states(mut self, hidden_states: Vec<Tensor>) -> Self {
        self.hidden_states = Some(hidden_states);
        self
    }

    /// Add attention weights
    pub fn with_attention_weights(mut self, attention_weights: Tensor) -> Self {
        self.attention_weights = Some(attention_weights);
        self
    }

    /// Add graph-level embedding
    pub fn with_graph_embedding(mut self, graph_embedding: Tensor) -> Self {
        self.graph_embedding = Some(graph_embedding);
        self
    }

    /// Get the shape of the output tensor
    pub fn shape(&self) -> &candle_core::Shape {
        self.output.shape()
    }
}

/// Information about a layer in the model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerInfo {
    pub name: String,
    pub layer_type: String,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub parameters: usize,
}

/// Summary of model architecture and parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSummary {
    pub model_type: ModelType,
    pub total_parameters: usize,
    pub trainable_parameters: usize,
    pub memory_usage_mb: f64,
    pub layers: Vec<LayerInfo>,
    pub architecture_details: HashMap<String, String>,
}

/// Unified configuration for all model types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedModelConfig {
    pub model_type: ModelType,
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub dropout: f64,
    pub activation: ActivationType,
    pub use_layer_norm: bool,
    pub use_residual: bool,
    pub use_edge_features: bool,
    pub num_relations: Option<usize>,
    pub edge_dim: Option<usize>,
    pub num_bases: Option<usize>,
    pub learning_rate: f64,
    pub weight_decay: f64,
    // Additional fields that other code expects
    pub input_dim: Option<usize>,
    pub output_dim: Option<usize>,
    pub use_batch_norm: bool,
    pub gradient_clipping: Option<f64>,
    pub num_heads: Option<usize>,
    pub use_attention: bool,
    pub attention_dropout: Option<f64>,
    pub device: DeviceConfig,
}

impl Default for UnifiedModelConfig {
    fn default() -> Self {
        Self {
            model_type: ModelType::RGCN,
            hidden_dim: 256,
            num_layers: 3,
            dropout: 0.1,
            activation: ActivationType::ReLU,
            use_layer_norm: true,
            use_residual: true,
            use_edge_features: true,
            num_relations: Some(10),
            edge_dim: Some(64),
            num_bases: Some(30),
            learning_rate: 0.001,
            weight_decay: 1e-4,
            input_dim: None,
            output_dim: None,
            use_batch_norm: false,
            gradient_clipping: None,
            num_heads: Some(8),
            use_attention: false,
            attention_dropout: Some(0.1),
            device: DeviceConfig::default(),
        }
    }
}

impl UnifiedModelConfig {
    /// Create RGCN configuration
    pub fn rgcn(hidden_dim: usize, num_relations: usize, num_bases: usize) -> Self {
        Self {
            model_type: ModelType::RGCN,
            hidden_dim,
            num_relations: Some(num_relations),
            num_bases: Some(num_bases),
            ..Default::default()
        }
    }

    /// Create StageGNN configuration
    pub fn stagegnn(hidden_dim: usize, num_layers: usize) -> Self {
        Self {
            model_type: ModelType::StageGNN,
            hidden_dim,
            num_layers,
            ..Default::default()
        }
    }

    /// Create LightRDL configuration
    pub fn lightrdl(hidden_dim: usize, num_relations: usize) -> Self {
        Self {
            model_type: ModelType::LightRDL,
            hidden_dim,
            num_relations: Some(num_relations),
            ..Default::default()
        }
    }

    /// Create GAT configuration
    pub fn gat(hidden_dim: usize, num_heads: usize) -> Self {
        Self {
            model_type: ModelType::GAT,
            hidden_dim,
            num_heads: Some(num_heads),
            use_attention: true,
            ..Default::default()
        }
    }

    /// Validate configuration (simple implementation)
    pub fn validate(&self) -> Result<()> {
        if self.hidden_dim == 0 {
            return Err(crate::ModelError::Candle(candle_core::Error::Msg("hidden_dim must be greater than 0".to_string())));
        }
        if self.num_layers == 0 {
            return Err(crate::ModelError::Candle(candle_core::Error::Msg("num_layers must be greater than 0".to_string())));
        }
        if !(0.0..=1.0).contains(&self.dropout) {
            return Err(crate::ModelError::Candle(candle_core::Error::Msg("dropout must be between 0.0 and 1.0".to_string())));
        }
        Ok(())
    }
}

/// Trait for relational graph transformer models
pub trait RelGTModel: Send + Sync {
    type Config: Clone + Send + Sync;

    /// Create a new model instance
    fn new(config: Self::Config, vb: VarBuilder) -> Result<Self> where Self: Sized;

    /// Forward pass through the model
    fn forward(&self, input: &ModelInput) -> Result<ModelOutput>;

    /// Make predictions on tabular data
    fn predict(&self, input: &Table) -> Result<Vec<f64>>;

    /// Make predictions with uncertainty estimation
    fn predict_with_uncertainty(&self, input: &Table) -> Result<(Vec<f64>, Vec<f64>)>;

    /// Batch predictions
    fn predict_batch(&self, inputs: &[Table]) -> Result<Vec<Vec<f64>>>;

    /// Generate explanations for predictions
    fn explain(&self, input: &Table) -> Result<HashMap<String, f64>>;

    /// Perform a training step
    fn train_step(&mut self, input: &ModelInput, targets: &Tensor) -> Result<f64>;

    /// Validate the model
    fn validate(&self, input: &ModelInput, targets: &Tensor) -> Result<HashMap<String, f64>>;

    /// Save model to file
    fn save(&self, path: &str) -> Result<()>;

    /// Load model from file
    fn load(&mut self, path: &str) -> Result<()>;

    /// Get the number of parameters
    fn parameter_count(&self) -> usize;

    /// Get memory usage in bytes
    fn memory_usage(&self) -> usize;

    /// Move model to device
    fn to_device(&mut self, device: &Device) -> Result<()>;

    /// Set training mode
    fn set_training(&mut self, training: bool);

    /// Get model configuration
    fn config(&self) -> &Self::Config;

    /// Get model summary
    fn summary(&self) -> ModelSummary;
}

/// Utility functions for model creation
pub fn create_model(config: UnifiedModelConfig) -> Result<Box<dyn RelGTModel<Config = UnifiedModelConfig>>> {
    config.validate()?;
    
    let device = match config.device.device_type {
        DeviceType::CPU => Device::Cpu,
        DeviceType::CUDA(id) => Device::new_cuda(id)?,
        DeviceType::Metal => Device::new_metal(0)?,
        DeviceType::Auto => Device::cuda_if_available(0)?,
    };
    
    let var_map = VarMap::new();
    let vb = VarBuilder::from_varmap(&var_map, DType::F32, &device);
    
    match config.model_type {
        ModelType::GAT => {
            let gat_config = GATConfig::from_unified(&config)?;
            let model = GAT::new(gat_config, vb)?;
            Ok(Box::new(ModelWrapper::new(model, config)))
        }
        ModelType::RGCN => {
            let rgcn_config = RGCNConfig::from_unified(&config)?;
            let model = RGCN::new(rgcn_config, vb)?;
            Ok(Box::new(ModelWrapper::new(model, config)))
        }
        ModelType::LightRDL => {
            let lightrdl_config = LightRDLConfig::from_unified(&config)?;
            let model = LightRDL::new(lightrdl_config, vb)?;
            Ok(Box::new(ModelWrapper::new(model, config)))
        }
        ModelType::StageGNN => {
            let stagegnn_config = StageGNNConfig::from_unified(&config)?;
            let model = StageGNN::new(stagegnn_config, vb)?;
            Ok(Box::new(ModelWrapper::new(model, config)))
        }
        ModelType::BaseGNN => {
            let base_config = BaseGNNConfig::from_unified(&config)?;
            let model = BaseGNN::new(base_config, vb)?;
            Ok(Box::new(ModelWrapper::new(model, config)))
        }
        ModelType::Custom(_) => {
            Err(crate::ModelError::Candle(candle_core::Error::Msg("Custom models not yet supported".to_string())))
        }
    }
}

/// Wrapper to make individual models compatible with the unified interface
pub struct ModelWrapper<M> {
    model: M,
    config: UnifiedModelConfig,
}

impl<M> ModelWrapper<M> {
    pub fn new(model: M, config: UnifiedModelConfig) -> Self {
        Self { model, config }
    }
}

// Implement RelGTModel for ModelWrapper to enable trait object usage
impl<M> RelGTModel for ModelWrapper<M> 
where 
    M: RelGTModel + Send + Sync,
    M::Config: Clone + Serialize + for<'de> Deserialize<'de>,
{
    type Config = UnifiedModelConfig;
    
    fn new(_config: Self::Config, _vb: VarBuilder) -> Result<Self> where Self: Sized {
        unimplemented!("Default implementation not provided")
    }
    
    fn forward(&self, inputs: &ModelInput) -> Result<ModelOutput> {
        self.model.forward(inputs)
    }
    
    fn predict(&self, input: &Table) -> Result<Vec<f64>> {
        self.model.predict(input)
    }
    
    fn predict_with_uncertainty(&self, input: &Table) -> Result<(Vec<f64>, Vec<f64>)> {
        self.model.predict_with_uncertainty(input)
    }
    
    fn predict_batch(&self, inputs: &[Table]) -> Result<Vec<Vec<f64>>> {
        self.model.predict_batch(inputs)
    }
    
    fn explain(&self, input: &Table) -> Result<HashMap<String, f64>> {
        self.model.explain(input)
    }
    
    fn train_step(&mut self, input: &ModelInput, targets: &Tensor) -> Result<f64> {
        self.model.train_step(input, targets)
    }
    
    fn validate(&self, input: &ModelInput, targets: &Tensor) -> Result<HashMap<String, f64>> {
        self.model.validate(input, targets)
    }
    
    fn save(&self, path: &str) -> Result<()> {
        self.model.save(path)
    }
    
    fn load(&mut self, path: &str) -> Result<()> {
        self.model.load(path)
    }
    
    fn parameter_count(&self) -> usize {
        self.model.parameter_count()
    }
    
    fn memory_usage(&self) -> usize {
        self.model.memory_usage()
    }
    
    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.model.to_device(device)
    }
    
    fn set_training(&mut self, training: bool) {
        self.model.set_training(training)
    }
    
    fn config(&self) -> &Self::Config {
        &self.config
    }
    
    fn summary(&self) -> ModelSummary {
        self.model.summary()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unified_config_creation() {
        let config = UnifiedModelConfig::gat(256, 8);
        assert_eq!(config.model_type, ModelType::GAT);
        assert_eq!(config.hidden_dim, 256);
        assert_eq!(config.num_heads, Some(8));
        assert_eq!(config.num_layers, 3);
        assert!(config.use_attention);
    }

    #[test]
    fn test_config_validation() {
        let mut config = UnifiedModelConfig::default();
        assert!(config.validate().is_ok());
        
        config.hidden_dim = 0;
        assert!(config.validate().is_err());
        
        config.hidden_dim = 256;
        config.dropout = 1.5;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_model_input_creation() -> Result<()> {
        let node_features = Tensor::zeros((10, 64), DType::F32, &Device::Cpu)?;
        let edge_index = Tensor::zeros((2, 20), DType::I64, &Device::Cpu)?;
        
        let input = ModelInput::homogeneous(node_features, edge_index);
        assert_eq!(input.num_nodes()?, 10);
        assert_eq!(input.num_edges()?, 20);
        assert_eq!(input.feature_dim()?, 64);
        
        Ok(())
    }
} 