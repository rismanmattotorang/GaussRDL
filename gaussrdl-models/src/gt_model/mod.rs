// src/model/mod.rs
use candle_core::Device;
use candle_nn::VarBuilder;
use crate::{Result, ModelInput, ModelOutput, RelGTModel, ModelSummary};
use gaussrdl_core::Table;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

// Re-export main modules
pub mod relgt;
pub mod tokenizer;
pub mod embeddings;
pub mod transformer;
pub mod attention;
pub mod config;
pub mod encoders;
pub mod vector_quantizer;
pub mod local_module;

// Re-export main types
pub use relgt::{RelgtModel, RelgtConfig, TaskType, ConvType};
pub use encoders::*;
pub use vector_quantizer::VectorQuantizerEMA;
pub use local_module::{LocalModule, EncoderLayer, FeedForwardNetwork};

/// Training batch structure
#[derive(Debug, Clone)]
pub struct TrainingBatch {
    pub neighbor_types: Vec<i64>,
    pub node_indices: Vec<i64>,
    pub neighbor_hops: Vec<i64>,
    pub neighbor_times: Vec<f32>,
    pub grouped_tf_dict: HashMap<String, Vec<f32>>,
    pub edge_index: Vec<(i64, i64)>,
    pub batch: Vec<i64>,
    pub targets: Vec<f32>,
}

impl TrainingBatch {
    pub fn new() -> Self {
        Self {
            neighbor_types: Vec::new(),
            node_indices: Vec::new(),
            neighbor_hops: Vec::new(),
            neighbor_times: Vec::new(),
            grouped_tf_dict: HashMap::new(),
            edge_index: Vec::new(),
            batch: Vec::new(),
            targets: Vec::new(),
        }
    }
    
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            neighbor_types: Vec::with_capacity(capacity),
            node_indices: Vec::with_capacity(capacity),
            neighbor_hops: Vec::with_capacity(capacity),
            neighbor_times: Vec::with_capacity(capacity),
            grouped_tf_dict: HashMap::new(),
            edge_index: Vec::with_capacity(capacity * 2),
            batch: Vec::with_capacity(capacity),
            targets: Vec::with_capacity(capacity),
        }
    }
    
    pub fn add_sample(
        &mut self,
        neighbor_types: Vec<i64>,
        node_indices: Vec<i64>,
        neighbor_hops: Vec<i64>,
        neighbor_times: Vec<f32>,
        grouped_tf_dict: HashMap<String, Vec<f32>>,
        edge_index: Vec<(i64, i64)>,
        batch: Vec<i64>,
        target: f32,
    ) {
        self.neighbor_types.extend(neighbor_types);
        self.node_indices.extend(node_indices);
        self.neighbor_hops.extend(neighbor_hops);
        self.neighbor_times.extend(neighbor_times);
        
        // Merge grouped_tf_dict
        for (key, values) in grouped_tf_dict {
            self.grouped_tf_dict.entry(key).or_insert_with(Vec::new).extend(values);
        }
        
        self.edge_index.extend(edge_index);
        self.batch.extend(batch);
        self.targets.push(target);
    }
    
    pub fn len(&self) -> usize {
        self.targets.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.targets.is_empty()
    }
    
    pub fn clear(&mut self) {
        self.neighbor_types.clear();
        self.node_indices.clear();
        self.neighbor_hops.clear();
        self.neighbor_times.clear();
        self.grouped_tf_dict.clear();
        self.edge_index.clear();
        self.batch.clear();
        self.targets.clear();
    }
}

/// Model configuration trait
pub trait ModelConfig: Clone + Serialize + for<'de> Deserialize<'de> {
    fn default_for_data(data: &TrainingBatch) -> Result<Self> where Self: Sized;
    fn validate(&self) -> Result<()>;
}

/// Model factory trait
pub trait ModelFactory {
    type Config: ModelConfig;
    type Model: RelGTModel<Config = Self::Config>;
    
    fn create_model(config: Self::Config, vb: VarBuilder) -> Result<Self::Model>;
    fn create_config() -> Self::Config;
}

/// Default model factory implementation
pub struct DefaultModelFactory;

impl ModelFactory for DefaultModelFactory {
    type Config = RelgtConfig;
    type Model = RelgtModel;
    
    fn create_model(config: Self::Config, vb: VarBuilder) -> Result<Self::Model> {
        RelgtModel::new(config, vb.device().clone(), &vb)
    }
    
    fn create_config() -> Self::Config {
        RelgtConfig::default()
    }
}

/// Model builder for fluent API
pub struct ModelBuilder<F = DefaultModelFactory> 
where 
    F: ModelFactory 
{
    factory: F,
    config: Option<<F as ModelFactory>::Config>,
}

impl<F: ModelFactory> ModelBuilder<F> {
    pub fn new(factory: F) -> Self {
        Self {
            factory,
            config: None,
        }
    }
    
    pub fn with_config(mut self, config: <F as ModelFactory>::Config) -> Self {
        self.config = Some(config);
        self
    }
    
    pub fn build(self, vb: VarBuilder) -> Result<<F as ModelFactory>::Model> {
        let config = self.config.unwrap_or_else(|| F::create_config());
        F::create_model(config, vb)
    }
}

impl Default for ModelBuilder {
    fn default() -> Self {
        Self::new(DefaultModelFactory)
    }
}

/// Model utilities
pub mod utils {
    use super::*;
    
    /// Create a default RelGT model
    pub fn create_default_relgt_model(vb: VarBuilder) -> Result<RelgtModel> {
        let config = RelgtConfig::default();
        RelgtModel::new(config, vb.device().clone(), &vb)
    }
    
    /// Create a RelGT model with custom configuration
    pub fn create_relgt_model(config: RelgtConfig, vb: VarBuilder) -> Result<RelgtModel> {
        RelgtModel::new(config, vb.device().clone(), &vb)
    }
    
    /// Validate model configuration
    pub fn validate_config(config: &RelgtConfig) -> Result<()> {
        if config.hidden_dim == 0 {
            return Err(crate::ModelError::Config("hidden_dim cannot be 0".to_string()));
        }
        
        if config.num_heads == 0 {
            return Err(crate::ModelError::Config("num_heads cannot be 0".to_string()));
        }
        
        if config.hidden_dim % config.num_heads != 0 {
            return Err(crate::ModelError::Config("hidden_dim must be divisible by num_heads".to_string()));
        }
        
        if config.dropout_rate < 0.0 || config.dropout_rate > 1.0 {
            return Err(crate::ModelError::Config("dropout_rate must be between 0.0 and 1.0".to_string()));
        }
        
        if config.attention_dropout < 0.0 || config.attention_dropout > 1.0 {
            return Err(crate::ModelError::Config("attention_dropout must be between 0.0 and 1.0".to_string()));
        }
        
        Ok(())
    }
}
