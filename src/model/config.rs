use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};
use crate::Result;
use std::path::PathBuf;

/// Attention mechanism types supported by the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AttentionType {
    /// Standard scaled dot-product attention
    Standard,
    /// Memory-efficient flash attention
    Flash,
    /// Linear attention for O(n) complexity
    Linear,
    /// Hybrid local-global attention
    LocalGlobal,
    /// Sparse attention patterns
    Sparse,
}

/// Positional encoding types for sequence/graph positions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PositionalEncodingType {
    /// Learned positional embeddings
    Learned,
    /// Fixed sinusoidal encodings
    Sinusoidal,
    /// Rotary position embeddings
    Rotary,
    /// Attention with Linear Biases
    ALiBi,
}

/// Activation functions supported by the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActivationType {
    /// Rectified Linear Unit
    ReLU,
    /// Gaussian Error Linear Unit
    GELU,
    /// Swish-Gated Linear Unit
    SwiGLU,
    /// Gated Linear Unit
    GLU,
}

/// Parameter initialization schemes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InitScheme {
    /// Normal distribution initialization
    Normal { mean: f32, std: f32 },
    /// Uniform distribution initialization
    Uniform { min: f32, max: f32 },
    /// Xavier/Glorot initialization
    Xavier,
    /// Kaiming/He initialization
    Kaiming,
}

/// Learning rate schedule types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LRScheduleType {
    /// Constant learning rate
    Constant(f32),
    /// Linear learning rate decay
    Linear { start: f32, end: f32 },
    /// Cosine learning rate decay
    Cosine { min_lr: f32, max_lr: f32 },
    /// Cosine decay with warmup
    WarmupCosine {
        warmup_steps: usize,
        total_steps: usize,
        min_lr: f32,
        max_lr: f32,
    },
}

/// Optimizer configuration parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerConfig {
    pub lr: f32,
    pub weight_decay: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
}

/// Model version information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
    pub config_hash: String,
    pub timestamp: DateTime<Utc>,
}

/// Main model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    // Common parameters
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub hidden_dropout_prob: f64,
    pub max_position_embeddings: usize,
    pub training: bool,
    pub use_bias: bool,
    pub layer_norm_eps: f64,
    pub num_classes: usize,

    // Stage-GNN specific parameters
    pub edge_dim: usize,
    pub use_layer_norm: bool,
    pub model_type: ModelType,
    pub improved_gcn: bool,
    pub cache_gc: bool,
    pub add_self_loops: bool,
    pub normalize_graph: bool,
    pub edge_dropout: f64,
    pub residual_connections: bool,

    // Architecture settings
    pub num_attention_heads: usize,
    pub intermediate_dim: usize,
    pub attention_dropout_prob: f64,
    
    // Advanced features
    pub use_rotary_embeddings: bool,
    pub use_flash_attention: bool,
    pub gradient_checkpointing: bool,
    pub use_mixed_precision: bool,
    
    // Memory management
    pub memory_efficient_attention: bool,
    pub max_memory_mb: usize,
    pub batch_size_auto_scaling: bool,
    
    // Distributed training
    pub distributed_strategy: DistributedStrategy,
    pub world_size: usize,
    pub local_rank: usize,
    
    // Optimization
    pub optimizer_type: OptimizerType,
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub warmup_steps: usize,
    pub max_grad_norm: f64,
    
    // Graph processing
    pub max_graph_size: usize,
    pub sampling_strategy: SamplingStrategy,
    pub num_sampling_workers: usize,
    
    // Logging and monitoring
    pub log_level: LogLevel,
    pub metrics_collection_interval: usize,
    pub checkpoint_dir: PathBuf,
}

/// Distributed training strategies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistributedStrategy {
    /// Single device training
    SingleDevice,
    /// Data parallel training
    DataParallel,
    /// Model parallel training
    ModelParallel,
    /// Pipeline parallel training
    PipelineParallel,
}

/// Optimizer types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizerType {
    /// Adam optimizer
    Adam,
    /// AdamW optimizer with weight decay
    AdamW,
    /// Lion optimizer
    Lion,
    /// Adafactor optimizer
    Adafactor,
}

/// Graph sampling strategies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SamplingStrategy {
    /// Random node sampling
    Random,
    /// Importance-based sampling
    Importance,
    /// Temporal-aware sampling
    TemporalAware,
    /// Cluster-based sampling
    ClusterBased,
}

/// Logging levels.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogLevel {
    Debug,
    Info,
    Warning,
    Error,
}

/// Model types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelType {
    LightRDL,
    StageGNN,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 128,
            num_layers: 4,
            hidden_dropout_prob: 0.1,
            max_position_embeddings: 512,
            training: true,
            use_bias: true,
            layer_norm_eps: 1e-12,
            num_classes: 1,
            edge_dim: 64,
            use_layer_norm: true,
            model_type: ModelType::LightRDL,
            improved_gcn: true,
            cache_gc: true,
            add_self_loops: true,
            normalize_graph: true,
            edge_dropout: 0.1,
            residual_connections: true,
            num_attention_heads: 12,
            intermediate_dim: 3072,
            attention_dropout_prob: 0.1,
            use_rotary_embeddings: true,
            use_flash_attention: true,
            gradient_checkpointing: false,
            use_mixed_precision: true,
            memory_efficient_attention: true,
            max_memory_mb: 16384,
            batch_size_auto_scaling: true,
            distributed_strategy: DistributedStrategy::SingleDevice,
            world_size: 1,
            local_rank: 0,
            optimizer_type: OptimizerType::AdamW,
            learning_rate: 1e-4,
            weight_decay: 0.01,
            warmup_steps: 1000,
            max_grad_norm: 1.0,
            max_graph_size: 1_000_000,
            sampling_strategy: SamplingStrategy::TemporalAware,
            num_sampling_workers: 4,
            log_level: LogLevel::Info,
            metrics_collection_interval: 100,
            checkpoint_dir: PathBuf::from("checkpoints"),
        }
    }
}

impl ModelConfig {
    /// Creates a new StageGNN configuration.
    pub fn new_stagegnn(hidden_dim: usize, edge_dim: usize, num_layers: usize) -> Self {
        Self {
            hidden_dim,
            edge_dim,
            num_layers,
            model_type: ModelType::StageGNN,
            ..Default::default()
        }
    }

    /// Sets the hidden dimension.
    pub fn with_hidden_dim(mut self, dim: usize) -> Self {
        self.hidden_dim = dim;
        self
    }
    
    /// Sets the number of layers.
    pub fn with_num_layers(mut self, layers: usize) -> Self {
        self.num_layers = layers;
        self
    }
    
    /// Sets the number of attention heads.
    pub fn with_attention_heads(mut self, heads: usize) -> Self {
        self.num_attention_heads = heads;
        self
    }
    
    /// Sets the optimizer type.
    pub fn with_optimizer(mut self, optimizer: OptimizerType) -> Self {
        self.optimizer_type = optimizer;
        self
    }
    
    /// Sets the distributed training strategy.
    pub fn with_distributed_strategy(mut self, strategy: DistributedStrategy) -> Self {
        self.distributed_strategy = strategy;
        self
    }
    
    /// Sets the sampling strategy.
    pub fn with_sampling_strategy(mut self, strategy: SamplingStrategy) -> Self {
        self.sampling_strategy = strategy;
        self
    }
    
    /// Validates distributed training configuration.
    pub fn validate_distributed(&self) -> Result<(), String> {
        match self.distributed_strategy {
            DistributedStrategy::SingleDevice => {
                if self.world_size != 1 {
                    return Err("world_size must be 1 for SingleDevice strategy".to_string());
                }
            }
            DistributedStrategy::DataParallel | DistributedStrategy::ModelParallel | DistributedStrategy::PipelineParallel => {
                if self.world_size < 2 {
                    return Err("world_size must be at least 2 for distributed strategies".to_string());
                }
                if self.local_rank >= self.world_size {
                    return Err("local_rank must be less than world_size".to_string());
                }
            }
        }
        Ok(())
    }

    /// Validates the configuration parameters.
    pub fn validate(&self) -> Result<(), String> {
        // Validate dimensions
        if self.hidden_dim == 0 {
            return Err("hidden_dim must be greater than 0".to_string());
        }
        
        if self.hidden_dim % self.num_attention_heads != 0 {
            return Err("hidden_dim must be divisible by num_attention_heads".to_string());
        }
        
        // Validate learning parameters
        if self.learning_rate <= 0.0 || self.learning_rate >= 1.0 {
            return Err("learning_rate must be between 0 and 1".to_string());
        }
        
        if self.weight_decay < 0.0 {
            return Err("weight_decay must be non-negative".to_string());
        }
        
        if self.max_grad_norm <= 0.0 {
            return Err("max_grad_norm must be positive".to_string());
        }
        
        // Validate memory settings
        if self.max_memory_mb < 1024 {
            return Err("max_memory_mb must be at least 1024".to_string());
        }

        // Validate distributed settings
        self.validate_distributed()?;
        
        Ok(())
    }

    /// Converts the configuration to a versioned JSON string.
    pub fn to_versioned_string(&self) -> Result<String, serde_json::Error> {
        #[derive(Serialize)]
        struct VersionedConfig {
            version: String,
            config: ModelConfig,
            timestamp: DateTime<Utc>,
        }

        let versioned = VersionedConfig {
            version: env!("CARGO_PKG_VERSION").to_string(),
            config: self.clone(),
            timestamp: Utc::now(),
        };

        serde_json::to_string_pretty(&versioned)
    }

    /// Creates a configuration from a versioned JSON string.
    pub fn from_versioned_string(json: &str) -> Result<Self, Box<dyn std::error::Error>> {
        #[derive(Deserialize)]
        struct VersionedConfig {
            version: String,
            config: ModelConfig,
            timestamp: DateTime<Utc>,
        }

        let versioned: VersionedConfig = serde_json::from_str(json)?;
        
        // Version compatibility check
        let current_version = env!("CARGO_PKG_VERSION");
        if versioned.version != current_version {
            eprintln!("Warning: Config version mismatch. Current: {}, Config: {}", 
                     current_version, versioned.version);
        }

        Ok(versioned.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ModelConfig::default();
        assert_eq!(config.hidden_dim, 128);
        assert_eq!(config.num_layers, 4);
        assert_eq!(config.edge_dim, 64);
    }

    #[test]
    fn test_stagegnn_config() {
        let config = ModelConfig::new_stagegnn(256, 128, 6);
        assert_eq!(config.hidden_dim, 256);
        assert_eq!(config.edge_dim, 128);
        assert_eq!(config.num_layers, 6);
        assert!(matches!(config.model_type, ModelType::StageGNN));
    }

    #[test]
    fn test_config_validation() {
        let mut config = ModelConfig::default();
        assert!(config.validate().is_ok());

        // Test invalid hidden_dim
        config.hidden_dim = 0;
        assert!(config.validate().is_err());

        // Test invalid learning_rate
        config = ModelConfig::default();
        config.learning_rate = 2.0;
        assert!(config.validate().is_err());

        // Test invalid distributed settings
        config = ModelConfig::default();
        config.distributed_strategy = DistributedStrategy::DataParallel;
        config.world_size = 1;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_serialization() {
        let config = ModelConfig::default();
        let json = config.to_versioned_string().unwrap();
        let deserialized = ModelConfig::from_versioned_string(&json).unwrap();
        assert_eq!(deserialized.hidden_dim, config.hidden_dim);
        assert_eq!(deserialized.num_layers, config.num_layers);
    }
} 