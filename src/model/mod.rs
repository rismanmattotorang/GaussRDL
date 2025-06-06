// src/model/mod.rs
use candle_core::Device;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use crate::Result;

pub mod relgt;
pub mod tokenizer;
pub mod transformer;
pub mod attention;
pub mod embeddings;

pub use relgt::*;
pub use tokenizer::*;
pub use transformer::*;
pub use attention::*;
pub use embeddings::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelgtConfig {
    // Core architecture (matching paper Section 3)
    pub hidden_dim: usize,                    // d in paper
    pub num_attention_heads: usize,           // Multi-head attention
    pub num_transformer_layers: usize,        // L in paper (Eq. 7)
    pub intermediate_size: usize,             // Feed-forward dimension
    
    // Graph-specific parameters (Section 3.1)
    pub max_sequence_length: usize,
    pub k_neighbors: usize,                   // K in paper (fixed-size sampling)
    pub max_hop_distance: usize,              // Maximum hops for sampling
    pub num_node_types: usize,                // |T| for node type embeddings
    pub num_global_centroids: usize,          // B in paper (Eq. 8)
    
    // Multi-element tokenization (Section 3.1.1)
    pub vocab_sizes: HashMap<String, usize>,
    pub max_text_length: usize,
    pub temporal_encoding_dim: usize,
    
    // Training parameters
    pub dropout_rate: f64,
    pub attention_dropout_rate: f64,
    pub hidden_dropout_rate: f64,
    pub layer_norm_epsilon: f64,
    
    // Task-specific parameters
    pub num_classes: usize,
    pub task_type: TaskType,
    
    // Advanced features (beyond paper)
    pub use_mixed_precision: bool,
    pub use_gradient_checkpointing: bool,
    pub use_flash_attention: bool,
    pub use_rotary_embeddings: bool,
    pub use_gated_mlp: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskType {
    BinaryClassification,
    MultiClassification,
    Regression,
    Ranking,
}

impl Default for RelgtConfig {
    fn default() -> Self {
        Self {
            // Paper defaults (Section 4.1)
            hidden_dim: 128,                  // d=128 from paper
            num_attention_heads: 8,
            num_transformer_layers: 4,        // L=4 from paper
            intermediate_size: 512,
            max_sequence_length: 512,
            k_neighbors: 300,                 // K=300 from paper
            max_hop_distance: 2,              // 2-hop sampling from paper
            num_node_types: 16,
            num_global_centroids: 4096,       // B=4096 from paper
            vocab_sizes: HashMap::new(),
            max_text_length: 128,
            temporal_encoding_dim: 64,
            dropout_rate: 0.1,
            attention_dropout_rate: 0.1,
            hidden_dropout_rate: 0.1,
            layer_norm_epsilon: 1e-12,
            num_classes: 2,
            task_type: TaskType::BinaryClassification,
            // Advanced features (disabled by default for stability)
            use_mixed_precision: false,
            use_gradient_checkpointing: false,
            use_flash_attention: false,
            use_rotary_embeddings: false,
            use_gated_mlp: false,
        }
    }
}

impl RelgtConfig {
    pub fn default_for_data(data: &crate::data::TrainingBatch) -> Result<Self> {
        let mut config = Self::default();
        config.num_node_types = data.graph.schema.tables.len();
        Ok(config)
    }
    
    pub fn load_from_file(path: &std::path::Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config = toml::from_str(&content)?;
        Ok(config)
    }
    
    pub fn save_to_file(&self, path: &std::path::Path) -> Result<()> {
        let content = toml::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }
}
