// src/model/embeddings.rs

use candle_core::{Tensor, Device};
use candle_nn::{Module, VarBuilder, Embedding, Linear};
use crate::ModelError;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// Configuration for multi-element embedding layers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Vocabulary sizes for different element types
    pub vocab_sizes: HashMap<String, usize>,
    /// Embedding dimension
    pub embedding_dim: usize,
    /// Whether to use positional embeddings
    pub use_positional: bool,
    /// Maximum sequence length for positional embeddings
    pub max_position: usize,
    /// Whether to use temporal embeddings
    pub use_temporal: bool,
    /// Temporal embedding dimension
    pub temporal_dim: usize,
    /// Dropout rate
    pub dropout: f64,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            vocab_sizes: HashMap::new(),
            embedding_dim: 128,
            use_positional: true,
            max_position: 512,
            use_temporal: true,
            temporal_dim: 64,
            dropout: 0.1,
        }
    }
}

/// Multi-element embedding layer for RelGT tokenization
pub struct MultiElementEmbedding {
    /// Token embeddings for different element types
    token_embeddings: HashMap<String, Embedding>,
    /// Positional embeddings
    positional_embedding: Option<Embedding>,
    /// Temporal embeddings
    temporal_embedding: Option<Linear>,
    /// Output projection layer
    projection: Linear,
    /// Configuration
    config: EmbeddingConfig,
}

impl MultiElementEmbedding {
    /// Create new multi-element embedding layer
    pub fn new(config: EmbeddingConfig, vb: VarBuilder) -> Result<Self, ModelError> {
        let mut token_embeddings = HashMap::new();
        
        // Create embeddings for each element type
        for (element_type, vocab_size) in &config.vocab_sizes {
            let embedding = candle_nn::embedding(
                *vocab_size,
                config.embedding_dim,
                vb.pp(&format!("token_emb_{}", element_type))
            )?;
            token_embeddings.insert(element_type.clone(), embedding);
        }
        
        // Positional embeddings
        let positional_embedding = if config.use_positional {
            Some(candle_nn::embedding(
                config.max_position,
                config.embedding_dim,
                vb.pp("pos_emb")
            )?)
        } else {
            None
        };
        
        // Temporal embeddings
        let temporal_embedding = if config.use_temporal {
            Some(candle_nn::linear(
                config.temporal_dim,
                config.embedding_dim,
                vb.pp("temporal_emb")
            )?)
        } else {
            None
        };
        
        // Output projection
        let projection = candle_nn::linear(
            config.embedding_dim,
            config.embedding_dim,
            vb.pp("projection")
        )?;
        
        Ok(Self {
            token_embeddings,
            positional_embedding,
            temporal_embedding,
            projection,
            config,
        })
    }
    
    /// Forward pass for multi-element tokens
    pub fn forward(
        &self,
        tokens: &HashMap<String, Tensor>,
        positions: Option<&Tensor>,
        temporal_features: Option<&Tensor>,
    ) -> Result<Tensor, ModelError> {
        let mut embeddings = Vec::new();
        
        // Process each element type
        for (element_type, token_ids) in tokens {
            if let Some(embedding_layer) = self.token_embeddings.get(element_type) {
                let emb = embedding_layer.forward(token_ids)?;
                embeddings.push(emb);
            }
        }
        
        if embeddings.is_empty() {
            return Err(ModelError::Model(
                "No valid token embeddings found".to_string()
            ));
        }
        
        // Combine embeddings (sum for now, could be more sophisticated)
        let mut combined = embeddings[0].clone();
        for emb in embeddings.iter().skip(1) {
            combined = (&combined + emb)?;
        }
        
        // Add positional embeddings
        if let (Some(pos_emb), Some(positions)) = (&self.positional_embedding, positions) {
            let pos_embeddings = pos_emb.forward(positions)?;
            combined = (&combined + &pos_embeddings)?;
        }
        
        // Add temporal embeddings
        if let (Some(temp_emb), Some(temporal_features)) = (&self.temporal_embedding, temporal_features) {
            let temp_embeddings = temp_emb.forward(temporal_features)?;
            combined = (&combined + &temp_embeddings)?;
        }
        
        // Apply projection
        let output = self.projection.forward(&combined)?;
        
        Ok(output)
    }
}

/// Node type embeddings for heterogeneous graphs
pub struct NodeTypeEmbedding {
    embedding: Embedding,
    num_types: usize,
    embedding_dim: usize,
}

impl NodeTypeEmbedding {
    pub fn new(num_types: usize, embedding_dim: usize, vb: VarBuilder) -> Result<Self, ModelError> {
        let embedding = candle_nn::embedding(num_types, embedding_dim, vb.pp("node_type"))
            .map_err(|e| ModelError::Candle(e))?;
        
        Ok(Self {
            embedding,
            num_types,
            embedding_dim,
        })
    }
    
    pub fn forward(&self, node_types: &Tensor) -> Result<Tensor, ModelError> {
        self.embedding.forward(node_types).map_err(|e| ModelError::Candle(e))
    }
}

/// Edge type embeddings for heterogeneous graphs
pub struct EdgeTypeEmbedding {
    embedding: Embedding,
    num_types: usize,
    embedding_dim: usize,
}

impl EdgeTypeEmbedding {
    pub fn new(num_types: usize, embedding_dim: usize, vb: VarBuilder) -> Result<Self, ModelError> {
        let embedding = candle_nn::embedding(num_types, embedding_dim, vb.pp("edge_type"))
            .map_err(|e| ModelError::Candle(e))?;
        
        Ok(Self {
            embedding,
            num_types,
            embedding_dim,
        })
    }
    
    pub fn forward(&self, edge_types: &Tensor) -> Result<Tensor, ModelError> {
        self.embedding.forward(edge_types).map_err(|e| ModelError::Candle(e))
    }
}

/// Learnable positional embeddings
pub struct LearnedPositionalEmbedding {
    embedding: Embedding,
    max_position: usize,
}

impl LearnedPositionalEmbedding {
    pub fn new(max_position: usize, embedding_dim: usize, vb: VarBuilder) -> Result<Self, ModelError> {
        let embedding = candle_nn::embedding(max_position, embedding_dim, vb.pp("pos_emb"))
            .map_err(|e| ModelError::Candle(e))?;
        
        Ok(Self {
            embedding,
            max_position,
        })
    }
    
    pub fn forward(&self, positions: &Tensor) -> Result<Tensor, ModelError> {
        self.embedding.forward(positions).map_err(|e| ModelError::Candle(e))
    }
}

/// Sinusoidal positional embeddings (fixed, not learned)
pub struct SinusoidalPositionalEmbedding {
    embedding_dim: usize,
    max_position: usize,
}

impl SinusoidalPositionalEmbedding {
    pub fn new(max_position: usize, embedding_dim: usize) -> Self {
        Self {
            embedding_dim,
            max_position,
        }
    }
    
    pub fn forward(&self, positions: &Tensor, device: &Device) -> Result<Tensor, ModelError> {
        let seq_len = positions.dim(0)?;
        let dim = self.embedding_dim;
        
        // Create sinusoidal embeddings
        let mut embeddings = Vec::new();
        
        for pos in 0..seq_len {
            let mut pos_emb = Vec::new();
            for i in 0..dim {
                let angle = pos as f32 / 10000.0_f32.powf(2.0 * (i as f32) / dim as f32);
                if i % 2 == 0 {
                    pos_emb.push(angle.sin());
                } else {
                    pos_emb.push(angle.cos());
                }
            }
            embeddings.push(pos_emb);
        }
        
        let flat_embeddings: Vec<f32> = embeddings.into_iter().flatten().collect();
        let tensor = Tensor::from_vec(flat_embeddings, (seq_len, dim), device)?;
        
        Ok(tensor)
    }
}

/// Rotary positional embeddings (RoPE)
pub struct RotaryPositionalEmbedding {
    dim: usize,
    base: f32,
}

impl RotaryPositionalEmbedding {
    pub fn new(dim: usize, base: f32) -> Self {
        Self { dim, base }
    }
    
    pub fn forward(&self, x: &Tensor, _positions: &Tensor) -> Result<Tensor, ModelError> {
        // Simplified RoPE implementation
        // In practice, this would involve more complex frequency computations
        let _seq_len = x.dim(1)?;
        let _head_dim = x.dim(x.dims().len() - 1)?;
        
        // For now, return input unchanged - proper RoPE is complex
        // TODO: Implement full RoPE when Candle supports more advanced operations
        Ok(x.clone())
    }
}

/// Combined embedding layer that handles all embedding types
pub struct CombinedEmbedding {
    multi_element: Option<MultiElementEmbedding>,
    node_type: Option<NodeTypeEmbedding>,
    edge_type: Option<EdgeTypeEmbedding>,
    positional: Option<LearnedPositionalEmbedding>,
    config: EmbeddingConfig,
}

impl CombinedEmbedding {
    pub fn new(config: EmbeddingConfig, vb: VarBuilder) -> Result<Self, ModelError> {
        let multi_element = if !config.vocab_sizes.is_empty() {
            Some(MultiElementEmbedding::new(config.clone(), vb.pp("multi_element"))?)
        } else {
            None
        };
        
        let positional = if config.use_positional {
            Some(LearnedPositionalEmbedding::new(
                config.max_position,
                config.embedding_dim,
                vb.pp("positional")
            )?)
        } else {
            None
        };
        
        Ok(Self {
            multi_element,
            node_type: None, // Will be set separately if needed
            edge_type: None, // Will be set separately if needed
            positional,
            config,
        })
    }
    
    pub fn with_node_types(mut self, num_types: usize, vb: VarBuilder) -> Result<Self, ModelError> {
        self.node_type = Some(NodeTypeEmbedding::new(
            num_types,
            self.config.embedding_dim,
            vb.pp("node_type")
        )?);
        Ok(self)
    }
    
    pub fn with_edge_types(mut self, num_types: usize, vb: VarBuilder) -> Result<Self, ModelError> {
        self.edge_type = Some(EdgeTypeEmbedding::new(
            num_types,
            self.config.embedding_dim,
            vb.pp("edge_type")
        )?);
        Ok(self)
    }
}

/// Embedding utilities
pub struct EmbeddingUtils;

impl EmbeddingUtils {
    /// Initialize embedding weights with Xavier uniform
    pub fn xavier_init(_embedding: &mut Embedding, _fan_in: usize, _fan_out: usize) -> Result<(), ModelError> {
        // TODO: Implement Xavier uniform initialization
        Ok(())
    }
    
    /// Initialize embedding weights with normal distribution
    pub fn normal_init(_embedding: &mut Embedding, _mean: f32, _std: f32) -> Result<(), ModelError> {
        // TODO: Implement normal initialization
        Ok(())
    }
    
    /// Compute embedding statistics
    pub fn compute_stats(embeddings: &Tensor) -> Result<HashMap<String, f32>, ModelError> {
        let mut stats = HashMap::new();
        
        // Mean
        let mean = embeddings.mean_all()
            .map_err(|e| ModelError::Candle(e))?
            .to_scalar::<f32>()
            .map_err(|e| ModelError::Candle(e))?;
        stats.insert("mean".to_string(), mean);
        
        // Standard deviation (approximate) - fix the tensor subtraction
        let mean_tensor = Tensor::from_slice(&[mean], (), embeddings.device())
            .map_err(|e| ModelError::Candle(e))?
            .broadcast_as(embeddings.shape())
            .map_err(|e| ModelError::Candle(e))?;
        let variance = embeddings.sub(&mean_tensor)
            .map_err(|e| ModelError::Candle(e))?
            .powf(2.0)
            .map_err(|e| ModelError::Candle(e))?
            .mean_all()
            .map_err(|e| ModelError::Candle(e))?
            .to_scalar::<f32>()
            .map_err(|e| ModelError::Candle(e))?;
        stats.insert("std".to_string(), variance.sqrt());
        
        // Min/Max would require more complex operations in Candle
        
        Ok(stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::VarMap;
    
    #[test]
    fn test_embedding_config() {
        let config = EmbeddingConfig::default();
        assert_eq!(config.embedding_dim, 128);
        assert!(config.use_positional);
    }
    
    #[test]
    fn test_node_type_embedding() -> Result<(), ModelError> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        
        let embedding = NodeTypeEmbedding::new(10, 64, vb)?;
        assert_eq!(embedding.num_types, 10);
        assert_eq!(embedding.embedding_dim, 64);
        
        Ok(())
    }
} 