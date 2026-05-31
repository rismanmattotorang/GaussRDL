use candle_core::{Tensor, Device, DType, Result as CandleResult};
use candle_nn::{Module, Embedding, VarBuilder};
use std::collections::HashMap;
use crate::ModelError;
use super::candle_utils::CandleHelper;

/// Unified embedding layer for relational graphs
pub struct RelationalEmbeddings {
    node_type_embeddings: Option<NodeTypeEmbeddings>,
    relation_embeddings: Option<RelationEmbeddings>,
    positional_embeddings: Option<PositionalEmbeddings>,
    temporal_embeddings: Option<TemporalEmbeddings>,
    feature_embeddings: HashMap<String, Embedding>,
    device: Device,
}

impl RelationalEmbeddings {
    /// Create new relational embeddings
    pub fn new(
        config: &RelationalEmbeddingConfig,
        device: Device,
        vb: VarBuilder,
    ) -> Result<Self, ModelError> {
        let node_type_embeddings = if let Some(ref node_config) = config.node_types {
            Some(NodeTypeEmbeddings::new(
                &node_config.type_names,
                node_config.embedding_dim,
                device.clone(),
                vb.pp("node_types"),
            )?)
        } else {
            None
        };
        
        let relation_embeddings = if let Some(ref rel_config) = config.relations {
            Some(RelationEmbeddings::new(
                rel_config.num_relations,
                rel_config.embedding_dim,
                vb.pp("relations"),
            )?)
        } else {
            None
        };
        
        let positional_embeddings = if let Some(ref pos_config) = config.positional {
            Some(PositionalEmbeddings::new(
                pos_config.max_length,
                pos_config.embedding_dim,
                &device,
            )?)
        } else {
            None
        };
        
        let temporal_embeddings = if let Some(ref temp_config) = config.temporal {
            Some(TemporalEmbeddings::new(
                temp_config.embedding_dim,
                vb.pp("temporal"),
            )?)
        } else {
            None
        };
        
        let mut feature_embeddings = HashMap::new();
        for (name, config) in &config.categorical_features {
            let emb = CandleHelper::embedding(
                config.vocab_size,
                config.embedding_dim,
                vb.pp(&format!("categorical_{}", name)),
                name,
            )?;
            feature_embeddings.insert(name.clone(), emb);
        }
        
        Ok(Self {
            node_type_embeddings,
            relation_embeddings,
            positional_embeddings,
            temporal_embeddings,
            feature_embeddings,
            device,
        })
    }
    
    /// Embed node types
    pub fn embed_node_types(&self, node_types: &[String], indices: &Tensor) -> CandleResult<Option<Tensor>> {
        if let Some(ref embeddings) = self.node_type_embeddings {
            let mut all_embeddings = Vec::new();
            
            for node_type in node_types {
                let emb = embeddings.forward(node_type, indices)?;
                all_embeddings.push(emb);
            }
            
            if all_embeddings.is_empty() {
                Ok(None)
            } else if all_embeddings.len() == 1 {
                Ok(Some(all_embeddings.into_iter().next().unwrap()))
            } else {
                Ok(Some(Tensor::stack(&all_embeddings, 1)?.mean(1)?))
            }
        } else {
            Ok(None)
        }
    }
    
    /// Embed relations
    pub fn embed_relations(&self, relation_indices: &Tensor) -> CandleResult<Option<Tensor>> {
        if let Some(ref embeddings) = self.relation_embeddings {
            Ok(Some(embeddings.forward(relation_indices)?))
        } else {
            Ok(None)
        }
    }
    
    /// Embed positions
    pub fn embed_positions(&self, positions: &Tensor) -> CandleResult<Option<Tensor>> {
        if let Some(ref embeddings) = self.positional_embeddings {
            Ok(Some(embeddings.forward(positions)?))
        } else {
            Ok(None)
        }
    }
    
    /// Embed temporal features
    pub fn embed_temporal(
        &self,
        day: &Tensor,
        month: &Tensor,
        year: &Tensor,
        hour: &Tensor,
    ) -> CandleResult<Option<Tensor>> {
        if let Some(ref embeddings) = self.temporal_embeddings {
            Ok(Some(embeddings.forward(day, month, year, hour)?))
        } else {
            Ok(None)
        }
    }
    
    /// Embed categorical features
    pub fn embed_categorical(&self, feature_name: &str, indices: &Tensor) -> CandleResult<Option<Tensor>> {
        if let Some(embedding) = self.feature_embeddings.get(feature_name) {
            Ok(Some(embedding.forward(indices)?))
        } else {
            Ok(None)
        }
    }
}

/// Configuration for relational embeddings
#[derive(Debug, Clone)]
pub struct RelationalEmbeddingConfig {
    pub node_types: Option<NodeTypeConfig>,
    pub relations: Option<RelationConfig>,
    pub positional: Option<PositionalConfig>,
    pub temporal: Option<TemporalConfig>,
    pub categorical_features: HashMap<String, CategoricalConfig>,
}

#[derive(Debug, Clone)]
pub struct NodeTypeConfig {
    pub type_names: Vec<String>,
    pub embedding_dim: usize,
}

#[derive(Debug, Clone)]
pub struct RelationConfig {
    pub num_relations: usize,
    pub embedding_dim: usize,
}

#[derive(Debug, Clone)]
pub struct PositionalConfig {
    pub max_length: usize,
    pub embedding_dim: usize,
}

#[derive(Debug, Clone)]
pub struct TemporalConfig {
    pub embedding_dim: usize,
}

#[derive(Debug, Clone)]
pub struct CategoricalConfig {
    pub vocab_size: usize,
    pub embedding_dim: usize,
}

/// Node type embeddings for relational graphs
pub struct NodeTypeEmbeddings {
    embeddings: HashMap<String, Embedding>,
    embedding_dim: usize,
    device: Device,
}

impl NodeTypeEmbeddings {
    pub fn new(
        node_types: &[String],
        embedding_dim: usize,
        device: Device,
        vb: VarBuilder,
    ) -> Result<Self, ModelError> {
        let mut embeddings = HashMap::new();
        
        for (i, node_type) in node_types.iter().enumerate() {
            let embedding = CandleHelper::embedding(
                1000, // Max vocab size per type
                embedding_dim,
                vb.pp(&format!("type_{}", i)),
                &format!("node_type_{}", node_type),
            )?;
            embeddings.insert(node_type.clone(), embedding);
        }
        
        Ok(Self {
            embeddings,
            embedding_dim,
            device,
        })
    }
    
    pub fn forward(&self, node_type: &str, indices: &Tensor) -> CandleResult<Tensor> {
        if let Some(embedding) = self.embeddings.get(node_type) {
            embedding.forward(indices)
        } else {
            // Return zero embeddings for unknown node types
            let batch_size = indices.shape().dims()[0];
            CandleHelper::zeros(&[batch_size, self.embedding_dim], DType::F32, &self.device)
        }
    }
    
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }
}

/// Relation embeddings for multi-relational graphs
pub struct RelationEmbeddings {
    embeddings: Embedding,
    embedding_dim: usize,
}

impl RelationEmbeddings {
    pub fn new(
        num_relations: usize,
        embedding_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self, ModelError> {
        let embeddings = CandleHelper::embedding(
            num_relations,
            embedding_dim,
            vb,
            "relations",
        )?;
        
        Ok(Self {
            embeddings,
            embedding_dim,
        })
    }
    
    pub fn forward(&self, relation_indices: &Tensor) -> CandleResult<Tensor> {
        self.embeddings.forward(relation_indices)
    }
    
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }
}

/// Positional embeddings for sequences
pub struct PositionalEmbeddings {
    embeddings: Tensor,
    max_length: usize,
}

impl PositionalEmbeddings {
    pub fn new(max_length: usize, embedding_dim: usize, device: &Device) -> Result<Self, ModelError> {
        let mut embeddings = Vec::new();
        
        for pos in 0..max_length {
            let mut pos_emb = Vec::new();
            for i in 0..embedding_dim {
                let angle = pos as f32 / 10000_f32.powf(2.0 * (i / 2) as f32 / embedding_dim as f32);
                if i % 2 == 0 {
                    pos_emb.push(angle.sin());
                } else {
                    pos_emb.push(angle.cos());
                }
            }
            embeddings.push(pos_emb);
        }
        
        let embeddings = Tensor::from_vec(
            embeddings.into_iter().flatten().collect::<Vec<f32>>(),
            (max_length, embedding_dim),
            device,
        )?;
        
        Ok(Self {
            embeddings,
            max_length,
        })
    }
    
    pub fn forward(&self, positions: &Tensor) -> CandleResult<Tensor> {
        // positions shape: (batch_size, seq_len)
        let shape = positions.shape();
        let batch_size = shape.dims()[0];
        let seq_len = shape.dims()[1];
        
        // Get embeddings for positions
        let pos_indices = positions.flatten_all()?;
        let pos_embs = CandleHelper::index_select(&self.embeddings, &pos_indices, 0)?;
        
        // Reshape to (batch_size, seq_len, embedding_dim)
        let embedding_dim = self.embeddings.shape().dims()[1];
        pos_embs.reshape((batch_size, seq_len, embedding_dim))
    }
}

/// Temporal embeddings for time-based features
pub struct TemporalEmbeddings {
    day_embedding: Embedding,
    month_embedding: Embedding,
    year_embedding: Embedding,
    hour_embedding: Embedding,
    embedding_dim: usize,
}

impl TemporalEmbeddings {
    pub fn new(embedding_dim: usize, vb: VarBuilder) -> Result<Self, ModelError> {
        let day_embedding = CandleHelper::embedding(31, embedding_dim, vb.pp("day"), "day")?;
        let month_embedding = CandleHelper::embedding(12, embedding_dim, vb.pp("month"), "month")?;
        let year_embedding = CandleHelper::embedding(100, embedding_dim, vb.pp("year"), "year")?; // Years 2000-2099
        let hour_embedding = CandleHelper::embedding(24, embedding_dim, vb.pp("hour"), "hour")?;
        
        Ok(Self {
            day_embedding,
            month_embedding,
            year_embedding,
            hour_embedding,
            embedding_dim,
        })
    }
    
    pub fn forward(
        &self,
        day: &Tensor,
        month: &Tensor,
        year: &Tensor,
        hour: &Tensor,
    ) -> CandleResult<Tensor> {
        let day_emb = self.day_embedding.forward(day)?;
        let month_emb = self.month_embedding.forward(month)?;
        let year_emb = self.year_embedding.forward(year)?;
        let hour_emb = self.hour_embedding.forward(hour)?;
        
        // Sum all temporal embeddings
        let temporal_emb = day_emb.add(&month_emb)?.add(&year_emb)?.add(&hour_emb)?;
        
        Ok(temporal_emb)
    }
    
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;

    #[test]
    fn test_node_type_embeddings() -> Result<(), ModelError> {
        let device = Device::Cpu;
        let var_map = VarMap::new();
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, &device);
        
        let node_types = vec!["user".to_string(), "item".to_string()];
        let embeddings = NodeTypeEmbeddings::new(
            &node_types,
            64,
            device.clone(),
            vb,
        )?;
        
        let indices = CandleHelper::zeros(&[10], DType::I64, &device)?;
        let user_emb = embeddings.forward("user", &indices)?;
        assert_eq!(user_emb.shape().dims(), &[10, 64]);
        
        Ok(())
    }
    
    #[test]
    fn test_positional_embeddings() -> Result<(), ModelError> {
        let device = Device::Cpu;
        let embeddings = PositionalEmbeddings::new(100, 64, &device)?;
        
        let positions = CandleHelper::zeros(&[2, 10], DType::I64, &device)?;
        let pos_emb = embeddings.forward(&positions)?;
        assert_eq!(pos_emb.shape().dims(), &[2, 10, 64]);
        
        Ok(())
    }
} 