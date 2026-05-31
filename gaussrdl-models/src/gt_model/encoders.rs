// src/model/encoders.rs
use candle_core::{Device, Tensor, DType, Module, IndexOp, Var};
use candle_nn::{VarBuilder, Linear, LayerNorm, Dropout, Embedding};
use std::collections::HashMap;
use crate::{ModelError, Result};
use crate::gnn_models::candle_utils::CandleHelper;

/// Neighbor node type encoder
pub struct NeighborNodeTypeEncoder {
    embedding: Embedding,
}

/// Neighbor hop encoder  
pub struct NeighborHopEncoder {
    embedding: Embedding,
}

/// Neighbor time encoder
pub struct NeighborTimeEncoder {
    pos_encoder: PositionalEncoding,
    linear: Linear,
    mask_vector: Var,
}

/// Neighbor TorchFrame features encoder
pub struct NeighborTfsEncoder {
    encoders: HashMap<String, Linear>,
    channels: usize,
}

/// GNN positional encoder
pub struct GNNPEEncoder {
    input_proj: Linear,
    conv_layers: Vec<Linear>,
    batch_norms: Vec<LayerNorm>,
    final_transform: Linear,
    pooling: String,
    num_layers: usize,
    layer_embedding_dim: usize,
    pe_dim: usize,
}

/// Positional encoding
pub struct PositionalEncoding {
    embedding_dim: usize,
}

impl NeighborNodeTypeEncoder {
    pub fn new(num_types: usize, embedding_dim: usize, vs: VarBuilder) -> Result<Self> {
        let embedding = CandleHelper::embedding(num_types + 1, embedding_dim, vs, "embedding")?;
        Ok(Self { embedding })
    }
    
    pub fn forward(&self, type_indices: &Tensor) -> Result<Tensor> {
        Ok(self.embedding.forward(type_indices)?)
    }
}

impl NeighborHopEncoder {
    pub fn new(max_neighbor_hop: usize, embedding_dim: usize, vs: VarBuilder) -> Result<Self> {
        let embedding = CandleHelper::embedding(max_neighbor_hop + 2, embedding_dim, vs, "embedding")?;
        Ok(Self { embedding })
    }
    
    pub fn forward(&self, hop_distances: &Tensor) -> Result<Tensor> {
        let shifted = hop_distances.add(&Tensor::ones_like(hop_distances)?)?;
        Ok(self.embedding.forward(&shifted)?)
    }
}

impl NeighborTimeEncoder {
    pub fn new(embedding_dim: usize, vs: VarBuilder) -> Result<Self> {
        let pos_encoder = PositionalEncoding::new(embedding_dim);
        let linear = CandleHelper::linear(vs.pp("linear"), embedding_dim, embedding_dim, "linear")?;
        let mask_vector = Var::randn(0f32, 0.02f32, (embedding_dim,), vs.device())?;
        
        Ok(Self {
            pos_encoder,
            linear,
            mask_vector,
        })
    }
    
    pub fn forward(&self, rel_time: &Tensor) -> Result<Tensor> {
        let (b, k) = (rel_time.dim(0)?, rel_time.dim(1)?);
        
        // Flatten and apply positional encoding
        let flattened_time = rel_time.flatten_from(0)?;
        let pos_encoded = self.pos_encoder.forward(&flattened_time)?;
        
        // Apply linear transformation
        let linear_out = self.linear.forward(&pos_encoded)?;
        let linear_out = linear_out.reshape((b, k, self.pos_encoder.embedding_dim))?;
        
        // Handle masking
        let mask = rel_time.lt(&Tensor::zeros_like(rel_time)?)?;
        let mask = mask.unsqueeze(1)?.to_dtype(DType::F32)?;
        let mask_vector = self.mask_vector.unsqueeze(0)?.unsqueeze(0)?.expand((b, k, self.pos_encoder.embedding_dim))?;
        
        let ones_mask = Tensor::ones_like(&mask)?;
        let diff_mask = ones_mask.sub(&mask)?;
        let masked_out = diff_mask.mul(&linear_out)?;
        let masked_vector = mask.mul(&mask_vector)?;
        let out = masked_out.add(&masked_vector)?;
        Ok(out)
    }
}

impl PositionalEncoding {
    pub fn new(embedding_dim: usize) -> Self {
        Self { embedding_dim }
    }
    
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Simple sinusoidal positional encoding
        let seq_len = x.dim(0)?;
        let device = x.device();
        
        let pos = Tensor::arange(0u32, seq_len as u32, device)?.to_dtype(DType::F32)?;
        let pos = pos.unsqueeze(1)?;
        
        let div_term = Tensor::exp(&Tensor::arange(0u32, self.embedding_dim as u32, device)?
            .to_dtype(DType::F32)?
            .mul(&Tensor::from_slice(&[-(2.0f32.ln() / self.embedding_dim as f32)], &[1], device)?)?)?;
        
        let pe = Tensor::zeros((seq_len, self.embedding_dim), DType::F32, device)?;
        
        // Apply sin/cos encoding
        let sin_terms = pos.matmul(&div_term.unsqueeze(0)?)?.sin()?;
        let cos_terms = pos.matmul(&div_term.unsqueeze(0)?)?.cos()?;
        
        // Create indices for scatter_add
        let indices = Tensor::arange(0u32, seq_len as u32, device)?.to_dtype(DType::U32)?;
        
        let pe = pe.scatter_add(&indices, &sin_terms, 0)?;
        let pe = pe.scatter_add(&indices, &cos_terms, 0)?;
        
        Ok(pe)
    }
}

impl NeighborTfsEncoder {
    pub fn new(channels: usize, _vs: VarBuilder) -> Result<Self> {
        let encoders = HashMap::new(); // Simplified for now
        Ok(Self { encoders, channels })
    }
    
    pub fn forward(&self, _batch_dict: &HashMap<String, Tensor>, _neighbor_types: &Tensor) -> Result<Tensor> {
        // Simplified implementation - return zeros for now
        let device = _neighbor_types.device();
        let (b, k) = (_neighbor_types.dim(0)?, _neighbor_types.dim(1)?);
        Ok(Tensor::zeros((b, k, self.channels), DType::F32, device)?)
    }
}

impl GNNPEEncoder {
    pub fn new(embedding_dim: usize, pe_dim: usize, vs: VarBuilder) -> Result<Self> {
        let layer_embedding_dim = embedding_dim / 4;
        let num_layers = 4;
        
        let input_proj = if pe_dim > 0 {
            CandleHelper::linear(vs.pp("input_proj"), pe_dim, layer_embedding_dim, "input_proj")?
        } else {
            CandleHelper::linear(vs.pp("input_proj"), 1, layer_embedding_dim, "input_proj")?
        };
        
        let mut conv_layers = Vec::new();
        let mut batch_norms = Vec::new();
        
        for i in 0..num_layers {
            let conv = CandleHelper::linear(
                vs.pp(&format!("conv.{}", i)), 
                layer_embedding_dim, 
                layer_embedding_dim, 
                &format!("conv.{}", i)
            )?;
            conv_layers.push(conv);
            
            let bn = CandleHelper::layer_norm(
                vs.pp(&format!("bn.{}", i)), 
                layer_embedding_dim, 
                &format!("bn.{}", i), 
                1e-5
            )?;
            batch_norms.push(bn);
        }
        
        let final_transform = CandleHelper::linear(
            vs.pp("final_transform"), 
            layer_embedding_dim, 
            embedding_dim, 
            "final_transform"
        )?;
        
        Ok(Self {
            input_proj,
            conv_layers,
            batch_norms,
            final_transform,
            pooling: "none".to_string(),
            num_layers,
            layer_embedding_dim,
            pe_dim,
        })
    }
    
    pub fn forward(&self, _edge_index: &Tensor, _batch: &Tensor) -> Result<Tensor> {
        // Simplified implementation - return random embeddings
        let device = _batch.device();
        let total_nodes = _batch.dim(0)?;
        let b = _batch.max(0)?.to_scalar::<u32>()? as usize + 1;
        let k = total_nodes / b;
        
        let x_input = Tensor::randn(0f32, 1f32, (total_nodes, 1), device)?;
        let x = self.input_proj.forward(&x_input)?;
        
        // Apply GNN layers
        let mut outputs = Vec::new();
        for i in 0..self.num_layers {
            let x_res = x.clone();
            let x_new = self.conv_layers[i].forward(&x)?;
            let x_new = self.batch_norms[i].forward(&x_new)?;
            let x_new = x_new.relu()?;
            let x = x_new.add(&x_res)?;
            
            outputs.push(x.clone());
        }
        
        // Final transformation
        let out = self.final_transform.forward(&x)?;
        let out = out.reshape((b, k, self.pe_dim + 1))?;
        
        Ok(out)
    }
} 