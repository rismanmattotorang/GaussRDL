// src/model/attention.rs
use candle_core::{Device, Tensor, Result as CandleResult};
use candle_nn::{Linear, Module};
use std::sync::Arc;
use parking_lot::RwLock;
use crate::model::config::ModelConfig;

use super::config::{AttentionType, RelgtConfig};

pub trait AttentionMechanism: Send + Sync {
    fn compute_attention(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        mask: Option<&Tensor>,
    ) -> CandleResult<Tensor>;
    
    fn get_attention_type(&self) -> AttentionType;
}

pub struct StandardAttention {
    scale: f32,
    dropout_rate: f32,
    device: Device,
}

impl StandardAttention {
    pub fn new(config: &RelgtConfig, device: Device) -> Self {
        let head_dim = config.hidden_dim / config.num_attention_heads;
        Self {
            scale: 1.0 / (head_dim as f32).sqrt(),
            dropout_rate: config.attention_dropout_rate,
            device,
        }
    }
}

impl AttentionMechanism for StandardAttention {
    fn compute_attention(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        mask: Option<&Tensor>,
    ) -> CandleResult<Tensor> {
        // Q * K^T
        let attention_scores = query.matmul(&key.transpose(1, 2)?)?;
        
        // Scale
        let scaled_scores = attention_scores.mul_scalar(self.scale)?;
        
        // Apply mask if provided
        let masked_scores = if let Some(mask) = mask {
            scaled_scores.add(mask)?
        } else {
            scaled_scores
        };
        
        // Softmax
        let attention_probs = masked_scores.softmax(2)?;
        
        // Apply dropout during training
        let dropped_probs = if self.dropout_rate > 0.0 {
            attention_probs.dropout(self.dropout_rate, None)?
        } else {
            attention_probs
        };
        
        // Multiply with values
        dropped_probs.matmul(value)
    }
    
    fn get_attention_type(&self) -> AttentionType {
        AttentionType::Standard
    }
}

pub struct FlashAttention {
    scale: f32,
    dropout_rate: f32,
    device: Device,
    block_size: usize,
}

impl FlashAttention {
    pub fn new(config: &RelgtConfig, device: Device) -> Self {
        let head_dim = config.hidden_dim / config.num_attention_heads;
        Self {
            scale: 1.0 / (head_dim as f32).sqrt(),
            dropout_rate: config.attention_dropout_rate,
            device,
            block_size: 128, // Optimal block size for most GPUs
        }
    }
    
    fn forward_optimized(&self, qkv: &Tensor) -> CandleResult<Tensor> {
        // Implement Flash Attention algorithm
        // This is a simplified version - real implementation would use
        // tiled matrix multiplication and shared memory optimizations
        let (batch_size, seq_len, num_heads, head_dim) = qkv.dims4()?;
        
        let qkv = qkv.reshape((batch_size, seq_len, 3, num_heads, head_dim))?;
        let q = qkv.narrow(2, 0, 1)?.squeeze(2)?;
        let k = qkv.narrow(2, 1, 1)?.squeeze(2)?;
        let v = qkv.narrow(2, 2, 1)?.squeeze(2)?;
        
        // Split sequence into blocks
        let num_blocks = (seq_len + self.block_size - 1) / self.block_size;
        let mut output = Tensor::zeros((batch_size, seq_len, num_heads, head_dim), &self.device)?;
        
        for i in 0..num_blocks {
            let start_idx = i * self.block_size;
            let end_idx = (start_idx + self.block_size).min(seq_len);
            let block_q = q.narrow(1, start_idx, end_idx - start_idx)?;
            
            // Compute attention for this block
            let block_scores = block_q.matmul(&k.transpose(2, 3)?)?;
            let block_probs = block_scores.softmax(-1)?;
            let block_output = block_probs.matmul(&v)?;
            
            // Update output
            output.narrow_mut(1, start_idx, end_idx - start_idx)?.copy_(&block_output)?;
        }
        
        Ok(output)
    }
}

impl AttentionMechanism for FlashAttention {
    fn compute_attention(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        _mask: Option<&Tensor>,
    ) -> CandleResult<Tensor> {
        // Combine QKV for optimized implementation
        let qkv = Tensor::cat(&[query, key, value], 2)?;
        self.forward_optimized(&qkv)
    }
    
    fn get_attention_type(&self) -> AttentionType {
        AttentionType::Flash
    }
}

pub struct LinearAttention {
    feature_map: Arc<dyn Fn(&Tensor) -> CandleResult<Tensor> + Send + Sync>,
    scale: f32,
    device: Device,
}

impl LinearAttention {
    pub fn new(config: &RelgtConfig, device: Device) -> Self {
        let head_dim = config.hidden_dim / config.num_attention_heads;
        let feature_map: Arc<dyn Fn(&Tensor) -> CandleResult<Tensor> + Send + Sync> = 
            Arc::new(move |x: &Tensor| {
                // Implement ELU + 1 feature map
                x.elu(1.0, None)?.add_scalar(1.0)
            });
        
        Self {
            feature_map,
            scale: 1.0 / (head_dim as f32).sqrt(),
            device,
        }
    }
}

impl AttentionMechanism for LinearAttention {
    fn compute_attention(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        _mask: Option<&Tensor>,
    ) -> CandleResult<Tensor> {
        // Apply feature map
        let q = (self.feature_map)(query)?;
        let k = (self.feature_map)(key)?;
        
        // Linear attention computation
        let kv = k.transpose(1, 2)?.matmul(&value)?;
        let normalizer = k.sum(1)?.unsqueeze(2)?;
        
        q.matmul(&kv)?.div(&normalizer)
    }
    
    fn get_attention_type(&self) -> AttentionType {
        AttentionType::Linear
    }
}

pub fn create_attention_mechanism(
    config: &RelgtConfig,
    device: Device,
) -> Box<dyn AttentionMechanism> {
    match config.attention_type {
        AttentionType::Standard => Box::new(StandardAttention::new(config, device)),
        AttentionType::Flash => Box::new(FlashAttention::new(config, device)),
        AttentionType::Linear => Box::new(LinearAttention::new(config, device)),
        AttentionType::LocalGlobal => Box::new(StandardAttention::new(config, device)), // TODO: Implement
        AttentionType::Sparse => Box::new(StandardAttention::new(config, device)), // TODO: Implement
    }
}

pub struct MultiHeadAttention {
    config: Arc<ModelConfig>,
    query_proj: Tensor,
    key_proj: Tensor,
    value_proj: Tensor,
    output_proj: Tensor,
    rotary_emb: Option<RotaryEmbeddings>,
    device: Device,
}

impl MultiHeadAttention {
    pub fn new(config: Arc<ModelConfig>, device: Device) -> CandleResult<Self> {
        let hidden_dim = config.hidden_dim;
        let num_heads = config.num_attention_heads;
        let head_dim = hidden_dim / num_heads;
        
        let query_proj = Tensor::randn(0.0, 0.02, (hidden_dim, hidden_dim), &device)?;
        let key_proj = Tensor::randn(0.0, 0.02, (hidden_dim, hidden_dim), &device)?;
        let value_proj = Tensor::randn(0.0, 0.02, (hidden_dim, hidden_dim), &device)?;
        let output_proj = Tensor::randn(0.0, 0.02, (hidden_dim, hidden_dim), &device)?;
        
        let rotary_emb = if config.use_rotary_embeddings {
            Some(RotaryEmbeddings::new(head_dim, config.max_position_embeddings))
        } else {
            None
        };
        
        Ok(Self {
            config,
            query_proj,
            key_proj,
            value_proj,
            output_proj,
            rotary_emb,
            device,
        })
    }
    
    pub fn forward(&self, query: &Tensor, key: &Tensor, value: &Tensor, mask: Option<&Tensor>) -> CandleResult<Tensor> {
        let batch_size = query.dim(0)?;
        let seq_len = query.dim(1)?;
        let num_heads = self.config.num_attention_heads;
        let head_dim = self.config.hidden_dim / num_heads;
        
        // Project inputs
        let query = query.matmul(&self.query_proj)?;
        let key = key.matmul(&self.key_proj)?;
        let value = value.matmul(&self.value_proj)?;
        
        // Reshape for multi-head attention
        let query = query.reshape((batch_size, seq_len, num_heads, head_dim))?;
        let key = key.reshape((batch_size, seq_len, num_heads, head_dim))?;
        let value = value.reshape((batch_size, seq_len, num_heads, head_dim))?;
        
        // Apply rotary embeddings if enabled
        let (query, key) = if let Some(rotary) = &self.rotary_emb {
            rotary.apply(query, key)?
        } else {
            (query, key)
        };
        
        // Compute attention scores
        let attention = if self.config.use_flash_attention {
            self.flash_attention(query, key, value, mask)?
        } else {
            self.standard_attention(query, key, value, mask)?
        };
        
        // Project output
        attention.matmul(&self.output_proj)
    }
    
    fn flash_attention(
        &self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Option<&Tensor>,
    ) -> CandleResult<Tensor> {
        // Implement Flash Attention algorithm
        // This is a memory-efficient attention implementation
        // that avoids materializing the full attention matrix
        
        let batch_size = query.dim(0)?;
        let seq_len = query.dim(1)?;
        let num_heads = self.config.num_attention_heads;
        let head_dim = self.config.hidden_dim / num_heads;
        
        // Split sequence into blocks for memory efficiency
        let block_size = (self.config.max_memory_mb * 1024 * 1024) / (4 * head_dim);
        let num_blocks = (seq_len + block_size - 1) / block_size;
        
        let mut output = Tensor::zeros((batch_size, seq_len, num_heads, head_dim), &self.device)?;
        
        for i in 0..num_blocks {
            let start_idx = i * block_size;
            let end_idx = (start_idx + block_size).min(seq_len);
            
            let query_block = query.slice(1, start_idx, end_idx)?;
            let key_block = key.slice(1, start_idx, end_idx)?;
            let value_block = value.slice(1, start_idx, end_idx)?;
            
            let block_mask = mask.map(|m| m.slice(1, start_idx, end_idx)).transpose()?;
            
            let attention_block = self.compute_block_attention(
                query_block,
                key_block,
                value_block,
                block_mask.as_ref(),
            )?;
            
            output.slice_assign(1, start_idx, end_idx, &attention_block)?;
        }
        
        Ok(output)
    }
    
    fn standard_attention(
        &self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Option<&Tensor>,
    ) -> CandleResult<Tensor> {
        // Standard scaled dot-product attention
        let scale = (self.config.hidden_dim as f64).sqrt();
        let scores = query.matmul(&key.transpose(2, 3)?)?.div(scale)?;
        
        if let Some(mask) = mask {
            scores.add(mask)?;
        }
        
        let attention_probs = scores.softmax(3)?;
        let attention_probs = attention_probs.dropout(self.config.attention_dropout_prob)?;
        
        attention_probs.matmul(&value)
    }
    
    fn compute_block_attention(
        &self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Option<&Tensor>,
    ) -> CandleResult<Tensor> {
        let scale = (self.config.hidden_dim as f64).sqrt();
        let scores = query.matmul(&key.transpose(2, 3)?)?.div(scale)?;
        
        if let Some(mask) = mask {
            scores.add(mask)?;
        }
        
        let attention_probs = scores.softmax(3)?;
        let attention_probs = attention_probs.dropout(self.config.attention_dropout_prob)?;
        
        attention_probs.matmul(&value)
    }
}

struct RotaryEmbeddings {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbeddings {
    fn new(dim: usize, max_position: usize) -> Self {
        let inv_freq: Vec<f32> = (0..dim/2)
            .map(|i| 1.0 / (10000_f32.powf(2.0 * i as f32 / dim as f32)))
            .collect();
            
        let pos: Vec<f32> = (0..max_position).map(|x| x as f32).collect();
        
        let freqs = pos.iter()
            .flat_map(|&p| inv_freq.iter().map(move |&f| p * f))
            .collect::<Vec<_>>();
            
        let sin = Tensor::from_slice(&freqs.iter().map(|x| x.sin()).collect::<Vec<_>>());
        let cos = Tensor::from_slice(&freqs.iter().map(|x| x.cos()).collect::<Vec<_>>());
        
        Self { sin, cos }
    }
    
    fn apply(&self, query: Tensor, key: Tensor) -> CandleResult<(Tensor, Tensor)> {
        let query_rot = self.rotate_half(&query)?;
        let key_rot = self.rotate_half(&key)?;
        
        let query = query.mul(&self.cos)?.sub(&query_rot.mul(&self.sin)?)?;
        let key = key.mul(&self.cos)?.sub(&key_rot.mul(&self.sin)?)?;
        
        Ok((query, key))
    }
    
    fn rotate_half(&self, x: &Tensor) -> CandleResult<Tensor> {
        let dims = x.dims().to_vec();
        let last_dim = dims[dims.len() - 1];
        let half_dim = last_dim / 2;
        
        let x1 = x.slice(dims.len() - 1, 0, half_dim)?;
        let x2 = x.slice(dims.len() - 1, half_dim, last_dim)?;
        
        Tensor::cat(&[x2.neg()?, x1], dims.len() - 1)
    }
}

#[derive(Debug)]
pub struct AttentionCache {
    key_cache: Option<Tensor>,
    value_cache: Option<Tensor>,
    size: usize,
}

impl AttentionCache {
    pub fn new() -> Self {
        Self {
            key_cache: None,
            value_cache: None,
            size: 0,
        }
    }

    pub fn update(&mut self, key: &Tensor, value: &Tensor) -> CandleResult<()> {
        if let (Some(k_cache), Some(v_cache)) = (&self.key_cache, &self.value_cache) {
            self.key_cache = Some(Tensor::cat(&[k_cache, key], 1)?);
            self.value_cache = Some(Tensor::cat(&[v_cache, value], 1)?);
        } else {
            self.key_cache = Some(key.clone());
            self.value_cache = Some(value.clone());
        }
        self.size += key.dim(1)?;
        Ok(())
    }

    pub fn clear(&mut self) {
        self.key_cache = None;
        self.value_cache = None;
        self.size = 0;
    }
}

impl MultiHeadAttention {
    pub fn forward_with_cache(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        mask: Option<&Tensor>,
        cache: Option<&mut AttentionCache>,
    ) -> CandleResult<Tensor> {
        let batch_size = query.dim(0)?;
        let seq_len = query.dim(1)?;
        
        // Input validation
        if key.dim(0)? != batch_size || value.dim(0)? != batch_size {
            return Err(candle_core::Error::Msg(
                "Batch size mismatch between query, key, and value".into()
            ));
        }

        // Project inputs
        let query = query.matmul(&self.query_proj)?;
        let key = key.matmul(&self.key_proj)?;
        let value = value.matmul(&self.value_proj)?;

        // Use cached key/value if available
        let (key_final, value_final) = if let Some(cache) = cache {
            if let (Some(k_cache), Some(v_cache)) = (&cache.key_cache, &cache.value_cache) {
                cache.update(&key, &value)?;
                (k_cache.clone(), v_cache.clone())
            } else {
                cache.update(&key, &value)?;
                (key, value)
            }
        } else {
            (key, value)
        };

        // Create attention dropout mask
        let dropout_mask = if self.training && self.config.attention_dropout_prob > 0.0 {
            Some(Tensor::rand(
                0.0,
                1.0,
                (batch_size, self.config.num_attention_heads, seq_len, key_final.dim(1)?),
                &self.device,
            )?.greater_equal(self.config.attention_dropout_prob)?)
        } else {
            None
        };

        // Compute attention with error handling
        let attention = match (self.config.use_flash_attention, mask, dropout_mask) {
            (true, None, _) => self.flash_attention(query, key_final, value_final, None)?,
            (true, Some(_), _) => {
                // Flash attention doesn't support masking, fall back to standard
                self.standard_attention(query, key_final, value_final, mask)?
            },
            (false, m, d) => self.standard_attention_with_dropout(query, key_final, value_final, m, d)?,
        };

        // Project output
        attention.matmul(&self.output_proj)
    }

    fn standard_attention_with_dropout(
        &self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Option<&Tensor>,
        dropout_mask: Option<Tensor>,
    ) -> CandleResult<Tensor> {
        let scale = (self.config.hidden_dim as f64).sqrt();
        let mut scores = query.matmul(&key.transpose(2, 3)?)?.div(scale)?;
        
        if let Some(mask) = mask {
            scores = scores.add(mask)?;
        }
        
        let attention_probs = scores.softmax(3)?;
        
        let attention_probs = if let Some(mask) = dropout_mask {
            attention_probs.mul(&mask)?
        } else {
            attention_probs
        };
        
        attention_probs.matmul(&value)
    }
}