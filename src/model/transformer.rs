// src/model/transformer.rs
use candle_core::{Device, Tensor, Result as CandleResult};
use candle_nn::{Linear, LayerNorm, Module};

use super::config::RelgtConfig;
use super::attention::{create_attention_mechanism, AttentionMechanism};

pub trait TransformerLayer: Send + Sync {
    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> CandleResult<Tensor>;
}

pub trait TransformerBlock: Send + Sync {
    fn attention(&self, x: &Tensor, mask: Option<&Tensor>) -> CandleResult<Tensor>;
    fn ffn(&self, x: &Tensor) -> CandleResult<Tensor>;
    fn norm(&self, x: &Tensor) -> CandleResult<Tensor>;
}

pub struct RelgtTransformerBlock {
    attention: Box<dyn AttentionMechanism>,
    attention_proj: Linear,
    ffn_1: Linear,
    ffn_2: Linear,
    layer_norm_1: LayerNorm,
    layer_norm_2: LayerNorm,
    dropout_rate: f32,
    device: Device,
}

impl RelgtTransformerBlock {
    pub fn new(config: &RelgtConfig, device: Device) -> CandleResult<Self> {
        let hidden_dim = config.hidden_dim;
        let intermediate_size = config.intermediate_size;
        
        Ok(Self {
            attention: create_attention_mechanism(config, device.clone()),
            attention_proj: Linear::new(hidden_dim, hidden_dim)?,
            ffn_1: Linear::new(hidden_dim, intermediate_size)?,
            ffn_2: Linear::new(intermediate_size, hidden_dim)?,
            layer_norm_1: LayerNorm::new(hidden_dim, config.layer_norm_epsilon)?,
            layer_norm_2: LayerNorm::new(hidden_dim, config.layer_norm_epsilon)?,
            dropout_rate: config.dropout_rate,
            device,
        })
    }
}

impl TransformerBlock for RelgtTransformerBlock {
    fn attention(&self, x: &Tensor, mask: Option<&Tensor>) -> CandleResult<Tensor> {
        let q = self.attention_proj.forward(x)?;
        let k = self.attention_proj.forward(x)?;
        let v = self.attention_proj.forward(x)?;
        
        self.attention.compute_attention(&q, &k, &v, mask)
    }
    
    fn ffn(&self, x: &Tensor) -> CandleResult<Tensor> {
        let intermediate = self.ffn_1.forward(x)?;
        let intermediate = intermediate.gelu()?;
        let output = self.ffn_2.forward(&intermediate)?;
        
        if self.dropout_rate > 0.0 {
            output.dropout(self.dropout_rate, None)
        } else {
            Ok(output)
        }
    }
    
    fn norm(&self, x: &Tensor) -> CandleResult<Tensor> {
        self.layer_norm_1.forward(x)
    }
}

impl TransformerLayer for RelgtTransformerBlock {
    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> CandleResult<Tensor> {
        // Pre-norm architecture
        let normed_states = self.layer_norm_1.forward(hidden_states)?;
        let attention_output = self.attention(&normed_states, attention_mask)?;
        let residual_1 = hidden_states.add(&attention_output)?;
        
        let normed_residual = self.layer_norm_2.forward(&residual_1)?;
        let ffn_output = self.ffn(&normed_residual)?;
        residual_1.add(&ffn_output)
    }
}

pub struct RelgtTransformer {
    layers: Vec<Box<dyn TransformerLayer>>,
    final_norm: LayerNorm,
    config: RelgtConfig,
    device: Device,
}

impl RelgtTransformer {
    pub fn new(config: RelgtConfig, device: Device) -> CandleResult<Self> {
        let mut layers = Vec::with_capacity(config.num_transformer_layers);
        for _ in 0..config.num_transformer_layers {
            layers.push(Box::new(RelgtTransformerBlock::new(&config, device.clone())?));
        }
        
        Ok(Self {
            layers,
            final_norm: LayerNorm::new(config.hidden_dim, config.layer_norm_epsilon)?,
            config,
            device,
        })
    }
    
    pub fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> CandleResult<Tensor> {
        let mut current_states = hidden_states.clone();
        
        for layer in &self.layers {
            current_states = layer.forward(&current_states, attention_mask)?;
        }
        
        self.final_norm.forward(&current_states)
    }
    
    pub fn get_config(&self) -> &RelgtConfig {
        &self.config
    }
}