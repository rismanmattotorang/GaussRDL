// src/model/transformer.rs
use candle_core::{Tensor, Device, Result as CandleResult};
use candle_nn::{Linear, LayerNorm, Module, VarBuilder};
use super::RelgtConfig;

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
    pub fn new(config: &RelgtConfig, device: Device, vb: VarBuilder) -> CandleResult<Self> {
        let hidden_dim = config.hidden_dim;
        let intermediate_size = config.intermediate_dim;
        
        // Create linear layers using VarBuilder
        let attention_proj = candle_nn::linear(hidden_dim, hidden_dim, vb.pp("attention_proj"))?;
        let ffn_1 = candle_nn::linear(hidden_dim, intermediate_size, vb.pp("ffn_1"))?;
        let ffn_2 = candle_nn::linear(intermediate_size, hidden_dim, vb.pp("ffn_2"))?;
        
        // Create layer norms
        let layer_norm_1 = candle_nn::layer_norm(hidden_dim, config.layer_norm_eps as f64, vb.pp("layer_norm_1"))?;
        let layer_norm_2 = candle_nn::layer_norm(hidden_dim, config.layer_norm_eps as f64, vb.pp("layer_norm_2"))?;
        
        Ok(Self {
            attention: create_attention_mechanism(config, device.clone()),
            attention_proj,
            ffn_1,
            ffn_2,
            layer_norm_1,
            layer_norm_2,
            dropout_rate: config.dropout_rate as f32,
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
        
        // Simple dropout simulation by scaling
        if self.dropout_rate > 0.0 {
            let scale = 1.0 / (1.0 - self.dropout_rate);
            let scale_tensor = Tensor::new(scale as f64, x.device())?;
            output.mul(&scale_tensor)
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
    pub fn new(config: RelgtConfig, device: Device, vb: VarBuilder) -> CandleResult<Self> {
        let mut layers: Vec<Box<dyn TransformerLayer>> = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let layer_vb = vb.pp(&format!("layer_{}", i));
            let layer = RelgtTransformerBlock::new(&config, device.clone(), layer_vb)?;
            layers.push(Box::new(layer));
        }
        
        let final_norm = candle_nn::layer_norm(config.hidden_dim, config.layer_norm_eps as f64, vb.pp("final_norm"))?;
        
        Ok(Self {
            layers,
            final_norm,
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