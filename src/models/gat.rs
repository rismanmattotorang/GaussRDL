use candle_core::{Device, Result as CandleResult, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use crate::error::Result;

/// GAT model configuration
#[derive(Debug, Clone)]
pub struct GATConfig {
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub dropout: f32,
    pub attention_dropout: f32,
    pub use_layer_norm: bool,
    pub use_residual: bool,
    pub activation: String,
}

impl Default for GATConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 256,
            num_heads: 8,
            num_layers: 3,
            dropout: 0.1,
            attention_dropout: 0.1,
            use_layer_norm: true,
            use_residual: true,
            activation: "relu".to_string(),
        }
    }
}

/// GAT model implementation
pub struct GAT {
    config: GATConfig,
    input_layer: Linear,
    gat_layers: Vec<GATLayer>,
    output_layer: Linear,
    device: Device,
}

impl GAT {
    /// Creates a new GAT model
    pub fn new(config: GATConfig, vb: VarBuilder) -> Result<Self> {
        let input_dim = config.hidden_dim;
        let hidden_dim = config.hidden_dim;
        let output_dim = config.hidden_dim;

        let input_layer = Linear::new(
            vb.pp("input"),
            input_dim,
            hidden_dim,
        )?;

        let mut gat_layers = Vec::new();
        for i in 0..config.num_layers {
            let layer = GATLayer::new(
                hidden_dim,
                hidden_dim,
                &config,
                vb.pp(&format!("gat_{}", i)),
            )?;
            gat_layers.push(layer);
        }

        let output_layer = Linear::new(
            vb.pp("output"),
            hidden_dim,
            output_dim,
        )?;

        Ok(Self {
            config,
            input_layer,
            gat_layers,
            output_layer,
            device: vb.device().clone(),
        })
    }

    /// Forward pass
    pub fn forward(&self, x: &Tensor, edge_index: &Tensor) -> Result<Tensor> {
        let mut h = self.input_layer.forward(x)?;

        for layer in &self.gat_layers {
            h = layer.forward(&h, edge_index)?;
        }

        let out = self.output_layer.forward(&h)?;
        Ok(out)
    }
}

/// GAT layer implementation
struct GATLayer {
    query: Linear,
    key: Linear,
    value: Linear,
    layer_norm: Option<candle_nn::LayerNorm>,
    config: GATConfig,
}

impl GATLayer {
    fn new(
        in_dim: usize,
        out_dim: usize,
        config: &GATConfig,
        vb: VarBuilder,
    ) -> Result<Self> {
        let head_dim = out_dim / config.num_heads;

        let query = Linear::new(
            vb.pp("query"),
            in_dim,
            out_dim,
        )?;

        let key = Linear::new(
            vb.pp("key"),
            in_dim,
            out_dim,
        )?;

        let value = Linear::new(
            vb.pp("value"),
            in_dim,
            out_dim,
        )?;

        let layer_norm = if config.use_layer_norm {
            Some(candle_nn::LayerNorm::new(
                vb.pp("layer_norm"),
                vec![out_dim],
                1e-5,
            )?)
        } else {
            None
        };

        Ok(Self {
            query,
            key,
            value,
            layer_norm,
            config: config.clone(),
        })
    }

    fn forward(&self, x: &Tensor, edge_index: &Tensor) -> Result<Tensor> {
        let batch_size = x.dim(0)?;
        let num_heads = self.config.num_heads;
        let head_dim = self.config.hidden_dim / num_heads;

        // Linear transformations
        let q = self.query.forward(x)?;
        let k = self.key.forward(x)?;
        let v = self.value.forward(x)?;

        // Reshape for multi-head attention
        let q = q.reshape((batch_size, num_heads, head_dim))?;
        let k = k.reshape((batch_size, num_heads, head_dim))?;
        let v = v.reshape((batch_size, num_heads, head_dim))?;

        // Get source and target nodes
        let row = edge_index.get(0)?;
        let col = edge_index.get(1)?;

        // Compute attention scores
        let q_i = q.index_select(&row)?;
        let k_j = k.index_select(&col)?;
        
        // Scaled dot-product attention
        let scores = q_i.matmul(&k_j.transpose(1, 2)?)?;
        let scores = scores.div(f64::sqrt(head_dim as f64))?;
        let attention = scores.softmax(2)?;

        // Apply attention dropout
        let attention = if self.config.attention_dropout > 0.0 {
            attention.dropout(self.config.attention_dropout)?
        } else {
            attention
        };

        // Compute weighted values
        let v_j = v.index_select(&col)?;
        let mut h = attention.matmul(&v_j)?;

        // Reshape back
        h = h.reshape((batch_size, self.config.hidden_dim))?;

        // Layer norm
        if let Some(ref ln) = self.layer_norm {
            h = ln.forward(&h)?;
        }

        // Residual connection
        if self.config.use_residual {
            h = h.add(x)?;
        }

        // Activation
        match self.config.activation.as_str() {
            "relu" => h = h.relu()?,
            "gelu" => h = h.gelu()?,
            _ => return Err(crate::error::Error::model("Unknown activation")),
        }

        Ok(h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_creation() -> Result<()> {
        let config = GATConfig::default();
        let vb = VarBuilder::zeros(Device::Cpu);
        let model = GAT::new(config, vb)?;
        
        // Test forward pass
        let x = Tensor::zeros((10, 256), Device::Cpu)?;
        let edge_index = Tensor::zeros((2, 20), Device::Cpu)?;
        let out = model.forward(&x, &edge_index)?;
        
        assert_eq!(out.shape().dims(), &[10, 256]);
        Ok(())
    }
} 