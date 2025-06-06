use candle_core::{Device, Result as CandleResult, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use crate::error::Result;

/// Base GNN model configuration
#[derive(Debug, Clone)]
pub struct BaseGNNConfig {
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub dropout: f32,
    pub use_layer_norm: bool,
    pub use_residual: bool,
    pub activation: String,
}

impl Default for BaseGNNConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 256,
            num_layers: 3,
            dropout: 0.1,
            use_layer_norm: true,
            use_residual: true,
            activation: "relu".to_string(),
        }
    }
}

/// Base GNN model implementation
pub struct BaseGNN {
    config: BaseGNNConfig,
    input_layer: Linear,
    gnn_layers: Vec<GNNLayer>,
    output_layer: Linear,
    device: Device,
}

impl BaseGNN {
    /// Creates a new base GNN model
    pub fn new(config: BaseGNNConfig, vb: VarBuilder) -> Result<Self> {
        let input_dim = config.hidden_dim;
        let hidden_dim = config.hidden_dim;
        let output_dim = config.hidden_dim;

        let input_layer = Linear::new(
            vb.pp("input"),
            input_dim,
            hidden_dim,
        )?;

        let mut gnn_layers = Vec::new();
        for i in 0..config.num_layers {
            let layer = GNNLayer::new(
                hidden_dim,
                hidden_dim,
                &config,
                vb.pp(&format!("gnn_{}", i)),
            )?;
            gnn_layers.push(layer);
        }

        let output_layer = Linear::new(
            vb.pp("output"),
            hidden_dim,
            output_dim,
        )?;

        Ok(Self {
            config,
            input_layer,
            gnn_layers,
            output_layer,
            device: vb.device().clone(),
        })
    }

    /// Forward pass
    pub fn forward(&self, x: &Tensor, edge_index: &Tensor) -> Result<Tensor> {
        let mut h = self.input_layer.forward(x)?;

        for layer in &self.gnn_layers {
            h = layer.forward(&h, edge_index)?;
        }

        let out = self.output_layer.forward(&h)?;
        Ok(out)
    }
}

/// GNN layer implementation
struct GNNLayer {
    weight: Linear,
    layer_norm: Option<candle_nn::LayerNorm>,
    config: BaseGNNConfig,
}

impl GNNLayer {
    fn new(
        in_dim: usize,
        out_dim: usize,
        config: &BaseGNNConfig,
        vb: VarBuilder,
    ) -> Result<Self> {
        let weight = Linear::new(
            vb.pp("weight"),
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
            weight,
            layer_norm,
            config: config.clone(),
        })
    }

    fn forward(&self, x: &Tensor, edge_index: &Tensor) -> Result<Tensor> {
        let mut h = self.weight.forward(x)?;

        // Message passing
        let row = edge_index.get(0)?;
        let col = edge_index.get(1)?;
        let msg = h.index_select(&col)?;
        
        // Aggregation
        h = msg.mean_dim(0, false)?;

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
        let config = BaseGNNConfig::default();
        let vb = VarBuilder::zeros(Device::Cpu);
        let model = BaseGNN::new(config, vb)?;
        
        // Test forward pass
        let x = Tensor::zeros((10, 256), Device::Cpu)?;
        let edge_index = Tensor::zeros((2, 20), Device::Cpu)?;
        let out = model.forward(&x, &edge_index)?;
        
        assert_eq!(out.shape().dims(), &[10, 256]);
        Ok(())
    }
} 