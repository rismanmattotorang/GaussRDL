use candle_core::{Device, Result as CandleResult, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use crate::error::Result;

/// RGCN model configuration
#[derive(Debug, Clone)]
pub struct RGCNConfig {
    pub hidden_dim: usize,
    pub num_relations: usize,
    pub num_bases: usize,
    pub num_layers: usize,
    pub dropout: f32,
    pub use_layer_norm: bool,
    pub use_residual: bool,
    pub activation: String,
}

impl Default for RGCNConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 256,
            num_relations: 5,
            num_bases: 3,
            num_layers: 3,
            dropout: 0.1,
            use_layer_norm: true,
            use_residual: true,
            activation: "relu".to_string(),
        }
    }
}

/// RGCN model implementation
pub struct RGCN {
    config: RGCNConfig,
    input_layer: Linear,
    rgcn_layers: Vec<RGCNLayer>,
    output_layer: Linear,
    device: Device,
}

impl RGCN {
    /// Creates a new RGCN model
    pub fn new(config: RGCNConfig, vb: VarBuilder) -> Result<Self> {
        let input_dim = config.hidden_dim;
        let hidden_dim = config.hidden_dim;
        let output_dim = config.hidden_dim;

        let input_layer = Linear::new(
            vb.pp("input"),
            input_dim,
            hidden_dim,
        )?;

        let mut rgcn_layers = Vec::new();
        for i in 0..config.num_layers {
            let layer = RGCNLayer::new(
                hidden_dim,
                hidden_dim,
                &config,
                vb.pp(&format!("rgcn_{}", i)),
            )?;
            rgcn_layers.push(layer);
        }

        let output_layer = Linear::new(
            vb.pp("output"),
            hidden_dim,
            output_dim,
        )?;

        Ok(Self {
            config,
            input_layer,
            rgcn_layers,
            output_layer,
            device: vb.device().clone(),
        })
    }

    /// Forward pass
    pub fn forward(
        &self,
        x: &Tensor,
        edge_index: &Tensor,
        edge_type: &Tensor,
    ) -> Result<Tensor> {
        let mut h = self.input_layer.forward(x)?;

        for layer in &self.rgcn_layers {
            h = layer.forward(&h, edge_index, edge_type)?;
        }

        let out = self.output_layer.forward(&h)?;
        Ok(out)
    }
}

/// RGCN layer implementation
struct RGCNLayer {
    weight: Vec<Linear>,
    basis: Vec<Linear>,
    layer_norm: Option<candle_nn::LayerNorm>,
    config: RGCNConfig,
}

impl RGCNLayer {
    fn new(
        in_dim: usize,
        out_dim: usize,
        config: &RGCNConfig,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut weight = Vec::new();
        for i in 0..config.num_relations {
            let w = Linear::new(
                vb.pp(&format!("weight_{}", i)),
                in_dim,
                out_dim,
            )?;
            weight.push(w);
        }

        let mut basis = Vec::new();
        for i in 0..config.num_bases {
            let b = Linear::new(
                vb.pp(&format!("basis_{}", i)),
                in_dim,
                out_dim,
            )?;
            basis.push(b);
        }

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
            basis,
            layer_norm,
            config: config.clone(),
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        edge_index: &Tensor,
        edge_type: &Tensor,
    ) -> Result<Tensor> {
        let mut output = None;

        // Process each relation type
        for r in 0..self.config.num_relations {
            // Get edges of current relation type
            let mask = edge_type.eq(r)?;
            let rel_edges = edge_index.masked_select(&mask)?;

            // Apply relation-specific transformation
            let mut h = self.weight[r].forward(x)?;

            // Message passing
            let row = rel_edges.get(0)?;
            let col = rel_edges.get(1)?;
            let msg = h.index_select(&col)?;
            
            // Aggregation
            h = msg.mean_dim(0, false)?;

            // Accumulate output
            if let Some(out) = output {
                output = Some(out.add(&h)?);
            } else {
                output = Some(h);
            }
        }

        let mut h = output.unwrap_or_else(|| x.clone());

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
        let config = RGCNConfig::default();
        let vb = VarBuilder::zeros(Device::Cpu);
        let model = RGCN::new(config, vb)?;
        
        // Test forward pass
        let x = Tensor::zeros((10, 256), Device::Cpu)?;
        let edge_index = Tensor::zeros((2, 20), Device::Cpu)?;
        let edge_type = Tensor::zeros((20,), Device::Cpu)?;
        let out = model.forward(&x, &edge_index, &edge_type)?;
        
        assert_eq!(out.shape().dims(), &[10, 256]);
        Ok(())
    }
} 