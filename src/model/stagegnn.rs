use candle_core::{Device, Tensor, Result as CandleResult, Shape};
use candle_nn::{Linear, Module, VarBuilder};
use std::collections::HashMap;
use std::sync::Arc;

use super::config::ModelConfig;
use crate::graph::RelationalGraph;
use crate::error::Error;

/// StageGNN: A state-of-the-art graph neural network for stage-wise message passing.
/// 
/// # Features
/// - Edge-aware graph convolution
/// - Multi-layer perceptron for edge feature processing
/// - Layer normalization and residual connections
/// - Efficient message passing mechanism
/// 
/// # Arguments
/// * `config` - Model configuration
/// * `vb` - Variable builder for parameter initialization
#[derive(Debug)]
pub struct StageGNN {
    // Core components
    gcn_edge: GCNEdgeConv,
    mpnn_layers: Vec<MPNNLayer>,
    output_layer: Linear,
    
    // Configuration
    config: Arc<ModelConfig>,
    hidden_dim: usize,
    edge_dim: usize,
    num_layers: usize,
    improved: bool,
    cached: bool,
    add_self_loops: bool,
    normalize: bool,
}

/// Edge-aware graph convolutional layer with advanced features.
#[derive(Debug)]
struct GCNEdgeConv {
    lin: Linear,
    edge_mlp: Vec<Linear>,
    bias: Option<Tensor>,
    cached_edge_index: Option<(Tensor, Option<Tensor>)>,
    improved: bool,
    add_self_loops: bool,
    normalize: bool,
}

impl GCNEdgeConv {
    /// Creates a new GCNEdgeConv layer.
    /// 
    /// # Arguments
    /// * `in_channels` - Number of input channels
    /// * `out_channels` - Number of output channels
    /// * `edge_dim` - Dimension of edge features
    /// * `vb` - Variable builder for parameter initialization
    /// * `config` - Model configuration
    fn new(in_channels: usize, out_channels: usize, edge_dim: usize, vb: VarBuilder, config: &ModelConfig) -> CandleResult<Self> {
        let lin = Linear::new(vb.pp("lin"), in_channels, out_channels)?;
        
        let edge_mlp = vec![
            Linear::new(vb.pp("edge_mlp.0"), edge_dim, out_channels)?,
            Linear::new(vb.pp("edge_mlp.1"), out_channels, out_channels)?,
        ];
        
        let bias = if config.use_bias {
            Some(vb.get_with_hints("bias", &[out_channels], "bias")?)
        } else {
            None
        };
        
        Ok(Self {
            lin,
            edge_mlp,
            bias,
            cached_edge_index: None,
            improved: config.improved_gcn,
            add_self_loops: config.add_self_loops,
            normalize: config.normalize_graph,
        })
    }
    
    /// Applies graph normalization to edge indices and weights.
    fn gcn_norm(&self, edge_index: &Tensor, edge_weight: Option<&Tensor>, num_nodes: usize) -> CandleResult<(Tensor, Option<Tensor>)> {
        let fill_value = if self.improved { 2.0 } else { 1.0 };
        
        // Add self loops if needed
        let (edge_index, edge_weight) = if self.add_self_loops {
            self.add_remaining_self_loops(edge_index, edge_weight, fill_value, num_nodes)?
        } else {
            (edge_index.clone(), edge_weight.cloned())
        };
        
        // Compute degree
        let row = edge_index.get(0)?;
        let col = edge_index.get(1)?;
        
        let edge_weight = if let Some(w) = edge_weight {
            w
        } else {
            Tensor::ones((edge_index.dim(1)?,), edge_index.device())?
        };
        
        let deg = edge_weight.scatter_add(0, &col, num_nodes)?;
        let deg_inv_sqrt = deg.pow_tensor_scalar(-0.5)?;
        let deg_inv_sqrt = deg_inv_sqrt.where_cond(&deg_inv_sqrt.eq(&f64::INFINITY)?, &Tensor::zeros_like(&deg_inv_sqrt)?)?;
        
        let norm = deg_inv_sqrt.index_select(&row)?.mul(&edge_weight)?.mul(&deg_inv_sqrt.index_select(&col)?)?;
        
        Ok((edge_index, Some(norm)))
    }
    
    /// Adds self loops to the graph.
    fn add_remaining_self_loops(
        &self,
        edge_index: &Tensor,
        edge_weight: Option<&Tensor>,
        fill_value: f64,
        num_nodes: usize,
    ) -> CandleResult<(Tensor, Option<Tensor>)> {
        let device = edge_index.device();
        let diag_idx = Tensor::arange(0, num_nodes as i64, device)?;
        let diag_edge_index = Tensor::stack(&[&diag_idx, &diag_idx], 0)?;
        
        let edge_index = Tensor::cat(&[edge_index, &diag_edge_index], 1)?;
        
        let edge_weight = if let Some(weight) = edge_weight {
            let diag_weight = Tensor::full(num_nodes, fill_value, device)?;
            Some(Tensor::cat(&[weight, &diag_weight], 0)?)
        } else {
            None
        };
        
        Ok((edge_index, edge_weight))
    }
    
    /// Forward pass of the GCNEdgeConv layer.
    fn forward(&self, x: &Tensor, edge_index: &Tensor, edge_weight: Option<&Tensor>, edge_attr: Option<&Tensor>) -> CandleResult<Tensor> {
        let x = self.lin.forward(x)?;
        
        // Process edge attributes if available
        let edge_features = if let Some(edge_attr) = edge_attr {
            let mut edge_hidden = self.edge_mlp[0].forward(edge_attr)?;
            edge_hidden = edge_hidden.relu()?;
            self.edge_mlp[1].forward(&edge_hidden)?
        } else {
            Tensor::zeros_like(&x)?
        };
        
        // Apply graph normalization if needed
        let (edge_index, edge_weight) = if self.normalize {
            if let Some(cached) = &self.cached_edge_index {
                cached.clone()
            } else {
                let norm = self.gcn_norm(edge_index, edge_weight, x.dim(0)?)?;
                if self.cached {
                    self.cached_edge_index = Some(norm.clone());
                }
                norm
            }
        } else {
            (edge_index.clone(), edge_weight.cloned())
        };
        
        // Message passing
        let row_idx = edge_index.get(0)?;
        let col_idx = edge_index.get(1)?;
        
        let x_j = x.index_select(0, &col_idx)?;
        let mut messages = x_j.add(&edge_features)?;
        
        // Apply edge weights if available
        if let Some(weight) = edge_weight {
            messages = messages.mul(&weight.unsqueeze(-1)?)?;
        }
        
        // Aggregate messages
        let out = messages.mean_dim(0, false)?;
        
        // Add bias if present
        if let Some(bias) = &self.bias {
            out.add(bias)
        } else {
            Ok(out)
        }
    }
}

/// Message passing neural network layer with normalization and dropout.
#[derive(Debug)]
struct MPNNLayer {
    conv: GCNEdgeConv,
    norm: candle_nn::LayerNorm,
    dropout: f64,
}

impl MPNNLayer {
    /// Creates a new MPNN layer.
    fn new(hidden_dim: usize, edge_dim: usize, vb: VarBuilder, config: &ModelConfig) -> CandleResult<Self> {
        Ok(Self {
            conv: GCNEdgeConv::new(hidden_dim, hidden_dim, edge_dim, vb.pp("conv"), config)?,
            norm: candle_nn::LayerNorm::new(vb.pp("norm"), vec![hidden_dim], config.layer_norm_eps)?,
            dropout: config.hidden_dropout_prob,
        })
    }
    
    /// Forward pass of the MPNN layer.
    fn forward(&self, x: &Tensor, edge_index: &Tensor, edge_weight: Option<&Tensor>, edge_attr: Option<&Tensor>) -> CandleResult<Tensor> {
        let out = self.conv.forward(x, edge_index, edge_weight, edge_attr)?;
        let out = out.relu()?;
        let out = self.norm.forward(&out)?;
        out.dropout(self.dropout)
    }
}

impl StageGNN {
    /// Creates a new StageGNN model.
    pub fn new(config: ModelConfig, vb: VarBuilder) -> CandleResult<Self> {
        let hidden_dim = config.hidden_dim;
        let edge_dim = config.edge_dim;
        let num_layers = config.num_layers;
        
        // Initialize GCN edge convolution
        let gcn_edge = GCNEdgeConv::new(hidden_dim, hidden_dim, edge_dim, vb.pp("gcn_edge"), &config)?;
        
        // Initialize MPNN layers
        let mut mpnn_layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            mpnn_layers.push(MPNNLayer::new(
                hidden_dim,
                edge_dim,
                vb.pp(&format!("mpnn_{}", i)),
                &config,
            )?);
        }
        
        // Initialize output layer
        let output_layer = Linear::new(vb.pp("output"), hidden_dim, config.num_classes)?;
        
        Ok(Self {
            gcn_edge,
            mpnn_layers,
            output_layer,
            config: Arc::new(config),
            hidden_dim,
            edge_dim,
            num_layers,
            improved: config.improved_gcn,
            cached: config.cache_gc,
            add_self_loops: config.add_self_loops,
            normalize: config.normalize_graph,
        })
    }
    
    /// Forward pass of the StageGNN model.
    pub fn forward(&self, x: &Tensor, edge_index: &Tensor, edge_weight: Option<&Tensor>, edge_attr: Option<&Tensor>) -> CandleResult<Tensor> {
        // Validate input dimensions
        self.validate_input_dims(x, edge_index, edge_attr)?;
        
        // Initial edge-aware convolution
        let mut hidden = self.gcn_edge.forward(x, edge_index, edge_weight, edge_attr)?;
        
        // Apply MPNN layers
        for layer in &self.mpnn_layers {
            hidden = layer.forward(&hidden, edge_index, edge_weight, edge_attr)?;
        }
        
        // Final prediction
        self.output_layer.forward(&hidden)
    }
    
    /// Validates input dimensions.
    fn validate_input_dims(&self, x: &Tensor, edge_index: &Tensor, edge_attr: Option<&Tensor>) -> CandleResult<()> {
        let x_shape = x.shape();
        if x_shape.dims()[1] != self.hidden_dim {
            return Err(Error::InvalidInputDimension(
                format!("Expected node features of dimension {}, got {}", self.hidden_dim, x_shape.dims()[1])
            ).into());
        }
        
        if let Some(edge_attr) = edge_attr {
            let edge_attr_shape = edge_attr.shape();
            if edge_attr_shape.dims()[1] != self.edge_dim {
                return Err(Error::InvalidInputDimension(
                    format!("Expected edge features of dimension {}, got {}", self.edge_dim, edge_attr_shape.dims()[1])
                ).into());
            }
        }
        
        Ok(())
    }
    
    /// Resets cached parameters.
    pub fn reset_parameters(&mut self) -> CandleResult<()> {
        // Reset cached values
        self.gcn_edge.cached_edge_index = None;
        
        // Reset layer parameters
        for layer in &self.mpnn_layers {
            layer.conv.cached_edge_index = None;
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::test_utils::TestDevice;
    
    #[test]
    fn test_stagegnn_forward() {
        let device = &TestDevice::Cpu;
        let config = ModelConfig::new_stagegnn(64, 32, 2);
        let vb = VarBuilder::zeros(device);
        
        let model = StageGNN::new(config, vb).unwrap();
        
        // Create test inputs
        let x = Tensor::randn(0.0, 1.0, (10, 64), device).unwrap();
        let edge_index = Tensor::from_slice(&[0, 1, 1, 2, 2, 3], (2, 3), device).unwrap();
        let edge_attr = Tensor::randn(0.0, 1.0, (3, 32), device).unwrap();
        
        let output = model.forward(&x, &edge_index, None, Some(&edge_attr)).unwrap();
        assert_eq!(output.shape().dims(), &[10, 1]);
    }
    
    #[test]
    fn test_gcn_edge_conv() {
        let device = &TestDevice::Cpu;
        let config = ModelConfig::new_stagegnn(32, 16, 1);
        let vb = VarBuilder::zeros(device);
        
        let conv = GCNEdgeConv::new(32, 32, 16, vb, &config).unwrap();
        
        // Create test inputs
        let x = Tensor::randn(0.0, 1.0, (5, 32), device).unwrap();
        let edge_index = Tensor::from_slice(&[0, 1, 1, 2], (2, 2), device).unwrap();
        let edge_attr = Tensor::randn(0.0, 1.0, (2, 16), device).unwrap();
        
        let output = conv.forward(&x, &edge_index, None, Some(&edge_attr)).unwrap();
        assert_eq!(output.shape().dims(), &[5, 32]);
    }
    
    #[test]
    fn test_mpnn_layer() {
        let device = &TestDevice::Cpu;
        let config = ModelConfig::new_stagegnn(32, 16, 1);
        let vb = VarBuilder::zeros(device);
        
        let layer = MPNNLayer::new(32, 16, vb, &config).unwrap();
        
        // Create test inputs
        let x = Tensor::randn(0.0, 1.0, (5, 32), device).unwrap();
        let edge_index = Tensor::from_slice(&[0, 1, 1, 2], (2, 2), device).unwrap();
        let edge_attr = Tensor::randn(0.0, 1.0, (2, 16), device).unwrap();
        
        let output = layer.forward(&x, &edge_index, None, Some(&edge_attr)).unwrap();
        assert_eq!(output.shape().dims(), &[5, 32]);
    }
    
    #[test]
    fn test_gcn_norm() {
        let device = &TestDevice::Cpu;
        let config = ModelConfig::new_stagegnn(32, 16, 1);
        let vb = VarBuilder::zeros(device);
        
        let conv = GCNEdgeConv::new(32, 32, 16, vb, &config).unwrap();
        
        // Create test inputs
        let edge_index = Tensor::from_slice(&[0, 1, 1, 2], (2, 2), device).unwrap();
        let (norm_edge_index, norm_edge_weight) = conv.gcn_norm(&edge_index, None, 3).unwrap();
        
        assert_eq!(norm_edge_index.shape().dims(), &[2, 5]); // Original edges + self loops
        assert!(norm_edge_weight.is_some());
    }
    
    #[test]
    fn test_self_loops() {
        let device = &TestDevice::Cpu;
        let config = ModelConfig::new_stagegnn(32, 16, 1);
        let vb = VarBuilder::zeros(device);
        
        let conv = GCNEdgeConv::new(32, 32, 16, vb, &config).unwrap();
        
        // Create test inputs
        let edge_index = Tensor::from_slice(&[0, 1], (2, 1), device).unwrap();
        let (loop_edge_index, _) = conv.add_remaining_self_loops(&edge_index, None, 1.0, 2).unwrap();
        
        assert_eq!(loop_edge_index.shape().dims(), &[2, 3]); // Original edge + 2 self loops
    }
    
    #[test]
    fn test_edge_features() {
        let device = &TestDevice::Cpu;
        let config = ModelConfig::new_stagegnn(32, 16, 1);
        let vb = VarBuilder::zeros(device);
        
        let conv = GCNEdgeConv::new(32, 32, 16, vb, &config).unwrap();
        
        // Create test inputs
        let x = Tensor::randn(0.0, 1.0, (5, 32), device).unwrap();
        let edge_index = Tensor::from_slice(&[0, 1, 1, 2], (2, 2), device).unwrap();
        
        // Test with and without edge features
        let output1 = conv.forward(&x, &edge_index, None, None).unwrap();
        let edge_attr = Tensor::randn(0.0, 1.0, (2, 16), device).unwrap();
        let output2 = conv.forward(&x, &edge_index, None, Some(&edge_attr)).unwrap();
        
        assert_eq!(output1.shape(), output2.shape());
    }
} 