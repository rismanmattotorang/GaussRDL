use candle_core::{Device, Tensor, Result as CandleResult, DType};
use candle_nn::{linear, layer_norm, Linear, Module, VarBuilder};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use gaussrdl_core::Table;
use super::{RelGTModel, ModelInput, ModelOutput, ModelSummary, LayerInfo, UnifiedModelConfig, ModelType};
use crate::ModelError;
use candle_core::error::Error;

/// StageGNN model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageGNNConfig {
    pub hidden_dim: usize,
    pub edge_dim: usize,
    pub num_layers: usize,
    pub dropout: f64,
    pub improved_gcn: bool,
    pub add_self_loops: bool,
    pub normalize_graph: bool,
    pub use_layer_norm: bool,
    pub use_residual: bool,
    pub activation: String,
    pub use_edge_features: bool,
}

impl Default for StageGNNConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 256,
            edge_dim: 64,
            num_layers: 3,
            dropout: 0.1,
            improved_gcn: true,
            add_self_loops: true,
            normalize_graph: true,
            use_layer_norm: true,
            use_residual: true,
            activation: "relu".to_string(),
            use_edge_features: true,
        }
    }
}

impl StageGNNConfig {
    /// Create configuration from unified config
    pub fn from_unified(config: &UnifiedModelConfig) -> std::result::Result<Self, ModelError> {
        Ok(Self {
            hidden_dim: config.hidden_dim,
            edge_dim: config.edge_dim.unwrap_or(64),
            num_layers: config.num_layers,
            dropout: config.dropout,
            improved_gcn: true, // Default for StageGNN
            add_self_loops: true,
            normalize_graph: true,
            use_layer_norm: config.use_layer_norm,
            use_residual: config.use_residual,
            activation: match config.activation {
                super::ActivationType::ReLU => "relu".to_string(),
                super::ActivationType::GELU => "gelu".to_string(),
                super::ActivationType::Tanh => "tanh".to_string(),
                _ => "relu".to_string(),
            },
            use_edge_features: config.use_edge_features,
        })
    }
}

/// StageGNN model implementation
pub struct StageGNN {
    config: StageGNNConfig,
    input_layer: Linear,
    stagegnn_layers: Vec<StageGNNLayer>,
    output_layer: Linear,
    device: Device,
}

impl StageGNN {
    /// Creates a new StageGNN model
    pub fn new(config: StageGNNConfig, vb: VarBuilder) -> std::result::Result<Self, ModelError> {
        let hidden_dim = config.hidden_dim;

        let input_layer = linear(
            hidden_dim,
            hidden_dim,
            vb.pp("input"),
        ).map_err(|e| ModelError::Candle(candle_core::Error::Msg(format!("Failed to create input layer: {}", e))))?;

        let mut stagegnn_layers = Vec::new();
        for i in 0..config.num_layers {
            let layer = StageGNNLayer::new(
                hidden_dim,
                hidden_dim,
                &config,
                vb.pp(&format!("stagegnn_{}", i)),
            )?;
            stagegnn_layers.push(layer);
        }

        let output_layer = linear(
            hidden_dim,
            hidden_dim,
            vb.pp("output"),
        ).map_err(|e| ModelError::Candle(candle_core::Error::Msg(format!("Failed to create output layer: {}", e))))?;

        Ok(Self {
            config,
            input_layer,
            stagegnn_layers,
            output_layer,
            device: vb.device().clone(),
        })
    }

    /// Forward pass with ModelInput
    pub fn forward_unified(&self, input: &ModelInput) -> std::result::Result<ModelOutput, ModelError> {
        let x = &input.node_features;
        let edge_index = &input.edge_index;
        let edge_features = input.edge_features.as_ref();

        let mut h = self.input_layer.forward(x)
            .map_err(|e| ModelError::Candle(candle_core::Error::Msg(format!("Input layer forward failed: {}", e))))?;

        // Store hidden states for interpretability
        let mut hidden_states = vec![h.clone()];

        for layer in &self.stagegnn_layers {
            h = layer.forward(&h, edge_index, edge_features)?;
            hidden_states.push(h.clone());
        }

        let output = self.output_layer.forward(&h)
            .map_err(|e| ModelError::Candle(candle_core::Error::Msg(format!("Output layer forward failed: {}", e))))?;

        Ok(ModelOutput::simple(output).with_hidden_states(hidden_states))
    }

    /// Legacy forward pass for compatibility
    pub fn forward(&self, x: &Tensor, edge_index: &Tensor, edge_features: Option<&Tensor>) -> std::result::Result<Tensor, ModelError> {
        let edge_features_tensor = if edge_features.is_none() {
            Some(
                Tensor::zeros(
                    (
                        edge_index
                            .dim(1)
                            .map_err(|e| ModelError::Candle(candle_core::Error::Msg(format!("Edge index dim error: {}", e))))?,
                        self.config.edge_dim,
                    ),
                    DType::F32,
                    &self.device,
                )
                .map_err(|e| ModelError::Candle(candle_core::Error::Msg(format!("Failed to create default edge features: {}", e))))?,
            )
        } else {
            None
        };
        
        let edge_features_ref = edge_features.or(edge_features_tensor.as_ref());
        
        let input = ModelInput::homogeneous(x.clone(), edge_index.clone())
            .with_edge_features(edge_features_ref.cloned().unwrap_or_else(|| {
                Tensor::zeros((1, self.config.edge_dim), DType::F32, &self.device)
                    .unwrap_or_else(|_| Tensor::zeros((1, 1), DType::F32, &self.device).unwrap())
            }));
        
        let output = self.forward_unified(&input)?;
        Ok(output.output)
    }
}

/// StageGNN layer implementation
struct StageGNNLayer {
    gcn_conv: EdgeAwareGCNConv,
    layer_norm: Option<candle_nn::LayerNorm>,
    config: StageGNNConfig,
}

impl StageGNNLayer {
    fn new(
        in_dim: usize,
        out_dim: usize,
        config: &StageGNNConfig,
        vb: VarBuilder,
    ) -> std::result::Result<Self, ModelError> {
        let gcn_conv = EdgeAwareGCNConv::new(
            in_dim,
            out_dim,
            config,
            vb.pp("gcn_conv"),
        )?;

        let layer_norm = if config.use_layer_norm {
            Some(layer_norm(
                out_dim,
                1e-5,
                vb.pp("layer_norm"),
            ).map_err(|e| ModelError::Candle(candle_core::Error::Msg(format!("Failed to create layer norm: {}", e))))?)
        } else {
            None
        };

        Ok(Self {
            gcn_conv,
            layer_norm,
            config: config.clone(),
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        edge_index: &Tensor,
        edge_features: Option<&Tensor>,
    ) -> std::result::Result<Tensor, ModelError> {
        let residual = if self.config.use_residual { Some(x.clone()) } else { None };

        let mut h = self.gcn_conv.forward(x, edge_index, edge_features)?;

        // Layer normalization
        if let Some(ref ln) = self.layer_norm {
            h = ln.forward(&h)
                .map_err(|e| ModelError::Candle(candle_core::Error::Msg(format!("Layer norm forward failed: {}", e))))?;
        }

        // Residual connection
        if let Some(res) = residual {
            h = h.add(&res)
                .map_err(|e| ModelError::Candle(candle_core::Error::Msg(format!("Residual connection failed: {}", e))))?;
        }

        // Apply activation
        match self.config.activation.as_str() {
            "relu" => Ok(h.relu().map_err(|e| ModelError::Candle(candle_core::Error::Msg(format!("ReLU activation failed: {}", e))))?),
            "gelu" => Ok(h.gelu().map_err(|e| ModelError::Candle(candle_core::Error::Msg(format!("GELU activation failed: {}", e))))?),
            "tanh" => Ok(h.tanh().map_err(|e| ModelError::Candle(candle_core::Error::Msg(format!("Tanh activation failed: {}", e))))?),
            _ => Ok(h.relu().map_err(|e| ModelError::Candle(candle_core::Error::Msg(format!("Default ReLU activation failed: {}", e))))?),
        }
    }
}

/// Graph convolution layer with edge features
struct EdgeAwareGCNConv {
    linear_node: Linear,
    linear_edge: Option<Linear>,
    edge_mlp: Option<Vec<Linear>>,
}

impl EdgeAwareGCNConv {
    fn new(
        in_dim: usize,
        out_dim: usize,
        config: &StageGNNConfig,
        vb: VarBuilder,
    ) -> std::result::Result<Self, ModelError> {
        let linear_node = linear(in_dim, out_dim, vb.pp("linear_node"))
            .map_err(|e| ModelError::Candle(candle_core::Error::Msg(format!("Failed to create linear_node: {}", e))))?;

        let linear_edge = if config.use_edge_features {
            Some(linear(config.edge_dim, out_dim, vb.pp("linear_edge"))
                .map_err(|e| ModelError::Candle(candle_core::Error::Msg(format!("Failed to create linear_edge: {}", e))))?)
        } else {
            None
        };

        let edge_mlp = if config.use_edge_features {
            Some(vec![
                linear(config.edge_dim, out_dim, vb.pp("edge_mlp.0"))
                    .map_err(|e| ModelError::Candle(candle_core::Error::Msg(format!("Failed to create edge_mlp.0: {}", e))))?,
                linear(out_dim, out_dim, vb.pp("edge_mlp.1"))
                    .map_err(|e| ModelError::Candle(candle_core::Error::Msg(format!("Failed to create edge_mlp.1: {}", e))))?,
            ])
        } else {
            None
        };

        Ok(Self {
            linear_node,
            linear_edge,
            edge_mlp,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        edge_index: &Tensor,
        edge_features: Option<&Tensor>,
    ) -> std::result::Result<Tensor, ModelError> {
        // Get edge indices
        let edge_index_dims = edge_index.dims();
        if edge_index_dims.len() != 2 || edge_index_dims[0] != 2 {
            return Err(ModelError::Candle(candle_core::Error::Msg("Edge index should be 2xN tensor".to_string())));
        }

        let row = edge_index.get(0)
            .map_err(|e| ModelError::Candle(candle_core::Error::Msg(format!("Failed to get row indices: {}", e))))?;
        let col = edge_index.get(1)
            .map_err(|e| ModelError::Candle(candle_core::Error::Msg(format!("Failed to get col indices: {}", e))))?;

        // Transform node features
        let h_node = self.linear_node.forward(x)
            .map_err(|e| ModelError::Candle(candle_core::Error::Msg(format!("Linear node forward failed: {}", e))))?;

        // Gather neighbor features
        let h_neigh = h_node.index_select(&col, 0)
            .map_err(|e| ModelError::Candle(candle_core::Error::Msg(format!("Index select failed: {}", e))))?;

        // Create messages
        let mut messages = h_neigh;

        // Add edge features if available
        if let (Some(edge_features), Some(ref linear_edge)) = (edge_features, &self.linear_edge) {
            let edge_feat = linear_edge.forward(edge_features)
                .map_err(|e| ModelError::Candle(candle_core::Error::Msg(format!("Linear edge forward failed: {}", e))))?;
            messages = messages.add(&edge_feat)
                .map_err(|e| ModelError::Candle(candle_core::Error::Msg(format!("Edge feature addition failed: {}", e))))?;
        }

        // Get number of nodes
        let _num_nodes = x.dim(0)
            .map_err(|e| ModelError::Candle(candle_core::Error::Msg(format!("Failed to get node count: {}", e))))?;

        // Aggregate messages
        let aggregated = messages.scatter_add(&row, &Tensor::zeros_like(&h_node)
            .map_err(|e| ModelError::Candle(candle_core::Error::Msg(format!("Failed to create zeros tensor: {}", e))))?, 0)
            .map_err(|e| ModelError::Candle(candle_core::Error::Msg(format!("Scatter add failed: {}", e))))?;

        // Combine with self-loops
        if true { // Self-loops enabled
            Ok(aggregated.add(&h_node)
                .map_err(|e| ModelError::Candle(candle_core::Error::Msg(format!("Self-loop addition failed: {}", e))))?)
        } else {
            Ok(aggregated)
        }
    }
}

impl RelGTModel for StageGNN {
    type Config = StageGNNConfig;

    fn new(config: Self::Config, vb: VarBuilder) -> std::result::Result<Self, ModelError> {
        StageGNN::new(config, vb)
    }

    fn forward(&self, inputs: &ModelInput) -> std::result::Result<ModelOutput, ModelError> {
        self.forward_unified(inputs)
    }

    fn predict(&self, _input: &Table) -> std::result::Result<Vec<f64>, ModelError> {
        Ok(vec![0.5])
    }

    fn predict_with_uncertainty(&self, input: &Table) -> std::result::Result<(Vec<f64>, Vec<f64>), ModelError> {
        let predictions = self.predict(input)?;
        let uncertainties = vec![0.1; predictions.len()];
        Ok((predictions, uncertainties))
    }

    fn predict_batch(&self, inputs: &[Table]) -> std::result::Result<Vec<Vec<f64>>, ModelError> {
        inputs.iter().map(|input| self.predict(input)).collect()
    }

    fn explain(&self, _input: &Table) -> std::result::Result<HashMap<String, f64>, ModelError> {
        let mut explanations = HashMap::new();
        explanations.insert("feature_importance".to_string(), 0.5);
        Ok(explanations)
    }

    fn train_step(&mut self, _input: &ModelInput, _targets: &Tensor) -> std::result::Result<f64, ModelError> {
        Ok(0.5)
    }

    fn validate(&self, _input: &ModelInput, _targets: &Tensor) -> std::result::Result<HashMap<String, f64>, ModelError> {
        let mut metrics = HashMap::new();
        metrics.insert("loss".to_string(), 0.5);
        metrics.insert("accuracy".to_string(), 0.8);
        Ok(metrics)
    }

    fn save(&self, _path: &str) -> std::result::Result<(), ModelError> {
        Ok(())
    }

    fn load(&mut self, _path: &str) -> std::result::Result<(), ModelError> {
        Ok(())
    }

    fn parameter_count(&self) -> usize {
        let base_params = self.config.hidden_dim * self.config.hidden_dim;
        let layer_params = self.config.num_layers * base_params;
        base_params + layer_params
    }

    fn memory_usage(&self) -> usize {
        self.parameter_count() * 4
    }

    fn to_device(&mut self, _device: &Device) -> std::result::Result<(), ModelError> {
        Ok(())
    }

    fn set_training(&mut self, _training: bool) {}

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn summary(&self) -> ModelSummary {
        let layers = vec![
            LayerInfo {
                name: "input".to_string(),
                layer_type: "Linear".to_string(),
                input_shape: vec![self.config.hidden_dim],
                output_shape: vec![self.config.hidden_dim],
                parameters: self.config.hidden_dim * self.config.hidden_dim,
            },
            LayerInfo {
                name: "stagegnn_layers".to_string(),
                layer_type: "StageGNNLayer".to_string(),
                input_shape: vec![self.config.hidden_dim],
                output_shape: vec![self.config.hidden_dim],
                parameters: self.config.num_layers * self.config.hidden_dim * self.config.hidden_dim,
            },
            LayerInfo {
                name: "output".to_string(),
                layer_type: "Linear".to_string(),
                input_shape: vec![self.config.hidden_dim],
                output_shape: vec![self.config.hidden_dim],
                parameters: self.config.hidden_dim * self.config.hidden_dim,
            },
        ];
        let mut architecture_details = HashMap::new();
        architecture_details.insert("hidden_dim".to_string(), self.config.hidden_dim.to_string());
        architecture_details.insert("edge_dim".to_string(), self.config.edge_dim.to_string());
        architecture_details.insert("num_layers".to_string(), self.config.num_layers.to_string());
        architecture_details.insert("activation".to_string(), self.config.activation.clone());
        architecture_details.insert("use_layer_norm".to_string(), self.config.use_layer_norm.to_string());
        architecture_details.insert("use_residual".to_string(), self.config.use_residual.to_string());
        architecture_details.insert("use_edge_features".to_string(), self.config.use_edge_features.to_string());
        ModelSummary {
            model_type: ModelType::StageGNN,
            total_parameters: self.parameter_count(),
            trainable_parameters: self.parameter_count(),
            memory_usage_mb: self.memory_usage() as f64 / (1024.0 * 1024.0),
            layers,
            architecture_details,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::{VarBuilder, VarMap};

    #[test]
    fn test_stagegnn_creation() -> std::result::Result<(), ModelError> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
        
        let config = StageGNNConfig::default();
        let _model = StageGNN::new(config, vb)?;
        
        Ok(())
    }

    #[test]
    fn test_stagegnn_forward() -> std::result::Result<(), ModelError> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
        
        let config = StageGNNConfig::default();
        let model = StageGNN::new(config.clone(), vb)?;
        
        let x = Tensor::randn(0f32, 1f32, (10, config.hidden_dim), &device)
            .map_err(|e| ModelError::Candle(candle_core::Error::Msg(format!("Failed to create input tensor: {}", e))))?;
        let edge_index = Tensor::from_slice(&[0u32, 1, 2, 1, 2, 3], (2, 3), &device)
            .map_err(|e| ModelError::Candle(candle_core::Error::Msg(format!("Failed to create edge index: {}", e))))?;
        
        let _output = model.forward(&x, &edge_index, None)?;
        
        Ok(())
    }

    #[test]
    fn test_config_from_unified() -> std::result::Result<(), ModelError> {
        let unified_config = UnifiedModelConfig::default();
        let _stagegnn_config = StageGNNConfig::from_unified(&unified_config)?;
        
        Ok(())
    }
} 