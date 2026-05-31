use candle_core::{Device, Tensor, DType, Result as CandleResult};
use candle_nn::{Linear, VarBuilder, Embedding, LayerNorm};
use gaussrdl_core::Result;

/// Helper functions for Candle API compatibility
pub struct CandleHelper;

impl CandleHelper {
    /// Create a Linear layer with proper parameter initialization
    pub fn linear(vb: VarBuilder, in_dim: usize, out_dim: usize, name: &str) -> Result<Linear> {
        let weight = vb.get((out_dim, in_dim), &format!("{}.weight", name))?;
        let bias = vb.get(out_dim, &format!("{}.bias", name))?;
        Ok(Linear::new(weight, Some(bias)))
    }
    
    /// Create a LayerNorm layer with proper parameter initialization
    pub fn layer_norm(vb: VarBuilder, normalized_shape: usize, name: &str, eps: f64) -> Result<LayerNorm> {
        let weight = vb.get(normalized_shape, &format!("{}.weight", name))?;
        let bias = vb.get(normalized_shape, &format!("{}.bias", name))?;
        Ok(LayerNorm::new(weight, bias, eps))
    }
    
    /// Apply dropout to a tensor
    pub fn dropout(tensor: &Tensor, dropout_prob: f64, training: bool) -> CandleResult<Tensor> {
        if !training || dropout_prob == 0.0 {
            Ok(tensor.clone())
        } else {
            let scale = 1.0 / (1.0 - dropout_prob);
            let mask = Tensor::rand(0.0, 1.0, tensor.shape(), tensor.device())?;
            let keep_mask = mask.gt(dropout_prob as f32)?;
            let scale_tensor = Tensor::from_slice(&[scale as f32], &[1], tensor.device())?;
            Ok(tensor.mul(&keep_mask.to_dtype(tensor.dtype())?)?.mul(&scale_tensor.broadcast_as(tensor.shape())?)?)
        }
    }
    
    /// Create an Embedding layer
    pub fn embedding(
        num_embeddings: usize,
        embedding_dim: usize,
        vb: VarBuilder,
        name: &str,
    ) -> Result<Embedding> {
        let embeddings = vb.get((num_embeddings, embedding_dim), &format!("{}.weight", name))?;
        Ok(Embedding::new(embeddings, embedding_dim))
    }
    
    /// Index select operation
    pub fn index_select(tensor: &Tensor, indices: &Tensor, dim: usize) -> CandleResult<Tensor> {
        tensor.index_select(indices, dim)
    }
    
    /// Scatter add operation
    pub fn scatter_add(tensor: &Tensor, indices: &Tensor, src: &Tensor, dim: usize) -> CandleResult<Tensor> {
        tensor.scatter_add(indices, src, dim)
    }
    
    /// Create zeros tensor
    pub fn zeros(shape: &[usize], dtype: DType, device: &Device) -> CandleResult<Tensor> {
        Tensor::zeros(shape, dtype, device)
    }
    
    /// Create ones tensor
    pub fn ones(shape: &[usize], dtype: DType, device: &Device) -> CandleResult<Tensor> {
        Tensor::ones(shape, dtype, device)
    }
    
    /// Create attention mask from edge index
    pub fn create_attention_mask(edge_index: &Tensor, _num_nodes: usize) -> CandleResult<Tensor> {
        // Simple implementation - return ones for all edges
        let num_edges = edge_index.shape().dims()[1];
        Tensor::ones(&[num_edges], DType::F32, edge_index.device())
    }
    
    /// L2 normalize a tensor
    pub fn l2_normalize(tensor: &Tensor, dim: usize, eps: f64) -> CandleResult<Tensor> {
        let norm = tensor.sqr()?.sum_keepdim(dim)?.sqrt()?.clamp(eps, f64::INFINITY)?;
        tensor.div(&norm)
    }
    
    /// Apply activation function
    pub fn activation(tensor: &Tensor, activation: &str) -> CandleResult<Tensor> {
        match activation {
            "relu" => tensor.relu(),
            "gelu" => tensor.gelu(),
            "tanh" => tensor.tanh(),
            "sigmoid" => {
                // Manual sigmoid implementation: 1 / (1 + exp(-x))
                let neg_x = tensor.neg()?;
                let exp_neg_x = neg_x.exp()?;
                let one = Tensor::ones(tensor.shape(), tensor.dtype(), tensor.device())?;
                let denominator = one.add(&exp_neg_x)?;
                one.div(&denominator)
            },
            "leaky_relu" => {
                let zero = Tensor::zeros(tensor.shape(), tensor.dtype(), tensor.device())?;
                let positive = tensor.maximum(&zero)?;
                let alpha_tensor = Tensor::from_slice(&[0.01f32], &[1], tensor.device())?;
                let negative = tensor.minimum(&zero)?.mul(&alpha_tensor.broadcast_as(tensor.shape())?)?;
                positive.add(&negative)
            },
            _ => Ok(tensor.clone()),
        }
    }
    
    /// Filter edges by type
    pub fn filter_edges_by_type(edge_types: &Tensor, target_type: usize) -> CandleResult<Tensor> {
        edge_types.eq(target_type as f64)
    }
    
    /// Batch update tensor
    pub fn batch_update(
        mut output: Tensor,
        updates: &[Tensor],
        indices: &[usize],
    ) -> CandleResult<Tensor> {
        for (idx, updated) in indices.iter().zip(updates.iter()) {
            if let Ok(unsqueezed) = updated.unsqueeze(0) {
                if let Ok(assigned) = output.slice_assign(&[*idx..*idx+1], &unsqueezed) {
                    output = assigned;
                }
            }
        }
        Ok(output)
    }
}

/// Trait for models that need Candle compatibility
pub trait CandleCompatible {
    /// Convert to compatible format
    fn to_compatible(&self) -> Result<()>;
    
    /// Validate tensor operations
    fn validate_tensors(&self) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;

    #[test]
    fn test_linear_creation() -> Result<()> {
        let device = Device::Cpu;
        let var_map = VarMap::new();
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, &device);
        
        let _linear = CandleHelper::linear(vb, 10, 5, "test")?;
        // Test would require actual tensor operations
        Ok(())
    }
    
    #[test]
    fn test_dropout() -> Result<()> {
        let device = Device::Cpu;
        let tensor = Tensor::ones(&[10, 5], DType::F32, &device)?;
        
        // Test training mode
        let _dropped = CandleHelper::dropout(&tensor, 0.5, true)?;
        
        // Test eval mode
        let not_dropped = CandleHelper::dropout(&tensor, 0.5, false)?;
        assert_eq!(tensor.shape(), not_dropped.shape());
        
        Ok(())
    }
    
    #[test]
    fn test_activation() -> Result<()> {
        let device = Device::Cpu;
        let tensor = Tensor::randn(0.0, 1.0, &[5, 3], &device)?;
        
        let _relu_out = CandleHelper::activation(&tensor, "relu")?;
        let _gelu_out = CandleHelper::activation(&tensor, "gelu")?;
        let _tanh_out = CandleHelper::activation(&tensor, "tanh")?;
        
        Ok(())
    }
} 