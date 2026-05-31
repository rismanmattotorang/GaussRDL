mod random_walk;
mod layer_wise;
mod cached;
mod adaptive;

pub use random_walk::RandomWalkSampler;
pub use layer_wise::LayerWiseSampler;
pub use cached::CachedGraphSampler;
pub use adaptive::AdaptiveGraphSampler;

use candle_core::{Device, Tensor, Result as CandleResult, DType};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use parking_lot::RwLock;
use std::default::Default;
use crate::graph::{RelationalGraph, NodeId, RelationType, SparseTensor};
use crate::graph::{GraphResult, GraphError};

/// Common trait for all graph sampling implementations
pub trait GraphSampler: Send + Sync {
    /// Sample a subgraph from the given graph
    fn sample_subgraph(&self, graph: &RelationalGraph, seed: Option<u64>) -> CandleResult<RelationalGraph>;
    
    /// Convert tensor to specified dtype safely
    fn convert_tensor_dtype(&self, tensor: &Tensor, dtype: DType) -> CandleResult<Tensor> {
        // For half-precision types, convert through f32 first
        match (tensor.dtype(), dtype) {
            (DType::F16, _) | (DType::BF16, _) => {
                let f32_tensor = tensor.to_dtype(DType::F32)?;
                f32_tensor.to_dtype(dtype)
            }
            (_, DType::F16) | (_, DType::BF16) => {
                let f32_tensor = tensor.to_dtype(DType::F32)?;
                f32_tensor.to_dtype(dtype)
            }
            _ => tensor.to_dtype(dtype)
        }
    }
    
    /// Create sparse tensor from edge indices and values
    fn create_sparse_tensor(
        &self,
        indices: Vec<i64>,
        values: Vec<f32>,
        size: (usize, usize),
        _device: &Device,
    ) -> CandleResult<SparseTensor> {
        let values_len = values.len();
        // Convert to the format expected by SparseTensor
        let indices_vec = if values_len > 0 {
            vec![
                indices.iter().step_by(2).map(|&x| x as usize).collect(),
                indices.iter().skip(1).step_by(2).map(|&x| x as usize).collect(),
            ]
        } else {
            vec![vec![], vec![]]
        };
        
        Ok(SparseTensor::new(
            indices_vec,
            values,
            vec![size.0, size.1],
        ))
    }
    
    /// Sample node features from the original graph
    fn sample_node_features(
        &self,
        graph: &RelationalGraph,
        sampled_nodes: &HashSet<usize>,
    ) -> CandleResult<Tensor> {
        let mut features = Vec::new();
        for &node in sampled_nodes {
            features.push(graph.node_features.as_ref().expect("Node features not available").get(node)?);
        }
        Tensor::stack(&features, 0)
    }
    
    /// Sample edge features from the original graph
    fn sample_edge_features(
        &self,
        graph: &RelationalGraph,
        sampled_edges: &[(usize, usize)],
    ) -> CandleResult<Option<Tensor>> {
        if let Some(ef) = &graph.edge_features {
            let mut features = Vec::new();
            for &(src, dst) in sampled_edges {
                // Find the edge index in the adjacency matrix
                let edge_idx = graph.adjacency.indices[0].iter()
                    .zip(graph.adjacency.indices[1].iter())
                    .position(|(&row, &col)| row == src && col == dst);
                
                if let Some(idx) = edge_idx {
                    features.push(ef.get(idx)?);
                }
            }
            if features.is_empty() {
                Ok(None)
            } else {
                Ok(Some(Tensor::stack(&features, 0)?))
            }
        } else {
            Ok(None)
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct GraphStats {
    pub num_nodes: usize,
    pub num_edges: usize,
    pub num_relation_types: usize,
    pub avg_degree: f64,
    pub density: f64,
    pub diameter: Option<f64>,
    pub clustering_coefficient: Option<f64>,
    pub max_degree: usize,
    pub min_degree: usize,
}

impl Default for GraphStats {
    fn default() -> Self {
        Self {
            num_nodes: 0,
            num_edges: 0,
            num_relation_types: 0,
            avg_degree: 0.0,
            density: 0.0,
            diameter: None,
            clustering_coefficient: None,
            max_degree: 0,
            min_degree: 0,
        }
    }
}

impl GraphStats {
    pub fn new(graph: &RelationalGraph) -> Self {
        let mut stats = Self::default();
        stats.num_nodes = graph.nodes.len();
        stats.num_edges = graph.edges.len();
        stats.num_relation_types = graph.relation_types.len();
        
        // Calculate degree statistics
        let mut degrees = Vec::new();
        for node_id in graph.nodes.keys() {
            let degree = graph.edges.iter()
                .filter(|e| e.from == *node_id || e.to == *node_id)
                .count();
            degrees.push(degree);
        }
        
        if !degrees.is_empty() {
            stats.avg_degree = degrees.iter().sum::<usize>() as f64 / degrees.len() as f64;
            stats.max_degree = *degrees.iter().max().unwrap();
            stats.min_degree = *degrees.iter().min().unwrap();
        }
        
        // Calculate density
        let max_edges = stats.num_nodes * (stats.num_nodes - 1) / 2;
        if max_edges > 0 {
            stats.density = stats.num_edges as f64 / max_edges as f64;
        }
        
        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_random_walk_sampling() {
        // Add tests for random walk sampling
    }
    
    #[test]
    fn test_layer_wise_sampling() {
        // Add tests for layer-wise sampling
    }
    
    #[test]
    fn test_cached_sampling() {
        // Add tests for cached sampling
    }
    
    #[test]
    fn test_adaptive_sampling() {
        // Add tests for adaptive sampling
    }
}