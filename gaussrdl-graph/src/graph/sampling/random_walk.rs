use candle_core::{Tensor, Result as CandleResult, DType};
use rand::{Rng, SeedableRng, seq::SliceRandom};
use rand::distributions::WeightedIndex;
use rand::rngs::StdRng;
use std::collections::{HashMap, HashSet};

use crate::graph::RelationalGraph;
use super::{GraphSampler, GraphStats};

pub struct RandomWalkSampler {
    walk_length: usize,
    num_walks: usize,
    p: f32,  // Return parameter
    q: f32,  // In-out parameter
}

impl RandomWalkSampler {
    pub fn new(walk_length: usize, num_walks: usize, p: f32, q: f32) -> Self {
        Self {
            walk_length,
            num_walks,
            p,
            q,
        }
    }
    
    fn get_value_as_f32(tensor: &Tensor) -> CandleResult<f32> {
        // Always convert to f32 first to avoid trait implementation issues
        let f32_tensor = tensor.to_dtype(DType::F32)?;
        f32_tensor.to_scalar::<f32>()
    }
    
    fn node2vec_walk(
        &self,
        graph: &RelationalGraph,
        start_node: usize,
        rng: &mut StdRng,
    ) -> CandleResult<Vec<usize>> {
        let mut walk = vec![start_node];
        let adj_matrix = graph.adjacency.to_dense()?;
        
        for _ in 0..self.walk_length {
            let curr = *walk.last().unwrap();
            let prev = if walk.len() > 1 { Some(walk[walk.len() - 2]) } else { None };
            
            // Get neighbors
            let mut neighbors = Vec::new();
            for i in 0..adj_matrix.dim(0)? {
                let val = adj_matrix.get(curr * adj_matrix.dim(1)? + i)?;
                if Self::get_value_as_f32(&val)? > 0.0 {
                    neighbors.push(i);
                }
            }
            
            if neighbors.is_empty() {
                break;
            }
            
            // Compute transition probabilities
            let mut probs = vec![1.0; neighbors.len()];
            if let Some(prev_node) = prev {
                for (i, &next) in neighbors.iter().enumerate() {
                    let val = adj_matrix.get(prev_node * adj_matrix.dim(1)? + next)?;
                    let edge_exists = Self::get_value_as_f32(&val)? > 0.0;
                    
                    probs[i] = if next == prev_node {
                        1.0 / self.p
                    } else if edge_exists {
                        1.0
                    } else {
                        1.0 / self.q
                    };
                }
            }
            
            // Normalize probabilities
            let sum: f32 = probs.iter().sum();
            for p in &mut probs {
                *p /= sum;
            }
            
            // Sample next node using weighted random selection
            let dist = WeightedIndex::new(&probs).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            let next_idx = rng.sample(dist);
            walk.push(neighbors[next_idx]);
        }
        
        Ok(walk)
    }
}

impl GraphSampler for RandomWalkSampler {
    fn sample_subgraph(&self, graph: &RelationalGraph, seed: Option<u64>) -> CandleResult<RelationalGraph> {
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        
        let num_nodes = graph.node_features.as_ref().expect("Node features not available").dim(0)?;
        let mut sampled_nodes = HashSet::new();
        let mut sampled_edges = Vec::new();
        
        // Perform random walks
        for start_node in 0..num_nodes {
            for _ in 0..self.num_walks {
                let walk = self.node2vec_walk(graph, start_node, &mut rng)?;
                
                // Add nodes and edges from walk
                for window in walk.windows(2) {
                    let src = window[0];
                    let dst = window[1];
                    sampled_nodes.insert(src);
                    sampled_nodes.insert(dst);
                    sampled_edges.push((src, dst));
                }
            }
        }
        
        // Create node mapping
        let node_map: HashMap<usize, usize> = sampled_nodes.iter()
            .enumerate()
            .map(|(i, &node)| (node, i))
            .collect();
        
        // Create new tensors
        let device = graph.device.clone();
        let num_sampled = sampled_nodes.len();
        
        // Create sparse adjacency matrix
        let mut indices = Vec::with_capacity(sampled_edges.len() * 2);
        let mut values = Vec::with_capacity(sampled_edges.len());
        
        for (src, dst) in &sampled_edges {
            let src_idx = *node_map.get(src).unwrap();
            let dst_idx = *node_map.get(dst).unwrap();
            indices.push(src_idx as i64);
            indices.push(dst_idx as i64);
            values.push(1.0f32);
        }
        
        // Create sparse tensor using helper method
        let adjacency = self.create_sparse_tensor(
            indices,
            values,
            (num_sampled, num_sampled),
            &device,
        )?;
        
        // Sample features using helper methods
        let node_features = self.sample_node_features(graph, &sampled_nodes)?;
        let edge_features = self.sample_edge_features(graph, &sampled_edges)?;
        
        Ok(RelationalGraph::new(
            adjacency,
            node_features,
            edge_features,
            graph.global_features.clone(),
            graph.node_types.clone(),
            graph.edge_types.clone(),
            device,
        ))
    }
} 