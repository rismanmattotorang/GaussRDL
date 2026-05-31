use candle_core::{Tensor, Result as CandleResult};
use rand::{Rng, SeedableRng};
use rand::seq::SliceRandom;
use rand::rngs::StdRng;
use std::collections::HashSet;

use crate::graph::RelationalGraph;
use super::{GraphSampler, GraphStats};

pub struct LayerWiseSampler {
    num_neighbors: Vec<usize>,
    replace: bool,
}

impl LayerWiseSampler {
    pub fn new(num_neighbors: Vec<usize>, replace: bool) -> Self {
        Self { num_neighbors, replace }
    }
    
    fn sample_neighbors(
        &self,
        neighbors: &[usize],
        k: usize,
        rng: &mut StdRng,
    ) -> Vec<usize> {
        if neighbors.is_empty() {
            return Vec::new();
        }
        
        let num_samples = k.min(neighbors.len());
        
        if self.replace {
            (0..k).map(|_| {
                neighbors[rng.gen_range(0..neighbors.len())]
            }).collect()
        } else {
            let mut indices: Vec<usize> = (0..neighbors.len()).collect();
            indices.shuffle(rng);
            indices.into_iter()
                .take(num_samples)
                .map(|i| neighbors[i])
                .collect()
        }
    }
}

impl GraphSampler for LayerWiseSampler {
    fn sample_subgraph(&self, graph: &RelationalGraph, seed: Option<u64>) -> CandleResult<RelationalGraph> {
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        
        let device = graph.device.clone();
        let adj_matrix = graph.adjacency.to_dense()?;
        let num_nodes = graph.node_features.as_ref().expect("Node features not available").dim(0)?;
        
        // Sample initial nodes
        let mut frontier: HashSet<usize> = (0..num_nodes).collect();
        let mut all_sampled = HashSet::new();
        
        // Sample neighbors for each layer
        for &k in self.num_neighbors.iter().rev() {
            let mut new_frontier = HashSet::new();
            
            for &node in &frontier {
                // Get neighbors
                let neighbors: Vec<usize> = (0..num_nodes)
                    .filter(|&i| {
                        adj_matrix.get(node * num_nodes + i)
                            .and_then(|v| v.to_scalar::<f32>())
                            .map_or(false, |v| v > 0.0)
                    })
                    .collect();
                
                // Sample k neighbors
                let sampled = self.sample_neighbors(&neighbors, k, &mut rng);
                new_frontier.extend(&sampled);
                all_sampled.extend(&sampled);
            }
            
            frontier = new_frontier;
        }
        
        // Create node mapping
        let node_map: Vec<usize> = all_sampled.iter().cloned().collect();
        let num_sampled = node_map.len();
        
        // Create edge list
        let mut sampled_edges = Vec::new();
        for (i, &src) in node_map.iter().enumerate() {
            for (j, &dst) in node_map.iter().enumerate() {
                if adj_matrix.get(src * num_nodes + dst)
                    .and_then(|v| v.to_scalar::<f32>())
                    .map_or(false, |v| v > 0.0)
                {
                    sampled_edges.push((i, j));
                }
            }
        }
        
        // Create sparse adjacency matrix
        let mut indices = Vec::with_capacity(sampled_edges.len() * 2);
        let mut values = Vec::with_capacity(sampled_edges.len());
        
        for (src, dst) in &sampled_edges {
            indices.push(*src as i64);
            indices.push(*dst as i64);
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
        let node_features = self.sample_node_features(graph, &all_sampled)?;
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