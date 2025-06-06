// src/graph/sampler.rs
use candle_core::{Device, Tensor, Result as CandleResult, DType};
use rand::{Rng, SeedableRng, seq::SliceRandom};
use rand::rngs::StdRng;
use std::collections::{HashSet, VecDeque, HashMap};
use crate::{Result, graph::*};

use super::{RelationalGraph, SparseTensor};

pub trait GraphSampler: Send + Sync {
    fn sample_subgraph(&self, graph: &RelationalGraph, seed: Option<u64>) -> CandleResult<RelationalGraph>;
}

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
            let neighbors: Vec<usize> = (0..adj_matrix.dim(0)?)
                .filter(|&i| adj_matrix.get((curr, i))?.to_scalar::<f32>()? > 0.0)
                .collect();
            
            if neighbors.is_empty() {
                break;
            }
            
            // Compute transition probabilities
            let mut probs = vec![1.0; neighbors.len()];
            if let Some(prev_node) = prev {
                for (i, &next) in neighbors.iter().enumerate() {
                    probs[i] = if next == prev_node {
                        1.0 / self.p
                    } else if adj_matrix.get((prev_node, next))?.to_scalar::<f32>()? > 0.0 {
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
            
            // Sample next node
            let next_idx = {
                let mut cumsum = 0.0;
                let rand_val = rng.gen::<f32>();
                let mut selected = 0;
                for (i, &p) in probs.iter().enumerate() {
                    cumsum += p;
                    if rand_val <= cumsum {
                        selected = i;
                        break;
                    }
                }
                selected
            };
            
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
        
        let num_nodes = graph.node_features.dim(0)?;
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
        
        for (src, dst) in sampled_edges {
            let src_idx = *node_map.get(&src).unwrap();
            let dst_idx = *node_map.get(&dst).unwrap();
            indices.push(src_idx as i64);
            indices.push(dst_idx as i64);
            values.push(1.0f32);
        }
        
        let indices_tensor = Tensor::from_slice2(&indices, (2, sampled_edges.len()), &device)?;
        let values_tensor = Tensor::from_slice(&values, DType::F32, &device)?;
        
        let adjacency = SparseTensor::new(
            indices_tensor,
            values_tensor,
            (num_sampled, num_sampled),
        );
        
        // Sample node features
        let mut sampled_features = Vec::new();
        for &node in &sampled_nodes {
            let features = graph.node_features.get(node)?;
            sampled_features.push(features);
        }
        
        let node_features = Tensor::stack(&sampled_features, 0)?;
        
        // Sample edge features if present
        let edge_features = if let Some(ef) = &graph.edge_features {
            let mut sampled_ef = Vec::new();
            for (src, dst) in &sampled_edges {
                let ef_idx = graph.adjacency.indices.get((0, *src))?.get(*dst)?;
                sampled_ef.push(ef.get(ef_idx.to_scalar::<usize>()?)?);
            }
            Some(Tensor::stack(&sampled_ef, 0)?)
        } else {
            None
        };
        
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

pub struct LayerWiseSampler {
    num_neighbors: Vec<usize>,
    replace: bool,
}

impl LayerWiseSampler {
    pub fn new(num_neighbors: Vec<usize>, replace: bool) -> Self {
        Self { num_neighbors, replace }
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
        let num_nodes = graph.node_features.dim(0)?;
        
        // Sample initial nodes
        let mut frontier: HashSet<usize> = (0..num_nodes).collect();
        let mut all_sampled = HashSet::new();
        
        // Sample neighbors for each layer
        for &k in self.num_neighbors.iter().rev() {
            let mut new_frontier = HashSet::new();
            
            for &node in &frontier {
                // Get neighbors
                let neighbors: Vec<usize> = (0..num_nodes)
                    .filter(|&i| adj_matrix.get((node, i))?.to_scalar::<f32>()? > 0.0)
                    .collect();
                
                if neighbors.is_empty() {
                    continue;
                }
                
                // Sample k neighbors
                let num_samples = k.min(neighbors.len());
                let sampled = if self.replace {
                    (0..k).map(|_| {
                        neighbors[rng.gen_range(0..neighbors.len())]
                    }).collect::<Vec<_>>()
                } else {
                    let mut indices: Vec<usize> = (0..neighbors.len()).collect();
                    indices.shuffle(&mut rng);
                    indices.into_iter()
                        .take(num_samples)
                        .map(|i| neighbors[i])
                        .collect()
                };
                
                new_frontier.extend(&sampled);
                all_sampled.extend(&sampled);
            }
            
            frontier = new_frontier;
        }
        
        // Create subgraph similar to RandomWalkSampler
        let node_map: HashMap<usize, usize> = all_sampled.iter()
            .enumerate()
            .map(|(i, &node)| (node, i))
            .collect();
        
        // Create adjacency matrix for sampled nodes
        let mut sampled_edges = Vec::new();
        for &src in &all_sampled {
            for &dst in &all_sampled {
                if adj_matrix.get((src, dst))?.to_scalar::<f32>()? > 0.0 {
                    sampled_edges.push((src, dst));
                }
            }
        }
        
        // Create tensors similar to RandomWalkSampler
        let mut indices = Vec::with_capacity(sampled_edges.len() * 2);
        let mut values = Vec::with_capacity(sampled_edges.len());
        
        for (src, dst) in sampled_edges {
            let src_idx = *node_map.get(&src).unwrap();
            let dst_idx = *node_map.get(&dst).unwrap();
            indices.push(src_idx as i64);
            indices.push(dst_idx as i64);
            values.push(1.0f32);
        }
        
        let num_sampled = all_sampled.len();
        let indices_tensor = Tensor::from_slice2(&indices, (2, sampled_edges.len()), &device)?;
        let values_tensor = Tensor::from_slice(&values, DType::F32, &device)?;
        
        let adjacency = SparseTensor::new(
            indices_tensor,
            values_tensor,
            (num_sampled, num_sampled),
        );
        
        // Sample node features
        let mut sampled_features = Vec::new();
        for &node in &all_sampled {
            let features = graph.node_features.get(node)?;
            sampled_features.push(features);
        }
        
        let node_features = Tensor::stack(&sampled_features, 0)?;
        
        Ok(RelationalGraph::new(
            adjacency,
            node_features,
            None, // Skip edge features for simplicity
            graph.global_features.clone(),
            graph.node_types.clone(),
            graph.edge_types.clone(),
            device,
        ))
    }
}