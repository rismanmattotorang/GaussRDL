//! Centrality measures for graph analysis
//! 
//! This module provides implementations of various centrality measures
//! including PageRank, Betweenness Centrality, Closeness Centrality,
//! and Eigenvector Centrality.

use crate::graph::{RelationalGraph, NodeId};
use candle_core::{Device, Tensor, Result as CandleResult};
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::Instant;
use rayon::prelude::*;

/// PageRank algorithm implementation
pub struct PageRank {
    damping_factor: f64,
    max_iterations: usize,
    tolerance: f64,
}

impl PageRank {
    pub fn new(damping_factor: f64, max_iterations: usize, tolerance: f64) -> Self {
        Self {
            damping_factor,
            max_iterations,
            tolerance,
        }
    }
    
    pub fn compute(&self, graph: &RelationalGraph) -> CandleResult<HashMap<NodeId, f64>> {
        let start_time = Instant::now();
        let num_nodes = graph.nodes.len();
        
        if num_nodes == 0 {
            return Ok(HashMap::new());
        }
        
        // Initialize PageRank values
        let initial_rank = 1.0 / num_nodes as f64;
        let mut ranks = HashMap::new();
        let mut new_ranks = HashMap::new();
        
        for node_id in graph.nodes.keys() {
            ranks.insert(node_id.clone(), initial_rank);
            new_ranks.insert(node_id.clone(), 0.0);
        }
        
        // Iterative PageRank computation
        for iteration in 0..self.max_iterations {
            let mut max_change: f64 = 0.0;
            
            // Reset new ranks
            for rank in new_ranks.values_mut() {
                *rank = 0.0;
            }
            
            // Distribute PageRank
            for node_id in graph.nodes.keys() {
                let current_rank = ranks[node_id];
                let out_degree = graph.edges.iter()
                    .filter(|e| e.from == *node_id)
                    .count();
                
                if out_degree > 0 {
                    let distributed_rank = current_rank / out_degree as f64;
                    for edge in &graph.edges {
                        if edge.from == *node_id {
                            *new_ranks.get_mut(&edge.to).unwrap() += distributed_rank;
                        }
                    }
                } else {
                    // Handle dangling nodes
                    let distributed_rank = current_rank / num_nodes as f64;
                    for rank in new_ranks.values_mut() {
                        *rank += distributed_rank;
                    }
                }
            }
            
            // Apply damping factor and check convergence
            for (node_id, new_rank) in &mut new_ranks {
                let old_rank = ranks[node_id];
                *new_rank = (1.0 - self.damping_factor) / num_nodes as f64 + 
                           self.damping_factor * *new_rank;
                
                let change = (*new_rank - old_rank).abs();
                max_change = max_change.max(change);
            }
            
            // Swap ranks
            std::mem::swap(&mut ranks, &mut new_ranks);
            
            // Check convergence
            if max_change < self.tolerance {
                tracing::debug!("PageRank converged after {} iterations", iteration + 1);
                break;
            }
        }
        
        let execution_time = start_time.elapsed().as_millis() as u64;
        tracing::info!("PageRank completed in {}ms", execution_time);
        
        Ok(ranks)
    }
}

/// Betweenness Centrality implementation
pub struct BetweennessCentrality;

impl BetweennessCentrality {
    pub fn compute(graph: &RelationalGraph) -> CandleResult<HashMap<NodeId, f64>> {
        let start_time = Instant::now();
        let num_nodes = graph.nodes.len();
        
        if num_nodes == 0 {
            return Ok(HashMap::new());
        }
        
        let mut betweenness = HashMap::new();
        for node_id in graph.nodes.keys() {
            betweenness.insert(node_id.clone(), 0.0);
        }
        
        // Compute betweenness for each node as source (simplified to avoid parallel processing issues)
        for source in graph.nodes.keys() {
            let mut local_betweenness: HashMap<NodeId, f64> = HashMap::new();
            
            // BFS to find shortest paths
            let mut distances = HashMap::new();
            let mut num_shortest_paths = HashMap::new();
            let mut predecessors = HashMap::new();
            let mut queue = VecDeque::new();
            
            // Initialize
            for node_id in graph.nodes.keys() {
                distances.insert(node_id.clone(), usize::MAX);
                num_shortest_paths.insert(node_id.clone(), 0);
                predecessors.insert(node_id.clone(), Vec::new());
            }
            
            distances.insert(source.clone(), 0);
            num_shortest_paths.insert(source.clone(), 1);
            queue.push_back(source.clone());
            
            // BFS traversal
            while let Some(current) = queue.pop_front() {
                let current_dist = distances[&current];
                
                for edge in &graph.edges {
                    if edge.from == current {
                        let neighbor = &edge.to;
                        let neighbor_dist = distances[neighbor];
                        
                        if neighbor_dist == usize::MAX {
                            distances.insert(neighbor.clone(), current_dist + 1);
                            num_shortest_paths.insert(neighbor.clone(), num_shortest_paths[&current]);
                            queue.push_back(neighbor.clone());
                        } else if neighbor_dist == current_dist + 1 {
                            let current_paths = num_shortest_paths[&current];
                            let neighbor_paths = num_shortest_paths[neighbor];
                            num_shortest_paths.insert(neighbor.clone(), neighbor_paths + current_paths);
                        }
                        
                        if neighbor_dist == current_dist + 1 {
                            predecessors.get_mut(neighbor).unwrap().push(current.clone());
                        }
                    }
                }
            }
            
            // Backward accumulation
            let mut dependency = HashMap::new();
            for node_id in graph.nodes.keys() {
                dependency.insert(node_id.clone(), 0.0);
            }
            
            // Process nodes in reverse distance order
            let mut nodes_by_distance: Vec<_> = distances.iter().collect();
            nodes_by_distance.sort_by_key(|(_, &dist)| dist);
            nodes_by_distance.reverse();
            
            for (node_id, _) in nodes_by_distance {
                if node_id != source {
                    for pred in &predecessors[node_id] {
                        let pred_paths = num_shortest_paths[pred];
                        let node_paths = num_shortest_paths[node_id];
                        let ratio = pred_paths as f64 / node_paths as f64;
                        let new_dependency = (1.0 + dependency[node_id]) * ratio;
                        *dependency.get_mut(pred).unwrap() += new_dependency;
                        *local_betweenness.get_mut(pred).unwrap() += new_dependency;
                    }
                }
            }
            
            // Accumulate to global betweenness
            for (node_id, score) in local_betweenness {
                if let Some(betweenness_score) = betweenness.get_mut(&node_id) {
                    *betweenness_score += score;
                }
            }
        }
        
        // Normalize by number of node pairs
        let num_pairs = (num_nodes - 1) * (num_nodes - 2);
        if num_pairs > 0 {
            for score in betweenness.values_mut() {
                *score /= num_pairs as f64;
            }
        }
        
        let execution_time = start_time.elapsed().as_millis() as u64;
        tracing::info!("Betweenness centrality completed in {}ms", execution_time);
        
        Ok(betweenness)
    }
}

/// Closeness Centrality implementation
pub struct ClosenessCentrality;

impl ClosenessCentrality {
    pub fn compute(graph: &RelationalGraph) -> CandleResult<HashMap<NodeId, f64>> {
        let start_time = Instant::now();
        let mut closeness = HashMap::new();
        
        for source in graph.nodes.keys() {
            let mut distances = HashMap::new();
            let mut queue = VecDeque::new();
            
            distances.insert(source.clone(), 0);
            queue.push_back(source.clone());
            
            while let Some(current) = queue.pop_front() {
                for edge in &graph.edges {
                    if edge.from == current {
                        let neighbor = &edge.to;
                        if !distances.contains_key(neighbor) {
                            distances.insert(neighbor.clone(), distances[&current] + 1);
                            queue.push_back(neighbor.clone());
                        }
                    }
                }
            }
            
            let total_distance: usize = distances.values().sum();
            let closeness_value = if total_distance > 0 {
                (graph.nodes.len() - 1) as f64 / total_distance as f64
            } else {
                0.0
            };
            
            closeness.insert(source.clone(), closeness_value);
        }
        
        let execution_time = start_time.elapsed().as_millis() as u64;
        tracing::info!("Closeness centrality completed in {}ms", execution_time);
        
        Ok(closeness)
    }
}

/// Eigenvector Centrality implementation
pub struct EigenvectorCentrality {
    max_iterations: usize,
    tolerance: f64,
}

impl EigenvectorCentrality {
    pub fn new(max_iterations: usize, tolerance: f64) -> Self {
        Self {
            max_iterations,
            tolerance,
        }
    }
    
    pub fn compute(&self, graph: &RelationalGraph) -> CandleResult<HashMap<NodeId, f64>> {
        let start_time = Instant::now();
        let num_nodes = graph.nodes.len();
        
        if num_nodes == 0 {
            return Ok(HashMap::new());
        }
        
        let mut centrality = HashMap::new();
        let mut new_centrality = HashMap::new();
        
        // Initialize centrality scores
        for node_id in graph.nodes.keys() {
            centrality.insert(node_id.clone(), 1.0);
            new_centrality.insert(node_id.clone(), 1.0);
        }
        
        for iteration in 0..self.max_iterations {
            let mut max_change: f64 = 0.0;
            
            // Calculate new centrality scores
            for node_id in graph.nodes.keys() {
                let mut new_score = 0.0;
                
                for edge in &graph.edges {
                    if edge.to == *node_id {
                        new_score += centrality[&edge.from];
                    }
                }
                
                let old_score = centrality[node_id];
                let change = if new_score > old_score { new_score - old_score } else { old_score - new_score };
                max_change = max_change.max(change);
                
                new_centrality.insert(node_id.clone(), new_score);
            }
            
            // Normalize
            let sum: f64 = new_centrality.values().sum();
            if sum > 0.0 {
                for score in new_centrality.values_mut() {
                    *score /= sum;
                }
            }
            
            // Update centrality
            std::mem::swap(&mut centrality, &mut new_centrality);
            
            if max_change < self.tolerance {
                tracing::debug!("Eigenvector centrality converged after {} iterations", iteration + 1);
                break;
            }
        }
        
        let execution_time = start_time.elapsed().as_millis() as u64;
        tracing::info!("Eigenvector centrality completed in {}ms", execution_time);
        
        Ok(centrality)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{RelationalGraph, SparseTensor};
    use candle_core::Device;
    
    #[test]
    fn test_pagerank() {
        let device = Device::Cpu;
        let graph = RelationalGraph::new_empty(device);
        let pagerank = PageRank::new(0.85, 100, 1e-6);
        let result = pagerank.compute(&graph);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_betweenness_centrality() {
        let device = Device::Cpu;
        let graph = RelationalGraph::new_empty(device);
        let result = BetweennessCentrality::compute(&graph);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_closeness_centrality() {
        let device = Device::Cpu;
        let graph = RelationalGraph::new_empty(device);
        let result = ClosenessCentrality::compute(&graph);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_eigenvector_centrality() {
        let device = Device::Cpu;
        let graph = RelationalGraph::new_empty(device);
        let eigenvector = EigenvectorCentrality::new(100, 1e-6);
        let result = eigenvector.compute(&graph);
        assert!(result.is_ok());
    }
} 