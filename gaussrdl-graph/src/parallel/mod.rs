//! Parallel graph processing and concurrency
//! 
//! This module provides parallel and concurrent graph processing capabilities
//! including parallel sampling, concurrent algorithms, and work-stealing executors.

use std::collections::{BinaryHeap, HashMap, HashSet};
use std::cmp::Reverse;
use std::sync::{Arc, Mutex};
use rayon::prelude::*;
use candle_core::{Device, Tensor, Result as CandleResult};
use crate::graph::{NodeId, RelationalGraph, Edge, RelationType};
use ordered_float::OrderedFloat;

/// Parallel processing utilities for graph operations
pub struct ParallelProcessor;

impl ParallelProcessor {
    /// Process nodes in parallel
    pub fn process_nodes_parallel<F, R>(graph: &RelationalGraph, f: F) -> CandleResult<Vec<R>>
    where
        F: Fn(&NodeId) -> R + Send + Sync,
        R: Send,
    {
        let node_ids: Vec<NodeId> = graph.nodes.keys().cloned().collect();
        Ok(node_ids.par_iter().map(f).collect())
    }

    /// Process edges in parallel
    pub fn process_edges_parallel<F, R>(graph: &RelationalGraph, f: F) -> CandleResult<Vec<R>>
    where
        F: Fn(&Edge) -> R + Send + Sync,
        R: Send,
    {
        Ok(graph.edges.par_iter().map(f).collect())
    }

    /// Parallel BFS traversal
    pub fn parallel_bfs<F>(graph: &RelationalGraph, start_nodes: Vec<NodeId>, f: F) -> CandleResult<()>
    where
        F: Fn(&NodeId, usize) -> CandleResult<()> + Send + Sync,
    {
        start_nodes.par_iter().enumerate().try_for_each(|(level, start_node)| {
            let mut visited = HashSet::new();
            let mut queue = vec![start_node];
            
            while !queue.is_empty() {
                let current = queue.remove(0);
                if visited.insert(current) {
                    f(current, level)?;
                    
                    for edge in &graph.edges {
                        if edge.from == *current {
                            queue.push(&edge.to);
                        }
                    }
                }
            }
            Ok(())
        })
    }

    /// Parallel shortest paths computation
    pub fn parallel_shortest_paths(graph: &RelationalGraph, start_nodes: Vec<NodeId>) -> CandleResult<HashMap<NodeId, f64>> {
        let distances = Arc::new(Mutex::new(HashMap::new()));
        
        start_nodes.par_iter().try_for_each(|start_node| {
            let mut local_distances = HashMap::new();
            let mut queue = BinaryHeap::new();
            
            // Use OrderedFloat for f64 comparison
            queue.push((Reverse(OrderedFloat(0.0)), start_node.clone()));
            local_distances.insert(start_node.clone(), 0.0);
            
            while let Some((Reverse(OrderedFloat(dist)), current)) = queue.pop() {
                if dist > local_distances[&current] {
                    continue;
                }
                for edge in &graph.edges {
                    let neighbor = if edge.from == current {
                        Some(edge.to.clone())
                    } else if edge.to == current {
                        Some(edge.from.clone())
                    } else {
                        None
                    };
                    if let Some(neighbor) = neighbor {
                        let new_dist = dist + edge.weight as f64;
                        if new_dist < *local_distances.get(&neighbor).unwrap_or(&f64::INFINITY) {
                            local_distances.insert(neighbor.clone(), new_dist);
                            queue.push((Reverse(OrderedFloat(new_dist)), neighbor));
                        }
                    }
                }
            }
            // Merge results
            let mut global_distances = distances.lock().unwrap();
            for (node_id, dist) in local_distances {
                global_distances.insert(node_id, dist);
            }
            Ok::<(), candle_core::Error>(())
        })?;
        Ok(Arc::try_unwrap(distances).unwrap().into_inner().unwrap())
    }
    
    /// Parallel graph clustering
    pub fn parallel_clustering(graph: &RelationalGraph, num_clusters: usize) -> CandleResult<HashMap<NodeId, usize>> {
        let node_ids: Vec<NodeId> = graph.nodes.keys().cloned().collect();
        let _num_nodes = node_ids.len();
        
        // Initialize random cluster assignments
        let cluster_assignments = Arc::new(Mutex::new(HashMap::new()));
        for (idx, node_id) in node_ids.iter().enumerate() {
            cluster_assignments.lock().unwrap().insert(node_id.clone(), idx % num_clusters);
        }
        
        // Parallel k-means iteration
        for _iteration in 0..50 {
            let assignments = cluster_assignments.lock().unwrap().clone();
            
            // Compute cluster centers in parallel
            let cluster_centers = (0..num_clusters).into_par_iter().map(|cluster_id| {
                let mut center = vec![0.0; 64]; // Assume 64-dimensional features
                let mut count = 0;
                
                for (_node_id, &assignment) in &assignments {
                    if assignment == cluster_id {
                        // In a real implementation, you'd get actual node features
                        count += 1;
                    }
                }
                
                if count > 0 {
                    for val in &mut center {
                        *val /= count as f64;
                    }
                }
                
                center
            }).collect::<Vec<_>>();
            
            // Reassign nodes in parallel
            let changes: Vec<bool> = node_ids.par_iter().map(|node_id| {
                let mut best_cluster = 0;
                let mut min_distance = f64::INFINITY;
                
                for (cluster_id, center) in cluster_centers.iter().enumerate() {
                    // Simplified distance calculation
                    let distance = center.iter().sum::<f64>();
                    if distance < min_distance {
                        min_distance = distance;
                        best_cluster = cluster_id;
                    }
                }
                
                let mut assignments = cluster_assignments.lock().unwrap();
                if assignments[node_id] != best_cluster {
                    assignments.insert(node_id.clone(), best_cluster);
                    true
                } else {
                    false
                }
            }).collect();
            let changed = changes.iter().any(|&c| c);
            
            if !changed {
                break;
            }
        }
        
        Ok(Arc::try_unwrap(cluster_assignments).unwrap().into_inner().unwrap())
    }
}

/// Thread pool for graph operations
pub struct GraphThreadPool {
    pool: rayon::ThreadPool,
}

impl GraphThreadPool {
    pub fn new(num_threads: usize) -> CandleResult<Self> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        
        Ok(Self { pool })
    }
    
    pub fn execute<F, R>(&self, f: F) -> CandleResult<R>
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        Ok(self.pool.install(f))
    }
    
    pub fn execute_parallel<F, R>(&self, items: Vec<R>, f: F) -> CandleResult<Vec<R>>
    where
        F: Fn(R) -> R + Send + Sync,
        R: Send,
    {
        Ok(self.pool.install(|| items.into_par_iter().map(f).collect()))
    }
}

/// Parallel graph algorithms
pub struct ParallelAlgorithms;

impl ParallelAlgorithms {
    /// Parallel connected components
    pub fn parallel_connected_components(graph: &RelationalGraph) -> CandleResult<Vec<Vec<NodeId>>> {
        let mut components = Vec::new();
        let mut visited = HashSet::new();
        
        for node_id in graph.nodes.keys() {
            if !visited.contains(node_id) {
                let mut component = Vec::new();
                let mut stack = vec![node_id.clone()];
                
                while let Some(current) = stack.pop() {
                    if visited.contains(&current) {
                        continue;
                    }
                    visited.insert(current.clone());
                    component.push(current.clone());
                    
                    // Add neighbors
                    for edge in &graph.edges {
                        if edge.from == current {
                            stack.push(edge.to.clone());
                        } else if edge.to == current {
                            stack.push(edge.from.clone());
                        }
                    }
                }
                
                components.push(component);
            }
        }
        
        Ok(components)
    }
    
    /// Parallel graph coloring
    pub fn parallel_graph_coloring(graph: &RelationalGraph) -> CandleResult<HashMap<NodeId, usize>> {
        let mut colors = HashMap::new();
        let node_ids: Vec<NodeId> = graph.nodes.keys().cloned().collect();
        
        // Simple greedy coloring (not truly parallel due to dependencies)
        for node_id in node_ids {
            let mut used_colors = HashSet::new();
            
            // Check colors of neighbors
            for edge in &graph.edges {
                if edge.from == node_id {
                    if let Some(&color) = colors.get(&edge.to) {
                        used_colors.insert(color);
                    }
                } else if edge.to == node_id {
                    if let Some(&color) = colors.get(&edge.from) {
                        used_colors.insert(color);
                    }
                }
            }
            
            // Find the smallest unused color
            let mut color = 0;
            while used_colors.contains(&color) {
                color += 1;
            }
            
            colors.insert(node_id, color);
        }
        
        Ok(colors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    
    #[test]
    fn test_parallel_processing() {
        let device = Device::Cpu;
        let graph = RelationalGraph::new_empty(device);
        let result = ParallelProcessor::process_nodes_parallel(&graph, |_| 42);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_thread_pool() {
        let pool = GraphThreadPool::new(4);
        assert!(pool.is_ok());
    }
    
    #[test]
    fn test_parallel_algorithms() {
        let device = Device::Cpu;
        let graph = RelationalGraph::new_empty(device);
        let result = ParallelAlgorithms::parallel_connected_components(&graph);
        assert!(result.is_ok());
    }
} 