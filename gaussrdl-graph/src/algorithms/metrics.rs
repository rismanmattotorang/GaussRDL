//! Graph metrics and statistics
//! 
//! This module provides implementations of various graph metrics
//! including density, clustering coefficient, diameter, and more.

use crate::graph::{RelationalGraph, NodeId};
use candle_core::{Tensor, Result, Device};
use std::collections::{HashMap, HashSet};
use rayon::prelude::*;

/// Graph metrics computation algorithms
pub struct GraphMetrics;

impl GraphMetrics {
    /// Compute degree centrality for all nodes
    pub fn degree_centrality(graph: &RelationalGraph) -> Result<HashMap<NodeId, f64>> {
        let mut centrality = HashMap::new();
        let total_nodes = graph.num_nodes() as f64;
        
        for (node_id, _) in &graph.nodes {
            let degree = graph.edges.iter()
                .filter(|edge| edge.from == *node_id || edge.to == *node_id)
                .count() as f64;
            
            centrality.insert(node_id.clone(), degree / (total_nodes - 1.0));
        }
        
        Ok(centrality)
    }
    
    /// Compute betweenness centrality using Brandes algorithm
    pub fn betweenness_centrality(graph: &RelationalGraph) -> Result<HashMap<NodeId, f64>> {
        let mut centrality = HashMap::new();
        let node_ids: Vec<NodeId> = graph.nodes.keys().cloned().collect();
        
        for node_id in &node_ids {
            centrality.insert(node_id.clone(), 0.0);
        }
        
        for source in &node_ids {
            let (distances, predecessors, sigma) = Self::brandes_bfs(graph, source)?;
            
            // Backward phase
            let mut delta = HashMap::new();
            for node_id in &node_ids {
                delta.insert(node_id.clone(), 0.0);
            }
            
            // Process nodes in non-increasing distance order
            let mut nodes_by_distance: Vec<_> = distances.iter().collect();
            nodes_by_distance.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
            
            for (node_id, _) in nodes_by_distance {
                if let Some(preds) = predecessors.get(node_id) {
                    for pred in preds {
                        let ratio = sigma.get(node_id).unwrap_or(&1.0) / sigma.get(pred).unwrap_or(&1.0);
                        let current_delta = delta.get(node_id).unwrap_or(&0.0);
                        delta.insert(pred.clone(), delta.get(pred).unwrap_or(&0.0) + ratio * current_delta);
                    }
                }
                
                if *node_id != *source {
                    let current_centrality = centrality.get(node_id).unwrap_or(&0.0);
                    centrality.insert(node_id.clone(), current_centrality + delta.get(node_id).unwrap_or(&0.0));
                }
            }
        }
        
        // Normalize by (n-1)(n-2) for undirected graphs
        let n = node_ids.len() as f64;
        let normalization = (n - 1.0) * (n - 2.0);
        
        for value in centrality.values_mut() {
            *value /= normalization;
        }
        
        Ok(centrality)
    }
    
    /// Compute closeness centrality
    pub fn closeness_centrality(graph: &RelationalGraph) -> Result<HashMap<NodeId, f64>> {
        let mut centrality = HashMap::new();
        let node_ids: Vec<NodeId> = graph.nodes.keys().cloned().collect();
        
        for node_id in &node_ids {
            let (distances, _, _) = Self::brandes_bfs(graph, node_id)?;
            
            let total_distance: f64 = distances.values().sum();
            let reachable_nodes = distances.len() as f64;
            
            if reachable_nodes > 1.0 {
                centrality.insert(node_id.clone(), (reachable_nodes - 1.0) / total_distance);
            } else {
                centrality.insert(node_id.clone(), 0.0);
            }
        }
        
        Ok(centrality)
    }
    
    /// Compute clustering coefficient
    pub fn clustering_coefficient(graph: &RelationalGraph) -> Result<HashMap<NodeId, f64>> {
        let mut coefficients = HashMap::new();
        
        for (node_id, _) in &graph.nodes {
            let neighbors: Vec<NodeId> = graph.edges.iter()
                .filter_map(|edge| {
                    if edge.from == *node_id {
                        Some(edge.to.clone())
                    } else if edge.to == *node_id {
                        Some(edge.from.clone())
                    } else {
                        None
                    }
                })
                .collect();
            
            let k = neighbors.len();
            if k < 2 {
                coefficients.insert(node_id.clone(), 0.0);
                continue;
            }
            
            let mut triangles = 0;
            for i in 0..k {
                for j in (i + 1)..k {
                    let has_edge = graph.edges.iter().any(|edge| {
                        (edge.from == neighbors[i] && edge.to == neighbors[j]) ||
                        (edge.from == neighbors[j] && edge.to == neighbors[i])
                    });
                    if has_edge {
                        triangles += 1;
                    }
                }
            }
            
            let max_triangles = k * (k - 1) / 2;
            coefficients.insert(node_id.clone(), triangles as f64 / max_triangles as f64);
        }
        
        Ok(coefficients)
    }
    
    /// Compute connected components
    pub fn connected_components(graph: &RelationalGraph) -> Result<Vec<Vec<NodeId>>> {
        let mut components = Vec::new();
        let mut visited = HashSet::new();
        
        for node_id in graph.nodes.keys() {
            if !visited.contains(node_id) {
                let mut component = Vec::new();
                Self::dfs_component(graph, node_id, &mut visited, &mut component)?;
                components.push(component);
            }
        }
        
        Ok(components)
    }
    
    /// Helper function for Brandes algorithm
    fn brandes_bfs(graph: &RelationalGraph, source: &NodeId) -> Result<(HashMap<NodeId, f64>, HashMap<NodeId, Vec<NodeId>>, HashMap<NodeId, f64>)> {
        let mut distances = HashMap::new();
        let mut predecessors = HashMap::new();
        let mut sigma = HashMap::new();
        let mut queue = std::collections::VecDeque::new();
        
        distances.insert(source.clone(), 0.0);
        sigma.insert(source.clone(), 1.0);
        queue.push_back(source.clone());
        
        while let Some(current) = queue.pop_front() {
            let current_distance = distances[&current];
            
            // Find neighbors
            for edge in &graph.edges {
                let neighbor = if edge.from == current {
                    Some(edge.to.clone())
                } else if edge.to == current {
                    Some(edge.from.clone())
                } else {
                    None
                };
                
                if let Some(neighbor) = neighbor {
                    if !distances.contains_key(&neighbor) {
                        distances.insert(neighbor.clone(), current_distance + 1.0);
                        queue.push_back(neighbor.clone());
                    }
                    
                    if distances[&neighbor] == current_distance + 1.0 {
                        sigma.insert(neighbor.clone(), sigma.get(&neighbor).unwrap_or(&0.0) + sigma[&current]);
                        predecessors.entry(neighbor.clone()).or_insert_with(Vec::new).push(current.clone());
                    }
                }
            }
        }
        
        Ok((distances, predecessors, sigma))
    }
    
    /// Helper function for DFS component discovery
    fn dfs_component(graph: &RelationalGraph, node_id: &NodeId, visited: &mut HashSet<NodeId>, component: &mut Vec<NodeId>) -> Result<()> {
        visited.insert(node_id.clone());
        component.push(node_id.clone());
        
        if let Some(_node) = graph.get_node(node_id) {
            for edge in &graph.edges {
                let neighbor = if edge.from == *node_id {
                    Some(edge.to.clone())
                } else if edge.to == *node_id {
                    Some(edge.from.clone())
                } else {
                    None
                };
                
                if let Some(neighbor) = neighbor {
                    if !visited.contains(&neighbor) {
                        Self::dfs_component(graph, &neighbor, visited, component)?;
                    }
                }
            }
        }
        
        Ok(())
    }
}

/// Comprehensive graph statistics
#[derive(Debug, Clone)]
pub struct GraphStatistics {
    pub num_nodes: usize,
    pub num_edges: usize,
    pub density: f64,
    pub average_degree: f64,
    pub degree_distribution: HashMap<usize, usize>,
    pub average_clustering_coefficient: f64,
    pub diameter: f64,
    pub average_path_length: f64,
    pub radius: f64,
    pub connected_components: usize,
}

impl GraphStatistics {
    /// Print statistics in a formatted way
    pub fn print_summary(&self) {
        println!("Graph Statistics:");
        println!("  Nodes: {}", self.num_nodes);
        println!("  Edges: {}", self.num_edges);
        println!("  Density: {:.4}", self.density);
        println!("  Average Degree: {:.2}", self.average_degree);
        println!("  Average Clustering Coefficient: {:.4}", self.average_clustering_coefficient);
        println!("  Diameter: {:.2}", self.diameter);
        println!("  Average Path Length: {:.2}", self.average_path_length);
        println!("  Radius: {:.2}", self.radius);
        println!("  Connected Components: {}", self.connected_components);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{RelationalGraph, SparseTensor};
    use candle_core::Device;
    
    #[test]
    fn test_graph_metrics() {
        let device = Device::Cpu;
        let graph = RelationalGraph::new_empty(device);
        
        assert_eq!(GraphMetrics::density(&graph), 0.0);
        assert_eq!(GraphMetrics::average_degree(&graph), 0.0);
        assert_eq!(GraphMetrics::average_clustering_coefficient(&graph), 0.0);
    }
    
    #[test]
    fn test_graph_statistics() {
        let device = Device::Cpu;
        let graph = RelationalGraph::new_empty(device);
        let stats = GraphMetrics::compute_all_metrics(&graph);
        assert!(stats.is_ok());
    }
} 