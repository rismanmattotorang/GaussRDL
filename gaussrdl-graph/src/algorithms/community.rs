//! Community detection algorithms
//! 
//! This module provides implementations of community detection algorithms
//! including Louvain method and Label Propagation.

use candle_core::{Device, Tensor, Result as CandleResult};
use std::collections::{HashMap, HashSet};
use crate::graph::{RelationalGraph, NodeId, Node, Edge, RelationType};
use rayon::prelude::*;

/// Louvain community detection algorithm
pub struct LouvainCommunityDetection {
    resolution: f64,
    max_iterations: usize,
}

impl LouvainCommunityDetection {
    pub fn new(resolution: f64, max_iterations: usize) -> Self {
        Self {
            resolution,
            max_iterations,
        }
    }
    
    pub fn detect(&self, graph: &RelationalGraph) -> CandleResult<HashMap<NodeId, usize>> {
        // Initialize each node in its own community
        let mut communities: HashMap<NodeId, usize> = HashMap::new();
        for (idx, node_id) in graph.nodes.keys().enumerate() {
            communities.insert(node_id.clone(), idx);
        }
        
        let mut modularity = 0.0;
        let mut iteration = 0;
        
        while iteration < self.max_iterations {
            let old_modularity = modularity;
            modularity = self.compute_modularity(graph, &communities)?;
            
            // Phase 1: Local optimization
            self.optimize_communities(graph, &mut communities)?;
            
            // Phase 2: Community aggregation
            let new_graph = self.aggregate_communities(graph, &communities)?;
            
            if (modularity - old_modularity).abs() < 1e-6 {
                break;
            }
            
            iteration += 1;
        }
        
        Ok(communities)
    }
    
    fn compute_modularity(&self, graph: &RelationalGraph, communities: &HashMap<NodeId, usize>) -> CandleResult<f64> {
        let total_edges = graph.num_edges() as f64;
        let mut modularity = 0.0;
        
        for edge in &graph.edges {
            let from_community = communities.get(&edge.from).unwrap_or(&0);
            let to_community = communities.get(&edge.to).unwrap_or(&0);
            
            if from_community == to_community {
                let ki = graph.edges.iter().filter(|e| e.from == edge.from || e.to == edge.from).count() as f64;
                let kj = graph.edges.iter().filter(|e| e.from == edge.to || e.to == edge.to).count() as f64;
                
                modularity += 1.0 - self.resolution * ki * kj / (2.0 * total_edges);
            }
        }
        
        modularity /= 2.0 * total_edges;
        Ok(modularity)
    }
    
    fn optimize_communities(&self, graph: &RelationalGraph, communities: &mut HashMap<NodeId, usize>) -> CandleResult<()> {
        // Simplified optimization - in practice, this would be more complex
        let node_ids: Vec<NodeId> = graph.nodes.keys().cloned().collect();
        
        for node_id in node_ids {
            let current_community = communities[&node_id];
            let mut best_community = current_community;
            let mut best_gain = 0.0;
            
            // Try moving to neighboring communities
            for edge in &graph.edges {
                if edge.from == node_id || edge.to == node_id {
                    let neighbor = if edge.from == node_id {
                        &edge.to
                    } else {
                        &edge.from
                    };
                    
                    let neighbor_community = communities[neighbor];
                    if neighbor_community != current_community {
                        // Compute modularity gain (simplified)
                        let gain = self.compute_modularity_gain(graph, &node_id, neighbor_community, communities)?;
                        if gain > best_gain {
                            best_gain = gain;
                            best_community = neighbor_community;
                        }
                    }
                }
            }
            
            if best_community != current_community {
                communities.insert(node_id, best_community);
            }
        }
        
        Ok(())
    }
    
    fn compute_modularity_gain(&self, graph: &RelationalGraph, node_id: &NodeId, new_community: usize, communities: &HashMap<NodeId, usize>) -> CandleResult<f64> {
        // Simplified modularity gain computation
        let total_edges = graph.num_edges() as f64;
        let node_degree = graph.edges.iter()
            .filter(|e| e.from == *node_id || e.to == *node_id)
            .count() as f64;
        
        let mut edges_to_community = 0.0;
        let mut community_degree = 0.0;
        
        for (other_id, &community) in communities {
            if community == new_community {
                let other_degree = graph.edges.iter()
                    .filter(|e| e.from == *other_id || e.to == *other_id)
                    .count() as f64;
                community_degree += other_degree;
                
                // Check if there's an edge between node_id and other_id
                for edge in &graph.edges {
                    if (edge.from == *node_id && edge.to == *other_id) ||
                       (edge.to == *node_id && edge.from == *other_id) {
                        edges_to_community += 1.0;
                    }
                }
            }
        }
        
        let gain = edges_to_community - self.resolution * node_degree * community_degree / (2.0 * total_edges);
        Ok(gain)
    }
    
    fn aggregate_communities(&self, graph: &RelationalGraph, communities: &HashMap<NodeId, usize>) -> CandleResult<RelationalGraph> {
        // Create a new graph where nodes represent communities
        let device = graph.device.clone();
        let mut new_graph = RelationalGraph::new_empty(device);
        
        // Group nodes by community
        let mut community_nodes: HashMap<usize, Vec<NodeId>> = HashMap::new();
        for (node_id, &community) in communities {
            community_nodes.entry(community).or_default().push(node_id.clone());
        }
        
        // Create new nodes for each community
        for (community_id, nodes) in &community_nodes {
            // Aggregate node features (simplified - just take first node's features)
            if let Some(first_node) = nodes.first() {
                if let Some(node) = graph.get_node(first_node) {
                    // Create aggregated features
                    let aggregated_features = graph.node_features.clone(); // Simplified
                    let community_id_node = NodeId::from_string(&format!("community_{}", community_id));
                    new_graph.nodes.insert(community_id_node.clone(), Node {
                        id: community_id_node,
                        attributes: HashMap::new(),
                        features: aggregated_features,
                    });
                }
            }
        }
        
        // Create edges between communities
        let mut community_edges: HashMap<(usize, usize), f64> = HashMap::new();
        
        for edge in &graph.edges {
            let from_community = communities.get(&edge.from).unwrap_or(&0);
            let to_community = communities.get(&edge.to).unwrap_or(&0);
            
            if from_community != to_community {
                let edge_key = if from_community < to_community {
                    (*from_community, *to_community)
                } else {
                    (*to_community, *from_community)
                };
                
                *community_edges.entry(edge_key).or_insert(0.0) += 1.0;
            }
        }
        
        // Add edges to new graph
        for ((from_comm, to_comm), weight) in community_edges {
            let from_id = NodeId::from_string(&format!("community_{}", from_comm));
            let to_id = NodeId::from_string(&format!("community_{}", to_comm));
            new_graph.edges.push(Edge {
                from: from_id,
                to: to_id,
                relation_type: RelationType {
                    name: "inter_community".to_string(),
                    attributes: HashMap::new(),
                },
                weight: weight as f32,
                attributes: HashMap::new(),
            });
        }
        
        Ok(new_graph)
    }
}

/// Label Propagation community detection
pub struct LabelPropagation {
    max_iterations: usize,
}

impl LabelPropagation {
    pub fn new(max_iterations: usize) -> Self {
        Self { max_iterations }
    }
    
    pub fn detect(&self, graph: &RelationalGraph) -> CandleResult<HashMap<NodeId, usize>> {
        let mut labels: HashMap<NodeId, usize> = HashMap::new();
        
        // Initialize each node with a unique label
        for (idx, node_id) in graph.nodes.keys().enumerate() {
            labels.insert(node_id.clone(), idx);
        }
        
        let mut iteration = 0;
        let mut changed = true;
        
        while changed && iteration < self.max_iterations {
            changed = false;
            
            // Random order for node updates
            let mut node_ids: Vec<NodeId> = graph.nodes.keys().cloned().collect();
            use rand::seq::SliceRandom;
            use rand::thread_rng;
            node_ids.shuffle(&mut thread_rng());
            
            for node_id in node_ids {
                let old_label = labels[&node_id];
                let new_label = self.get_majority_label(graph, &node_id, &labels)?;
                
                if new_label != old_label {
                    labels.insert(node_id, new_label);
                    changed = true;
                }
            }
            
            iteration += 1;
        }
        
        tracing::info!("Label Propagation converged after {} iterations", iteration);
        Ok(labels)
    }
    
    fn get_majority_label(&self, graph: &RelationalGraph, node_id: &NodeId, labels: &HashMap<NodeId, usize>) -> CandleResult<usize> {
        let mut label_counts: HashMap<usize, usize> = HashMap::new();
        
        // Find all neighbors of the node
        for edge in &graph.edges {
            let neighbor = if edge.from == *node_id {
                &edge.to
            } else if edge.to == *node_id {
                &edge.from
            } else {
                continue;
            };
            
            let neighbor_label = labels.get(neighbor).unwrap_or(&0);
            *label_counts.entry(*neighbor_label).or_insert(0) += 1;
        }
        
        // Find the most frequent label
        let majority_label = label_counts
            .iter()
            .max_by_key(|(_, &count)| count)
            .map(|(&label, _)| label)
            .unwrap_or(0);
        
        Ok(majority_label)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{RelationalGraph, SparseTensor};
    use candle_core::Device;
    
    #[test]
    fn test_louvain_community_detection() {
        let device = Device::Cpu;
        let graph = RelationalGraph::new_empty(device);
        let louvain = LouvainCommunityDetection::new(1.0, 10);
        let result = louvain.detect(&graph);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_label_propagation() {
        let device = Device::Cpu;
        let graph = RelationalGraph::new_empty(device);
        let label_prop = LabelPropagation::new(10);
        let result = label_prop.detect(&graph);
        assert!(result.is_ok());
    }
} 