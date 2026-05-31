use candle_core::{Tensor, Result as CandleResult};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::collections::{HashMap, HashSet};
use chrono::Utc;

use crate::graph::{RelationalGraph, NodeId};
use crate::graph::{GraphResult, GraphError};
use super::{GraphSampler, GraphStats};

pub struct AdaptiveGraphSampler {
    max_nodes: usize,
    min_nodes: usize,
    importance_threshold: f64,
    temporal_decay: f64,
}

impl AdaptiveGraphSampler {
    pub fn new(max_nodes: usize, min_nodes: usize, importance_threshold: f64, temporal_decay: f64) -> Self {
        Self {
            max_nodes,
            min_nodes,
            importance_threshold,
            temporal_decay,
        }
    }
    
    fn calculate_node_importance(
        &self,
        node_id: &NodeId,
        node: &crate::graph::Node,
        graph: &RelationalGraph,
        stats: &GraphStats,
    ) -> f64 {
        // Calculate degree by counting edges
        let degree = graph.edges.iter()
            .filter(|e| e.from == *node_id || e.to == *node_id)
            .count() as f64;
        
        // Degree centrality
        let degree_centrality = degree / (graph.nodes.len() - 1) as f64;
        
        // Temporal importance
        let temporal_score = node.attributes.get("timestamp")
            .and_then(|ts| match ts {
                crate::graph::AttributeValue::Integer(i) => Some(*i),
                _ => None,
            })
            .map_or(0.5, |ts| {
                let current_time = Utc::now().timestamp();
                let age = (current_time - ts) as f64;
                (-age / (24.0 * 3600.0 * self.temporal_decay)).exp()
            });
        
        // Structural importance
        let avg_degree = stats.avg_degree;
        let structural_score = if degree > avg_degree {
            1.0 + (degree - avg_degree) / avg_degree
        } else {
            degree / avg_degree
        };
        
        // Combine scores
        0.4 * degree_centrality + 0.3 * temporal_score + 0.3 * structural_score
    }
    
    fn select_important_nodes(
        &self,
        graph: &RelationalGraph,
        stats: &GraphStats,
    ) -> GraphResult<HashSet<NodeId>> {
        let mut important_nodes = HashSet::new();
        let mut node_scores: Vec<(NodeId, f64)> = Vec::new();
        
        // Calculate importance scores for all nodes
        for (node_id, node) in &graph.nodes {
            let importance = self.calculate_node_importance(node_id, node, graph, stats);
            if importance >= self.importance_threshold {
                node_scores.push((node_id.clone(), importance));
            }
        }
        
        // Sort nodes by importance
        node_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Select nodes within limits
        let num_nodes = node_scores.len().min(self.max_nodes).max(self.min_nodes);
        for (node_id, _) in node_scores.into_iter().take(num_nodes) {
            important_nodes.insert(node_id);
        }
        
        Ok(important_nodes)
    }
}

impl GraphSampler for AdaptiveGraphSampler {
    fn sample_subgraph(&self, graph: &RelationalGraph, seed: Option<u64>) -> CandleResult<RelationalGraph> {
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        
        // Calculate graph statistics
        let stats = GraphStats::new(graph);
        
        // Select important nodes
        let important_nodes = self.select_important_nodes(graph, &stats)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        
        // Create node mapping
        let mut node_map = HashMap::new();
        let mut sampled_edges = Vec::new();
        
        // Add important nodes and their edges
        for node_id in &important_nodes {
            let idx = node_map.len();
            node_map.insert(node_id.clone(), idx);
            
            // Find all edges connected to this node
            for edge in &graph.edges {
                if edge.from == *node_id || edge.to == *node_id {
                    if important_nodes.contains(&edge.from) && important_nodes.contains(&edge.to) {
                        if let (Some(&src_idx), Some(&dst_idx)) = (
                            node_map.get(&edge.from),
                            node_map.get(&edge.to)
                        ) {
                            sampled_edges.push((src_idx, dst_idx));
                        }
                    }
                }
            }
        }
        
        let device = graph.device.clone();
        let num_sampled = node_map.len();
        
        // Create sparse adjacency matrix
        let mut indices = Vec::with_capacity(sampled_edges.len() * 2);
        let mut values = Vec::with_capacity(sampled_edges.len());
        
        for (src, dst) in &sampled_edges {
            indices.push(*src as i64);
            indices.push(*dst as i64);
            values.push(1.0f32);
        }
        
        // Create sparse adjacency matrix
        let mut indices_vec = Vec::new();
        for i in 0..indices.len() / 2 {
            indices_vec.push(vec![indices[i * 2] as usize, indices[i * 2 + 1] as usize]);
        }
        
        let adjacency = crate::graph::SparseTensor::new(
            indices_vec,
            values,
            vec![num_sampled, num_sampled],
        );
        
        // Create empty tensors for features
        let node_features = candle_core::Tensor::zeros((num_sampled, 64), candle_core::DType::F32, &device)?;
        
        Ok(RelationalGraph::new(
            adjacency,
            node_features,
            None,
            None,
            HashMap::new(),
            HashMap::new(),
            device,
        ))
    }
} 