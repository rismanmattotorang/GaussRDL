//! Graph visualization and export
//! 
//! This module provides graph visualization capabilities including DOT export and plotting.

use crate::graph::RelationalGraph;
use crate::graph::NodeId;
use candle_core::{Tensor, Result, Device};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;

/// Graph visualization utilities
pub struct GraphVisualizer;

impl GraphVisualizer {
    /// Export graph to DOT format for visualization
    pub fn export_to_dot(graph: &RelationalGraph, path: &str) -> Result<()> {
        let mut file = File::create(path)?;
        
        writeln!(file, "digraph G {{")?;
        writeln!(file, "  rankdir=LR;")?;
        writeln!(file, "  node [shape=circle];")?;
        
        // Add nodes
        for (node_id, _) in &graph.nodes {
            writeln!(file, "  \"{}\" [label=\"{}\"];", node_id.to_string(), node_id.to_string())?;
        }
        
        // Add edges
        for edge in &graph.edges {
            writeln!(file, "  \"{}\" -> \"{}\" [label=\"{}\"];", 
                edge.from.to_string(), 
                edge.to.to_string(), 
                edge.relation_type.name)?;
        }
        
        writeln!(file, "}}")?;
        Ok(())
    }
    
    /// Export graph statistics to text format
    pub fn export_statistics(graph: &RelationalGraph, path: &str) -> Result<()> {
        let mut file = File::create(path)?;
        
        writeln!(file, "Graph Statistics")?;
        writeln!(file, "================")?;
        writeln!(file, "Nodes: {}", graph.num_nodes())?;
        writeln!(file, "Edges: {}", graph.num_edges())?;
        writeln!(file, "Relation Types: {}", graph.num_relation_types())?;
        
        // Node type distribution
        writeln!(file, "\nNode Type Distribution:")?;
        let mut type_counts: HashMap<usize, usize> = HashMap::new();
        for (_, node_type) in &graph.node_types {
            *type_counts.entry(*node_type).or_insert(0) += 1;
        }
        for (node_type, count) in type_counts {
            writeln!(file, "  Type {}: {} nodes", node_type, count)?;
        }
        
        // Edge type distribution
        writeln!(file, "\nEdge Type Distribution:")?;
        let mut edge_type_counts: HashMap<String, usize> = HashMap::new();
        for edge in &graph.edges {
            *edge_type_counts.entry(edge.relation_type.name.clone()).or_insert(0) += 1;
        }
        for (edge_type, count) in edge_type_counts {
            writeln!(file, "  {}: {} edges", edge_type, count)?;
        }
        
        Ok(())
    }
    
    /// Generate adjacency matrix visualization
    pub fn export_adjacency_matrix(graph: &RelationalGraph, path: &str) -> Result<()> {
        let mut file = File::create(path)?;
        
        let num_nodes = graph.num_nodes();
        let mut adjacency = vec![vec![0.0; num_nodes]; num_nodes];
        
        // Create node mapping
        let node_mapping: HashMap<NodeId, usize> = graph.nodes.keys()
            .enumerate()
            .map(|(idx, node_id)| (node_id.clone(), idx))
            .collect();
        
        // Fill adjacency matrix
        for edge in &graph.edges {
            if let (Some(&from_idx), Some(&to_idx)) = (
                node_mapping.get(&edge.from),
                node_mapping.get(&edge.to)
            ) {
                adjacency[from_idx][to_idx] = edge.weight;
                adjacency[to_idx][from_idx] = edge.weight; // Undirected
            }
        }
        
        // Write matrix
        writeln!(file, "Adjacency Matrix")?;
        writeln!(file, "================")?;
        for row in adjacency {
            for val in row {
                write!(file, "{:.2} ", val)?;
            }
            writeln!(file)?;
        }
        
        Ok(())
    }
    
    /// Export node features to CSV
    pub fn export_node_features(graph: &RelationalGraph, path: &str) -> Result<()> {
        let mut file = File::create(path)?;
        
        if let Some(node_features) = graph.get_node_features() {
            let num_nodes = node_features.dim(0)?;
            let num_features = node_features.dim(1)?;
            
            // Write header
            write!(file, "node_id")?;
            for i in 0..num_features {
                write!(file, ",feature_{}", i)?;
            }
            writeln!(file)?;
            
            // Write data
            let node_ids: Vec<NodeId> = graph.nodes.keys().cloned().collect();
            for (idx, node_id) in node_ids.iter().enumerate() {
                if idx < num_nodes {
                    write!(file, "{}", node_id.to_string())?;
                    let node_feature = node_features.get(idx)?;
                    for j in 0..num_features {
                        let value = node_feature.get(j)?.to_scalar::<f32>()?;
                        write!(file, ",{:.6}", value)?;
                    }
                    writeln!(file)?;
                }
            }
        }
        
        Ok(())
    }
    
    /// Export edge list to CSV
    pub fn export_edge_list(graph: &RelationalGraph, path: &str) -> Result<()> {
        let mut file = File::create(path)?;
        
        writeln!(file, "source,target,relation,weight")?;
        
        for edge in &graph.edges {
            writeln!(file, "{},{},{},{:.6}", 
                edge.from.to_string(),
                edge.to.to_string(),
                edge.relation_type.name,
                edge.weight)?;
        }
        
        Ok(())
    }
    
    /// Generate simple text-based graph visualization
    pub fn print_graph_summary(graph: &RelationalGraph) -> Result<()> {
        println!("Graph Summary");
        println!("=============");
        println!("Nodes: {}", graph.num_nodes());
        println!("Edges: {}", graph.num_edges());
        println!("Relation Types: {}", graph.num_relation_types());
        
        println!("\nSample Nodes:");
        let mut count = 0;
        for (node_id, _) in &graph.nodes {
            if count < 5 {
                println!("  {}", node_id.to_string());
                count += 1;
            } else {
                break;
            }
        }
        
        println!("\nSample Edges:");
        let mut count = 0;
        for edge in &graph.edges {
            if count < 5 {
                println!("  {} -> {} [{}] (weight: {:.3})", 
                    edge.from.to_string(),
                    edge.to.to_string(),
                    edge.relation_type.name,
                    edge.weight);
                count += 1;
            } else {
                break;
            }
        }
        
        Ok(())
    }
}

/// Graph layout algorithms (simplified)
pub struct GraphLayout;

impl GraphLayout {
    /// Simple force-directed layout (simplified)
    pub fn force_directed_layout(graph: &RelationalGraph, iterations: usize) -> Result<HashMap<NodeId, (f64, f64)>> {
        let mut positions = HashMap::new();
        let node_ids: Vec<NodeId> = graph.nodes.keys().cloned().collect();
        
        // Initialize random positions
        for node_id in &node_ids {
            positions.insert(node_id.clone(), (
                rand::random::<f64>() * 100.0,
                rand::random::<f64>() * 100.0
            ));
        }
        
        // Simple force-directed iteration
        for _ in 0..iterations {
            let mut new_positions = positions.clone();
            
            for node_id in &node_ids {
                let (mut fx, mut fy) = (0.0, 0.0);
                
                // Repulsive forces from all other nodes
                for other_id in &node_ids {
                    if node_id != other_id {
                        let (x1, y1) = positions[node_id];
                        let (x2, y2) = positions[other_id];
                        let dx = x1 - x2;
                        let dy = y1 - y2;
                        let distance = (dx * dx + dy * dy).sqrt().max(0.1);
                        let force = 100.0 / (distance * distance);
                        fx += force * dx / distance;
                        fy += force * dy / distance;
                    }
                }
                
                // Attractive forces from connected nodes
                for edge in &graph.edges {
                    if edge.from == *node_id {
                        if let Some((x2, y2)) = positions.get(&edge.to) {
                            let (x1, y1) = positions[node_id];
                            let dx = x1 - x2;
                            let dy = y1 - y2;
                            let distance = (dx * dx + dy * dy).sqrt().max(0.1);
                            let force = distance * 0.01;
                            fx -= force * dx / distance;
                            fy -= force * dy / distance;
                        }
                    } else if edge.to == *node_id {
                        if let Some((x2, y2)) = positions.get(&edge.from) {
                            let (x1, y1) = positions[node_id];
                            let dx = x1 - x2;
                            let dy = y1 - y2;
                            let distance = (dx * dx + dy * dy).sqrt().max(0.1);
                            let force = distance * 0.01;
                            fx -= force * dx / distance;
                            fy -= force * dy / distance;
                        }
                    }
                }
                
                // Update position
                let (x, y) = positions[node_id];
                new_positions.insert(node_id.clone(), (x + fx * 0.1, y + fy * 0.1));
            }
            
            positions = new_positions;
        }
        
        Ok(positions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    
    #[test]
    fn test_export_to_dot() {
        let device = Device::Cpu;
        let graph = RelationalGraph::new_empty(device);
        let result = GraphVisualizer::export_to_dot(&graph, "test_graph.dot");
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_export_statistics() {
        let device = Device::Cpu;
        let graph = RelationalGraph::new_empty(device);
        let result = GraphVisualizer::export_statistics(&graph, "test_stats.txt");
        assert!(result.is_ok());
    }
} 