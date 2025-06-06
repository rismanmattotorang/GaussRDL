use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;
use rayon::prelude::*;
use candle_core::{Tensor, Device, Result as CandleResult};
use crate::graph::{RelationalGraph, Node, Edge, RelationType};
use crate::graph::error::{GraphResult, GraphError};
use crate::monitoring::metrics::GraphMetrics;

#[derive(Debug, Clone)]
pub struct BatchConfig {
    pub batch_size: usize,
    pub num_workers: usize,
    pub memory_limit: usize,  // in MB
    pub use_gpu: bool,
    pub optimize_memory: bool,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            batch_size: 1000,
            num_workers: 4,
            memory_limit: 8192,  // 8GB
            use_gpu: true,
            optimize_memory: true,
        }
    }
}

pub struct BatchProcessor {
    config: BatchConfig,
    device: Device,
    metrics: Arc<RwLock<GraphMetrics>>,
}

impl BatchProcessor {
    pub fn new(config: BatchConfig, device: Device) -> Self {
        Self {
            config,
            device,
            metrics: Arc::new(RwLock::new(GraphMetrics::default())),
        }
    }
    
    pub fn process_in_batches<F, T>(
        &self,
        graph: &RelationalGraph,
        operation: F,
    ) -> GraphResult<Vec<T>>
    where
        F: Fn(&[Node], &[Edge], &Device) -> CandleResult<Vec<T>> + Send + Sync,
        T: Send,
    {
        let total_nodes = graph.nodes.len();
        let batch_size = self.config.batch_size;
        let num_batches = (total_nodes + batch_size - 1) / batch_size;
        
        // Pre-allocate results vector
        let mut results = Vec::with_capacity(total_nodes);
        
        // Process batches in parallel
        let node_batches: Vec<_> = graph.nodes.values().collect();
        let node_chunks: Vec<_> = node_batches.chunks(batch_size).collect();
        
        let batch_results: Vec<_> = node_chunks
            .into_par_iter()
            .enumerate()
            .try_fold(
                Vec::new,
                |mut acc, (batch_idx, nodes)| -> GraphResult<Vec<T>> {
                    // Get relevant edges for this batch of nodes
                    let relevant_edges = self.get_relevant_edges(graph, nodes);
                    
                    // Process batch
                    let start = std::time::Instant::now();
                    let batch_result = operation(nodes, &relevant_edges, &self.device)
                        .map_err(|e| GraphError::BatchError(format!("Batch {} failed: {}", batch_idx, e)))?;
                    
                    // Update metrics
                    let elapsed = start.elapsed();
                    self.update_metrics(nodes.len(), &relevant_edges, elapsed);
                    
                    // Store results
                    acc.extend(batch_result);
                    Ok(acc)
                },
            )
            .try_fold(
                || Vec::new(),
                |mut acc, batch_results| -> GraphResult<Vec<T>> {
                    acc.extend(batch_results);
                    Ok(acc)
                },
            )
            .try_reduce(
                || Vec::new(),
                |mut a, mut b| {
                    a.append(&mut b);
                    Ok(a)
                },
            )?;
        
        results.extend(batch_results);
        Ok(results)
    }
    
    pub fn process_edges_in_batches<F, T>(
        &self,
        graph: &RelationalGraph,
        operation: F,
    ) -> GraphResult<Vec<T>>
    where
        F: Fn(&[Edge], &Device) -> CandleResult<Vec<T>> + Send + Sync,
        T: Send,
    {
        let total_edges = graph.edges.len();
        let batch_size = self.config.batch_size;
        let num_batches = (total_edges + batch_size - 1) / batch_size;
        
        // Process edge batches in parallel
        let edge_chunks: Vec<_> = graph.edges.chunks(batch_size).collect();
        
        let results: Vec<_> = edge_chunks
            .into_par_iter()
            .enumerate()
            .try_fold(
                Vec::new,
                |mut acc, (batch_idx, edges)| -> GraphResult<Vec<T>> {
                    let start = std::time::Instant::now();
                    let batch_result = operation(edges, &self.device)
                        .map_err(|e| GraphError::BatchError(format!("Edge batch {} failed: {}", batch_idx, e)))?;
                    
                    let elapsed = start.elapsed();
                    self.update_edge_metrics(edges.len(), elapsed);
                    
                    acc.extend(batch_result);
                    Ok(acc)
                },
            )
            .try_fold(
                || Vec::new(),
                |mut acc, batch_results| -> GraphResult<Vec<T>> {
                    acc.extend(batch_results);
                    Ok(acc)
                },
            )
            .try_reduce(
                || Vec::new(),
                |mut a, mut b| {
                    a.append(&mut b);
                    Ok(a)
                },
            )?;
        
        Ok(results)
    }
    
    pub fn batch_subgraph_extraction(
        &self,
        graph: &RelationalGraph,
        node_indices: &[usize],
    ) -> GraphResult<RelationalGraph> {
        let batch_size = self.config.batch_size;
        let mut subgraph = RelationalGraph::new(self.device.clone());
        
        // Process nodes in batches
        for node_batch in node_indices.chunks(batch_size) {
            let start = std::time::Instant::now();
            
            // Extract nodes and their immediate edges
            for &idx in node_batch {
                if let Some(node) = graph.nodes.values().nth(idx) {
                    // Add node to subgraph
                    subgraph.nodes.insert(node.id.clone(), node.clone());
                    
                    // Add connected edges
                    for &edge_idx in &node.edges {
                        if let Some(edge) = graph.edges.get(edge_idx) {
                            subgraph.edges.push(edge.clone());
                            subgraph.relation_types.insert(edge.relation.clone());
                        }
                    }
                }
            }
            
            let elapsed = start.elapsed();
            self.update_metrics(node_batch.len(), &subgraph.edges, elapsed);
        }
        
        // Update subgraph metrics
        let mut metrics = subgraph.metrics.write();
        metrics.num_nodes = subgraph.nodes.len();
        metrics.num_edges = subgraph.edges.len();
        metrics.avg_degree = (2.0 * subgraph.edges.len() as f64) / subgraph.nodes.len() as f64;
        metrics.density = self.calculate_density(&subgraph);
        
        Ok(subgraph)
    }
    
    fn get_relevant_edges<'a>(
        &self,
        graph: &'a RelationalGraph,
        nodes: &[&'a Node],
    ) -> Vec<&'a Edge> {
        let mut relevant_edges = Vec::new();
        let node_ids: std::collections::HashSet<_> = nodes.iter().map(|n| &n.id).collect();
        
        for node in nodes {
            for &edge_idx in &node.edges {
                if let Some(edge) = graph.edges.get(edge_idx) {
                    if node_ids.contains(&edge.from) || node_ids.contains(&edge.to) {
                        relevant_edges.push(edge);
                    }
                }
            }
        }
        
        relevant_edges
    }
    
    fn update_metrics(&self, num_nodes: usize, edges: &[impl AsRef<Edge>], elapsed: std::time::Duration) {
        let mut metrics = self.metrics.write();
        metrics.num_nodes += num_nodes;
        metrics.num_edges += edges.len();
        metrics.processing_time_ms += elapsed.as_secs_f64() * 1000.0;
        
        // Update memory metrics if configured
        if self.config.optimize_memory {
            metrics.memory_used = self.estimate_memory_usage(num_nodes, edges.len());
        }
    }
    
    fn update_edge_metrics(&self, num_edges: usize, elapsed: std::time::Duration) {
        let mut metrics = self.metrics.write();
        metrics.num_edges += num_edges;
        metrics.processing_time_ms += elapsed.as_secs_f64() * 1000.0;
    }
    
    fn calculate_density(&self, graph: &RelationalGraph) -> f64 {
        let n = graph.nodes.len();
        if n <= 1 {
            return 0.0;
        }
        let max_edges = n * (n - 1) / 2;
        graph.edges.len() as f64 / max_edges as f64
    }
    
    fn estimate_memory_usage(&self, num_nodes: usize, num_edges: usize) -> f64 {
        // Rough estimation in MB
        let node_size = 64; // bytes per node
        let edge_size = 48; // bytes per edge
        let total_bytes = (num_nodes * node_size + num_edges * edge_size) as f64;
        total_bytes / (1024.0 * 1024.0) // Convert to MB
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_batch_processing() {
        let config = BatchConfig::default();
        let device = Device::Cpu;
        let processor = BatchProcessor::new(config, device);
        
        // Create a test graph
        let mut graph = RelationalGraph::new(Device::Cpu);
        
        // Add test nodes and edges
        for i in 0..1000 {
            let id = i.to_string();
            let node = Node {
                id: id.clone(),
                attributes: HashMap::new(),
                edges: Vec::new(),
            };
            graph.nodes.insert(id, node);
        }
        
        // Process nodes in batches
        let result = processor.process_in_batches(&graph, |nodes, edges, device| {
            // Simple test operation: count nodes
            Ok(vec![nodes.len()])
        });
        
        assert!(result.is_ok());
        let counts = result.unwrap();
        assert_eq!(counts.iter().sum::<usize>(), 1000);
    }
    
    #[test]
    fn test_subgraph_extraction() {
        let config = BatchConfig::default();
        let device = Device::Cpu;
        let processor = BatchProcessor::new(config, device);
        
        // Create a test graph
        let mut graph = RelationalGraph::new(Device::Cpu);
        
        // Add test nodes and edges
        for i in 0..100 {
            let id = i.to_string();
            let node = Node {
                id: id.clone(),
                attributes: HashMap::new(),
                edges: Vec::new(),
            };
            graph.nodes.insert(id, node);
        }
        
        // Extract subgraph
        let indices: Vec<_> = (0..10).collect();
        let result = processor.batch_subgraph_extraction(&graph, &indices);
        
        assert!(result.is_ok());
        let subgraph = result.unwrap();
        assert_eq!(subgraph.nodes.len(), 10);
    }
} 