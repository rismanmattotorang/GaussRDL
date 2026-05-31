//! Advanced graph algorithms for GaussRDL
//! 
//! This module provides high-performance implementations of classic and modern
//! graph algorithms optimized for relational graphs.

pub mod centrality;
pub mod community;
pub mod paths;
pub mod metrics;
pub mod clustering;

pub use centrality::*;
pub use community::*;
pub use paths::*;
pub use metrics::*;
pub use clustering::*;

use crate::graph::{RelationalGraph, EntityNode, RelationalEdge, NodeId};
use candle_core::{Device, Tensor, Result as CandleResult};
use std::collections::{HashMap, HashSet, VecDeque};
use rayon::prelude::*;

/// Algorithm configuration for performance tuning
#[derive(Debug, Clone)]
pub struct AlgorithmConfig {
    /// Number of threads for parallel algorithms
    pub num_threads: usize,
    /// Memory limit in bytes
    pub memory_limit: Option<usize>,
    /// Convergence tolerance for iterative algorithms
    pub tolerance: f64,
    /// Maximum iterations for iterative algorithms
    pub max_iterations: usize,
    /// Whether to use approximate algorithms for large graphs
    pub use_approximation: bool,
}

impl Default for AlgorithmConfig {
    fn default() -> Self {
        Self {
            num_threads: rayon::current_num_threads(),
            memory_limit: None,
            tolerance: 1e-6,
            max_iterations: 1000,
            use_approximation: true,
        }
    }
}

/// Algorithm result with metadata
#[derive(Debug, Clone)]
pub struct AlgorithmResult<T> {
    /// The computed result
    pub result: T,
    /// Number of iterations performed
    pub iterations: usize,
    /// Final convergence value
    pub convergence: f64,
    /// Execution time in milliseconds
    pub execution_time_ms: u64,
    /// Memory usage in bytes
    pub memory_usage_bytes: usize,
}

impl<T> AlgorithmResult<T> {
    pub fn new(result: T, iterations: usize, convergence: f64, execution_time_ms: u64, memory_usage_bytes: usize) -> Self {
        Self {
            result,
            iterations,
            convergence,
            execution_time_ms,
            memory_usage_bytes,
        }
    }
}

/// Trait for graph algorithms
pub trait GraphAlgorithm {
    type Input;
    type Output;
    type Config;
    
    /// Execute the algorithm
    fn execute(&self, input: Self::Input, config: Self::Config) -> CandleResult<AlgorithmResult<Self::Output>>;
    
    /// Get algorithm name
    fn name(&self) -> &'static str;
    
    /// Get algorithm complexity
    fn complexity(&self) -> &'static str;
}

/// Parallel algorithm executor
pub struct ParallelExecutor {
    config: AlgorithmConfig,
}

impl ParallelExecutor {
    pub fn new(config: AlgorithmConfig) -> Self {
        Self { config }
    }
    
    pub fn execute_parallel<T, F, R>(&self, items: Vec<T>, f: F) -> Vec<R>
    where
        T: Send + Sync,
        R: Send,
        F: Fn(T) -> R + Send + Sync,
    {
        items.into_par_iter().map(f).collect()
    }
    
    pub fn execute_parallel_with_reduce<T, F, R, G>(&self, items: Vec<T>, map: F, reduce: G) -> R
    where
        T: Send + Sync,
        R: Send + Sync,
        F: Fn(T) -> R + Send + Sync,
        G: Fn(R, R) -> R + Send + Sync,
    {
        items.into_par_iter().map(map).reduce(|| {
            // This is a placeholder - in practice, you'd need a proper identity element
            panic!("Reduce operation requires identity element")
        }, reduce)
    }
}

/// Memory-aware algorithm executor
pub struct MemoryAwareExecutor {
    config: AlgorithmConfig,
    memory_tracker: MemoryTracker,
}

impl MemoryAwareExecutor {
    pub fn new(config: AlgorithmConfig) -> Self {
        Self {
            config,
            memory_tracker: MemoryTracker::new(),
        }
    }
    
    pub fn check_memory_limit(&self, required_bytes: usize) -> bool {
        if let Some(limit) = self.config.memory_limit {
            required_bytes <= limit
        } else {
            true
        }
    }
    
    pub fn track_memory(&mut self, bytes: usize) {
        self.memory_tracker.track(bytes);
    }
}

/// Memory usage tracker
#[derive(Debug)]
struct MemoryTracker {
    peak_usage: usize,
    current_usage: usize,
}

impl MemoryTracker {
    fn new() -> Self {
        Self {
            peak_usage: 0,
            current_usage: 0,
        }
    }
    
    fn track(&mut self, bytes: usize) {
        self.current_usage = bytes;
        self.peak_usage = self.peak_usage.max(bytes);
    }
    
    fn peak_usage(&self) -> usize {
        self.peak_usage
    }
    
    fn current_usage(&self) -> usize {
        self.current_usage
    }
}

/// Utility functions for graph algorithms
pub mod utils {
    use super::*;
    
    /// Convert node ID to index
    pub fn node_id_to_index(node_id: &NodeId, node_mapping: &HashMap<NodeId, usize>) -> Option<usize> {
        node_mapping.get(node_id).copied()
    }
    
    /// Convert index to node ID
    pub fn index_to_node_id(index: usize, reverse_mapping: &HashMap<usize, NodeId>) -> Option<NodeId> {
        reverse_mapping.get(&index).cloned()
    }
    
    /// Create node mapping for efficient lookups
    pub fn create_node_mapping(graph: &RelationalGraph) -> (HashMap<NodeId, usize>, HashMap<usize, NodeId>) {
        let mut forward = HashMap::new();
        let mut reverse = HashMap::new();
        
        for (idx, node_id) in graph.nodes.keys().enumerate() {
            forward.insert(node_id.clone(), idx);
            reverse.insert(idx, node_id.clone());
        }
        
        (forward, reverse)
    }
    
    /// Compute graph density
    pub fn compute_density(graph: &RelationalGraph) -> f64 {
        let num_nodes = graph.num_nodes();
        let num_edges = graph.num_edges();
        
        if num_nodes <= 1 {
            0.0
        } else {
            let max_edges = num_nodes * (num_nodes - 1) / 2;
            num_edges as f64 / max_edges as f64
        }
    }
    
    /// Compute average degree
    pub fn compute_avg_degree(graph: &RelationalGraph) -> f64 {
        let num_nodes = graph.num_nodes();
        let num_edges = graph.num_edges();
        
        if num_nodes == 0 {
            0.0
        } else {
            (2 * num_edges) as f64 / num_nodes as f64
        }
    }
    
    /// Check if graph is connected using BFS
    pub fn is_connected(graph: &RelationalGraph) -> bool {
        if graph.num_nodes() <= 1 {
            return true;
        }
        
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        
        // Start from first node
        if let Some(first_node) = graph.nodes.keys().next() {
            queue.push_back(first_node.clone());
            visited.insert(first_node.clone());
            
            while let Some(node_id) = queue.pop_front() {
                // Find all neighbors of the node
                for edge in &graph.edges {
                    let neighbor = if edge.from == node_id {
                        &edge.to
                    } else if edge.to == node_id {
                        &edge.from
                    } else {
                        continue;
                    };
                    
                    if !visited.contains(neighbor) {
                        visited.insert(neighbor.clone());
                        queue.push_back(neighbor.clone());
                    }
                }
            }
        }
        
        visited.len() == graph.num_nodes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{RelationalGraph, SparseTensor};
    use candle_core::Device;
    
    #[test]
    fn test_algorithm_config_default() {
        let config = AlgorithmConfig::default();
        assert!(config.num_threads > 0);
        assert!(config.tolerance > 0.0);
        assert!(config.max_iterations > 0);
    }
    
    #[test]
    fn test_memory_tracker() {
        let mut tracker = MemoryTracker::new();
        assert_eq!(tracker.peak_usage(), 0);
        assert_eq!(tracker.current_usage(), 0);
        
        tracker.track(100);
        assert_eq!(tracker.peak_usage(), 100);
        assert_eq!(tracker.current_usage(), 100);
        
        tracker.track(50);
        assert_eq!(tracker.peak_usage(), 100);
        assert_eq!(tracker.current_usage(), 50);
        
        tracker.track(200);
        assert_eq!(tracker.peak_usage(), 200);
        assert_eq!(tracker.current_usage(), 200);
    }
    
    #[test]
    fn test_utils_density() {
        let device = Device::Cpu;
        let empty_graph = RelationalGraph::new_empty(device);
        assert_eq!(utils::compute_density(&empty_graph), 0.0);
    }
} 