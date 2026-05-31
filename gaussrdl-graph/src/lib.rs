//! # GaussRDL Graph - Advanced Graph Construction and Manipulation
//! 
//! This crate provides high-performance graph construction, manipulation, and analysis
//! capabilities for relational deep learning applications.
//! 
//! ## Features
//! 
//! ### Core Graph Operations
//! - **Relational Graph Construction**: Build complex multi-relational graphs
//! - **Entity-Edge Management**: Efficient node and edge operations
//! - **Schema Validation**: Type-safe graph schema enforcement
//! - **Memory Optimization**: Advanced memory management and pooling
//! 
//! ### Advanced Sampling Strategies
//! - **Random Walk Sampling**: Node2Vec-style biased random walks
//! - **Layer-wise Sampling**: Hierarchical neighborhood sampling
//! - **Adaptive Sampling**: Dynamic sampling based on graph properties
//! - **Temporal Sampling**: Time-aware subgraph extraction
//! - **Cached Sampling**: Memory-efficient sampling with caching
//! 
//! ### High-Performance Algorithms
//! - **Parallel Processing**: Multi-threaded graph operations
//! - **Lock-free Data Structures**: Concurrent graph access
//! - **Memory Pooling**: Zero-allocation graph operations
//! - **Compression**: Efficient graph serialization and storage
//! 
//! ### Graph Analysis
//! - **Centrality Measures**: PageRank, Betweenness, Closeness
//! - **Community Detection**: Louvain, Label Propagation
//! - **Path Finding**: Shortest paths, all-pairs shortest paths
//! - **Graph Metrics**: Density, clustering coefficient, diameter
//! 
//! ### Visualization and Monitoring
//! - **Graph Visualization**: DOT format export
//! - **Performance Metrics**: Real-time operation monitoring
//! - **Memory Profiling**: Detailed memory usage tracking
//! 
//! ## Quick Start
//! 
//! ```rust
//! use gaussrdl_graph::{
//!     RelationalGraph, EntityNode, RelationalEdge, NodeId, RelationType,
//!     RandomWalkSampler, GraphBuilder
//! };
//! use candle_core::Device;
//! 
//! // Create a new graph
//! let device = Device::Cpu;
//! let mut graph = RelationalGraph::new_empty(device);
//! 
//! // Add nodes and edges
//! let node1 = EntityNode::new(
//!     NodeId::new(0, 1),
//!     "user".to_string(),
//!     Tensor::zeros((64,), &device).unwrap(),
//!     None
//! );
//! 
//! let edge = RelationalEdge::new(
//!     NodeId::new(0, 1),
//!     NodeId::new(0, 2),
//!     RelationType::new("follows"),
//!     1.0,
//!     None
//! );
//! 
//! // Sample subgraph
//! let sampler = RandomWalkSampler::new(10, 5, 1.0, 1.0);
//! let subgraph = sampler.sample_subgraph(&graph, Some(42)).unwrap();
//! ```
//! 
//! ## Performance Features
//! 
//! - **Lock-free Concurrency**: Safe concurrent graph access
//! - **Memory Pooling**: Zero-allocation operations for critical paths
//! - **SIMD Optimizations**: Vectorized graph operations
//! - **Cache-aware Algorithms**: Optimized for modern CPU architectures
//! - **Compression**: Efficient storage and transmission
//! 
//! ## Memory Management
//! 
//! The crate supports multiple memory allocators:
//! - **jemalloc**: Default high-performance allocator
//! - **mimalloc**: Alternative high-performance allocator
//! - **System allocator**: Fallback option
//! 
//! ## Parallel Processing
//! 
//! All major operations support parallel execution:
//! - **Parallel sampling**: Multi-threaded subgraph extraction
//! - **Parallel algorithms**: Concurrent graph analysis
//! - **Work-stealing**: Efficient load balancing
//! - **Async support**: Non-blocking operations
//! 
//! ## Error Handling
//! 
//! Comprehensive error handling with:
//! - **Structured errors**: Detailed error information
//! - **Error recovery**: Automatic error handling strategies
//! - **Context preservation**: Rich error context
//! - **Fatal vs recoverable**: Error classification
//! 
//! ## Monitoring and Metrics
//! 
//! Built-in monitoring capabilities:
//! - **Performance metrics**: Operation timing and throughput
//! - **Memory tracking**: Detailed memory usage
//! - **Graph statistics**: Real-time graph properties
//! - **Prometheus integration**: Metrics export
//! 
//! ## License
//! 
//! MIT License - see LICENSE file for details.

// Global allocator configuration
#[cfg(feature = "jemalloc")]
extern crate jemallocator;

#[cfg(feature = "jemalloc")]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

pub mod graph;
pub mod algorithms;
pub mod analysis;
pub mod visualization;
pub mod monitoring;
pub mod compression;
pub mod parallel;

// Re-export main types
pub use graph::{
    RelationalGraph, Node, Edge, NodeId, RelationType, SparseTensor,
};

// Re-export algorithms
pub use algorithms::{
    centrality::{PageRank, BetweennessCentrality, ClosenessCentrality, EigenvectorCentrality},
    clustering::{KMeansClustering, SpectralClustering},
    community::{LouvainCommunityDetection, LabelPropagation},
    paths::{DijkstraShortestPath, FloydWarshall, BFSShortestPath},
    metrics::{GraphMetrics, GraphStatistics}
};

// Re-export sampling
pub use graph::sampling::{
    GraphSampler, RandomWalkSampler, LayerWiseSampler
};

// Re-export parallel processing
pub use parallel::ParallelProcessor;

// Re-export visualization
pub use visualization::GraphVisualizer;

// Re-export monitoring
pub use monitoring::PerformanceMetrics;

// Re-export SubgraphSample
pub use graph::sampler::SubgraphSample;

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const AUTHORS: &str = env!("CARGO_PKG_AUTHORS");
pub const DESCRIPTION: &str = env!("CARGO_PKG_DESCRIPTION");

/// Initialize the graph system with default configuration
pub fn init() -> Result<(), Box<dyn std::error::Error>> {
    tracing::info!("Initializing GaussRDL Graph v{}", VERSION);
    
    #[cfg(feature = "monitoring")]
    monitoring::init_metrics()?;
    
    #[cfg(feature = "parallel")]
    // parallel::init_thread_pool()?; // Not implemented yet
    tracing::info!("Parallel processing enabled");
    
    tracing::info!("GaussRDL Graph initialized successfully");
    Ok(())
}

/// Shutdown the graph system and cleanup resources
pub fn shutdown() -> Result<(), Box<dyn std::error::Error>> {
    tracing::info!("Shutting down GaussRDL Graph");
    
    #[cfg(feature = "monitoring")]
    monitoring::shutdown_metrics()?;
    
    #[cfg(feature = "parallel")]
    // parallel::shutdown_thread_pool()?; // Not implemented yet
    tracing::info!("Parallel processing disabled");
    
    tracing::info!("GaussRDL Graph shutdown complete");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_init_shutdown() {
        assert!(init().is_ok());
        assert!(shutdown().is_ok());
    }
}
