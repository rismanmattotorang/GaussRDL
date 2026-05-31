// src/graph/sampler.rs
use candle_core::{Device, Tensor, Result as CandleResult, DType};
use rand::{Rng, SeedableRng, seq::SliceRandom, thread_rng};
use rand::rngs::StdRng;
use std::collections::{HashSet, VecDeque, HashMap, BTreeSet};
use crate::graph::{Result, Node, RelationalEdge, NodeId};
use gaussrdl_core::Error;
use serde::{Serialize, Deserialize};
use std::time::Instant;
use chrono::DateTime;
use chrono::Utc;
use crate::graph::{EntityNode, EdgeId};
use crate::graph::error::GraphError;

#[derive(Debug, Clone)]
pub struct SubgraphSample {
    pub id: String,
    pub nodes: Vec<EntityNode>,
    pub edges: Vec<RelationalEdge>,
    pub node_mapping: std::collections::HashMap<NodeId, usize>,
    pub statistics: SamplerStatistics,
    pub timestamps: Vec<DateTime<Utc>>,
}

impl SubgraphSample {
    pub fn new(id: String, seed_nodes: Vec<NodeId>, _strategy: SamplingStrategy) -> Self {
        Self {
            id,
            nodes: Vec::new(),
            edges: Vec::new(),
            node_mapping: seed_nodes.into_iter().enumerate().map(|(i, n)| (n, i)).collect(),
            statistics: SamplerStatistics {
                total_samples: 0,
                avg_sampling_time_ms: 0.0,
                cache_hit_rate: 0.0,
                avg_subgraph_size: 0.0,
                quality_distribution: Vec::new(),
                error_count: 0,
            },
            timestamps: Vec::new(),
        }
    }
    pub fn add_node(&mut self, node: EntityNode, idx: usize) {
        self.node_mapping.insert(node.id.clone(), idx);
        self.nodes.push(node);
    }
    pub fn add_edge(&mut self, edge: RelationalEdge) {
        self.edges.push(edge);
    }
}

/// Enhanced graph sampler trait with comprehensive sampling strategies
pub trait GraphSampler: Send + Sync {
    /// Sample a subgraph from the given graph
    fn sample_subgraph(
        &self,
        graph: &dyn RelationalGraph,
        seed_nodes: &[NodeId],
        config: &SamplingConfig,
    ) -> Result<SubgraphSample>;
    
    /// Sample multiple subgraphs for batch processing
    fn sample_batch(
        &self,
        graph: &dyn RelationalGraph,
        seed_nodes: &[Vec<NodeId>],
        config: &SamplingConfig,
    ) -> Result<Vec<SubgraphSample>>;
    
    /// Get sampler statistics
    fn get_statistics(&self) -> SamplerStatistics;
    
    /// Validate sampling configuration
    fn validate_config(&self, config: &SamplingConfig) -> Result<()>;
}

/// Configuration for graph sampling
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    pub strategy: SamplingStrategy,
    pub batch_size: usize,
    pub device: Device,
    pub seed: Option<u64>,
    pub max_samples_per_node: usize,
    pub enable_caching: bool,
    pub parallel_sampling: bool,
    pub quality_threshold: f64,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            strategy: SamplingStrategy {
                strategy_type: SamplingType::NeighborSampling {
                    fanouts: vec![10, 5],
                    replace: false,
                },
                parameters: HashMap::new(),
                max_nodes: 1000,
                max_edges: 5000,
                max_hops: 3,
                temporal_window: None,
            },
            batch_size: 32,
            device: Device::Cpu,
            seed: None,
            max_samples_per_node: 100,
            enable_caching: true,
            parallel_sampling: false,
            quality_threshold: 0.7,
        }
    }
}

/// Statistics for sampling operations
#[derive(Debug, Clone)]
pub struct SamplerStatistics {
    pub total_samples: usize,
    pub avg_sampling_time_ms: f64,
    pub cache_hit_rate: f64,
    pub avg_subgraph_size: f64,
    pub quality_distribution: Vec<f64>,
    pub error_count: usize,
}

/// Enhanced random walk sampler with node2vec support
pub struct RandomWalkSampler {
    walk_length: usize,
    num_walks: usize,
    p: f64,  // Return parameter
    q: f64,  // In-out parameter
    statistics: SamplerStatistics,
    cache: HashMap<String, SubgraphSample>,
}

impl RandomWalkSampler {
    /// Create a new random walk sampler
    pub fn new(walk_length: usize, num_walks: usize, p: f64, q: f64) -> Self {
        Self {
            walk_length,
            num_walks,
            p,
            q,
            statistics: SamplerStatistics {
                total_samples: 0,
                avg_sampling_time_ms: 0.0,
                cache_hit_rate: 0.0,
                avg_subgraph_size: 0.0,
                quality_distribution: Vec::new(),
                error_count: 0,
            },
            cache: HashMap::new(),
        }
    }
    
    /// Perform a node2vec biased random walk
    fn node2vec_walk(
        &self,
        graph: &dyn RelationalGraph,
        start_node: &NodeId,
        rng: &mut StdRng,
    ) -> Result<Vec<NodeId>> {
        let mut walk = vec![start_node.clone()];
        
        for step in 0..self.walk_length {
            let current_node = &walk[step];
            let neighbors = graph.get_neighbors(current_node)?;
            
            if neighbors.is_empty() {
                break;
            }
            
            let next_node = if step == 0 {
                // First step: uniform random selection
                neighbors.choose(rng).unwrap().clone()
            } else {
                // Subsequent steps: biased selection based on p and q
                let prev_node = &walk[step - 1];
                self.select_next_node_biased(current_node, prev_node, &neighbors, graph, rng)?
            };
            
            walk.push(next_node);
        }
        
        Ok(walk)
    }
    
    /// Select next node with bias based on node2vec parameters
    fn select_next_node_biased(
        &self,
        current: &NodeId,
        previous: &NodeId,
        neighbors: &[NodeId],
        graph: &dyn RelationalGraph,
        rng: &mut StdRng,
    ) -> Result<NodeId> {
        let mut weights = Vec::with_capacity(neighbors.len());
        
        for neighbor in neighbors {
            let weight = if neighbor == previous {
                // Return to previous node
                1.0 / self.p
            } else if graph.has_edge(previous, neighbor)? {
                // Move to a node that was also reachable from previous
                1.0
            } else {
                // Move to a new node
                1.0 / self.q
            };
            weights.push(weight);
        }
        
        // Normalize weights
        let total_weight: f64 = weights.iter().sum();
        if total_weight == 0.0 {
            return Ok(neighbors.choose(rng).unwrap().clone());
        }
        
        for weight in &mut weights {
            *weight /= total_weight;
        }
        
        // Sample based on weights
        let mut cumsum = 0.0;
        let rand_val = rng.gen::<f64>();
        
        for (i, &weight) in weights.iter().enumerate() {
            cumsum += weight;
            if rand_val <= cumsum {
                return Ok(neighbors[i].clone());
            }
        }
        
        // Fallback
        Ok(neighbors.last().unwrap().clone())
    }
    
    /// Convert walk to subgraph
    fn walk_to_subgraph(
        &self,
        walks: &[Vec<NodeId>],
        graph: &dyn RelationalGraph,
        config: &SamplingConfig,
    ) -> Result<SubgraphSample> {
        let start_time = Instant::now();
        
        let mut subgraph = SubgraphSample::new(
            uuid::Uuid::new_v4().to_string(),
            walks.first().map(|w| vec![w[0].clone()]).unwrap_or_default(),
            config.strategy.clone(),
        );
        
        let mut visited_nodes = HashSet::new();
        let mut visited_edges = HashSet::new();
        
        // Collect all nodes and edges from walks
        for walk in walks {
            for (i, node_id) in walk.iter().enumerate() {
                if visited_nodes.insert(node_id.clone()) {
                    let node = graph.get_node(node_id)?;
                    subgraph.add_node(node, i.min(config.strategy.max_hops));
                }
                
                if i > 0 {
                    let src = &walk[i - 1];
                    let dst = node_id;
                    let edge_id = EdgeId {
                        source: src.clone(),
                        target: dst.clone(),
                        relation_id: 0, // Would get actual relation ID
                    };
                    
                    if visited_edges.insert(edge_id.clone()) {
                        if let Ok(edge) = graph.get_edge(src, dst) {
                            subgraph.add_edge(edge);
                        }
                    }
                }
            }
        }
        
        let sampling_time = start_time.elapsed().as_millis() as u64;
        subgraph.statistics.avg_sampling_time_ms = sampling_time as f64;
        
        Ok(subgraph)
    }
}

impl GraphSampler for RandomWalkSampler {
    fn sample_subgraph(
        &self,
        graph: &dyn RelationalGraph,
        seed_nodes: &[NodeId],
        config: &SamplingConfig,
    ) -> Result<SubgraphSample> {
        let mut rng = match config.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        
        let mut all_walks = Vec::new();
        
        for seed_node in seed_nodes {
            for _ in 0..self.num_walks {
                let walk = self.node2vec_walk(graph, seed_node, &mut rng)?;
                all_walks.push(walk);
            }
        }
        
        self.walk_to_subgraph(&all_walks, graph, config)
    }
    
    fn sample_batch(
        &self,
        graph: &dyn RelationalGraph,
        seed_nodes: &[Vec<NodeId>],
        config: &SamplingConfig,
    ) -> Result<Vec<SubgraphSample>> {
        let mut samples = Vec::with_capacity(seed_nodes.len());
        
        for seed_batch in seed_nodes {
            let sample = self.sample_subgraph(graph, seed_batch, config)?;
            samples.push(sample);
        }
        
        Ok(samples)
    }
    
    fn get_statistics(&self) -> SamplerStatistics {
        self.statistics.clone()
    }
    
    fn validate_config(&self, config: &SamplingConfig) -> Result<()> {
        if let SamplingType::RandomWalk { walk_length, num_walks, p, q } = &config.strategy.strategy_type {
            if *walk_length == 0 || *num_walks == 0 {
                return Err(Error::validation("Walk length and num_walks must be > 0"));
            }
            if *p <= 0.0 || *q <= 0.0 {
                return Err(Error::validation("Parameters p and q must be > 0"));
            }
        }
        Ok(())
    }
}

/// Layer-wise neighbor sampling (FastGraphSAINT style)
pub struct LayerWiseSampler {
    fanouts: Vec<usize>,
    replace: bool,
    statistics: SamplerStatistics,
    cache: HashMap<String, SubgraphSample>,
}

impl LayerWiseSampler {
    /// Create a new layer-wise sampler
    pub fn new(fanouts: Vec<usize>, replace: bool) -> Self {
        Self {
            fanouts,
            replace,
            statistics: SamplerStatistics {
                total_samples: 0,
                avg_sampling_time_ms: 0.0,
                cache_hit_rate: 0.0,
                avg_subgraph_size: 0.0,
                quality_distribution: Vec::new(),
                error_count: 0,
            },
            cache: HashMap::new(),
        }
    }
    
    /// Sample neighbors for a single layer
    fn sample_layer(
        &self,
        graph: &dyn RelationalGraph,
        nodes: &[NodeId],
        fanout: usize,
        rng: &mut StdRng,
    ) -> Result<(Vec<NodeId>, Vec<RelationalEdge>)> {
        let mut sampled_nodes = HashSet::new();
        let mut sampled_edges = Vec::new();
        
        for node in nodes {
            let neighbors = graph.get_neighbors(node)?;
            
            let selected_neighbors = if self.replace || neighbors.len() <= fanout {
                // Sample with replacement or if we have fewer neighbors than fanout
                (0..fanout.min(neighbors.len()))
                    .map(|_| neighbors.choose(rng).unwrap().clone())
                    .collect::<Vec<_>>()
            } else {
                // Sample without replacement
                neighbors.choose_multiple(rng, fanout).cloned().collect()
            };
            
            for neighbor in selected_neighbors {
                sampled_nodes.insert(neighbor.clone());
                
                if let Ok(edge) = graph.get_edge(node, &neighbor) {
                    sampled_edges.push(edge);
                }
            }
        }
        
        Ok((sampled_nodes.into_iter().collect(), sampled_edges))
    }
}

impl GraphSampler for LayerWiseSampler {
    fn sample_subgraph(
        &self,
        graph: &dyn RelationalGraph,
        seed_nodes: &[NodeId],
        config: &SamplingConfig,
    ) -> Result<SubgraphSample> {
        let start_time = Instant::now();
        let mut rng = match config.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        
        let mut subgraph = SubgraphSample::new(
            uuid::Uuid::new_v4().to_string(),
            seed_nodes.to_vec(),
            config.strategy.clone(),
        );
        
        let mut current_layer = seed_nodes.to_vec();
        let mut all_nodes = HashSet::new();
        let mut all_edges = Vec::new();
        
        // Add seed nodes
        for (i, seed_node) in seed_nodes.iter().enumerate() {
            all_nodes.insert(seed_node.clone());
            let node = graph.get_node(seed_node)?;
            subgraph.add_node(node, 0);
        }
        
        // Sample layers
        for (layer_idx, &fanout) in self.fanouts.iter().enumerate() {
            if current_layer.is_empty() {
                break;
            }
            
            let (next_nodes, edges) = self.sample_layer(graph, &current_layer, fanout, &mut rng)?;
            
            // Add new nodes
            for node_id in &next_nodes {
                if all_nodes.insert(node_id.clone()) {
                    let node = graph.get_node(node_id)?;
                    subgraph.add_node(node, layer_idx + 1);
                }
            }
            
            // Add edges
            for edge in edges {
                subgraph.add_edge(edge.clone());
                all_edges.push(edge.clone());
            }
            
            current_layer = next_nodes;
        }
        
        let sampling_time = start_time.elapsed().as_millis() as u64;
        subgraph.statistics.avg_sampling_time_ms = sampling_time as f64;
        
        Ok(subgraph)
    }
    
    fn sample_batch(
        &self,
        graph: &dyn RelationalGraph,
        seed_nodes: &[Vec<NodeId>],
        config: &SamplingConfig,
    ) -> Result<Vec<SubgraphSample>> {
        seed_nodes.iter()
            .map(|seeds| self.sample_subgraph(graph, seeds, config))
            .collect()
    }
    
    fn get_statistics(&self) -> SamplerStatistics {
        self.statistics.clone()
    }
    
    fn validate_config(&self, config: &SamplingConfig) -> Result<()> {
        if let SamplingType::NeighborSampling { fanouts, .. } = &config.strategy.strategy_type {
            if fanouts.is_empty() {
                return Err(Error::validation("Fanouts cannot be empty"));
            }
            if fanouts.iter().any(|&f| f == 0) {
                return Err(Error::validation("All fanouts must be > 0"));
            }
        }
        Ok(())
    }
}

/// Factory function to create samplers based on strategy type
pub fn create_sampler(strategy: &SamplingType) -> Result<Box<dyn GraphSampler>> {
    match strategy {
        SamplingType::RandomWalk { walk_length, num_walks, p, q } => {
            Ok(Box::new(RandomWalkSampler::new(*walk_length, *num_walks, *p, *q)))
        }
        SamplingType::NeighborSampling { fanouts, replace } => {
            Ok(Box::new(LayerWiseSampler::new(fanouts.clone(), *replace)))
        }
        SamplingType::Temporal { window_size, max_events } => {
            Err(Error::validation("Temporal sampler not yet implemented"))
        }
        SamplingType::Heterogeneous { type_constraints } => {
            Err(Error::validation("Heterogeneous sampler not yet implemented"))
        }
    }
}

// Add SamplingType and SamplingStrategy definitions
#[derive(Debug, Clone)]
pub enum SamplingType {
    RandomWalk {
        walk_length: usize,
        num_walks: usize,
        p: f64,
        q: f64,
    },
    NeighborSampling {
        fanouts: Vec<usize>,
        replace: bool,
    },
    Temporal {
        window_size: usize,
        max_events: usize,
    },
    Heterogeneous {
        type_constraints: Vec<String>,
    },
}

#[derive(Debug, Clone)]
pub struct SamplingStrategy {
    pub strategy_type: SamplingType,
    pub parameters: HashMap<String, String>,
    pub max_nodes: usize,
    pub max_edges: usize,
    pub max_hops: usize,
    pub temporal_window: Option<chrono::Duration>,
}

// Placeholder RelationalGraph trait for compilation
pub trait RelationalGraph {
    fn get_neighbors(&self, node: &NodeId) -> Result<Vec<NodeId>>;
    fn get_node(&self, node_id: &NodeId) -> Result<EntityNode>;
    fn get_edge(&self, src: &NodeId, dst: &NodeId) -> Result<RelationalEdge>;
    fn has_edge(&self, src: &NodeId, dst: &NodeId) -> Result<bool>;
    fn get_temporal_edges(
        &self,
        start: chrono::DateTime<chrono::Utc>,
        end: chrono::DateTime<chrono::Utc>,
    ) -> Result<Vec<RelationalEdge>>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_sampling_config_validation() {
        let config = SamplingConfig::default();
        let sampler = LayerWiseSampler::new(vec![10, 5], false);
        assert!(sampler.validate_config(&config).is_ok());
    }
    
    #[test]
    fn test_sampler_creation() {
        let strategy = SamplingType::NeighborSampling {
            fanouts: vec![10, 5],
            replace: false,
        };
        
        let sampler = create_sampler(&strategy);
        assert!(sampler.is_ok());
    }
    
    #[test]
    fn test_random_walk_config() {
        let strategy = SamplingType::RandomWalk {
            walk_length: 10,
            num_walks: 5,
            p: 1.0,
            q: 1.0,
        };
        
        let sampler = create_sampler(&strategy);
        assert!(sampler.is_ok());
    }
} 