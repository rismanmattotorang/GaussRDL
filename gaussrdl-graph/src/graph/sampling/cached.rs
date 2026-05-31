use std::sync::Arc;
use parking_lot::RwLock;
use lru::LruCache;
use std::num::NonZeroUsize;

use candle_core::Result as CandleResult;
use crate::graph::RelationalGraph;
use crate::graph::{GraphResult, GraphError};
use super::GraphSampler;

pub struct CachedGraphSampler {
    inner: Box<dyn GraphSampler>,
    cache: Arc<RwLock<LruCache<String, RelationalGraph>>>,
}

impl CachedGraphSampler {
    pub fn new(sampler: Box<dyn GraphSampler>, cache_size: usize) -> Self {
        Self {
            inner: sampler,
            cache: Arc::new(RwLock::new(LruCache::new(NonZeroUsize::new(cache_size).unwrap()))),
        }
    }
    
    pub fn sample_subgraph(&mut self, graph: &RelationalGraph, seed_nodes: &[crate::graph::NodeId]) -> GraphResult<RelationalGraph> {
        let cache_key = self.compute_cache_key(seed_nodes);
        
        // Check cache first
        {
            let mut cache_guard = self.cache.write();
            if let Some(cached) = cache_guard.get(&cache_key) {
                return Ok((*cached).clone());
            }
        }
        
        // Sample new subgraph
        let subgraph = self.inner.sample_subgraph(graph, None)
            .map_err(|e| GraphError::SamplingError(format!("Sampling failed: {}", e)))?;
        
        // Cache result - clone before caching since LruCache takes ownership
        let subgraph_for_cache = subgraph.clone();
        self.cache.write().put(cache_key, subgraph_for_cache);
        
        Ok(subgraph)
    }
    
    fn compute_cache_key(&self, seed_nodes: &[crate::graph::NodeId]) -> String {
        let mut nodes = seed_nodes.to_vec();
        nodes.sort();
        nodes.iter().map(|n| format!("{}_{}", n.table_type, n.id)).collect::<Vec<_>>().join("_")
    }
    
    pub fn clear_cache(&mut self) {
        self.cache.write().clear();
    }
    
    pub fn get_cache_size(&self) -> usize {
        self.cache.read().len()
    }
    
    pub fn get_cache_capacity(&self) -> usize {
        self.cache.read().cap().get()
    }
} 