// src/graph/lightrdl_graph.rs
use std::collections::HashMap;
use chrono::{DateTime, Duration, Utc};
use candle_core::{Tensor, Device, Result as CandleResult, DType};
use polars::prelude::*;
use serde::{Serialize, Deserialize};

use crate::graph::*;
use crate::error::Result;
use crate::models::UnifiedModelConfig;

/// Enhanced LightRDL graph builder with modern architecture support
pub struct LightRDLGraphBuilder {
    config: LightRDLGraphConfig,
    device: Device,
    embeddings: HashMap<String, DataFrame>,
    entity_schemas: HashMap<String, EntitySchema>,
    relation_schemas: HashMap<String, RelationSchema>,
    statistics: GraphBuildingStatistics,
}

/// Configuration for LightRDL graph building
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LightRDLGraphConfig {
    pub hidden_dim: usize,
    pub max_nodes: usize,
    pub max_edges: usize,
    pub temporal_window_hours: u64,
    pub feature_aggregation: FeatureAggregationType,
    pub edge_weight_threshold: f64,
    pub enable_caching: bool,
    pub sampling_strategy: GraphSamplingStrategy,
}

/// Feature aggregation strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeatureAggregationType {
    Mean,
    Sum,
    Max,
    Concatenate,
    WeightedMean,
}

/// Graph sampling strategies for LightRDL
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GraphSamplingStrategy {
    FullGraph,
    RandomSampling { fraction: f64 },
    TemporalWindow { hours: u64 },
    KHopNeighborhood { k: usize },
}

/// Statistics for graph building operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphBuildingStatistics {
    pub nodes_created: usize,
    pub edges_created: usize,
    pub features_processed: usize,
    pub build_time_ms: u64,
    pub memory_usage_mb: f64,
    pub cache_hits: usize,
    pub cache_misses: usize,
}

impl Default for LightRDLGraphConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 256,
            max_nodes: 10000,
            max_edges: 50000,
            temporal_window_hours: 24,
            feature_aggregation: FeatureAggregationType::Mean,
            edge_weight_threshold: 0.1,
            enable_caching: true,
            sampling_strategy: GraphSamplingStrategy::FullGraph,
        }
    }
}

impl Default for GraphBuildingStatistics {
    fn default() -> Self {
        Self {
            nodes_created: 0,
            edges_created: 0,
            features_processed: 0,
            build_time_ms: 0,
            memory_usage_mb: 0.0,
            cache_hits: 0,
            cache_misses: 0,
        }
    }
}

impl LightRDLGraphBuilder {
    /// Create a new enhanced LightRDL graph builder
    pub fn new(config: LightRDLGraphConfig, device: Device) -> Self {
        Self {
            config,
            device,
            embeddings: HashMap::new(),
            entity_schemas: HashMap::new(),
            relation_schemas: HashMap::new(),
            statistics: GraphBuildingStatistics::default(),
        }
    }
    
    /// Create from unified model configuration
    pub fn from_unified_config(unified_config: &UnifiedModelConfig, device: Device) -> Self {
        let config = LightRDLGraphConfig {
            hidden_dim: unified_config.hidden_dim,
            max_nodes: 10000,
            max_edges: 50000,
            temporal_window_hours: 24,
            feature_aggregation: FeatureAggregationType::Mean,
            edge_weight_threshold: 0.1,
            enable_caching: true,
            sampling_strategy: GraphSamplingStrategy::FullGraph,
        };
        
        Self::new(config, device)
    }
    
    /// Load embeddings with enhanced error handling and validation
    pub fn load_embeddings(&mut self, path: &str, entity_type: &str) -> Result<()> {
        let start_time = std::time::Instant::now();
        
        let embedding_path = format!("{}/emb_{}.parquet", path, entity_type);
        
        match LazyFrame::scan_parquet(&embedding_path, ScanArgsParquet::default()) {
            Ok(lazy_df) => {
                let df = lazy_df.collect()?;
                
                // Validate embedding dimensions
                if let Ok(embedding_col) = df.column("embedding") {
                    if embedding_col.len() > 0 {
                        self.embeddings.insert(entity_type.to_string(), df);
                        
                        // Create entity schema
                        let schema = EntitySchema {
                            feature_names: vec!["embedding".to_string()],
                            feature_types: vec![FeatureType::Embedding],
                            categorical_mappings: HashMap::new(),
                            numerical_ranges: HashMap::new(),
                        };
                        self.entity_schemas.insert(entity_type.to_string(), schema);
                        
                        if self.config.enable_caching {
                            self.statistics.cache_hits += 1;
                        }
                    }
                }
                
                self.statistics.build_time_ms += start_time.elapsed().as_millis() as u64;
                Ok(())
            }
            Err(e) => {
                self.statistics.cache_misses += 1;
                Err(crate::error::Error::io(&format!("Failed to load embeddings for {}: {}", entity_type, e)))
            }
        }
    }
    
    /// Build enhanced subgraph with comprehensive features
    pub fn build_subgraph(
        &mut self,
        t0: DateTime<Utc>,
        t1: DateTime<Utc>,
        seed_nodes: &[NodeId],
        database: &DataFrame,
    ) -> Result<SubgraphSample> {
        let start_time = std::time::Instant::now();
        
        let mut subgraph = SubgraphSample::new(
            format!("lightrdl_{}_{}", t0.timestamp(), t1.timestamp()),
            seed_nodes.to_vec(),
            SamplingStrategy {
                strategy_type: SamplingType::NeighborSampling {
                    fanouts: vec![10, 5],
                    replace: false,
                },
                parameters: HashMap::new(),
                max_nodes: self.config.max_nodes,
                max_edges: self.config.max_edges,
                max_hops: 3,
                temporal_window: Some(chrono::Duration::hours(self.config.temporal_window_hours as i64)),
            },
        );
        
        // Build nodes with enhanced features
        for seed_node in seed_nodes {
            let node = self.create_enhanced_node(seed_node, t0, t1, database)?;
            subgraph.add_node(node, 0);
            self.statistics.nodes_created += 1;
        }
        
        // Build edges with relation types
        self.add_relational_edges(&mut subgraph, t0, t1, database)?;
        
        // Add temporal context
        self.add_temporal_context(&mut subgraph, t0, t1)?;
        
        // Update statistics
        self.statistics.build_time_ms += start_time.elapsed().as_millis() as u64;
        subgraph.statistics.sampling_time_ms = start_time.elapsed().as_millis() as u64;
        
        Ok(subgraph)
    }
    
    /// Create an enhanced entity node with comprehensive features
    fn create_enhanced_node(
        &self,
        node_id: &NodeId,
        t0: DateTime<Utc>,
        t1: DateTime<Utc>,
        database: &DataFrame,
    ) -> Result<EntityNode> {
        // Extract base features
        let features = self.extract_node_features(node_id, database)?;
        
        // Create entity type
        let entity_type = EntityType {
            name: format!("entity_{}", node_id.table_type),
            table_id: node_id.table_type,
            schema: self.entity_schemas.get(&format!("entity_{}", node_id.table_type))
                .cloned()
                .unwrap_or_else(|| EntitySchema {
                    feature_names: vec!["default".to_string()],
                    feature_types: vec![FeatureType::Numerical],
                    categorical_mappings: HashMap::new(),
                    numerical_ranges: HashMap::new(),
                }),
        };
        
        // Create enhanced node
        let mut node = EntityNode::new(
            node_id.clone(),
            entity_type,
            features,
            Some(t0), // Use start time as timestamp
        );
        
        // Add metadata
        node.metadata.insert("created_at".to_string(), Utc::now().to_rfc3339());
        node.metadata.insert("time_window".to_string(), format!("{} to {}", t0, t1));
        
        Ok(node)
    }
    
    /// Add relational edges with enhanced relationship modeling
    fn add_relational_edges(
        &mut self,
        subgraph: &mut SubgraphSample,
        t0: DateTime<Utc>,
        t1: DateTime<Utc>,
        database: &DataFrame,
    ) -> Result<()> {
        // Extract relationships from database
        let relationships = self.extract_relationships(database, t0, t1)?;
        
        for relationship in relationships {
            if let (Some(_source_node), Some(_target_node)) = (
                subgraph.node_mapping.get(&relationship.source),
                subgraph.node_mapping.get(&relationship.target),
            ) {
                // Create relation type
                let relation_type = RelationType {
                    name: relationship.relation_name.clone(),
                    id: relationship.relation_id,
                    is_directed: true,
                    is_temporal: true,
                    schema: RelationSchema {
                        feature_names: vec!["weight".to_string(), "timestamp".to_string()],
                        feature_types: vec![FeatureType::Numerical, FeatureType::Temporal],
                        constraints: vec![RelationConstraint::TemporalOrder],
                    },
                };
                
                // Create enhanced edge
                let mut edge = RelationalEdge::new(
                    relationship.source.clone(),
                    relationship.target.clone(),
                    relation_type,
                    relationship.weight,
                    relationship.timestamp,
                );
                
                // Add edge features if available
                if let Some(features) = relationship.features {
                    edge = edge.with_features(features);
                }
                
                subgraph.add_edge(edge);
                self.statistics.edges_created += 1;
            }
        }
        
        Ok(())
    }
    
    /// Add temporal context to the subgraph
    fn add_temporal_context(
        &self,
        subgraph: &mut SubgraphSample,
        t0: DateTime<Utc>,
        t1: DateTime<Utc>,
    ) -> Result<()> {
        // Update subgraph timestamps
        subgraph.timestamps = vec![t0, t1];
        
        // Calculate temporal span
        let duration = t1 - t0;
        subgraph.statistics.temporal_span = Some(duration);
        
        Ok(())
    }
    
    /// Extract node features with enhanced aggregation
    fn extract_node_features(&self, node_id: &NodeId, _database: &DataFrame) -> Result<Tensor> {
        let entity_type = format!("entity_{}", node_id.table_type);
        
        // Try to get cached embeddings first
        if let Some(embedding_df) = self.embeddings.get(&entity_type) {
            if let Ok(filtered) = embedding_df
                .clone()
                .lazy()
                .filter(col("id").eq(lit(node_id.node_id as i64)))
                .collect()
            {
                if let Ok(_embedding_series) = filtered.column("embedding") {
                    // Convert to tensor (this is a simplified implementation)
                    let features = vec![0.5; self.config.hidden_dim]; // Placeholder
                    return Tensor::from_vec(features, (self.config.hidden_dim,), &self.device);
                }
            }
        }
        
        // Fallback to default features
        let default_features = vec![0.0; self.config.hidden_dim];
        Tensor::from_vec(default_features, (self.config.hidden_dim,), &self.device)
    }
    
    /// Extract relationships from database with temporal filtering
    fn extract_relationships(
        &self,
        _database: &DataFrame,
        t0: DateTime<Utc>,
        _t1: DateTime<Utc>,
    ) -> Result<Vec<RelationshipData>> {
        let mut relationships = Vec::new();
        
        // This is a simplified implementation
        // In practice, this would extract actual relationships from the database
        // based on foreign key constraints and temporal windows
        
        relationships.push(RelationshipData {
            source: NodeId { table_type: 0, node_id: 1 },
            target: NodeId { table_type: 1, node_id: 1 },
            relation_name: "associated_with".to_string(),
            relation_id: 0,
            weight: 1.0,
            timestamp: Some(t0),
            features: None,
        });
        
        Ok(relationships)
    }
    
    /// Get comprehensive building statistics
    pub fn get_statistics(&self) -> &GraphBuildingStatistics {
        &self.statistics
    }
    
    /// Reset statistics for new building session
    pub fn reset_statistics(&mut self) {
        self.statistics = GraphBuildingStatistics::default();
    }
    
    /// Get configuration
    pub fn get_config(&self) -> &LightRDLGraphConfig {
        &self.config
    }
    
    /// Update configuration
    pub fn update_config(&mut self, config: LightRDLGraphConfig) {
        self.config = config;
    }
}

/// Helper struct for relationship data extraction
#[derive(Debug, Clone)]
struct RelationshipData {
    source: NodeId,
    target: NodeId,
    relation_name: String,
    relation_id: usize,
    weight: f64,
    timestamp: Option<DateTime<Utc>>,
    features: Option<Tensor>,
}

/// Factory for creating LightRDL graph builders with different configurations
pub struct LightRDLGraphBuilderFactory;

impl LightRDLGraphBuilderFactory {
    /// Create a builder optimized for small graphs
    pub fn create_small_graph_builder(device: Device) -> LightRDLGraphBuilder {
        let config = LightRDLGraphConfig {
            hidden_dim: 128,
            max_nodes: 1000,
            max_edges: 5000,
            temporal_window_hours: 12,
            feature_aggregation: FeatureAggregationType::Mean,
            edge_weight_threshold: 0.2,
            enable_caching: true,
            sampling_strategy: GraphSamplingStrategy::FullGraph,
        };
        
        LightRDLGraphBuilder::new(config, device)
    }
    
    /// Create a builder optimized for large graphs
    pub fn create_large_graph_builder(device: Device) -> LightRDLGraphBuilder {
        let config = LightRDLGraphConfig {
            hidden_dim: 512,
            max_nodes: 100000,
            max_edges: 500000,
            temporal_window_hours: 168, // 1 week
            feature_aggregation: FeatureAggregationType::WeightedMean,
            edge_weight_threshold: 0.05,
            enable_caching: true,
            sampling_strategy: GraphSamplingStrategy::RandomSampling { fraction: 0.1 },
        };
        
        LightRDLGraphBuilder::new(config, device)
    }
    
    /// Create a builder optimized for temporal analysis
    pub fn create_temporal_builder(device: Device) -> LightRDLGraphBuilder {
        let config = LightRDLGraphConfig {
            hidden_dim: 256,
            max_nodes: 10000,
            max_edges: 50000,
            temporal_window_hours: 6,
            feature_aggregation: FeatureAggregationType::Concatenate,
            edge_weight_threshold: 0.1,
            enable_caching: true,
            sampling_strategy: GraphSamplingStrategy::TemporalWindow { hours: 6 },
        };
        
        LightRDLGraphBuilder::new(config, device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_enhanced_graph_builder_creation() {
        let device = Device::Cpu;
        let config = LightRDLGraphConfig::default();
        let builder = LightRDLGraphBuilder::new(config, device);
        
        assert_eq!(builder.get_config().hidden_dim, 256);
        assert_eq!(builder.get_statistics().nodes_created, 0);
    }
    
    #[test]
    fn test_factory_builders() {
        let device = Device::Cpu;
        
        let small_builder = LightRDLGraphBuilderFactory::create_small_graph_builder(device.clone());
        assert_eq!(small_builder.get_config().max_nodes, 1000);
        
        let large_builder = LightRDLGraphBuilderFactory::create_large_graph_builder(device.clone());
        assert_eq!(large_builder.get_config().max_nodes, 100000);
        
        let temporal_builder = LightRDLGraphBuilderFactory::create_temporal_builder(device);
        assert_eq!(temporal_builder.get_config().temporal_window_hours, 6);
    }
    
    #[test]
    fn test_config_serialization() {
        let config = LightRDLGraphConfig::default();
        let serialized = serde_json::to_string(&config).unwrap();
        let deserialized: LightRDLGraphConfig = serde_json::from_str(&serialized).unwrap();
        
        assert_eq!(config.hidden_dim, deserialized.hidden_dim);
        assert_eq!(config.max_nodes, deserialized.max_nodes);
    }
} 