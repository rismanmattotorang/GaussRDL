// src/graph/mod.rs
use candle_core::{Device, Tensor, Result as CandleResult, DType};
use std::collections::{HashMap, HashSet};
use serde::{Serialize, Deserialize};
use serde::ser::SerializeStruct;
use gaussrdl_core::Result;
use chrono;
use crate::graph::error::{GraphError, GraphResult};
pub mod builder;
pub mod sampling;
pub mod error;
pub mod sampler;

pub use builder::*;
pub use sampling::*;
pub use error::*;

// Core types moved to top level to avoid path issues
pub struct NodeId {
    pub id: String,
    pub table_type: String,
}

// Manual trait implementations to bypass derive macro issues
impl std::fmt::Debug for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NodeId")
            .field("node_id", &self.id)
            .field("table_type", &self.table_type)
            .finish()
    }
}

impl Clone for NodeId {
    fn clone(&self) -> Self {
        Self {
            id: self.id.clone(),
            table_type: self.table_type.clone(),
        }
    }
}

impl PartialEq for NodeId {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.table_type == other.table_type
    }
}

impl Eq for NodeId {}

impl std::hash::Hash for NodeId {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
        self.table_type.hash(state);
    }
}

impl PartialOrd for NodeId {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for NodeId {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.table_type.cmp(&other.table_type)
            .then(self.id.cmp(&other.id))
    }
}

// Simple serde implementation using derive macros
#[derive(Serialize, Deserialize)]
struct NodeIdSerde {
    node_id: u64,
    table_type: usize,
}

impl serde::Serialize for NodeId {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let serde_struct = NodeIdSerde {
            node_id: self.id.parse::<u64>().unwrap(),
            table_type: self.table_type.parse::<usize>().unwrap(),
        };
        serde_struct.serialize(serializer)
    }
}

impl<'de> serde::Deserialize<'de> for NodeId {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let serde_struct = NodeIdSerde::deserialize(deserializer)?;
        Ok(NodeId {
            id: serde_struct.node_id.to_string(),
            table_type: serde_struct.table_type.to_string(),
        })
    }
}

impl NodeId {
    pub fn new(table_type: usize, node_id: u64) -> Self {
        Self {
            id: node_id.to_string(),
            table_type: table_type.to_string(),
        }
    }
    
    pub fn from_string(s: &str) -> Self {
        // Simple hash-based conversion for compatibility
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        s.hash(&mut hasher);
        let hash = hasher.finish();
        
        Self {
            id: hash.to_string(),
            table_type: (hash % 1000).to_string(),
        }
    }
    
    pub fn to_string(&self) -> String {
        format!("{}:{}", self.table_type, self.id)
    }
}

impl From<String> for NodeId {
    fn from(s: String) -> Self {
        Self::from_string(&s)
    }
}

impl From<&str> for NodeId {
    fn from(s: &str) -> Self {
        Self::from_string(s)
    }
}

/// Entity type definition with enhanced schema
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntityType {
    pub name: String,
    pub table_id: usize,
    pub schema: EntitySchema,
}

/// Schema definition for entities
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntitySchema {
    pub feature_names: Vec<String>,
    pub feature_types: Vec<FeatureType>,
    pub categorical_mappings: HashMap<String, Vec<String>>,
    pub numerical_ranges: HashMap<String, (f64, f64)>,
}

/// Feature type enumeration
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FeatureType {
    Numerical,
    Categorical,
    Text,
    Temporal,
    Binary,
    Embedding,
}

/// Enhanced attribute value types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AttributeValue {
    Integer(i64),
    Float(f64),
    String(String),
    Boolean(bool),
    Timestamp(chrono::DateTime<chrono::Utc>),
    Vector(Vec<f64>),
    Text(String),
    Categorical(String),
    Null,
}

/// Enhanced entity node
#[derive(Debug, Clone)]
pub struct EntityNode {
    pub id: NodeId,
    pub entity_type: EntityType,
    pub features: Tensor,
    pub timestamp: Option<chrono::DateTime<chrono::Utc>>,
    pub attributes: HashMap<String, AttributeValue>,
    pub metadata: HashMap<String, String>,
}

impl EntityNode {
    /// Create a new entity node
    pub fn new(
        id: NodeId,
        entity_type: EntityType,
        features: Tensor,
        timestamp: Option<chrono::DateTime<chrono::Utc>>,
    ) -> Self {
        Self {
            id,
            entity_type,
            features,
            timestamp,
            attributes: HashMap::new(),
            metadata: HashMap::new(),
        }
    }
    
    /// Add an attribute to the node
    pub fn add_attribute(&mut self, key: String, value: AttributeValue) {
        self.attributes.insert(key, value);
    }
    
    /// Get feature dimension
    pub fn feature_dim(&self) -> Result<usize> {
        Ok(self.features.dims()[0])
    }
}

/// Relation type with weight
pub struct RelationType {
    pub name: String,
    pub attributes: HashMap<String, AttributeValue>,
}

// Manual trait implementations to bypass derive macro issues
impl std::fmt::Debug for RelationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RelationType")
            .field("name", &self.name)
            .field("attributes", &self.attributes)
            .finish()
    }
}

impl Clone for RelationType {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            attributes: self.attributes.clone(),
        }
    }
}

impl PartialEq for RelationType {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.attributes == other.attributes
    }
}

impl Eq for RelationType {}

impl std::hash::Hash for RelationType {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.name.hash(state);
        // Note: HashMap doesn't implement Hash, so we skip attributes for now
        // In a production system, you might want to sort the attributes first
    }
}

impl std::fmt::Display for RelationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name)
    }
}

impl serde::Serialize for RelationType {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut state = serializer.serialize_struct("RelationType", 2)?;
        state.serialize_field("name", &self.name)?;
        state.serialize_field("attributes", &self.attributes)?;
        state.end()
    }
}

impl<'de> serde::Deserialize<'de> for RelationType {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct RelationTypeHelper {
            name: String,
            attributes: HashMap<String, AttributeValue>,
        }
        
        let helper = RelationTypeHelper::deserialize(deserializer)?;
        Ok(RelationType {
            name: helper.name,
            attributes: helper.attributes,
        })
    }
}

/// Schema for relation types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RelationSchema {
    pub feature_names: Vec<String>,
    pub feature_types: Vec<FeatureType>,
    pub constraints: Vec<RelationConstraint>,
}

/// Constraints for relations
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RelationConstraint {
    SameEntityType,
    DifferentEntityType,
    TemporalOrder,
    Unique,
    Weighted,
}

/// Unique edge identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct EdgeId {
    pub source: NodeId,
    pub target: NodeId,
    pub relation_id: usize,
}

/// Enhanced relational edge
#[derive(Debug, Clone)]
pub struct RelationalEdge {
    pub id: EdgeId,
    pub source: NodeId,
    pub target: NodeId,
    pub relation_type: RelationType,
    pub features: Option<Tensor>,
    pub weight: f64,
    pub timestamp: Option<chrono::DateTime<chrono::Utc>>,
    pub attributes: HashMap<String, AttributeValue>,
}

impl RelationalEdge {
    /// Create a new relational edge
    pub fn new(
        source: NodeId,
        target: NodeId,
        relation_type: RelationType,
        weight: f64,
        timestamp: Option<chrono::DateTime<chrono::Utc>>,
    ) -> Self {
        let id = EdgeId {
            source: source.clone(),
            target: target.clone(),
            relation_id: 0, // Use a default value since RelationType doesn't have an id field
        };
        
        Self {
            id,
            source,
            target,
            relation_type,
            features: None,
            weight,
            timestamp,
            attributes: HashMap::new(),
        }
    }
    
    /// Add features to the edge
    pub fn with_features(mut self, features: Tensor) -> Self {
        self.features = Some(features);
        self
    }
    
    /// Add an attribute to the edge
    pub fn add_attribute(&mut self, key: String, value: AttributeValue) {
        self.attributes.insert(key, value);
    }
}

#[derive(Debug, Clone)]
pub struct Node {
    pub id: NodeId,
    pub attributes: HashMap<String, AttributeValue>,
    pub features: Option<Tensor>,
}

// Remove Serialize/Deserialize from Edge
#[derive(Debug, Clone)]
pub struct Edge {
    pub from: NodeId,
    pub to: NodeId,
    pub relation_type: RelationType,
    pub weight: f32,
    pub attributes: HashMap<String, AttributeValue>,
}

#[derive(Debug, Clone)]
pub struct SparseTensor {
    pub indices: Vec<Vec<usize>>,
    pub values: Vec<f32>,
    pub shape: Vec<usize>,
}

impl SparseTensor {
    pub fn new(indices: Vec<Vec<usize>>, values: Vec<f32>, shape: Vec<usize>) -> Self {
        Self { indices, values, shape }
    }
    
    /// Convert sparse tensor to dense format
    pub fn to_dense(&self) -> CandleResult<Tensor> {
        let size = (self.shape[0], self.shape[1]);
        let mut dense_data = vec![0.0f32; size.0 * size.1];
        
        // Extract indices and values
        let indices_data = self.indices.iter().map(|row| row.iter().map(|&x| x as i64).collect::<Vec<_>>()).collect::<Vec<_>>();
        let values_data = self.values.iter().map(|&x| x).collect::<Vec<_>>();
        
        // Fill dense matrix
        for i in 0..values_data.len() {
            let row = indices_data[0][i] as usize;
            let col = indices_data[1][i] as usize;
            let idx = row * size.1 + col;
            if idx < dense_data.len() {
                dense_data[idx] = values_data[i];
            }
        }
        
        Tensor::from_vec(dense_data, (size.0, size.1), &Device::Cpu)
    }
    
    /// Matrix-vector multiplication
    pub fn spmv(&self, x: &Tensor) -> CandleResult<Tensor> {
        let dense = self.to_dense()?;
        dense.matmul(x)
    }
}

#[derive(Clone)]
pub struct RelationalGraph {
    /// Graph nodes indexed by ID
    pub nodes: HashMap<NodeId, Node>,
    
    /// Graph edges
    pub edges: Vec<Edge>,
    
    /// Set of relation types in the graph
    pub relation_types: HashSet<RelationType>,
    
    /// Node feature tensor
    pub node_features: Option<Tensor>,
    
    /// Optional edge feature tensor
    pub edge_features: Option<Tensor>,
    
    /// Optional global feature tensor
    pub global_features: Option<Tensor>,
    
    /// Node type mapping
    pub node_types: HashMap<NodeId, usize>,
    
    /// Edge type mapping
    pub edge_types: HashMap<String, usize>,
    
    /// Device for tensors
    pub device: Device,
    
    /// Sparse adjacency matrix
    pub adjacency: SparseTensor,
}

impl RelationalGraph {
    pub fn new(
        adjacency: SparseTensor,
        node_features: Tensor,
        edge_features: Option<Tensor>,
        global_features: Option<Tensor>,
        node_types: HashMap<NodeId, usize>,
        edge_types: HashMap<String, usize>,
        device: Device,
    ) -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            relation_types: HashSet::new(),
            node_features: Some(node_features),
            edge_features: edge_features,
            global_features: global_features,
            node_types,
            edge_types,
            device,
            adjacency,
        }
    }
    
    pub fn new_empty(device: Device) -> Self {
        let empty_tensor = Tensor::zeros((0, 0), candle_core::DType::F32, &device).unwrap();
        let empty_adjacency = SparseTensor::new(
            vec![vec![0usize; 0]; 2],
            vec![],
            vec![0, 0],
        );
        
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            relation_types: HashSet::new(),
            node_features: None,
            edge_features: None,
            global_features: None,
            node_types: HashMap::new(),
            edge_types: HashMap::new(),
            device,
            adjacency: empty_adjacency,
        }
    }
    
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }
    
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }
    
    pub fn num_relation_types(&self) -> usize {
        self.relation_types.len()
    }
    
    pub fn get_node(&self, id: &NodeId) -> Option<&Node> {
        self.nodes.get(id)
    }
    
    pub fn get_edge(&self, idx: usize) -> Option<&Edge> {
        self.edges.get(idx)
    }
    
    pub fn get_node_features(&self) -> Option<&Tensor> {
        self.node_features.as_ref()
    }
    
    pub fn get_edge_features(&self) -> Option<&Tensor> {
        self.edge_features.as_ref()
    }
    
    pub fn get_global_features(&self) -> Option<&Tensor> {
        self.global_features.as_ref()
    }
    
    pub fn get_adjacency(&self) -> Option<&SparseTensor> {
        Some(&self.adjacency)
    }
    
    pub fn set_node_features(&mut self, features: Tensor) {
        self.node_features = Some(features);
    }
    
    pub fn set_edge_features(&mut self, features: Tensor) {
        self.edge_features = Some(features);
    }
    
    pub fn set_global_features(&mut self, features: Tensor) {
        self.global_features = Some(features);
    }
    
    pub fn set_adjacency(&mut self, adjacency: SparseTensor) {
        self.adjacency = adjacency;
    }
    
    pub fn message_passing(&self) -> CandleResult<Tensor> {
        // Use simple dense matrix multiplication instead of complex sparse operations
        let dense_adj = self.adjacency.to_dense()?;
        dense_adj.matmul(self.node_features.as_ref().unwrap())
    }

    pub fn batch_message_passing(&self, batch_size: usize) -> CandleResult<Tensor> {
        // Simplified batch processing
        let dense_adj = self.adjacency.to_dense()?;
        
        // Perform batched matrix multiplication
        dense_adj.matmul(self.node_features.as_ref().unwrap())
    }
    
    pub fn optimize_memory_layout(&self) -> CandleResult<Self> {
        // Create contiguous tensors
        let node_features = self.node_features.as_ref().map(|t| t.contiguous()).transpose()?;
        let adjacency = self.adjacency.clone();
        let edge_features = self.edge_features.as_ref().map(|t| t.contiguous()).transpose()?;
        let global_features = self.global_features.as_ref().map(|t| t.contiguous()).transpose()?;
        
        Ok(Self {
            nodes: self.nodes.clone(),
            edges: self.edges.clone(),
            relation_types: self.relation_types.clone(),
            node_features,
            edge_features,
            global_features,
            node_types: self.node_types.clone(),
            edge_types: self.edge_types.clone(),
            device: self.device.clone(),
            adjacency,
        })
    }
    
    pub fn get_node_type_embeddings(&self, embeddings: &HashMap<usize, Tensor>) -> CandleResult<Tensor> {
        let num_nodes = self.node_features.as_ref().map(|t| t.dim(0)).unwrap_or(Ok(0))?;
        let embed_dim = embeddings.values().next().map(|t| t.dim(0)).unwrap_or(Ok(64))?;
        
        let mut output_data = vec![0.0f32; num_nodes * embed_dim];
        
        for (i, node_type) in self.node_types.values().enumerate() {
            if let Some(embedding) = embeddings.get(node_type) {
                let embed_data = embedding.to_vec1::<f32>()?;
                for j in 0..embed_dim.min(embed_data.len()) {
                    if i * embed_dim + j < output_data.len() {
                        output_data[i * embed_dim + j] = embed_data[j];
                    }
                }
            }
        }
        
        Tensor::from_vec(output_data, (num_nodes, embed_dim), &self.device)
    }
}

#[derive(Debug, Clone)]
pub struct RelationalEntityGraph {
    pub nodes: HashMap<(usize, u64), EntityNode>,
    pub edges: HashMap<(usize, u64), Vec<(usize, u64)>>,
    pub device: Device,
}

impl RelationalEntityGraph {
    pub fn new(device: Device) -> Self {
        Self {
            nodes: HashMap::new(),
            edges: HashMap::new(),
            device,
        }
    }
    
    pub fn add_node(&mut self, node: EntityNode) -> Result<()> {
        let key = (node.id.table_type.parse::<usize>().unwrap(), node.id.id.parse::<u64>().unwrap());
        self.nodes.insert(key, node);
        self.edges.entry(key).or_insert_with(Vec::new);
        Ok(())
    }
    
    pub fn add_edge(&mut self, from: (usize, u64), to: (usize, u64)) -> Result<()> {
        self.edges.entry(from).or_default().push(to);
        self.edges.entry(to).or_default().push(from); // Undirected
        Ok(())
    }
    
    pub fn get_node(&self, key: (usize, u64)) -> Option<&EntityNode> {
        self.nodes.get(&key)
    }
    
    pub fn get_neighbors(&self, key: (usize, u64)) -> Option<&Vec<(usize, u64)>> {
        self.edges.get(&key)
    }
}

// Types are already available in this module
pub use sampler::SubgraphSample;