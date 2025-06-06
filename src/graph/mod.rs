// src/graph/mod.rs
use candle_core::{Device, Tensor, Result as CandleResult, DType};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use parking_lot::RwLock;
use serde::{Serialize, Deserialize};
use crate::{Result, database::DatabaseSchema};
use crate::monitoring::metrics::GraphMetrics;

pub mod builder;
pub mod sampler;
pub mod types;

pub use builder::*;
pub use sampler::*;
pub use types::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RelationType(String);

impl RelationType {
    pub fn new(name: &str) -> Self {
        Self(name.to_string())
    }
    
    pub fn name(&self) -> &str {
        &self.0
    }
}

impl ToString for RelationType {
    fn to_string(&self) -> String {
        self.0.clone()
    }
}

#[derive(Debug, Clone)]
pub struct Node {
    pub id: String,
    pub attributes: HashMap<String, String>,
    pub edges: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct Edge {
    pub from: String,
    pub to: String,
    pub relation: RelationType,
    pub attributes: HashMap<String, String>,
}

#[derive(Debug)]
pub struct RelationalGraph {
    pub nodes: HashMap<String, Node>,
    pub edges: Vec<Edge>,
    pub relation_types: HashSet<RelationType>,
    pub metrics: Arc<RwLock<GraphMetrics>>,
    
    // Features
    pub node_features: Option<Tensor>,
    pub edge_features: Option<Tensor>,
    pub global_features: Option<Tensor>,
    
    // Type mappings
    pub node_types: HashMap<String, usize>,
    pub edge_types: HashMap<RelationType, usize>,
    
    // Device
    pub device: Device,
    
    // Adjacency information
    pub adjacency: Option<Tensor>,
}

impl RelationalGraph {
    pub fn new(device: Device) -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            relation_types: HashSet::new(),
            metrics: Arc::new(RwLock::new(GraphMetrics {
                num_nodes: 0,
                num_edges: 0,
                avg_degree: 0.0,
                density: 0.0,
                processing_time_ms: 0.0,
                memory_used: 0.0,
            })),
            node_features: None,
            edge_features: None,
            global_features: None,
            node_types: HashMap::new(),
            edge_types: HashMap::new(),
            device,
            adjacency: None,
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
    
    pub fn get_node(&self, id: &str) -> Option<&Node> {
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
    
    pub fn get_adjacency(&self) -> Option<&Tensor> {
        self.adjacency.as_ref()
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
    
    pub fn set_adjacency(&mut self, adjacency: Tensor) {
        self.adjacency = Some(adjacency);
    }
    
    pub fn get_metrics(&self) -> GraphMetrics {
        self.metrics.read().clone()
    }
}

#[derive(Debug, Clone)]
pub struct RelationalEntityGraph {
    pub nodes: HashMap<(usize, u64), EntityNode>,
    pub edges: HashMap<(usize, u64), Vec<(usize, u64)>>,
    pub schema: DatabaseSchema,
    pub device: Device,
}

impl RelationalEntityGraph {
    pub fn new(schema: DatabaseSchema, device: Device) -> Self {
        Self {
            nodes: HashMap::new(),
            edges: HashMap::new(),
            schema,
            device,
        }
    }
    
    pub fn add_node(&mut self, node: EntityNode) -> Result<()> {
        let key = (node.table_type, node.node_id);
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

#[derive(Debug, Clone)]
pub struct SparseTensor {
    pub indices: Tensor,  // (2, num_edges) tensor of node indices
    pub values: Tensor,   // (num_edges,) tensor of edge values
    pub size: (usize, usize), // (num_nodes, num_nodes)
}

impl SparseTensor {
    pub fn new(indices: Tensor, values: Tensor, size: (usize, usize)) -> Self {
        Self { indices, values, size }
    }
    
    pub fn to_dense(&self) -> CandleResult<Tensor> {
        let device = self.indices.device();
        let mut dense = Tensor::zeros((self.size.0, self.size.1), DType::F32, device)?;
        
        // Convert sparse to dense format
        let num_edges = self.values.dim(0)?;
        for i in 0..num_edges {
            let row = self.indices.get(0)?.get(i)?;
            let col = self.indices.get(1)?.get(i)?;
            let val = self.values.get(i)?;
            dense.get_mut((row.to_scalar::<usize>()?, col.to_scalar::<usize>()?))?
                .copy_(&val)?;
        }
        
        Ok(dense)
    }
}

impl RelationalGraph {
    pub fn sparse_message_passing(&self) -> CandleResult<Tensor> {
        let num_nodes = self.node_features.as_ref().unwrap().dim(0)?;
        let hidden_dim = self.node_features.as_ref().unwrap().dim(1)?;
        
        // Initialize output tensor
        let mut output = Tensor::zeros((num_nodes, hidden_dim), DType::F32, &self.device)?;
        
        // Get adjacency indices and values
        let indices = self.adjacency.as_ref().unwrap();
        let edge_weights = self.edge_features.as_ref().unwrap();
        
        // Perform sparse message passing
        let num_edges = edge_weights.dim(0)?;
        for i in 0..num_edges {
            let src = indices.get(0)?.get(i)?.to_scalar::<usize>()?;
            let dst = indices.get(1)?.get(i)?.to_scalar::<usize>()?;
            let weight = edge_weights.get(i)?;
            
            // Get source node features and apply edge weight
            let msg = self.node_features.as_ref().unwrap().get(src)?.mul(&weight)?;
            
            // Update destination node
            let dst_slice = output.get_mut(dst)?;
            dst_slice.add_(&msg)?;
        }
        
        Ok(output)
    }
    
    pub fn batch_graph_operations(&self) -> CandleResult<Tensor> {
        // Convert sparse adjacency to dense for batch operations
        let dense_adj = self.adjacency.as_ref().unwrap().to_dense()?;
        
        // Perform batched matrix multiplication
        dense_adj.matmul(self.node_features.as_ref().unwrap())
    }
    
    pub fn optimize_memory_layout(&self) -> CandleResult<Self> {
        // Create contiguous tensors
        let node_features = self.node_features.as_ref().map(|t| t.contiguous()).transpose()?;
        let adjacency = self.adjacency.as_ref().map(|t| t.contiguous()).transpose()?;
        let edge_features = self.edge_features.as_ref().map(|t| t.contiguous()).transpose()?;
        let global_features = self.global_features.as_ref().map(|t| t.contiguous()).transpose()?;
        
        Ok(Self {
            nodes: self.nodes.clone(),
            edges: self.edges.clone(),
            relation_types: self.relation_types.clone(),
            metrics: self.metrics.clone(),
            node_features,
            edge_features,
            global_features,
            node_types: self.node_types.clone(),
            edge_types: self.edge_types.clone(),
            device: self.device.clone(),
            adjacency,
        })
    }
    
    pub fn get_node_type_embeddings(&self, embeddings: &HashMap<String, Tensor>) -> CandleResult<Tensor> {
        let num_nodes = self.node_features.as_ref().unwrap().dim(0)?;
        let embedding_dim = embeddings.values().next().unwrap().dim(1)?;
        let mut output = Tensor::zeros((num_nodes, embedding_dim), DType::F32, &self.device)?;
        
        for (i, node_type) in self.node_types.iter() {
            if let Some(embedding) = embeddings.get(node_type) {
                output.get_mut(i)?.copy_(embedding)?;
            }
        }
        
        Ok(output)
    }
}