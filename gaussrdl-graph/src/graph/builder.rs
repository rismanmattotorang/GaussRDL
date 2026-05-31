// src/graph/builder.rs
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use parking_lot::RwLock;
use serde::{Serialize, Deserialize};
use crate::graph::{RelationalGraph, Node, Edge, RelationType, SparseTensor};
use crate::graph::{GraphResult, GraphError};
use crate::graph::NodeId;
use gaussrdl_metrics::monitoring::metrics::GraphMetrics;
// Removed monitoring import - not implemented yet
use candle_core::{Device, Tensor, Result as CandleResult, DType};
use std::collections::BinaryHeap;
use ordered_float::OrderedFloat;
use std::cmp::Reverse;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphBuilderConfig {
    pub max_nodes: usize,
    pub max_edges: usize,
    pub max_density: f64,
    pub allow_cycles: bool,
    pub require_connected: bool,
    pub validate_attributes: bool,
    pub batch_size: usize,
    pub num_workers: usize,
}

impl Default for GraphBuilderConfig {
    fn default() -> Self {
        Self {
            max_nodes: 1_000_000,
            max_edges: 10_000_000,
            max_density: 0.1,
            allow_cycles: false,
            require_connected: true,
            validate_attributes: true,
            batch_size: 10000,
            num_workers: 4,
        }
    }
}

pub struct GraphBuilder {
    config: GraphBuilderConfig,
    nodes: HashMap<NodeId, Node>,
    edges: Vec<Edge>,
    relation_types: HashSet<RelationType>,
    metrics: Option<Arc<RwLock<GraphMetrics>>>,
}

impl GraphBuilder {
    pub fn new(config: GraphBuilderConfig) -> Self {
        Self {
            config,
            nodes: HashMap::new(),
            edges: Vec::new(),
            relation_types: HashSet::new(),
            metrics: None,
        }
    }
    
    pub fn add_node(&mut self, id: NodeId, attributes: HashMap<String, crate::graph::AttributeValue>) -> GraphResult<()> {
        // Check node limit
        if self.nodes.len() >= self.config.max_nodes {
            return Err(GraphError::GraphTooLarge {
                nodes: self.nodes.len() + 1,
                max_nodes: self.config.max_nodes,
            });
        }
        
        // Validate node ID
        if self.nodes.contains_key(&id) {
            return Err(GraphError::DuplicateNode(id.to_string()));
        }
        
        // Validate attributes if configured
        if self.config.validate_attributes {
            self.validate_node_attributes(&attributes)?;
        }
        
        // Add node
        self.nodes.insert(id.clone(), Node {
            id,
            attributes: attributes.clone(),
            features: None,
        });
        
        // Update metrics
        let num_nodes = self.nodes.len();
        let density = self.calculate_density();
        if let Some(metrics) = &mut self.metrics {
            let mut metrics_guard = metrics.write();
            metrics_guard.num_nodes = num_nodes;
            metrics_guard.density = density;
        }
        
        Ok(())
    }
    
    pub fn add_edge(
        &mut self,
        from: NodeId,
        to: NodeId,
        relation: RelationType,
        attributes: HashMap<String, crate::graph::AttributeValue>,
    ) -> GraphResult<()> {
        // Check edge limit
        if self.edges.len() >= self.config.max_edges {
            return Err(GraphError::InvalidOperation(
                format!("Maximum number of edges ({}) exceeded", self.config.max_edges)
            ));
        }
        
        // Validate nodes exist
        if !self.nodes.contains_key(&from) {
            return Err(GraphError::NodeNotFound(from.to_string()));
        }
        if !self.nodes.contains_key(&to) {
            return Err(GraphError::NodeNotFound(to.to_string()));
        }
        
        // Check for duplicate edge
        if self.has_edge(&from, &to, &relation) {
            return Err(GraphError::DuplicateEdge {
                from: from.to_string(),
                to: to.to_string(),
                relation: format!("{:?}", relation),
            });
        }
        
        // Validate attributes if configured
        if self.config.validate_attributes {
            self.validate_edge_attributes(&attributes)?;
        }
        
        // Check cycle constraints
        if !self.config.allow_cycles && self.would_create_cycle(&from, &to) {
            return Err(GraphError::CycleDetected);
        }
        
        // Add edge
        self.edges.push(Edge {
            from: from.clone(),
            to: to.clone(),
            relation_type: relation.clone(),
            weight: 1.0,
            attributes: attributes.clone(),
        });
        self.relation_types.insert(relation);
        
        // Update node edge lists
        if let Some(_from_node) = self.nodes.get_mut(&from) {
            // Note: Edge tracking is now handled separately
        }
        if let Some(_to_node) = self.nodes.get_mut(&to) {
            // Note: Edge tracking is now handled separately
        }
        
        // Update metrics
        let num_edges = self.edges.len();
        let avg_degree = self.calculate_avg_degree();
        let density = self.calculate_density();
        if let Some(metrics) = &mut self.metrics {
            let mut metrics_guard = metrics.write();
            metrics_guard.num_edges = num_edges;
            metrics_guard.avg_degree = avg_degree;
            metrics_guard.density = density;
        }
        
        // Check density constraint
        if let Some(metrics) = &self.metrics {
            let metrics = metrics.read();
            if metrics.density > self.config.max_density {
                return Err(GraphError::GraphTooDense {
                    density: metrics.density,
                    max_density: self.config.max_density,
                });
            }
        }
        
        Ok(())
    }
    
    pub fn build(self) -> GraphResult<RelationalGraph> {
        // Validate final graph state
        self.validate_final_graph()?;
        
        // Store lengths before moving
        let num_nodes = self.nodes.len();
        let num_edges = self.edges.len();
        
        // Create empty tensors for the new fields
        let device = candle_core::Device::Cpu;
        let node_features = Some(candle_core::Tensor::zeros((num_nodes, 64), candle_core::DType::F32, &device)
            .map_err(|e| GraphError::MemoryError(format!("Failed to create node features: {}", e)))?);
        
        Ok(RelationalGraph {
            nodes: self.nodes,
            edges: self.edges,
            relation_types: self.relation_types,
            node_features,
            edge_features: None,
            global_features: None,
            node_types: HashMap::new(),
            edge_types: HashMap::new(),
            device: device.clone(),
            adjacency: SparseTensor {
                indices: vec![vec![0; num_edges], vec![0; num_edges]],
                values: vec![1.0; num_edges],
                shape: vec![num_nodes, num_nodes],
            },
        })
    }
    
    pub fn build_from_edges(edges: Vec<(String, String, RelationType, HashMap<String, String>)>) -> GraphResult<RelationalGraph> {
        let config = GraphBuilderConfig::default();
        let mut builder = GraphBuilder::new(config);
        
        // First pass: collect all nodes
        let mut node_ids = HashSet::new();
        for (from, to, _, _) in &edges {
            node_ids.insert(NodeId::from_string(from));
            node_ids.insert(NodeId::from_string(to));
        }
        
        // Add nodes
        for id in node_ids {
            builder.add_node(id, HashMap::new())?;
        }
        
        // Add edges in batches
        for chunk in edges.chunks(builder.config.batch_size) {
            for (from, to, relation, attributes) in chunk {
                // Convert String attributes to AttributeValue
                let converted_attributes: HashMap<String, crate::graph::AttributeValue> = attributes
                    .iter()
                    .map(|(k, v)| (k.clone(), crate::graph::AttributeValue::String(v.clone())))
                    .collect();
                builder.add_edge(NodeId::from_string(from), NodeId::from_string(to), relation.clone(), converted_attributes)?;
            }
        }
        
        builder.build()
    }
    
    fn validate_node_attributes(&self, _attributes: &HashMap<String, crate::graph::AttributeValue>) -> GraphResult<()> {
        // Add custom node attribute validation logic here
        Ok(())
    }
    
    fn validate_edge_attributes(&self, _attributes: &HashMap<String, crate::graph::AttributeValue>) -> GraphResult<()> {
        // Add custom edge attribute validation logic here
        Ok(())
    }
    
    fn has_edge(&self, from: &NodeId, to: &NodeId, relation: &RelationType) -> bool {
        self.edges.iter().any(|e| {
            e.from == *from && e.to == *to && e.relation_type == relation.clone()
        })
    }
    
    fn would_create_cycle(&self, from: &NodeId, to: &NodeId) -> bool {
        // Simple DFS to detect cycles
        let mut visited = HashSet::new();
        let mut stack = vec![to.clone()];
        
        while let Some(current) = stack.pop() {
            if current == *from {
                return true;
            }
            
            if visited.insert(current.clone()) {
                // Find edges where current is the source
                for edge in &self.edges {
                    if edge.from == current {
                        stack.push(edge.to.clone());
                    }
                }
            }
        }
        
        false
    }
    
    fn calculate_avg_degree(&self) -> f64 {
        if self.nodes.is_empty() {
            return 0.0;
        }
        (2 * self.edges.len()) as f64 / self.nodes.len() as f64
    }
    
    fn calculate_density(&self) -> f64 {
        let n = self.nodes.len();
        if n <= 1 {
            return 0.0;
        }
        let max_edges = n * (n - 1) / 2;
        self.edges.len() as f64 / max_edges as f64
    }
    
    fn validate_final_graph(&self) -> GraphResult<()> {
        // Check if graph is empty
        if self.nodes.is_empty() {
            return Err(GraphError::ValidationError("Graph has no nodes".to_string()));
        }
        
        // Check connectivity if required
        if self.config.require_connected && !self.is_connected() {
            return Err(GraphError::Disconnected);
        }
        
        Ok(())
    }
    
    fn is_connected(&self) -> bool {
        if self.nodes.is_empty() {
            return true;
        }
        
        let start_node = self.nodes.keys().next().unwrap();
        let mut visited = HashSet::new();
        let mut stack = vec![start_node.clone()];
        
        while let Some(current) = stack.pop() {
            if visited.insert(current.clone()) {
                // Find all edges connected to current node
                for edge in &self.edges {
                    if edge.from == current {
                        stack.push(edge.to.clone());
                    } else if edge.to == current {
                        stack.push(edge.from.clone());
                    }
                }
            }
        }
        
        visited.len() == self.nodes.len()
    }
}




#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_graph_builder_basic() {
        let config = GraphBuilderConfig::default();
        let mut builder = GraphBuilder::new(config);
        
        // Add nodes
        builder.add_node(NodeId::from_string("1"), HashMap::new()).unwrap();
        builder.add_node(NodeId::from_string("2"), HashMap::new()).unwrap();
        
        // Add edge
        builder.add_edge(
            NodeId::from_string("1"),
            NodeId::from_string("2"),
            RelationType { name: "CONNECTS_TO".to_string(), attributes: HashMap::new() },
            HashMap::new(),
        ).unwrap();
        
        // Build graph
        let graph = builder.build().unwrap();
        assert_eq!(graph.nodes.len(), 2);
        assert_eq!(graph.edges.len(), 1);
    }
    
    #[test]
    fn test_cycle_detection() {
        let mut config = GraphBuilderConfig::default();
        config.allow_cycles = false;
        let mut builder = GraphBuilder::new(config);
        
        // Create a potential cycle
        builder.add_node(NodeId::from_string("1"), HashMap::new()).unwrap();
        builder.add_node(NodeId::from_string("2"), HashMap::new()).unwrap();
        builder.add_node(NodeId::from_string("3"), HashMap::new()).unwrap();
        
        builder.add_edge(
            NodeId::from_string("1"),
            NodeId::from_string("2"),
            RelationType { name: "CONNECTS_TO".to_string(), attributes: HashMap::new() },
            HashMap::new(),
        ).unwrap();
        
        builder.add_edge(
            NodeId::from_string("2"),
            NodeId::from_string("3"),
            RelationType { name: "CONNECTS_TO".to_string(), attributes: HashMap::new() },
            HashMap::new(),
        ).unwrap();
        
        // This should fail due to cycle creation
        let result = builder.add_edge(
            NodeId::from_string("3"),
            NodeId::from_string("1"),
            RelationType { name: "CONNECTS_TO".to_string(), attributes: HashMap::new() },
            HashMap::new(),
        );
        
        assert!(matches!(result, Err(GraphError::CycleDetected)));
    }
    
    #[test]
    fn test_density_constraint() {
        let mut config = GraphBuilderConfig::default();
        config.max_density = 0.5;
        let mut builder = GraphBuilder::new(config);
        
        // Add nodes
        for i in 0..4 {
            builder.add_node(NodeId::from_string(&i.to_string()), HashMap::new()).unwrap();
        }
        
        // Add edges until we exceed density
        let mut edges = vec![
            ("0", "1"),
            ("1", "2"),
            ("2", "3"),
            ("3", "0"),
            ("0", "2"),  // This should exceed max density
        ];
        
        let mut results = Vec::new();
        for (from, to) in edges {
            results.push(builder.add_edge(
                NodeId::from_string(from),
                NodeId::from_string(to),
                RelationType { name: "CONNECTS_TO".to_string(), attributes: HashMap::new() },
                HashMap::new(),
            ));
        }
        
        // The last edge should fail due to density constraint
        assert!(matches!(results.last().unwrap(), Err(GraphError::GraphTooDense { .. })));
    }
}