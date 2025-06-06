// src/graph/builder.rs
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use parking_lot::RwLock;
use serde::{Serialize, Deserialize};
use crate::graph::{RelationalGraph, Node, Edge, RelationType};
use crate::graph::error::{GraphResult, GraphError, ErrorContext};
use crate::monitoring::metrics::GraphMetrics;

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
    nodes: HashMap<String, Node>,
    edges: Vec<Edge>,
    relation_types: HashSet<RelationType>,
    metrics: Arc<RwLock<GraphMetrics>>,
}

impl GraphBuilder {
    pub fn new(config: GraphBuilderConfig) -> Self {
        Self {
            config,
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
        }
    }
    
    pub fn add_node(&mut self, id: String, attributes: HashMap<String, String>) -> GraphResult<()> {
        // Check node limit
        if self.nodes.len() >= self.config.max_nodes {
            return Err(GraphError::GraphTooLarge {
                nodes: self.nodes.len() + 1,
                max_nodes: self.config.max_nodes,
            });
        }
        
        // Validate node ID
        if self.nodes.contains_key(&id) {
            return Err(GraphError::DuplicateNode(id));
        }
        
        // Validate attributes if configured
        if self.config.validate_attributes {
            self.validate_node_attributes(&attributes)?;
        }
        
        // Add node
        self.nodes.insert(id.clone(), Node {
            id,
            attributes,
            edges: Vec::new(),
        });
        
        // Update metrics
        let mut metrics = self.metrics.write();
        metrics.num_nodes = self.nodes.len();
        metrics.density = self.calculate_density();
        
        Ok(())
    }
    
    pub fn add_edge(
        &mut self,
        from: String,
        to: String,
        relation: RelationType,
        attributes: HashMap<String, String>,
    ) -> GraphResult<()> {
        // Check edge limit
        if self.edges.len() >= self.config.max_edges {
            return Err(GraphError::InvalidOperation(
                format!("Maximum number of edges ({}) exceeded", self.config.max_edges)
            ));
        }
        
        // Validate nodes exist
        if !self.nodes.contains_key(&from) {
            return Err(GraphError::NodeNotFound(from));
        }
        if !self.nodes.contains_key(&to) {
            return Err(GraphError::NodeNotFound(to));
        }
        
        // Check for duplicate edge
        if self.has_edge(&from, &to, &relation) {
            return Err(GraphError::DuplicateEdge {
                from,
                to,
                relation: relation.to_string(),
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
        let edge = Edge {
            from: from.clone(),
            to: to.clone(),
            relation: relation.clone(),
            attributes,
        };
        
        self.edges.push(edge);
        self.relation_types.insert(relation);
        
        // Update node edge lists
        if let Some(node) = self.nodes.get_mut(&from) {
            node.edges.push(self.edges.len() - 1);
        }
        if let Some(node) = self.nodes.get_mut(&to) {
            node.edges.push(self.edges.len() - 1);
        }
        
        // Update metrics
        let mut metrics = self.metrics.write();
        metrics.num_edges = self.edges.len();
        metrics.avg_degree = self.calculate_avg_degree();
        metrics.density = self.calculate_density();
        
        // Check density constraint
        if metrics.density > self.config.max_density {
            return Err(GraphError::GraphTooDense {
                density: metrics.density,
                max_density: self.config.max_density,
            });
        }
        
        Ok(())
    }
    
    pub fn build(self) -> GraphResult<RelationalGraph> {
        // Validate final graph state
        self.validate_final_graph()?;
        
        Ok(RelationalGraph {
            nodes: self.nodes,
            edges: self.edges,
            relation_types: self.relation_types,
            metrics: self.metrics,
        })
    }
    
    pub fn build_from_edges(edges: Vec<(String, String, RelationType, HashMap<String, String>)>) -> GraphResult<RelationalGraph> {
        let config = GraphBuilderConfig::default();
        let mut builder = GraphBuilder::new(config);
        
        // First pass: collect all nodes
        let mut node_ids = HashSet::new();
        for (from, to, _, _) in &edges {
            node_ids.insert(from.clone());
            node_ids.insert(to.clone());
        }
        
        // Add nodes
        for id in node_ids {
            builder.add_node(id, HashMap::new())?;
        }
        
        // Add edges in batches
        for chunk in edges.chunks(builder.config.batch_size) {
            for (from, to, relation, attributes) in chunk {
                builder.add_edge(from.clone(), to.clone(), relation.clone(), attributes.clone())?;
            }
        }
        
        builder.build()
    }
    
    fn validate_node_attributes(&self, attributes: &HashMap<String, String>) -> GraphResult<()> {
        // Add custom node attribute validation logic here
        Ok(())
    }
    
    fn validate_edge_attributes(&self, attributes: &HashMap<String, String>) -> GraphResult<()> {
        // Add custom edge attribute validation logic here
        Ok(())
    }
    
    fn has_edge(&self, from: &str, to: &str, relation: &RelationType) -> bool {
        self.edges.iter().any(|e| {
            e.from == from && e.to == to && e.relation == *relation
        })
    }
    
    fn would_create_cycle(&self, from: &str, to: &str) -> bool {
        // Simple DFS to detect cycles
        let mut visited = HashSet::new();
        let mut stack = vec![to];
        
        while let Some(current) = stack.pop() {
            if current == from {
                return true;
            }
            
            if visited.insert(current) {
                if let Some(node) = self.nodes.get(current) {
                    for &edge_idx in &node.edges {
                        let edge = &self.edges[edge_idx];
                        if edge.from == current {
                            stack.push(&edge.to);
                        }
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
        let mut stack = vec![start_node];
        
        while let Some(current) = stack.pop() {
            if visited.insert(current) {
                if let Some(node) = self.nodes.get(current) {
                    for &edge_idx in &node.edges {
                        let edge = &self.edges[edge_idx];
                        if edge.from == *current {
                            stack.push(&edge.to);
                        } else {
                            stack.push(&edge.from);
                        }
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
        builder.add_node("1".to_string(), HashMap::new()).unwrap();
        builder.add_node("2".to_string(), HashMap::new()).unwrap();
        
        // Add edge
        builder.add_edge(
            "1".to_string(),
            "2".to_string(),
            RelationType::new("CONNECTS_TO"),
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
        builder.add_node("1".to_string(), HashMap::new()).unwrap();
        builder.add_node("2".to_string(), HashMap::new()).unwrap();
        builder.add_node("3".to_string(), HashMap::new()).unwrap();
        
        builder.add_edge(
            "1".to_string(),
            "2".to_string(),
            RelationType::new("CONNECTS_TO"),
            HashMap::new(),
        ).unwrap();
        
        builder.add_edge(
            "2".to_string(),
            "3".to_string(),
            RelationType::new("CONNECTS_TO"),
            HashMap::new(),
        ).unwrap();
        
        // This should fail due to cycle creation
        let result = builder.add_edge(
            "3".to_string(),
            "1".to_string(),
            RelationType::new("CONNECTS_TO"),
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
            builder.add_node(i.to_string(), HashMap::new()).unwrap();
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
                from.to_string(),
                to.to_string(),
                RelationType::new("CONNECTS_TO"),
                HashMap::new(),
            ));
        }
        
        // The last edge should fail due to density constraint
        assert!(matches!(results.last().unwrap(), Err(GraphError::GraphTooDense { .. })));
    }
}