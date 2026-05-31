use std::fmt;
use thiserror::Error;
use std::collections::HashMap;

#[derive(Error, Debug)]
pub enum GraphError {
    #[error("Invalid node index: {0}")]
    InvalidNodeIndex(usize),
    
    #[error("Invalid edge index: {0}")]
    InvalidEdgeIndex(usize),
    
    #[error("Invalid relation type: {0}")]
    InvalidRelationType(String),
    
    #[error("Node already exists with ID: {0}")]
    DuplicateNode(String),
    
    #[error("Edge already exists between nodes {from} and {to} with relation {relation}")]
    DuplicateEdge {
        from: String,
        to: String,
        relation: String,
    },
    
    #[error("Node not found with ID: {0}")]
    NodeNotFound(String),
    
    #[error("Edge not found between nodes {from} and {to}")]
    EdgeNotFound {
        from: String,
        to: String,
    },
    
    #[error("Invalid graph operation: {0}")]
    InvalidOperation(String),
    
    #[error("Graph validation error: {0}")]
    ValidationError(String),
    
    #[error("Graph construction error: {0}")]
    ConstructionError(String),
    
    #[error("Memory error: {0}")]
    MemoryError(String),
    
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    
    #[error("Database error: {0}")]
    DatabaseError(String),
    
    #[error("Batch processing error: {0}")]
    BatchError(String),
    
    #[error("Graph is too large: {nodes} nodes (max: {max_nodes})")]
    GraphTooLarge {
        nodes: usize,
        max_nodes: usize,
    },
    
    #[error("Graph is too dense: {density} (max: {max_density})")]
    GraphTooDense {
        density: f64,
        max_density: f64,
    },
    
    #[error("Cycle detected in graph")]
    CycleDetected,
    
    #[error("Graph is disconnected")]
    Disconnected,
    
    #[error("Invalid attribute type: expected {expected}, got {actual}")]
    InvalidAttributeType {
        expected: String,
        actual: String,
    },
    
    #[error("Missing required attribute: {0}")]
    MissingAttribute(String),
    
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
    
    #[error("Graph sampling error: {0}")]
    SamplingError(String),
    
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl GraphError {
    pub fn is_fatal(&self) -> bool {
        matches!(
            self,
            GraphError::MemoryError(_) |
            GraphError::DatabaseError(_) |
            GraphError::GraphTooLarge { .. } |
            GraphError::InvalidConfig(_)
        )
    }
    
    pub fn is_recoverable(&self) -> bool {
        !self.is_fatal()
    }
    
    pub fn error_code(&self) -> &'static str {
        match self {
            GraphError::InvalidNodeIndex(_) => "INVALID_NODE_INDEX",
            GraphError::InvalidEdgeIndex(_) => "INVALID_EDGE_INDEX",
            GraphError::InvalidRelationType(_) => "INVALID_RELATION_TYPE",
            GraphError::DuplicateNode(_) => "DUPLICATE_NODE",
            GraphError::DuplicateEdge { .. } => "DUPLICATE_EDGE",
            GraphError::NodeNotFound(_) => "NODE_NOT_FOUND",
            GraphError::EdgeNotFound { .. } => "EDGE_NOT_FOUND",
            GraphError::InvalidOperation(_) => "INVALID_OPERATION",
            GraphError::ValidationError(_) => "VALIDATION_ERROR",
            GraphError::ConstructionError(_) => "CONSTRUCTION_ERROR",
            GraphError::MemoryError(_) => "MEMORY_ERROR",
            GraphError::IoError(_) => "IO_ERROR",
            GraphError::SerializationError(_) => "SERIALIZATION_ERROR",
            GraphError::DatabaseError(_) => "DATABASE_ERROR",
            GraphError::BatchError(_) => "BATCH_ERROR",
            GraphError::GraphTooLarge { .. } => "GRAPH_TOO_LARGE",
            GraphError::GraphTooDense { .. } => "GRAPH_TOO_DENSE",
            GraphError::CycleDetected => "CYCLE_DETECTED",
            GraphError::Disconnected => "DISCONNECTED",
            GraphError::InvalidAttributeType { .. } => "INVALID_ATTRIBUTE_TYPE",
            GraphError::MissingAttribute(_) => "MISSING_ATTRIBUTE",
            GraphError::InvalidConfig(_) => "INVALID_CONFIG",
            GraphError::SamplingError(_) => "SAMPLING_ERROR",
            GraphError::Other(_) => "UNKNOWN_ERROR",
        }
    }
}

pub type GraphResult<T> = Result<T, GraphError>;

#[derive(Debug)]
pub struct ErrorContext {
    pub operation: String,
    pub node_id: Option<String>,
    pub edge_ids: Option<(String, String)>,
    pub relation_type: Option<String>,
    pub additional_info: Option<String>,
}

impl fmt::Display for ErrorContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Operation: {}", self.operation)?;
        
        if let Some(node_id) = &self.node_id {
            write!(f, ", Node: {}", node_id)?;
        }
        
        if let Some((from, to)) = &self.edge_ids {
            write!(f, ", Edge: {} -> {}", from, to)?;
        }
        
        if let Some(rel_type) = &self.relation_type {
            write!(f, ", Relation: {}", rel_type)?;
        }
        
        if let Some(info) = &self.additional_info {
            write!(f, ", Info: {}", info)?;
        }
        
        Ok(())
    }
}

pub trait ErrorHandler {
    fn handle_error(&self, error: GraphError, context: Option<ErrorContext>) -> GraphResult<()>;
}

#[derive(Default)]
pub struct DefaultErrorHandler;

impl ErrorHandler for DefaultErrorHandler {
    fn handle_error(&self, error: GraphError, context: Option<ErrorContext>) -> GraphResult<()> {
        use tracing::{error, warn};
        
        match &error {
            GraphError::InvalidNodeIndex(_) |
            GraphError::InvalidEdgeIndex(_) |
            GraphError::InvalidRelationType(_) |
            GraphError::DuplicateNode(_) |
            GraphError::DuplicateEdge { .. } |
            GraphError::NodeNotFound(_) |
            GraphError::EdgeNotFound { .. } => {
                warn!(
                    error = %error,
                    context = ?context,
                    code = error.error_code(),
                    "Recoverable graph error occurred"
                );
            }
            _ => {
                error!(
                    error = %error,
                    context = ?context,
                    code = error.error_code(),
                    is_fatal = error.is_fatal(),
                    "Graph error occurred"
                );
            }
        }
        
        Err(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_error_context_display() {
        let context = ErrorContext {
            operation: "add_edge".to_string(),
            node_id: Some("node1".to_string()),
            edge_ids: Some(("node1".to_string(), "node2".to_string())),
            relation_type: Some("CONNECTS_TO".to_string()),
            additional_info: Some("max retries exceeded".to_string()),
        };
        
        let display = context.to_string();
        assert!(display.contains("Operation: add_edge"));
        assert!(display.contains("Node: node1"));
        assert!(display.contains("Edge: node1 -> node2"));
        assert!(display.contains("Relation: CONNECTS_TO"));
        assert!(display.contains("Info: max retries exceeded"));
    }
    
    #[test]
    fn test_error_handling() {
        let handler = DefaultErrorHandler::default();
        let error = GraphError::NodeNotFound("test_node".to_string());
        let context = ErrorContext {
            operation: "get_node".to_string(),
            node_id: Some("test_node".to_string()),
            edge_ids: None,
            relation_type: None,
            additional_info: None,
        };
        
        let result = handler.handle_error(error, Some(context));
        assert!(result.is_err());
    }
    
    #[test]
    fn test_error_properties() {
        let error = GraphError::MemoryError("Out of memory".to_string());
        assert!(error.is_fatal());
        assert!(!error.is_recoverable());
        assert_eq!(error.error_code(), "MEMORY_ERROR");
        
        let error = GraphError::NodeNotFound("test".to_string());
        assert!(!error.is_fatal());
        assert!(error.is_recoverable());
        assert_eq!(error.error_code(), "NODE_NOT_FOUND");
    }
} 