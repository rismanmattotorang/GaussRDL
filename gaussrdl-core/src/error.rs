use std::backtrace::Backtrace;

/// Comprehensive error type for GaussRDL operations
#[derive(Debug)]
pub enum Error {
    /// IO error with context
    Io {
        message: String,
        // Underlying IO error
        source: std::io::Error,
        backtrace: Backtrace,
    },
    
    /// Serialization error with context
    Serialization {
        message: String,
        // Underlying serde_json error
        source: serde_json::Error,
        backtrace: Backtrace,
    },
    
    /// Deserialization error with context
    Deserialization {
        message: String,
        // Underlying serde_yaml error
        source: serde_yaml::Error,
        backtrace: Backtrace,
    },
    
    /// Database error with context
    Database {
        message: String,
        context: Option<String>,
        backtrace: Backtrace,
    },
    
    /// Validation error with context
    Validation {
        message: String,
        field: Option<String>,
        value: Option<String>,
        backtrace: Backtrace,
    },
    
    /// Configuration error with context
    Configuration {
        message: String,
        field: Option<String>,
        backtrace: Backtrace,
    },
    
    /// Model error with context
    Model {
        message: String,
        model_name: Option<String>,
        backtrace: Backtrace,
    },
    
    /// Training error with context
    Training {
        message: String,
        epoch: Option<usize>,
        step: Option<usize>,
        backtrace: Backtrace,
    },
    
    /// Graph error with context
    Graph {
        message: String,
        node_count: Option<usize>,
        edge_count: Option<usize>,
        backtrace: Backtrace,
    },
    
    /// Data error with context
    Data {
        message: String,
        table_name: Option<String>,
        column_name: Option<String>,
        backtrace: Backtrace,
    },
    
    /// Dataset error with context
    Dataset {
        message: String,
        dataset_name: Option<String>,
        backtrace: Backtrace,
    },
    
    /// Task error with context
    Task {
        message: String,
        task_name: Option<String>,
        backtrace: Backtrace,
    },
    
    /// Metrics error with context
    Metrics {
        message: String,
        metric_name: Option<String>,
        backtrace: Backtrace,
    },
    
    /// Server error with context
    Server {
        message: String,
        status_code: Option<u16>,
        backtrace: Backtrace,
    },
    
    /// CLI error with context
    Cli {
        message: String,
        command: Option<String>,
        backtrace: Backtrace,
    },
    
    /// Network error with context
    Network {
        message: String,
        url: Option<String>,
        backtrace: Backtrace,
    },
    
    /// Timeout error with context
    Timeout {
        message: String,
        duration: Option<std::time::Duration>,
        backtrace: Backtrace,
    },
    
    /// Resource error with context
    Resource {
        message: String,
        resource_type: Option<String>,
        resource_id: Option<String>,
        backtrace: Backtrace,
    },
    
    /// Other error with context
    Other {
        message: String,
        error_type: Option<String>,
        backtrace: Backtrace,
    },
}

impl Error {
    /// Create a database error
    pub fn database(msg: impl Into<String>) -> Self {
        Self::Database {
            message: msg.into(),
            context: None,
            backtrace: Backtrace::capture(),
        }
    }
    
    /// Create a database error with context
    pub fn database_with_context(msg: impl Into<String>, context: impl Into<String>) -> Self {
        Self::Database {
            message: msg.into(),
            context: Some(context.into()),
            backtrace: Backtrace::capture(),
        }
    }
    
    /// Create a validation error
    pub fn validation(msg: impl Into<String>) -> Self {
        Self::Validation {
            message: msg.into(),
            field: None,
            value: None,
            backtrace: Backtrace::capture(),
        }
    }
    
    /// Create a validation error with field and value
    pub fn validation_with_field(msg: impl Into<String>, field: impl Into<String>, value: impl Into<String>) -> Self {
        Self::Validation {
            message: msg.into(),
            field: Some(field.into()),
            value: Some(value.into()),
            backtrace: Backtrace::capture(),
        }
    }
    
    /// Create a configuration error
    pub fn configuration(msg: impl Into<String>) -> Self {
        Self::Configuration {
            message: msg.into(),
            field: None,
            backtrace: Backtrace::capture(),
        }
    }
    
    /// Create a configuration error with field
    pub fn configuration_with_field(msg: impl Into<String>, field: impl Into<String>) -> Self {
        Self::Configuration {
            message: msg.into(),
            field: Some(field.into()),
            backtrace: Backtrace::capture(),
        }
    }
    
    /// Create a model error
    pub fn model(msg: impl Into<String>) -> Self {
        Self::Model {
            message: msg.into(),
            model_name: None,
            backtrace: Backtrace::capture(),
        }
    }
    
    /// Create a model error with model name
    pub fn model_with_name(msg: impl Into<String>, model_name: impl Into<String>) -> Self {
        Self::Model {
            message: msg.into(),
            model_name: Some(model_name.into()),
            backtrace: Backtrace::capture(),
        }
    }
    
    /// Create a training error
    pub fn training(msg: impl Into<String>) -> Self {
        Self::Training {
            message: msg.into(),
            epoch: None,
            step: None,
            backtrace: Backtrace::capture(),
        }
    }
    
    /// Create a training error with epoch and step
    pub fn training_with_progress(msg: impl Into<String>, epoch: usize, step: usize) -> Self {
        Self::Training {
            message: msg.into(),
            epoch: Some(epoch),
            step: Some(step),
            backtrace: Backtrace::capture(),
        }
    }
    
    /// Create a graph error
    pub fn graph(msg: impl Into<String>) -> Self {
        Self::Graph {
            message: msg.into(),
            node_count: None,
            edge_count: None,
            backtrace: Backtrace::capture(),
        }
    }
    
    /// Create a graph error with node and edge counts
    pub fn graph_with_stats(msg: impl Into<String>, node_count: usize, edge_count: usize) -> Self {
        Self::Graph {
            message: msg.into(),
            node_count: Some(node_count),
            edge_count: Some(edge_count),
            backtrace: Backtrace::capture(),
        }
    }
    
    /// Create a data error
    pub fn data(msg: impl Into<String>) -> Self {
        Self::Data {
            message: msg.into(),
            table_name: None,
            column_name: None,
            backtrace: Backtrace::capture(),
        }
    }
    
    /// Create a data error with table and column names
    pub fn data_with_location(msg: impl Into<String>, table_name: impl Into<String>, column_name: impl Into<String>) -> Self {
        Self::Data {
            message: msg.into(),
            table_name: Some(table_name.into()),
            column_name: Some(column_name.into()),
            backtrace: Backtrace::capture(),
        }
    }
    
    /// Create a dataset error
    pub fn dataset(msg: impl Into<String>) -> Self {
        Self::Dataset {
            message: msg.into(),
            dataset_name: None,
            backtrace: Backtrace::capture(),
        }
    }
    
    /// Create a dataset error with dataset name
    pub fn dataset_with_name(msg: impl Into<String>, dataset_name: impl Into<String>) -> Self {
        Self::Dataset {
            message: msg.into(),
            dataset_name: Some(dataset_name.into()),
            backtrace: Backtrace::capture(),
        }
    }
    
    /// Create a task error
    pub fn task(msg: impl Into<String>) -> Self {
        Self::Task {
            message: msg.into(),
            task_name: None,
            backtrace: Backtrace::capture(),
        }
    }
    
    /// Create a task error with task name
    pub fn task_with_name(msg: impl Into<String>, task_name: impl Into<String>) -> Self {
        Self::Task {
            message: msg.into(),
            task_name: Some(task_name.into()),
            backtrace: Backtrace::capture(),
        }
    }
    
    /// Create a metrics error
    pub fn metrics(msg: impl Into<String>) -> Self {
        Self::Metrics {
            message: msg.into(),
            metric_name: None,
            backtrace: Backtrace::capture(),
        }
    }
    
    /// Create a metrics error with metric name
    pub fn metrics_with_name(msg: impl Into<String>, metric_name: impl Into<String>) -> Self {
        Self::Metrics {
            message: msg.into(),
            metric_name: Some(metric_name.into()),
            backtrace: Backtrace::capture(),
        }
    }
    
    /// Create a server error
    pub fn server(msg: impl Into<String>) -> Self {
        Self::Server {
            message: msg.into(),
            status_code: None,
            backtrace: Backtrace::capture(),
        }
    }
    
    /// Create a server error with status code
    pub fn server_with_status(msg: impl Into<String>, status_code: u16) -> Self {
        Self::Server {
            message: msg.into(),
            status_code: Some(status_code),
            backtrace: Backtrace::capture(),
        }
    }
    
    /// Create a CLI error
    pub fn cli(msg: impl Into<String>) -> Self {
        Self::Cli {
            message: msg.into(),
            command: None,
            backtrace: Backtrace::capture(),
        }
    }
    
    /// Create a CLI error with command
    pub fn cli_with_command(msg: impl Into<String>, command: impl Into<String>) -> Self {
        Self::Cli {
            message: msg.into(),
            command: Some(command.into()),
            backtrace: Backtrace::capture(),
        }
    }
    
    /// Create a network error
    pub fn network(msg: impl Into<String>) -> Self {
        Self::Network {
            message: msg.into(),
            url: None,
            backtrace: Backtrace::capture(),
        }
    }
    
    /// Create a network error with URL
    pub fn network_with_url(msg: impl Into<String>, url: impl Into<String>) -> Self {
        Self::Network {
            message: msg.into(),
            url: Some(url.into()),
            backtrace: Backtrace::capture(),
        }
    }
    
    /// Create a timeout error
    pub fn timeout(msg: impl Into<String>) -> Self {
        Self::Timeout {
            message: msg.into(),
            duration: None,
            backtrace: Backtrace::capture(),
        }
    }
    
    /// Create a timeout error with duration
    pub fn timeout_with_duration(msg: impl Into<String>, duration: std::time::Duration) -> Self {
        Self::Timeout {
            message: msg.into(),
            duration: Some(duration),
            backtrace: Backtrace::capture(),
        }
    }
    
    /// Create a resource error
    pub fn resource(msg: impl Into<String>) -> Self {
        Self::Resource {
            message: msg.into(),
            resource_type: None,
            resource_id: None,
            backtrace: Backtrace::capture(),
        }
    }
    
    /// Create a resource error with type and ID
    pub fn resource_with_info(msg: impl Into<String>, resource_type: impl Into<String>, resource_id: impl Into<String>) -> Self {
        Self::Resource {
            message: msg.into(),
            resource_type: Some(resource_type.into()),
            resource_id: Some(resource_id.into()),
            backtrace: Backtrace::capture(),
        }
    }
    
    /// Create an other error
    pub fn other(msg: impl Into<String>) -> Self {
        Self::Other {
            message: msg.into(),
            error_type: None,
            backtrace: Backtrace::capture(),
        }
    }
    
    /// Create an other error with type
    pub fn other_with_type(msg: impl Into<String>, error_type: impl Into<String>) -> Self {
        Self::Other {
            message: msg.into(),
            error_type: Some(error_type.into()),
            backtrace: Backtrace::capture(),
        }
    }
    
    /// Get error context as a string
    pub fn context(&self) -> String {
        match self {
            Self::Database { context, .. } => context.clone().unwrap_or_default(),
            Self::Validation { field, value, .. } => {
                let mut ctx = String::new();
                if let Some(field) = field {
                    ctx.push_str(&format!("field: {}, ", field));
                }
                if let Some(value) = value {
                    ctx.push_str(&format!("value: {}", value));
                }
                ctx
            }
            Self::Configuration { field, .. } => field.clone().unwrap_or_default(),
            Self::Model { model_name, .. } => model_name.clone().unwrap_or_default(),
            Self::Training { epoch, step, .. } => {
                let mut ctx = String::new();
                if let Some(epoch) = epoch {
                    ctx.push_str(&format!("epoch: {}, ", epoch));
                }
                if let Some(step) = step {
                    ctx.push_str(&format!("step: {}", step));
                }
                ctx
            }
            Self::Graph { node_count, edge_count, .. } => {
                let mut ctx = String::new();
                if let Some(node_count) = node_count {
                    ctx.push_str(&format!("nodes: {}, ", node_count));
                }
                if let Some(edge_count) = edge_count {
                    ctx.push_str(&format!("edges: {}", edge_count));
                }
                ctx
            }
            Self::Data { table_name, column_name, .. } => {
                let mut ctx = String::new();
                if let Some(table_name) = table_name {
                    ctx.push_str(&format!("table: {}, ", table_name));
                }
                if let Some(column_name) = column_name {
                    ctx.push_str(&format!("column: {}", column_name));
                }
                ctx
            }
            Self::Dataset { dataset_name, .. } => dataset_name.clone().unwrap_or_default(),
            Self::Task { task_name, .. } => task_name.clone().unwrap_or_default(),
            Self::Metrics { metric_name, .. } => metric_name.clone().unwrap_or_default(),
            Self::Server { status_code, .. } => status_code.map(|s| s.to_string()).unwrap_or_default(),
            Self::Cli { command, .. } => command.clone().unwrap_or_default(),
            Self::Network { url, .. } => url.clone().unwrap_or_default(),
            Self::Timeout { duration, .. } => duration.map(|d| format!("{:?}", d)).unwrap_or_default(),
            Self::Resource { resource_type, resource_id, .. } => {
                let mut ctx = String::new();
                if let Some(resource_type) = resource_type {
                    ctx.push_str(&format!("type: {}, ", resource_type));
                }
                if let Some(resource_id) = resource_id {
                    ctx.push_str(&format!("id: {}", resource_id));
                }
                ctx
            }
            Self::Other { error_type, .. } => error_type.clone().unwrap_or_default(),
            _ => String::new(),
        }
    }
    
    /// Check if this is a validation error
    pub fn is_validation(&self) -> bool {
        matches!(self, Self::Validation { .. })
    }
    
    /// Check if this is a configuration error
    pub fn is_configuration(&self) -> bool {
        matches!(self, Self::Configuration { .. })
    }
    
    /// Check if this is a data error
    pub fn is_data(&self) -> bool {
        matches!(self, Self::Data { .. })
    }
    
    /// Check if this is a network error
    pub fn is_network(&self) -> bool {
        matches!(self, Self::Network { .. })
    }
    
    /// Check if this is a timeout error
    pub fn is_timeout(&self) -> bool {
        matches!(self, Self::Timeout { .. })
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Io { message, .. } => write!(f, "IO error: {}", message),
            Error::Serialization { message, .. } => write!(f, "Serialization error: {}", message),
            Error::Deserialization { message, .. } => write!(f, "Deserialization error: {}", message),
            Error::Database { message, context, .. } => {
                if let Some(ctx) = context {
                    write!(f, "Database error: {} (context: {})", message, ctx)
                } else {
                    write!(f, "Database error: {}", message)
                }
            }
            Error::Validation { message, field, value, .. } => {
                if let (Some(fld), Some(val)) = (field, value) {
                    write!(f, "Validation error: {} (field: {}, value: {})", message, fld, val)
                } else if let Some(fld) = field {
                    write!(f, "Validation error: {} (field: {})", message, fld)
                } else {
                    write!(f, "Validation error: {}", message)
                }
            }
            Error::Configuration { message, field, .. } => {
                if let Some(fld) = field {
                    write!(f, "Configuration error: {} (field: {})", message, fld)
                } else {
                    write!(f, "Configuration error: {}", message)
                }
            }
            Error::Model { message, model_name, .. } => {
                if let Some(name) = model_name {
                    write!(f, "Model error: {} (model: {})", message, name)
                } else {
                    write!(f, "Model error: {}", message)
                }
            }
            Error::Training { message, epoch, step, .. } => {
                match (epoch, step) {
                    (Some(e), Some(s)) => write!(f, "Training error: {} (epoch: {}, step: {})", message, e, s),
                    (Some(e), None) => write!(f, "Training error: {} (epoch: {})", message, e),
                    _ => write!(f, "Training error: {}", message),
                }
            }
            Error::Graph { message, node_count, edge_count, .. } => {
                match (node_count, edge_count) {
                    (Some(n), Some(e)) => write!(f, "Graph error: {} (nodes: {}, edges: {})", message, n, e),
                    (Some(n), None) => write!(f, "Graph error: {} (nodes: {})", message, n),
                    _ => write!(f, "Graph error: {}", message),
                }
            }
            Error::Data { message, table_name, column_name, .. } => {
                match (table_name, column_name) {
                    (Some(t), Some(c)) => write!(f, "Data error: {} (table: {}, column: {})", message, t, c),
                    (Some(t), None) => write!(f, "Data error: {} (table: {})", message, t),
                    _ => write!(f, "Data error: {}", message),
                }
            }
            Error::Dataset { message, dataset_name, .. } => {
                if let Some(name) = dataset_name {
                    write!(f, "Dataset error: {} (dataset: {})", message, name)
                } else {
                    write!(f, "Dataset error: {}", message)
                }
            }
            Error::Task { message, task_name, .. } => {
                if let Some(name) = task_name {
                    write!(f, "Task error: {} (task: {})", message, name)
                } else {
                    write!(f, "Task error: {}", message)
                }
            }
            Error::Metrics { message, metric_name, .. } => {
                if let Some(name) = metric_name {
                    write!(f, "Metrics error: {} (metric: {})", message, name)
                } else {
                    write!(f, "Metrics error: {}", message)
                }
            }
            Error::Server { message, status_code, .. } => {
                if let Some(code) = status_code {
                    write!(f, "Server error: {} (status: {})", message, code)
                } else {
                    write!(f, "Server error: {}", message)
                }
            }
            Error::Cli { message, command, .. } => {
                if let Some(cmd) = command {
                    write!(f, "CLI error: {} (command: {})", message, cmd)
                } else {
                    write!(f, "CLI error: {}", message)
                }
            }
            Error::Network { message, url, .. } => {
                if let Some(u) = url {
                    write!(f, "Network error: {} (url: {})", message, u)
                } else {
                    write!(f, "Network error: {}", message)
                }
            }
            Error::Timeout { message, duration, .. } => {
                if let Some(dur) = duration {
                    write!(f, "Timeout error: {} (duration: {:?})", message, dur)
                } else {
                    write!(f, "Timeout error: {}", message)
                }
            }
            Error::Resource { message, resource_type, resource_id, .. } => {
                match (resource_type, resource_id) {
                    (Some(t), Some(id)) => write!(f, "Resource error: {} (type: {}, id: {})", message, t, id),
                    (Some(t), None) => write!(f, "Resource error: {} (type: {})", message, t),
                    _ => write!(f, "Resource error: {}", message),
                }
            }
            Error::Other { message, error_type, .. } => {
                if let Some(typ) = error_type {
                    write!(f, "Other error: {} (type: {})", message, typ)
                } else {
                    write!(f, "Other error: {}", message)
                }
            }
        }
    }
}

// From implementations for external error types
impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Self::Io {
            message: err.to_string(),
            source: err,
            backtrace: Backtrace::capture(),
        }
    }
}

impl From<serde_json::Error> for Error {
    fn from(err: serde_json::Error) -> Self {
        Self::Serialization {
            message: err.to_string(),
            source: err,
            backtrace: Backtrace::capture(),
        }
    }
}

impl From<serde_yaml::Error> for Error {
    fn from(err: serde_yaml::Error) -> Self {
        Self::Deserialization {
            message: err.to_string(),
            source: err,
            backtrace: Backtrace::capture(),
        }
    }
}

impl From<candle_core::Error> for Error {
    fn from(err: candle_core::Error) -> Self {
        Self::Model {
            message: err.to_string(),
            model_name: None,
            backtrace: Backtrace::capture(),
        }
    }
}

impl From<polars::prelude::PolarsError> for Error {
    fn from(err: polars::prelude::PolarsError) -> Self {
        Self::Data {
            message: err.to_string(),
            table_name: None,
            column_name: None,
            backtrace: Backtrace::capture(),
        }
    }
}

impl From<reqwest::Error> for Error {
    fn from(err: reqwest::Error) -> Self {
        Self::Network {
            message: err.to_string(),
            url: None,
            backtrace: Backtrace::capture(),
        }
    }
}

impl From<std::time::SystemTimeError> for Error {
    fn from(err: std::time::SystemTimeError) -> Self {
        Self::Other {
            message: err.to_string(),
            error_type: Some("SystemTimeError".to_string()),
            backtrace: Backtrace::capture(),
        }
    }
}

/// Validation-specific error type
#[derive(Debug, Clone)]
pub enum ValidationError {
    /// Field validation error
    Field {
        field: String,
        message: String,
        value: Option<String>,
    },
    
    /// Schema validation error
    Schema {
        message: String,
        table_name: Option<String>,
    },
    
    /// Data validation error
    Data {
        message: String,
        row: Option<usize>,
        column: Option<String>,
    },
    
    /// Constraint validation error
    Constraint {
        message: String,
        constraint_name: Option<String>,
    },
}

impl ValidationError {
    /// Create a field validation error
    pub fn field(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Field {
            field: field.into(),
            message: message.into(),
            value: None,
        }
    }
    
    /// Create a field validation error with value
    pub fn field_with_value(field: impl Into<String>, message: impl Into<String>, value: impl Into<String>) -> Self {
        Self::Field {
            field: field.into(),
            message: message.into(),
            value: Some(value.into()),
        }
    }
    
    /// Create a schema validation error
    pub fn schema(message: impl Into<String>) -> Self {
        Self::Schema {
            message: message.into(),
            table_name: None,
        }
    }
    
    /// Create a schema validation error with table name
    pub fn schema_with_table(message: impl Into<String>, table_name: impl Into<String>) -> Self {
        Self::Schema {
            message: message.into(),
            table_name: Some(table_name.into()),
        }
    }
    
    /// Create a data validation error
    pub fn data(message: impl Into<String>) -> Self {
        Self::Data {
            message: message.into(),
            row: None,
            column: None,
        }
    }
    
    /// Create a data validation error with row and column
    pub fn data_with_location(message: impl Into<String>, row: usize, column: impl Into<String>) -> Self {
        Self::Data {
            message: message.into(),
            row: Some(row),
            column: Some(column.into()),
        }
    }
    
    /// Create a constraint validation error
    pub fn constraint(message: impl Into<String>) -> Self {
        Self::Constraint {
            message: message.into(),
            constraint_name: None,
        }
    }
    
    /// Create a constraint validation error with constraint name
    pub fn constraint_with_name(message: impl Into<String>, constraint_name: impl Into<String>) -> Self {
        Self::Constraint {
            message: message.into(),
            constraint_name: Some(constraint_name.into()),
        }
    }
}

/// Result type for GaussRDL operations
pub type Result<T> = std::result::Result<T, Error>;

/// Result type for validation operations
pub type ValidationResult<T> = std::result::Result<T, ValidationError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let db_error = Error::database("Connection failed");
        assert!(db_error.is_data());
        
        let val_error = Error::validation("Invalid value");
        assert!(val_error.is_validation());
        
        let config_error = Error::configuration("Missing field");
        assert!(config_error.is_configuration());
    }

    #[test]
    fn test_error_context() {
        let error = Error::validation_with_field("Invalid value", "age", "150");
        let context = error.context();
        assert!(context.contains("field: age"));
        assert!(context.contains("value: 150"));
    }

    #[test]
    fn test_validation_error() {
        let error = ValidationError::field_with_value("age", "Must be positive", "-5");
        assert!(matches!(error, ValidationError::Field { .. }));
    }

    #[test]
    fn test_from_implementations() {
        let io_error = std::io::Error::new(std::io::ErrorKind::NotFound, "File not found");
        let error: Error = io_error.into();
        assert!(matches!(error, Error::Io { .. }));
    }
} 