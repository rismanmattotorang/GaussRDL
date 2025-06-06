use std::io;
use thiserror::Error;

/// Result type for RelBench operations
pub type Result<T> = std::result::Result<T, Error>;

/// RelBench error types
#[derive(Error, Debug)]
pub enum Error {
    /// IO errors
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
    
    /// Serialization errors
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    /// YAML errors
    #[error("YAML error: {0}")]
    Yaml(#[from] serde_yaml::Error),
    
    /// Polars errors
    #[error("Data frame error: {0}")]
    DataFrame(#[from] polars::error::PolarsError),
    
    /// Dataset errors
    #[error("Dataset error: {0}")]
    Dataset(String),
    
    /// Task errors
    #[error("Task error: {0}")]
    Task(String),
    
    /// Model errors
    #[error("Model error: {0}")]
    Model(String),
    
    /// Training errors
    #[error("Training error: {0}")]
    Training(String),
    
    /// Validation errors
    #[error("Validation error: {0}")]
    Validation(String),
    
    /// Configuration errors
    #[error("Configuration error: {0}")]
    Config(String),
    
    /// Network errors
    #[error("Network error: {0}")]
    Network(String),
    
    /// GPU errors
    #[error("GPU error: {0}")]
    Gpu(String),
    
    /// Memory errors
    #[error("Memory error: {0}")]
    Memory(String),
    
    /// Other errors
    #[error("Other error: {0}")]
    Other(String),
}

impl Error {
    /// Create a new dataset error
    pub fn dataset<T: ToString>(msg: T) -> Self {
        Self::Dataset(msg.to_string())
    }
    
    /// Create a new task error
    pub fn task<T: ToString>(msg: T) -> Self {
        Self::Task(msg.to_string())
    }
    
    /// Create a new model error
    pub fn model<T: ToString>(msg: T) -> Self {
        Self::Model(msg.to_string())
    }
    
    /// Create a new training error
    pub fn training<T: ToString>(msg: T) -> Self {
        Self::Training(msg.to_string())
    }
    
    /// Create a new validation error
    pub fn validation<T: ToString>(msg: T) -> Self {
        Self::Validation(msg.to_string())
    }
    
    /// Create a new configuration error
    pub fn config<T: ToString>(msg: T) -> Self {
        Self::Config(msg.to_string())
    }
    
    /// Create a new network error
    pub fn network<T: ToString>(msg: T) -> Self {
        Self::Network(msg.to_string())
    }
    
    /// Create a new GPU error
    pub fn gpu<T: ToString>(msg: T) -> Self {
        Self::Gpu(msg.to_string())
    }
    
    /// Create a new memory error
    pub fn memory<T: ToString>(msg: T) -> Self {
        Self::Memory(msg.to_string())
    }
    
    /// Create a new other error
    pub fn other<T: ToString>(msg: T) -> Self {
        Self::Other(msg.to_string())
    }
}

/// Result extension trait
pub trait ResultExt<T> {
    /// Adds context to an error
    fn context<C>(self, context: C) -> Result<T>
    where
        C: ToString;
}

impl<T, E> ResultExt<T> for std::result::Result<T, E>
where
    E: std::error::Error + Send + Sync + 'static,
{
    fn context<C>(self, context: C) -> Result<T>
    where
        C: ToString,
    {
        self.map_err(|e| Error::Other(format!("{}: {}", context.to_string(), e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_error_creation() {
        let err = Error::dataset("test error");
        assert!(matches!(err, Error::Dataset(_)));
        
        let err = Error::task("test error");
        assert!(matches!(err, Error::Task(_)));
        
        let err = Error::model("test error");
        assert!(matches!(err, Error::Model(_)));
    }
    
    #[test]
    fn test_error_display() {
        let err = Error::dataset("test error");
        assert_eq!(err.to_string(), "Dataset error: test error");
        
        let err = Error::task("test error");
        assert_eq!(err.to_string(), "Task error: test error");
        
        let err = Error::model("test error");
        assert_eq!(err.to_string(), "Model error: test error");
    }
    
    #[test]
    fn test_error_conversion() {
        let io_err = io::Error::new(io::ErrorKind::NotFound, "file not found");
        let err: Error = io_err.into();
        assert!(matches!(err, Error::Io(_)));
        
        let json_err = serde_json::from_str::<serde_json::Value>("invalid json").unwrap_err();
        let err: Error = json_err.into();
        assert!(matches!(err, Error::Serialization(_)));
    }
} 