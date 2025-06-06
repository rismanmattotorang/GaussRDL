use std::path::Path;
use serde::{Serialize, Deserialize};
use crate::error::Result;
use crate::base::Table;

/// Model trait defining common functionality for all models
pub trait ModelTrait: Send + Sync {
    /// Get the model name
    fn name(&self) -> &str;
    
    /// Get the model version
    fn version(&self) -> &str;
    
    /// Save the model to a file
    fn save<P: AsRef<Path>>(&self, path: P) -> Result<()>;
    
    /// Load the model from a file
    fn load<P: AsRef<Path>>(path: P) -> Result<Self> where Self: Sized;
    
    /// Make predictions on a table
    fn predict(&self, table: &Table) -> Result<Vec<f64>>;
    
    /// Make predictions from JSON inputs
    fn predict_json(&self, inputs: &[serde_json::Value]) -> Result<Vec<f64>>;
}

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub name: String,
    pub version: String,
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub dropout: f32,
    pub use_layer_norm: bool,
    pub use_residual: bool,
    pub activation: String,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            name: "base".to_string(),
            version: "1.0.0".to_string(),
            hidden_dim: 256,
            num_layers: 3,
            dropout: 0.1,
            use_layer_norm: true,
            use_residual: true,
            activation: "relu".to_string(),
        }
    }
}

/// Base model implementation
pub struct Model {
    config: ModelConfig,
}

impl Model {
    /// Create a new model
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelTrait for Model {
    fn name(&self) -> &str {
        &self.config.name
    }
    
    fn version(&self) -> &str {
        &self.config.version
    }
    
    fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = std::fs::File::create(path)?;
        serde_json::to_writer_pretty(file, &self.config)?;
        Ok(())
    }
    
    fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        let config = serde_json::from_reader(file)?;
        Ok(Self::new(config))
    }
    
    fn predict(&self, _table: &Table) -> Result<Vec<f64>> {
        // Implement prediction logic
        unimplemented!()
    }
    
    fn predict_json(&self, _inputs: &[serde_json::Value]) -> Result<Vec<f64>> {
        // Implement JSON prediction logic
        unimplemented!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    
    #[test]
    fn test_model_config() {
        let config = ModelConfig::default();
        assert_eq!(config.name, "base");
        assert_eq!(config.version, "1.0.0");
        assert_eq!(config.hidden_dim, 256);
    }
    
    #[test]
    fn test_model_save_load() -> Result<()> {
        let temp_dir = tempdir()?;
        let model_path = temp_dir.path().join("model.json");
        
        let config = ModelConfig::default();
        let model = Model::new(config.clone());
        
        // Save model
        model.save(&model_path)?;
        
        // Load model
        let loaded = Model::load(&model_path)?;
        assert_eq!(loaded.name(), model.name());
        assert_eq!(loaded.version(), model.version());
        
        Ok(())
    }
} 