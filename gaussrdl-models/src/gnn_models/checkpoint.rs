use std::path::Path;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};
use sha2::{Sha256, Digest};
use crate::Result;
use super::{RelGTModel, UnifiedModelConfig, ModelSummary};

/// Model checkpoint structure for saving and loading unified models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCheckpoint {
    pub version: ModelVersion,
    pub config: UnifiedModelConfig,
    pub model_summary: ModelSummary,
    pub state_dict: HashMap<String, Vec<f32>>, // Serialized tensors
    pub optimizer_state: Option<OptimizerState>,
    pub training_metrics: TrainingMetrics,
}

/// Model version information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
    pub config_hash: String,
    pub timestamp: DateTime<Utc>,
    pub git_commit: Option<String>,
    pub training_step: usize,
}

/// Optimizer state for resuming training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerState {
    pub step: usize,
    pub learning_rate: f64,
    pub momentum_buffers: HashMap<String, Vec<f32>>,
    pub variance_buffers: HashMap<String, Vec<f32>>,
    pub optimizer_type: String,
}

/// Training metrics and progress
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    pub train_loss: f64,
    pub val_loss: f64,
    pub best_val_score: f64,
    pub learning_rate: f64,
    pub epoch: usize,
    pub total_steps: usize,
    pub convergence_metrics: HashMap<String, f64>,
    pub training_time: f64, // in seconds
}

impl ModelCheckpoint {
    /// Create a new checkpoint from a trained model
    pub fn new<M: RelGTModel<Config = UnifiedModelConfig>>(
        model: &M,
        config: &UnifiedModelConfig,
        metrics: &TrainingMetrics,
        optimizer_state: OptimizerState,
    ) -> Result<Self> {
        let version = Self::generate_version(config, metrics.total_steps);
        
        // Create basic state dict entries
        let mut state_dict = HashMap::new();
        state_dict.insert("parameter_count".to_string(), vec![model.parameter_count() as f32]);
        
        let checkpoint = Self {
            version,
            config: config.clone(),
            model_summary: model.summary(),
            state_dict,
            optimizer_state: Some(optimizer_state),
            training_metrics: metrics.clone(),
        };
        
        Ok(checkpoint)
    }
    
    /// Serialize model parameters to a HashMap
    fn serialize_model_state<M: RelGTModel<Config = UnifiedModelConfig>>(model: &M) -> Result<HashMap<String, Vec<f32>>> {
        let mut state_dict = HashMap::new();
        
        // Add model metadata
        state_dict.insert("parameter_count".to_string(), vec![model.parameter_count() as f32]);
        state_dict.insert("memory_usage".to_string(), vec![model.memory_usage() as f32]);
        
        // In a real implementation, we would serialize actual model parameters
        // For now, we'll store model metadata and configuration
        
        Ok(state_dict)
    }
    
    /// Generate version information
    fn generate_version(config: &UnifiedModelConfig, step: usize) -> ModelVersion {
        let mut hasher = Sha256::new();
        let config_str = serde_json::to_string(config).unwrap_or_default();
        hasher.update(config_str.as_bytes());
        let config_hash = format!("{:x}", hasher.finalize());
        
        ModelVersion {
            major: 1,
            minor: 0,
            patch: 0,
            config_hash,
            timestamp: Utc::now(),
            git_commit: std::env::var("GIT_COMMIT").ok(),
            training_step: step,
        }
    }
    
    /// Save checkpoint to file
    pub fn save(&self, path: &Path) -> Result<()> {
        let file = std::fs::File::create(path)?;
        let writer = std::io::BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self)?;
        Ok(())
    }
    
    /// Load checkpoint from file
    pub fn load(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        let checkpoint: Self = serde_json::from_reader(reader)?;
        Ok(checkpoint)
    }
    
    /// Validate checkpoint compatibility
    pub fn validate_compatibility(&self, current_config: &UnifiedModelConfig) -> Result<()> {
        // Check model type compatibility
        if self.config.model_type != current_config.model_type {
            return Err(crate::ModelError::Model(format!(
                "Model type mismatch: checkpoint has {:?}, current is {:?}", 
                self.config.model_type, current_config.model_type
            )));
        }
        
        // Check architectural compatibility
        if self.config.hidden_dim != current_config.hidden_dim {
            return Err(crate::ModelError::Model(format!(
                "Hidden dimension mismatch: checkpoint has {}, current is {}", 
                self.config.hidden_dim, current_config.hidden_dim
            )));
        }
        
        if self.config.num_layers != current_config.num_layers {
            return Err(crate::ModelError::Model(format!(
                "Number of layers mismatch: checkpoint has {}, current is {}", 
                self.config.num_layers, current_config.num_layers
            )));
        }
        
        Ok(())
    }
    
    /// Get checkpoint file size in bytes
    pub fn file_size(&self, path: &Path) -> Result<u64> {
        let metadata = std::fs::metadata(path)?;
        Ok(metadata.len())
    }
    
    /// Get human-readable description
    pub fn description(&self) -> String {
        format!(
            "{:?} model (v{}.{}.{}) - {} parameters - Step {} - {:.3} val_loss",
            self.config.model_type,
            self.version.major,
            self.version.minor, 
            self.version.patch,
            self.model_summary.total_parameters,
            self.version.training_step,
            self.training_metrics.val_loss
        )
    }
}

/// Checkpoint manager for handling multiple checkpoints
pub struct CheckpointManager {
    checkpoint_dir: std::path::PathBuf,
    max_checkpoints: usize,
    save_optimizer_state: bool,
    save_best_only: bool,
}

impl CheckpointManager {
    /// Create a new checkpoint manager
    pub fn new(
        checkpoint_dir: &Path,
        max_checkpoints: usize,
        save_optimizer_state: bool,
        save_best_only: bool,
    ) -> Result<Self> {
        std::fs::create_dir_all(checkpoint_dir)?;
        Ok(Self {
            checkpoint_dir: checkpoint_dir.to_path_buf(),
            max_checkpoints,
            save_optimizer_state,
            save_best_only,
        })
    }
    
    /// Save a new checkpoint
    pub fn save_checkpoint<M: RelGTModel<Config = UnifiedModelConfig>>(
        &self,
        model: &M,
        optimizer_state: Option<OptimizerState>,
        metrics: TrainingMetrics,
    ) -> Result<std::path::PathBuf> {
        let config = model.config();
        
        let checkpoint = ModelCheckpoint::new(
            model,
            config,
            &metrics,
            optimizer_state.unwrap_or_else(|| OptimizerState {
                step: 0,
                learning_rate: 0.001,
                momentum_buffers: HashMap::new(),
                variance_buffers: HashMap::new(),
                optimizer_type: "adam".to_string(),
            }),
        )?;
        
        let checkpoint_path = self.checkpoint_dir.join(format!(
            "checkpoint_step_{}.json",
            checkpoint.version.training_step
        ));
        
        checkpoint.save(&checkpoint_path)?;
        
        // Save as latest checkpoint
        let latest_path = self.checkpoint_dir.join("latest.json");
        checkpoint.save(&latest_path)?;
        
        // Save as best if applicable
        if self.is_best_checkpoint(&checkpoint)? {
            let best_path = self.checkpoint_dir.join("best.json");
            checkpoint.save(&best_path)?;
        }
        
        self.cleanup_old_checkpoints()?;
        
        Ok(checkpoint_path)
    }
    
    /// Load the latest checkpoint
    pub fn load_latest_checkpoint(&self) -> Result<Option<ModelCheckpoint>> {
        let latest_path = self.checkpoint_dir.join("latest.json");
        if latest_path.exists() {
            Ok(Some(ModelCheckpoint::load(&latest_path)?))
        } else {
            Ok(None)
        }
    }
    
    /// Load the best checkpoint
    pub fn load_best_checkpoint(&self) -> Result<Option<ModelCheckpoint>> {
        let best_path = self.checkpoint_dir.join("best.json");
        if best_path.exists() {
            Ok(Some(ModelCheckpoint::load(&best_path)?))
        } else {
            Ok(None)
        }
    }
    
    /// List all available checkpoints
    pub fn list_checkpoints(&self) -> Result<Vec<ModelCheckpoint>> {
        let mut checkpoints = Vec::new();
        
        for entry in std::fs::read_dir(&self.checkpoint_dir)? {
            let entry = entry?;
            let path = entry.path();
            
            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                if let Ok(checkpoint) = ModelCheckpoint::load(&path) {
                    checkpoints.push(checkpoint);
                }
            }
        }
        
        // Sort by training step
        checkpoints.sort_by_key(|c| c.version.training_step);
        Ok(checkpoints)
    }
    
    /// Check if this is the best checkpoint so far
    fn is_best_checkpoint(&self, checkpoint: &ModelCheckpoint) -> Result<bool> {
        if let Ok(Some(current_best)) = self.load_best_checkpoint() {
            Ok(checkpoint.training_metrics.best_val_score > current_best.training_metrics.best_val_score)
        } else {
            Ok(true) // First checkpoint is always the best
        }
    }
    
    /// Clean up old checkpoints
    fn cleanup_old_checkpoints(&self) -> Result<()> {
        let mut checkpoint_files: Vec<_> = std::fs::read_dir(&self.checkpoint_dir)?
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                let path = entry.path();
                path.extension().and_then(|s| s.to_str()) == Some("json") &&
                !path.file_name().unwrap().to_str().unwrap().starts_with("latest") &&
                !path.file_name().unwrap().to_str().unwrap().starts_with("best")
            })
            .collect();
        
        if checkpoint_files.len() > self.max_checkpoints {
            checkpoint_files.sort_by_key(|entry| {
                entry.metadata().unwrap().modified().unwrap()
            });
            
            let to_remove = checkpoint_files.len() - self.max_checkpoints;
            for entry in checkpoint_files.iter().take(to_remove) {
                std::fs::remove_file(entry.path())?;
            }
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use candle_core::DType;
    use crate::{UnifiedModelConfig, ModelType};

    #[test]
    fn test_checkpoint_creation() -> Result<()> {
        let config = UnifiedModelConfig::default();
        let metrics = TrainingMetrics {
            train_loss: 0.5,
            val_loss: 0.4,
            best_val_score: 0.8,
            learning_rate: 0.001,
            epoch: 10,
            total_steps: 100,
            convergence_metrics: HashMap::new(),
            training_time: 300.0,
        };
        
        // Would need a real model for full test
        // let checkpoint = ModelCheckpoint::new(&model, None, metrics)?;
        
        Ok(())
    }
    
    #[test]
    fn test_checkpoint_manager() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let manager = CheckpointManager::new(
            temp_dir.path(),
            3,
            true,
            false,
        )?;
        
        assert!(temp_dir.path().exists());
        Ok(())
    }
} 