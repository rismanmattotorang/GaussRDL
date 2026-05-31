// src/utils/checkpoint.rs
use std::path::PathBuf;
use serde::{Serialize, Deserialize};
use gaussrdl_core::Result;
use gaussrdl_models::gt_model::relgt::RelgtModel;

/// Checkpoint configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointConfig {
    /// Directory to save checkpoints
    pub checkpoint_dir: PathBuf,
    /// Save interval in steps
    pub save_interval: usize,
    /// Keep only the last N checkpoints
    pub keep_last: usize,
    /// Whether to save optimizer state
    pub save_optimizer: bool,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            checkpoint_dir: PathBuf::from("./checkpoints"),
            save_interval: 1000,
            keep_last: 5,
            save_optimizer: true,
        }
    }
}

/// Checkpoint manager
pub struct CheckpointManager {
    config: CheckpointConfig,
    current_step: usize,
}

impl CheckpointManager {
    /// Create a new checkpoint manager
    pub fn new(config: CheckpointConfig) -> Self {
        Self {
            config,
            current_step: 0,
        }
    }
    
    // TODO: Implement checkpointing for RelgtModel when supported
    /*
    /// Save checkpoint
    pub fn save_checkpoint(&mut self, model: &RelgtModel, step: usize) -> Result<()> {
        if step % self.config.save_interval != 0 {
            return Ok(());
        }
        
        let checkpoint_path = self.config.checkpoint_dir.join(format!("checkpoint_{}.pt", step));
        match model.save_checkpoint(&checkpoint_path) {
            Ok(()) => {
                self.current_step = step;
                self.cleanup_old_checkpoints()?;
                Ok(())
            }
            Err(e) => Err(gaussrdl_core::Error::model(e.to_string())),
        }
    }
    
    /// Load checkpoint
    pub fn load_checkpoint(&self, model: &mut RelgtModel, step: usize) -> Result<()> {
        let checkpoint_path = self.config.checkpoint_dir.join(format!("checkpoint_{}.pt", step));
        match RelgtModel::load_from_checkpoint(&checkpoint_path, model.device().clone()) {
            Ok(new_model) => {
                *model = new_model;
                Ok(())
            }
            Err(e) => Err(gaussrdl_core::Error::model(e.to_string())),
        }
    }
    
    /// Load latest checkpoint
    pub fn load_latest_checkpoint(&self, model: &mut RelgtModel) -> Result<usize> {
        let checkpoints = self.get_checkpoint_files()?;
        if let Some(latest) = checkpoints.last() {
            let step = self.extract_step_from_filename(latest)?;
            self.load_checkpoint(model, step)?;
            Ok(step)
        } else {
            Err(gaussrdl_core::Error::Resource {
                message: "No checkpoints found".to_string(),
                resource_type: None,
                resource_id: None,
                backtrace: std::backtrace::Backtrace::capture(),
            })
        }
    }
    */
    
    /// Get checkpoint files
    fn get_checkpoint_files(&self) -> Result<Vec<PathBuf>> {
        if !self.config.checkpoint_dir.exists() {
            return Ok(vec![]);
        }
        
        let mut files = Vec::new();
        for entry in std::fs::read_dir(&self.config.checkpoint_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() && path.extension().map_or(false, |ext| ext == "pt") {
                files.push(path);
            }
        }
        
        files.sort();
        Ok(files)
    }
    
    /// Extract step from filename
    fn extract_step_from_filename(&self, path: &PathBuf) -> Result<usize> {
        let filename = path.file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| gaussrdl_core::Error::Validation {
                message: "Invalid checkpoint filename".to_string(),
                field: None,
                value: None,
                backtrace: std::backtrace::Backtrace::capture(),
            })?;
        
        if let Some(step_str) = filename.strip_prefix("checkpoint_").and_then(|s| s.strip_suffix(".pt")) {
            step_str.parse().map_err(|_| gaussrdl_core::Error::Validation {
                message: "Invalid step number in filename".to_string(),
                field: None,
                value: None,
                backtrace: std::backtrace::Backtrace::capture(),
            })
        } else {
            Err(gaussrdl_core::Error::Validation {
                message: "Invalid checkpoint filename format".to_string(),
                field: None,
                value: None,
                backtrace: std::backtrace::Backtrace::capture(),
            })
        }
    }
    
    /// Cleanup old checkpoints
    fn cleanup_old_checkpoints(&self) -> Result<()> {
        let checkpoints = self.get_checkpoint_files()?;
        if checkpoints.len() <= self.config.keep_last {
            return Ok(());
        }
        
        let to_remove = checkpoints.len() - self.config.keep_last;
        for checkpoint in checkpoints.into_iter().take(to_remove) {
            std::fs::remove_file(checkpoint)?;
        }
        
        Ok(())
    }
}