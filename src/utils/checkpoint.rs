// src/utils/checkpoint.rs
use std::path::Path;
use crate::{Result, model::RelgtModel};

pub struct CheckpointManager {
    checkpoint_dir: std::path::PathBuf,
}

impl CheckpointManager {
    pub fn new(checkpoint_dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(checkpoint_dir)?;
        Ok(Self {
            checkpoint_dir: checkpoint_dir.to_path_buf(),
        })
    }
    
    pub fn save_checkpoint(&self, model: &RelgtModel, epoch: usize) -> Result<()> {
        let checkpoint_path = self.checkpoint_dir.join(format!("checkpoint_epoch_{}.pt", epoch));
        model.save_checkpoint(&checkpoint_path)?;
        Ok(())
    }
    
    pub fn load_latest_checkpoint(&self) -> Result<Option<std::path::PathBuf>> {
        // Find latest checkpoint file
        let mut checkpoints = Vec::new();
        for entry in std::fs::read_dir(&self.checkpoint_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("pt") {
                checkpoints.push(path);
            }
        }
        
        checkpoints.sort();
        Ok(checkpoints.into_iter().last())
    }
}