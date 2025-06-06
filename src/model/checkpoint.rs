use std::path::Path;
use std::collections::HashMap;
use candle_core::{Tensor, Device, Result as CandleResult};
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};
use sha2::{Sha256, Digest};
use crate::{Result, model::{RelgtModel, RelgtConfig}};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCheckpoint {
    pub version: ModelVersion,
    pub state_dict: HashMap<String, Vec<f32>>, // Serialized tensors
    pub config: RelgtConfig,
    pub optimizer_state: Option<OptimizerState>,
    pub training_metrics: TrainingMetrics,
}

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerState {
    pub step: usize,
    pub momentum_buffers: HashMap<String, Vec<f32>>,
    pub variance_buffers: HashMap<String, Vec<f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    pub train_loss: f64,
    pub val_loss: f64,
    pub best_val_score: f64,
    pub learning_rate: f64,
    pub epoch: usize,
    pub total_steps: usize,
    pub convergence_metrics: HashMap<String, f64>,
}

impl ModelCheckpoint {
    pub fn new(
        model: &RelgtModel,
        optimizer_state: Option<OptimizerState>,
        metrics: TrainingMetrics,
    ) -> Result<Self> {
        let config = model.get_config().clone();
        let state_dict = Self::serialize_state_dict(model)?;
        let version = Self::generate_version(&config, metrics.total_steps);
        
        Ok(Self {
            version,
            state_dict,
            config,
            optimizer_state,
            training_metrics: metrics,
        })
    }
    
    fn serialize_state_dict(model: &RelgtModel) -> Result<HashMap<String, Vec<f32>>> {
        let mut state_dict = HashMap::new();
        // Implementation would serialize model parameters to Vec<f32>
        Ok(state_dict)
    }
    
    fn generate_version(config: &RelgtConfig, step: usize) -> ModelVersion {
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
    
    pub fn save(&self, path: &Path) -> Result<()> {
        let file = std::fs::File::create(path)?;
        let writer = std::io::BufWriter::new(file);
        serde_json::to_writer(writer, self)?;
        Ok(())
    }
    
    pub fn load(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        let checkpoint: Self = serde_json::from_reader(reader)?;
        Ok(checkpoint)
    }
}

pub struct CheckpointManager {
    checkpoint_dir: std::path::PathBuf,
    max_checkpoints: usize,
    save_optimizer_state: bool,
}

impl CheckpointManager {
    pub fn new(
        checkpoint_dir: &Path,
        max_checkpoints: usize,
        save_optimizer_state: bool,
    ) -> Result<Self> {
        std::fs::create_dir_all(checkpoint_dir)?;
        Ok(Self {
            checkpoint_dir: checkpoint_dir.to_path_buf(),
            max_checkpoints,
            save_optimizer_state,
        })
    }
    
    pub fn save_checkpoint(
        &self,
        model: &RelgtModel,
        optimizer_state: Option<OptimizerState>,
        metrics: TrainingMetrics,
    ) -> Result<()> {
        let checkpoint = ModelCheckpoint::new(
            model,
            if self.save_optimizer_state { optimizer_state } else { None },
            metrics,
        )?;
        
        let checkpoint_path = self.checkpoint_dir.join(format!(
            "checkpoint_step_{}.pt",
            checkpoint.version.training_step
        ));
        
        checkpoint.save(&checkpoint_path)?;
        self.cleanup_old_checkpoints()?;
        
        Ok(())
    }
    
    pub fn load_latest_checkpoint(&self) -> Result<Option<ModelCheckpoint>> {
        let mut checkpoints: Vec<_> = std::fs::read_dir(&self.checkpoint_dir)?
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry.path()
                    .extension()
                    .and_then(|s| s.to_str())
                    .map_or(false, |ext| ext == "pt")
            })
            .collect();
        
        checkpoints.sort_by_key(|entry| entry.path());
        
        if let Some(latest) = checkpoints.last() {
            Ok(Some(ModelCheckpoint::load(&latest.path())?))
        } else {
            Ok(None)
        }
    }
    
    fn cleanup_old_checkpoints(&self) -> Result<()> {
        let mut checkpoints: Vec<_> = std::fs::read_dir(&self.checkpoint_dir)?
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry.path()
                    .extension()
                    .and_then(|s| s.to_str())
                    .map_or(false, |ext| ext == "pt")
            })
            .collect();
        
        if checkpoints.len() > self.max_checkpoints {
            checkpoints.sort_by_key(|entry| entry.path());
            for entry in checkpoints.iter().take(checkpoints.len() - self.max_checkpoints) {
                std::fs::remove_file(entry.path())?;
            }
        }
        
        Ok(())
    }
} 