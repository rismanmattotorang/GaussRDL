// src/model/embeddings.rs
use candle_core::{Device, Tensor, Result as CandleResult};
use std::collections::HashMap;

pub struct MultiModalEmbedding {
    config: super::RelgtConfig,
    device: Device,
}

impl MultiModalEmbedding {
    pub fn new(config: super::RelgtConfig, device: Device) -> Result<Self> {
        Ok(Self { config, device })
    }
    
    pub fn forward(&self, features: &FeatureMap) -> CandleResult<Tensor> {
        let hidden_dim = self.config.hidden_dim;
        // Return dummy embedding
        Tensor::randn(0.0f32, 1.0, (1, hidden_dim), &self.device)
    }
}

#[derive(Debug)]
pub struct FeatureMap {
    pub numeric: Option<Vec<Tensor>>,
    pub categorical: Option<HashMap<String, Tensor>>,
    pub text: Option<Tensor>,
    pub temporal: Option<Tensor>,
}

// src/training/mod.rs
use std::time::Duration;
use std::path::PathBuf;
use crate::Result;

pub mod trainer;
pub mod optimizer;
pub mod scheduler;

pub use trainer::*;
pub use optimizer::*;
pub use scheduler::*;

#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub warmup_steps: usize,
    pub grad_clip: f64,
    pub patience: usize,
    pub val_freq: usize,
    pub save_freq: usize,
    pub mixed_precision: bool,
    pub distributed: bool,
    pub output_dir: PathBuf,
    pub resume_checkpoint: Option<PathBuf>,
    pub early_stopping_metric: String,
    pub early_stopping_mode: String,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 100,
            batch_size: 256,
            learning_rate: 1e-4,
            weight_decay: 0.01,
            warmup_steps: 1000,
            grad_clip: 1.0,
            patience: 10,
            val_freq: 1,
            save_freq: 5,
            mixed_precision: false,
            distributed: false,
            output_dir: "./checkpoints".into(),
            resume_checkpoint: None,
            early_stopping_metric: "val_loss".to_string(),
            early_stopping_mode: "min".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TrainingResult {
    pub best_val_score: f64,
    pub best_epoch: usize,
    pub total_time: Duration,
    pub final_train_loss: f64,
    pub final_val_loss: f64,
    pub convergence_epoch: Option<usize>,
}