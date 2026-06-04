//! Training monitoring: per-epoch metrics, LR scheduling, early stopping,
//! gradient-norm tracking, and checkpoint save/load.

use crate::error::Result;
use candle_core::backprop::GradStore;
use candle_core::Var;
use serde::{Deserialize, Serialize};

/// Metrics captured at the end of one training epoch.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EpochMetrics {
    pub epoch: usize,
    pub train_loss: f32,
    /// Validation headline metric (ROC-AUC for classification, MAE for regression).
    pub val_metric: f32,
    pub val_loss: f32,
    pub learning_rate: f32,
    /// L2 norm of the parameter gradients this step (training-health signal).
    pub grad_norm: f32,
    pub seconds: f32,
}

/// Full training history plus the best checkpoint selector.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TrainingHistory {
    pub epochs: Vec<EpochMetrics>,
    pub best_epoch: usize,
    pub best_val: f32,
    pub stopped_early: bool,
}

/// Learning-rate schedule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LrSchedule {
    Constant,
    /// Cosine decay from `lr` to `lr * 0.01`.
    Cosine,
    /// Linear warmup (10% of epochs) then cosine decay.
    WarmupCosine,
}

impl LrSchedule {
    /// Learning rate at epoch `t` of `total`.
    pub fn at(&self, base_lr: f64, t: usize, total: usize) -> f64 {
        let total = total.max(1) as f64;
        let t = t as f64;
        match self {
            LrSchedule::Constant => base_lr,
            LrSchedule::Cosine => {
                let min = base_lr * 0.01;
                min + 0.5 * (base_lr - min) * (1.0 + (std::f64::consts::PI * t / total).cos())
            }
            LrSchedule::WarmupCosine => {
                let warmup = (total * 0.1).max(1.0);
                if t < warmup {
                    base_lr * (t + 1.0) / warmup
                } else {
                    let min = base_lr * 0.01;
                    let p = (t - warmup) / (total - warmup).max(1.0);
                    min + 0.5 * (base_lr - min) * (1.0 + (std::f64::consts::PI * p).cos())
                }
            }
        }
    }
}

/// Tracks the best validation score and decides when to stop early.
pub struct EarlyStopper {
    pub patience: usize,
    pub maximize: bool,
    best: f32,
    since_improved: usize,
    pub best_epoch: usize,
}

impl EarlyStopper {
    pub fn new(patience: usize, maximize: bool) -> Self {
        Self {
            patience,
            maximize,
            best: if maximize { f32::NEG_INFINITY } else { f32::INFINITY },
            since_improved: 0,
            best_epoch: 0,
        }
    }

    /// Record a validation score; returns `true` if it improved on the best.
    pub fn update(&mut self, epoch: usize, val: f32) -> bool {
        let improved = if self.maximize { val > self.best } else { val < self.best };
        if improved {
            self.best = val;
            self.best_epoch = epoch;
            self.since_improved = 0;
        } else {
            self.since_improved += 1;
        }
        improved
    }

    pub fn should_stop(&self) -> bool {
        self.patience > 0 && self.since_improved >= self.patience
    }

    pub fn best(&self) -> f32 {
        self.best
    }
}

/// L2 norm of all parameter gradients in `grads`.
pub fn grad_norm(grads: &GradStore, vars: &[Var]) -> Result<f32> {
    let mut total = 0.0f64;
    for v in vars {
        if let Some(g) = grads.get(v.as_tensor()) {
            let sq = g.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
            total += sq;
        }
    }
    Ok((total.sqrt()) as f32)
}
