use crate::error::Result;
use super::traits::*;
use std::time::Duration;

/// Training configuration builder
#[derive(Debug, Clone)]
pub struct TrainingConfigBuilder {
    lr_schedule: Option<Box<dyn LearningRateScheduler>>,
    initial_lr: f32,
    min_lr: f32,
    lr_decay: f32,
    grad_clip: Option<f32>,
    early_stop_patience: Option<usize>,
    early_stop_min_delta: f32,
    pruning: Option<Box<dyn ModelPruner>>,
    device: Option<Box<dyn Device>>,
    augmentation: Option<Box<dyn DataAugmenter>>,
    ensemble: Option<Box<dyn ModelEnsemble<Model = Box<dyn std::any::Any + Send + Sync>>>>,
    memory_optimizer: Option<Box<dyn MemoryOptimizer>>,
    batch_size: usize,
    epochs: usize,
    num_workers: usize,
}

impl Default for TrainingConfigBuilder {
    fn default() -> Self {
        Self {
            lr_schedule: None,
            initial_lr: 0.001,
            min_lr: 1e-6,
            lr_decay: 0.1,
            grad_clip: Some(1.0),
            early_stop_patience: Some(10),
            early_stop_min_delta: 1e-4,
            pruning: None,
            device: None,
            augmentation: None,
            ensemble: None,
            memory_optimizer: None,
            batch_size: 32,
            epochs: 100,
            num_workers: num_cpus::get(),
        }
    }
}

impl TrainingConfigBuilder {
    /// Create new builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Set learning rate scheduler
    pub fn with_lr_scheduler(mut self, scheduler: Box<dyn LearningRateScheduler>) -> Self {
        self.lr_schedule = Some(scheduler);
        self
    }

    /// Set initial learning rate
    pub fn with_initial_lr(mut self, lr: f32) -> Self {
        self.initial_lr = lr;
        self
    }

    /// Set minimum learning rate
    pub fn with_min_lr(mut self, lr: f32) -> Self {
        self.min_lr = lr;
        self
    }

    /// Set learning rate decay
    pub fn with_lr_decay(mut self, decay: f32) -> Self {
        self.lr_decay = decay;
        self
    }

    /// Set gradient clipping
    pub fn with_grad_clip(mut self, clip: Option<f32>) -> Self {
        self.grad_clip = clip;
        self
    }

    /// Set early stopping parameters
    pub fn with_early_stopping(mut self, patience: Option<usize>, min_delta: f32) -> Self {
        self.early_stop_patience = patience;
        self.early_stop_min_delta = min_delta;
        self
    }

    /// Set model pruning
    pub fn with_pruning(mut self, pruner: Box<dyn ModelPruner>) -> Self {
        self.pruning = Some(pruner);
        self
    }

    /// Set device
    pub fn with_device(mut self, device: Box<dyn Device>) -> Self {
        self.device = Some(device);
        self
    }

    /// Set data augmentation
    pub fn with_augmentation(mut self, augmenter: Box<dyn DataAugmenter>) -> Self {
        self.augmentation = Some(augmenter);
        self
    }

    /// Set model ensemble
    pub fn with_ensemble(mut self, ensemble: Box<dyn ModelEnsemble<Model = Box<dyn std::any::Any + Send + Sync>>>) -> Self {
        self.ensemble = Some(ensemble);
        self
    }

    /// Set memory optimizer
    pub fn with_memory_optimizer(mut self, optimizer: Box<dyn MemoryOptimizer>) -> Self {
        self.memory_optimizer = Some(optimizer);
        self
    }

    /// Set batch size
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Set number of epochs
    pub fn with_epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    /// Set number of workers
    pub fn with_num_workers(mut self, workers: usize) -> Self {
        self.num_workers = workers;
        self
    }

    /// Build configuration
    pub fn build(self) -> Result<TrainingConfig> {
        Ok(TrainingConfig {
            lr_schedule: self.lr_schedule,
            initial_lr: self.initial_lr,
            min_lr: self.min_lr,
            lr_decay: self.lr_decay,
            grad_clip: self.grad_clip,
            early_stop_patience: self.early_stop_patience,
            early_stop_min_delta: self.early_stop_min_delta,
            pruning: self.pruning,
            device: self.device,
            augmentation: self.augmentation,
            ensemble: self.ensemble,
            memory_optimizer: self.memory_optimizer,
            batch_size: self.batch_size,
            epochs: self.epochs,
            num_workers: self.num_workers,
        })
    }
}

/// Training configuration
#[derive(Debug)]
pub struct TrainingConfig {
    pub(crate) lr_schedule: Option<Box<dyn LearningRateScheduler>>,
    pub(crate) initial_lr: f32,
    pub(crate) min_lr: f32,
    pub(crate) lr_decay: f32,
    pub(crate) grad_clip: Option<f32>,
    pub(crate) early_stop_patience: Option<usize>,
    pub(crate) early_stop_min_delta: f32,
    pub(crate) pruning: Option<Box<dyn ModelPruner>>,
    pub(crate) device: Option<Box<dyn Device>>,
    pub(crate) augmentation: Option<Box<dyn DataAugmenter>>,
    pub(crate) ensemble: Option<Box<dyn ModelEnsemble<Model = Box<dyn std::any::Any + Send + Sync>>>>,
    pub(crate) memory_optimizer: Option<Box<dyn MemoryOptimizer>>,
    pub(crate) batch_size: usize,
    pub(crate) epochs: usize,
    pub(crate) num_workers: usize,
}

impl TrainingConfig {
    /// Create new builder
    pub fn builder() -> TrainingConfigBuilder {
        TrainingConfigBuilder::new()
    }
} 