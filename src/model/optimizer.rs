// src/training/optimizer.rs
pub struct AdamWOptimizer {
    learning_rate: f64,
    weight_decay: f64,
}

impl AdamWOptimizer {
    pub fn new(learning_rate: f64, weight_decay: f64) -> Self {
        Self { learning_rate, weight_decay }
    }
}

// src/training/scheduler.rs
pub struct CosineAnnealingScheduler {
    base_lr: f64,
    max_epochs: usize,
    warmup_steps: usize,
}

impl CosineAnnealingScheduler {
    pub fn new(base_lr: f64, max_epochs: usize, warmup_steps: usize) -> Self {
        Self { base_lr, max_epochs, warmup_steps }
    }
}