use std::time::Duration;
use crate::error::Result;

/// Trait for learning rate scheduling
pub trait LearningRateScheduler: Send + Sync {
    /// Step the learning rate
    fn step(&mut self, epoch: usize) -> f32;
    
    /// Get current learning rate
    fn get_lr(&self) -> f32;
}

/// Trait for gradient operations
pub trait GradientOperator: Send + Sync {
    /// Apply operation to gradients
    fn apply(&self, gradients: &mut [f32]);
}

/// Trait for model pruning
pub trait ModelPruner: Send + Sync {
    /// Update pruning mask
    fn update(&mut self, params: &[f32], epoch: usize) -> &[bool];
    
    /// Get current sparsity
    fn sparsity(&self) -> f32;
}

/// Trait for data augmentation
pub trait DataAugmenter: Send + Sync {
    /// Augment features
    fn augment_features(&mut self, features: &mut [f32]);
    
    /// Augment edges
    fn augment_edges(&mut self, edges: &mut Vec<(usize, usize)>);
    
    /// Augment nodes
    fn augment_nodes(&mut self, nodes: &mut [bool]);
}

/// Trait for model ensembling
pub trait ModelEnsemble: Send + Sync {
    type Model;
    
    /// Add model to ensemble
    fn add_model(&mut self, model: Self::Model);
    
    /// Get number of models
    fn num_models(&self) -> usize;
    
    /// Get models
    fn models(&self) -> &[Self::Model];
    
    /// Get mutable models
    fn models_mut(&mut self) -> &mut [Self::Model];
}

/// Trait for early stopping
pub trait EarlyStopping: Send + Sync {
    /// Update early stopping state
    fn update(&mut self, value: f32, epoch: usize) -> bool;
    
    /// Get best value
    fn best_value(&self) -> f32;
    
    /// Get best epoch
    fn best_epoch(&self) -> usize;
}

/// Trait for memory optimization
pub trait MemoryOptimizer: Send + Sync {
    /// Track memory usage
    fn track_memory(&mut self, current_memory: usize);
    
    /// Get elapsed time
    fn elapsed(&self) -> Duration;
    
    /// Get peak memory
    fn peak_memory(&self) -> usize;
}

/// Trait for device management
pub trait Device: Send + Sync {
    /// Check if device is available
    fn is_available(&self) -> bool;
    
    /// Get device name
    fn name(&self) -> &str;
    
    /// Get device type
    fn device_type(&self) -> DeviceType;
}

/// Device types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    /// CPU device
    CPU,
    /// CUDA GPU device
    CUDA,
    /// Metal device
    Metal,
} 