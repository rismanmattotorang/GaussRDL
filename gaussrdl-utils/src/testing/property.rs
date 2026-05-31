use proptest::prelude::*;
use crate::error::Result;
use crate::training::{TrainingConfig, LearningRateSchedule};
use crate::data::DataLoaderConfig;
use std::path::PathBuf;

/// Generate valid training configuration
pub fn arb_training_config() -> impl Strategy<Value = TrainingConfig> {
    (
        prop::num::f32::POSITIVE,  // initial_lr
        prop::num::f32::POSITIVE,  // min_lr
        prop::num::f32::POSITIVE,  // lr_decay
        prop::option::of(prop::num::f32::POSITIVE),  // grad_clip
        prop::option::of(1usize..100),  // early_stop_patience
        prop::num::f32::POSITIVE,  // early_stop_min_delta
        1usize..1024,  // batch_size
        1usize..1000,  // epochs
        1usize..32,    // num_workers
    ).prop_map(|(
        initial_lr,
        min_lr,
        lr_decay,
        grad_clip,
        early_stop_patience,
        early_stop_min_delta,
        batch_size,
        epochs,
        num_workers,
    )| {
        TrainingConfig::builder()
            .with_initial_lr(initial_lr)
            .with_min_lr(min_lr)
            .with_lr_decay(lr_decay)
            .with_grad_clip(grad_clip)
            .with_early_stopping(early_stop_patience, early_stop_min_delta)
            .with_batch_size(batch_size)
            .with_epochs(epochs)
            .with_num_workers(num_workers)
            .build()
            .unwrap()
    })
}

/// Generate valid data loader configuration
pub fn arb_data_loader_config() -> impl Strategy<Value = DataLoaderConfig> {
    (
        1usize..1024,  // batch_size
        1usize..32,    // num_workers
        prop::bool::ANY,  // shuffle
        prop::bool::ANY,  // use_mmap
        1usize..16,    // prefetch_size
    ).prop_map(|(
        batch_size,
        num_workers,
        shuffle,
        use_mmap,
        prefetch_size,
    )| {
        DataLoaderConfig {
            batch_size,
            num_workers,
            shuffle,
            use_mmap,
            prefetch_size,
        }
    })
}

/// Generate valid file path
pub fn arb_file_path() -> impl Strategy<Value = PathBuf> {
    "[a-zA-Z0-9_][a-zA-Z0-9_/-]*[a-zA-Z0-9_]".prop_map(PathBuf::from)
}

/// Property tests for training configuration
proptest! {
    #[test]
    fn test_training_config_properties(config in arb_training_config()) {
        // Learning rates should be valid
        prop_assert!(config.initial_lr >= config.min_lr);
        prop_assert!(config.lr_decay > 0.0);
        
        // Batch size should be positive
        prop_assert!(config.batch_size > 0);
        
        // Epochs should be positive
        prop_assert!(config.epochs > 0);
        
        // Workers should be positive
        prop_assert!(config.num_workers > 0);
        
        // Early stopping should be valid
        if let Some(patience) = config.early_stop_patience {
            prop_assert!(patience > 0);
            prop_assert!(config.early_stop_min_delta >= 0.0);
        }
    }
    
    #[test]
    fn test_data_loader_config_properties(config in arb_data_loader_config()) {
        // Batch size should be positive
        prop_assert!(config.batch_size > 0);
        
        // Workers should be positive
        prop_assert!(config.num_workers > 0);
        
        // Prefetch size should be positive
        prop_assert!(config.prefetch_size > 0);
    }
}

/// Property test utilities
pub mod utils {
    use super::*;
    
    /// Test that a function preserves ordering
    pub fn preserves_ordering<F, T>(mut f: F, x: T, y: T) -> bool
    where
        F: FnMut(T) -> T,
        T: PartialOrd + Clone,
    {
        let fx = f(x.clone());
        let fy = f(y.clone());
        (x <= y) == (fx <= fy)
    }
    
    /// Test that a function is idempotent
    pub fn is_idempotent<F, T>(mut f: F, x: T) -> bool
    where
        F: FnMut(T) -> T,
        T: PartialEq + Clone,
    {
        let fx = f(x.clone());
        let ffx = f(fx.clone());
        fx == ffx
    }
    
    /// Test that a function is associative
    pub fn is_associative<F, T>(mut f: F, x: T, y: T, z: T) -> bool
    where
        F: FnMut(T, T) -> T,
        T: PartialEq + Clone,
    {
        let fxy = f(x.clone(), y.clone());
        let fyz = f(y, z.clone());
        f(fxy, z) == f(x, fyz)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    proptest! {
        #[test]
        fn test_ordering_preservation(
            x in prop::num::f32::POSITIVE,
            y in prop::num::f32::POSITIVE
        ) {
            let f = |x: f32| x * 2.0;
            prop_assert!(utils::preserves_ordering(f, x, y));
        }
        
        #[test]
        fn test_idempotence(x in prop::num::f32::POSITIVE) {
            let f = |x: f32| x.abs();
            prop_assert!(utils::is_idempotent(f, x));
        }
        
        #[test]
        fn test_associativity(
            x in prop::num::f32::POSITIVE,
            y in prop::num::f32::POSITIVE,
            z in prop::num::f32::POSITIVE
        ) {
            let f = |x: f32, y: f32| x + y;
            prop_assert!(utils::is_associative(f, x, y, z));
        }
    }
} 