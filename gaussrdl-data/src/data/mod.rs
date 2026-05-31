// src/data/mod.rs
use candle_core::Tensor;
use gaussrdl_core::Result;

pub mod loader;
pub mod pipeline;

pub use pipeline::*;

#[derive(Debug, Clone)]
pub struct TrainingBatch {
    pub graph: Tensor, // Simplified for now
    pub seed_nodes: Vec<(usize, u64)>,
    pub subgraphs: Vec<Vec<(usize, u64)>>,
    pub timestamps: Vec<f64>,
    pub targets: Tensor,
    pub batch_size: usize,
}

#[derive(Debug, Clone)]
pub struct TestBatch {
    pub graph: Tensor, // Simplified for now
    pub seed_nodes: Vec<(usize, u64)>,
    pub subgraphs: Vec<Vec<(usize, u64)>>,
    pub timestamps: Vec<f64>,
    pub targets: Tensor,
    pub batch_size: usize,
}

#[derive(Debug, Clone)]
pub struct PreparedData {
    pub train_size: usize,
    pub val_size: usize,
    pub test_size: usize,
    pub num_features: usize,
    pub num_classes: usize,
    pub task_type: String,
}

/// Data loader configuration
#[derive(Debug, Clone)]
pub struct DataLoaderConfig {
    /// Batch size
    pub batch_size: usize,
    
    /// Number of worker threads
    pub num_workers: usize,
    
    /// Whether to shuffle data
    pub shuffle: bool,
    
    /// Whether to use memory mapping
    pub use_mmap: bool,
    
    /// Prefetch size
    pub prefetch_size: usize,
}

impl Default for DataLoaderConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            num_workers: num_cpus::get(),
            shuffle: true,
            use_mmap: true,
            prefetch_size: 2,
        }
    }
}

/// Batch of data
#[derive(Debug)]
pub struct Batch {
    /// Features
    pub features: Vec<f32>,
    /// Labels
    pub labels: Vec<f32>,
    /// Batch size
    pub size: usize,
}

/// Data loader
pub struct DataLoader {
    /// Configuration
    config: DataLoaderConfig,
    
    /// Data indices
    indices: Vec<usize>,
    
    /// Current position
    position: usize,
}

impl DataLoader {
    /// Create new data loader
    pub fn new(config: DataLoaderConfig) -> Result<Self> {
        Ok(Self {
            config,
            indices: Vec::new(),
            position: 0,
        })
    }
    
    /// Load data from file
    pub fn load_data(&mut self, _path: &std::path::Path) -> Result<()> {
        // Initialize indices
        self.indices = (0..1000).collect(); // Placeholder
        if self.config.shuffle {
            use rand::seq::SliceRandom;
            self.indices.shuffle(&mut rand::thread_rng());
        }
        
        Ok(())
    }
    
    /// Get next batch
    pub fn next_batch(&mut self) -> Result<Option<Batch>> {
        if self.position >= self.indices.len() {
            return Ok(None);
        }
        
        let end = (self.position + self.config.batch_size)
            .min(self.indices.len());
            
        let batch_indices = &self.indices[self.position..end];
        let batch = self.load_batch(batch_indices)?;
        
        self.position = end;
        
        Ok(Some(batch))
    }
    
    /// Reset loader
    pub fn reset(&mut self) {
        self.position = 0;
        if self.config.shuffle {
            use rand::seq::SliceRandom;
            self.indices.shuffle(&mut rand::thread_rng());
        }
    }
    
    /// Load batch of data
    fn load_batch(&self, _indices: &[usize]) -> Result<Batch> {
        // Placeholder implementation
        Ok(Batch {
            features: vec![0.0; 32],
            labels: vec![0.0; 32],
            size: 32,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_loader() {
        let config = DataLoaderConfig::default();
        let mut loader = DataLoader::new(config).unwrap();
        assert_eq!(loader.config.batch_size, 32);
    }
}
