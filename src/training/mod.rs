use std::path::PathBuf;
use serde::{Serialize, Deserialize};
use crate::error::Result;
use crate::tasks::Task;
use crate::models::{Model, ModelConfig};

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f32,
    pub workers: Option<usize>,
    pub gpu: bool,
    pub output_dir: Option<PathBuf>,
    pub checkpoint_interval: usize,
    pub early_stopping_patience: usize,
    pub gradient_clip_norm: Option<f32>,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            epochs: 100,
            batch_size: 32,
            learning_rate: 0.001,
            workers: None,
            gpu: false,
            output_dir: None,
            checkpoint_interval: 10,
            early_stopping_patience: 5,
            gradient_clip_norm: Some(1.0),
        }
    }
}

/// Training metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainMetrics {
    pub epoch: usize,
    pub train_loss: f64,
    pub val_loss: f64,
    pub val_metrics: Vec<f64>,
    pub elapsed_time: f64,
}

/// Model trainer
pub struct Trainer {
    task: Box<dyn Task>,
    model: Model,
    config: TrainConfig,
}

impl Trainer {
    /// Create a new trainer
    pub fn new(
        task: Box<dyn Task>,
        model_config: serde_yaml::Value,
        train_config: TrainConfig,
    ) -> Result<Self> {
        let model_config: ModelConfig = serde_yaml::from_value(model_config)?;
        let model = Model::new(model_config);
        
        Ok(Self {
            task,
            model,
            config: train_config,
        })
    }
    
    /// Train the model
    pub fn train(&self) -> Result<()> {
        // Get data
        let train_table = self.task.get_train_table()?;
        let val_table = self.task.get_val_table()?;
        
        // Training loop
        let mut best_val_loss = f64::INFINITY;
        let mut patience_counter = 0;
        
        for epoch in 0..self.config.epochs {
            // Train epoch
            let train_loss = self.train_epoch(&train_table)?;
            
            // Validate
            let val_loss = self.validate(&val_table)?;
            
            // Early stopping
            if val_loss < best_val_loss {
                best_val_loss = val_loss;
                patience_counter = 0;
                
                // Save checkpoint
                if let Some(dir) = &self.config.output_dir {
                    let path = dir.join(format!("model_epoch_{}.json", epoch));
                    self.model.save(path)?;
                }
            } else {
                patience_counter += 1;
                if patience_counter >= self.config.early_stopping_patience {
                    println!("Early stopping triggered");
                    break;
                }
            }
            
            // Log progress
            println!(
                "Epoch {}/{}: train_loss={:.4}, val_loss={:.4}",
                epoch + 1,
                self.config.epochs,
                train_loss,
                val_loss
            );
        }
        
        Ok(())
    }
    
    /// Train one epoch
    fn train_epoch(&self, table: &crate::base::Table) -> Result<f64> {
        // Implement training logic
        unimplemented!()
    }
    
    /// Validate the model
    fn validate(&self, table: &crate::base::Table) -> Result<f64> {
        // Implement validation logic
        unimplemented!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    
    struct MockTask;
    
    impl Task for MockTask {
        fn task_type(&self) -> crate::tasks::TaskType {
            crate::tasks::TaskType::Entity
        }
        
        fn dataset(&self) -> &dyn crate::base::Dataset {
            unimplemented!()
        }
        
        fn timedelta(&self) -> std::time::Duration {
            std::time::Duration::from_secs(86400)
        }
        
        fn num_eval_timestamps(&self) -> usize {
            7
        }
        
        fn make_table(
            &self,
            _db: &crate::base::Database,
            _timestamps: Vec<chrono::DateTime<chrono::Utc>>,
        ) -> Result<crate::base::Table> {
            unimplemented!()
        }
        
        fn evaluate(&self, _predictions: &[f64], _metrics: Option<Vec<Box<dyn crate::metrics::Metric>>>) -> Result<Vec<f64>> {
            unimplemented!()
        }
    }
    
    #[test]
    fn test_trainer_config() {
        let config = TrainConfig::default();
        assert_eq!(config.epochs, 100);
        assert_eq!(config.batch_size, 32);
        assert!((config.learning_rate - 0.001).abs() < f32::EPSILON);
    }
    
    #[test]
    fn test_trainer_creation() -> Result<()> {
        let task = Box::new(MockTask);
        let model_config = serde_yaml::to_value(ModelConfig::default())?;
        let train_config = TrainConfig::default();
        
        let trainer = Trainer::new(task, model_config, train_config)?;
        assert_eq!(trainer.model.name(), "base");
        assert_eq!(trainer.config.epochs, 100);
        
        Ok(())
    }
} 