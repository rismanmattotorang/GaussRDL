// src/training/trainer.rs
use candle_core::Device;
use std::time::Instant;
use tracing::{info, warn};
use crate::{Result, model::RelgtModel, data::TrainingBatch};

pub struct RelgtTrainer {
    model: RelgtModel,
    config: super::TrainingConfig,
    device: Device,
    current_epoch: usize,
    best_val_score: f64,
    patience_counter: usize,
}

impl RelgtTrainer {
    pub fn new(
        model: RelgtModel,
        config: super::TrainingConfig,
        device: Device,
    ) -> Result<Self> {
        let best_val_score = match config.early_stopping_mode.as_str() {
            "min" => f64::INFINITY,
            "max" => f64::NEG_INFINITY,
            _ => f64::INFINITY,
        };
        
        Ok(Self {
            model,
            config,
            device,
            current_epoch: 0,
            best_val_score,
            patience_counter: 0,
        })
    }
    
    pub async fn train(
        &mut self,
        train_data: Vec<TrainingBatch>,
        val_data: Vec<TrainingBatch>,
    ) -> Result<super::TrainingResult> {
        info!("Starting RELGT training with {} epochs", self.config.epochs);
        let start_time = Instant::now();
        
        for epoch in 0..self.config.epochs {
            self.current_epoch = epoch;
            
            // Training phase
            let train_loss = self.train_epoch(&train_data).await?;
            
            // Validation phase
            let val_loss = if epoch % self.config.val_freq == 0 {
                Some(self.validate_epoch(&val_data).await?)
            } else {
                None
            };
            
            info!("Epoch {}: train_loss={:.4}, val_loss={:.4}", 
                epoch, train_loss, val_loss.unwrap_or(0.0));
            
            // Early stopping check
            if let Some(val_score) = val_loss {
                let improved = match self.config.early_stopping_mode.as_str() {
                    "min" => val_score < self.best_val_score,
                    "max" => val_score > self.best_val_score,
                    _ => val_score < self.best_val_score,
                };
                
                if improved {
                    self.best_val_score = val_score;
                    self.patience_counter = 0;
                } else {
                    self.patience_counter += 1;
                    
                    if self.patience_counter >= self.config.patience {
                        warn!("Early stopping triggered after {} epochs", epoch + 1);
                        break;
                    }
                }
            }
        }
        
        let total_time = start_time.elapsed();
        info!("Training completed in {:?}", total_time);
        
        Ok(super::TrainingResult {
            best_val_score: self.best_val_score,
            best_epoch: self.current_epoch - self.patience_counter,
            total_time,
            final_train_loss: 0.0, // Would track actual final loss
            final_val_loss: self.best_val_score,
            convergence_epoch: None,
        })
    }
    
    async fn train_epoch(&mut self, train_data: &[TrainingBatch]) -> Result<f64> {
        let mut total_loss = 0.0;
        
        for batch in train_data {
            // Simplified training step
            let _predictions = self.model.forward(
                &batch.graph,
                &batch.seed_nodes,
                &batch.subgraphs,
                &batch.timestamps,
            ).map_err(crate::GaussRelgtError::from)?;
            
            // Would compute actual loss and backpropagate
            total_loss += 0.5; // Dummy loss
        }
        
        Ok(total_loss / train_data.len() as f64)
    }
    
    async fn validate_epoch(&self, val_data: &[TrainingBatch]) -> Result<f64> {
        let mut total_loss = 0.0;
        
        for batch in val_data {
            // Simplified validation step
            let _predictions = self.model.forward(
                &batch.graph,
                &batch.seed_nodes,
                &batch.subgraphs,
                &batch.timestamps,
            ).map_err(crate::GaussRelgtError::from)?;
            
            // Would compute actual validation loss
            total_loss += 0.4; // Dummy loss
        }
        
        Ok(total_loss / val_data.len() as f64)
    }
}