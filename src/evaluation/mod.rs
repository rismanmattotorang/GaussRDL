// src/evaluation/mod.rs
use std::collections::HashMap;
use crate::Result;

pub mod evaluator;
pub mod benchmark;

pub use evaluator::*;
pub use benchmark::*;

#[derive(Debug, Clone)]
pub struct EvaluationConfig {
    pub batch_size: usize,
    pub detailed_analysis: bool,
    pub generate_visualizations: bool,
    pub export_predictions: bool,
    pub output_path: Option<std::path::PathBuf>,
    pub compute_feature_importance: bool,
    pub compute_attention_maps: bool,
}

impl Default for EvaluationConfig {
    fn default() -> Self {
        Self {
            batch_size: 512,
            detailed_analysis: false,
            generate_visualizations: false,
            export_predictions: false,
            output_path: None,
            compute_feature_importance: false,
            compute_attention_maps: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EvaluationResult {
    pub accuracy: f64,
    pub auc: f64,
    pub f1_score: f64,
    pub precision: f64,
    pub recall: f64,
    pub mae: f64,
    pub rmse: f64,
}

impl EvaluationResult {
    pub fn print_summary(&self) {
        println!("=== RELGT Evaluation Results ===");
        println!("Accuracy: {:.4}", self.accuracy);
        println!("AUC: {:.4}", self.auc);
        println!("F1 Score: {:.4}", self.f1_score);
        println!("Precision: {:.4}", self.precision);
        println!("Recall: {:.4}", self.recall);
        if self.mae > 0.0 {
            println!("MAE: {:.4}", self.mae);
            println!("RMSE: {:.4}", self.rmse);
        }
    }
    
    pub fn save_to_file(&self, path: &std::path::Path) -> Result<()> {
        let json_content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json_content)?;
        Ok(())
    }
}