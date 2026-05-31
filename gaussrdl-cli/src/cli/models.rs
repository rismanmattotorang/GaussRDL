use gaussrdl_core::{Result, Error, Utc};
use serde::{Serialize, Deserialize};
use std::path::PathBuf;
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub model_type: String,
    pub architecture: String,
    pub parameters: usize,
    pub memory_usage_mb: f64,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub last_modified: chrono::DateTime<chrono::Utc>,
    pub status: ModelStatus,
    pub metrics: Option<ModelMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelStatus {
    Created,
    Training,
    Trained,
    Validated,
    Deployed,
    Archived,
    Error(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetrics {
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub loss: f64,
    pub inference_time_ms: f64,
    pub training_time_hours: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub model_type: String,
    pub architecture: ModelArchitecture,
    pub training: TrainingConfig,
    pub dataset: DatasetConfig,
    pub optimization: OptimizationConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelArchitecture {
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub num_heads: Option<usize>,
    pub dropout: f64,
    pub activation: String,
    pub use_layer_norm: bool,
    pub use_residual: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub learning_rate: f64,
    pub batch_size: usize,
    pub epochs: u32,
    pub optimizer: String,
    pub scheduler: Option<String>,
    pub early_stopping: Option<EarlyStoppingConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyStoppingConfig {
    pub patience: u32,
    pub min_delta: f64,
    pub monitor: String,
    pub mode: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetConfig {
    pub name: String,
    pub split_ratio: (f32, f32, f32), // train, val, test
    pub preprocessing: Vec<String>,
    pub augmentation: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    pub mixed_precision: bool,
    pub gradient_clipping: Option<f64>,
    pub weight_decay: f64,
    pub distributed: bool,
}

pub struct ModelManager {
    models_dir: PathBuf,
    registry: HashMap<String, ModelInfo>,
}

impl ModelManager {
    pub fn new(models_dir: PathBuf) -> Self {
        Self {
            models_dir,
            registry: HashMap::new(),
        }
    }

    pub fn list_models(&self, model_type_filter: Option<&str>, trained_only: bool) -> Vec<&ModelInfo> {
        self.registry
            .values()
            .filter(|model| {
                if let Some(filter_type) = model_type_filter {
                    if model.model_type != filter_type {
                        return false;
                    }
                }
                if trained_only {
                    matches!(model.status, ModelStatus::Trained | ModelStatus::Validated | ModelStatus::Deployed)
                } else {
                    true
                }
            })
            .collect()
    }

    pub fn get_model_info(&self, name: &str) -> Result<&ModelInfo> {
        self.registry
            .get(name)
            .ok_or_else(|| Error::other(&format!("Model '{}' not found", name)))
    }

    pub fn create_model(&mut self, name: String, config: ModelConfig) -> Result<()> {
        if self.registry.contains_key(&name) {
            return Err(Error::other(&format!("Model '{}' already exists", name)));
        }

        let model_info = ModelInfo {
            name: name.clone(),
            model_type: config.model_type.clone(),
            architecture: format!("{}-{}-{}", config.model_type, config.architecture.num_layers, config.architecture.hidden_dim),
            parameters: estimate_parameters(&config),
            memory_usage_mb: estimate_memory_usage(&config),
            created_at: Utc::now(),
            last_modified: Utc::now(),
            status: ModelStatus::Created,
            metrics: None,
        };

        self.registry.insert(name, model_info);
        Ok(())
    }

    pub fn validate_model_config(&self, config: &ModelConfig, dataset: Option<&str>) -> Result<Vec<String>> {
        let mut issues = Vec::new();

        // Validate architecture
        if config.architecture.hidden_dim == 0 {
            issues.push("Hidden dimension must be greater than 0".to_string());
        }

        if config.architecture.num_layers == 0 {
            issues.push("Number of layers must be greater than 0".to_string());
        }

        if config.architecture.dropout < 0.0 || config.architecture.dropout >= 1.0 {
            issues.push("Dropout must be between 0.0 and 1.0".to_string());
        }

        // Validate training config
        if config.training.learning_rate <= 0.0 {
            issues.push("Learning rate must be positive".to_string());
        }

        if config.training.batch_size == 0 {
            issues.push("Batch size must be greater than 0".to_string());
        }

        if config.training.epochs == 0 {
            issues.push("Number of epochs must be greater than 0".to_string());
        }

        // Validate dataset compatibility if provided
        if let Some(_dataset_name) = dataset {
            // TODO: Add dataset-specific validation
        }

        Ok(issues)
    }

    pub fn export_model(&self, name: &str, format: &str, output_path: &PathBuf) -> Result<()> {
        let model_info = self.get_model_info(name)?;

        match format.to_lowercase().as_str() {
            "json" => {
                let json = serde_json::to_string_pretty(model_info)
                    .map_err(|e| Error::other(&format!("JSON serialization failed: {}", e)))?;
                std::fs::write(output_path, json)
                    .map_err(|e| Error::other(&format!("Failed to write file: {}", e)))?;
            }
            "onnx" => {
                // TODO: Implement ONNX export
                return Err(Error::other("ONNX export not yet implemented"));
            }
            _ => {
                return Err(Error::other("Unsupported export format"));
            }
        }

        Ok(())
    }

    pub fn compare_models(&self, model1: &str, model2: &str) -> Result<ModelComparison> {
        let info1 = self.get_model_info(model1)?;
        let info2 = self.get_model_info(model2)?;

        Ok(ModelComparison {
            model1: info1.clone(),
            model2: info2.clone(),
            parameter_diff: info2.parameters as i64 - info1.parameters as i64,
            memory_diff: info2.memory_usage_mb - info1.memory_usage_mb,
            metrics_comparison: compare_metrics(&info1.metrics, &info2.metrics),
        })
    }

    pub fn update_model_status(&mut self, name: &str, status: ModelStatus) -> Result<()> {
        if let Some(model) = self.registry.get_mut(name) {
            model.status = status;
            model.last_modified = Utc::now();
            Ok(())
        } else {
            Err(Error::other(&format!("Model '{}' not found", name)))
        }
    }

    pub fn update_model_metrics(&mut self, name: &str, metrics: ModelMetrics) -> Result<()> {
        if let Some(model) = self.registry.get_mut(name) {
            model.metrics = Some(metrics);
            model.last_modified = Utc::now();
            Ok(())
        } else {
            Err(Error::other(&format!("Model '{}' not found", name)))
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelComparison {
    pub model1: ModelInfo,
    pub model2: ModelInfo,
    pub parameter_diff: i64,
    pub memory_diff: f64,
    pub metrics_comparison: Option<MetricsComparison>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsComparison {
    pub accuracy_diff: f64,
    pub loss_diff: f64,
    pub inference_time_diff: f64,
    pub better_model: String,
}

fn estimate_parameters(config: &ModelConfig) -> usize {
    // Simplified parameter estimation
    let hidden_dim = config.architecture.hidden_dim;
    let num_layers = config.architecture.num_layers;
    
    // Basic estimation: input + hidden layers + output
    let base_params = hidden_dim * hidden_dim * num_layers;
    let input_params = hidden_dim * 1000; // Assume 1000 input features
    let output_params = hidden_dim * 100; // Assume 100 output classes
    
    base_params + input_params + output_params
}

fn estimate_memory_usage(config: &ModelConfig) -> f64 {
    // Simplified memory estimation in MB
    let params = estimate_parameters(config) as f64;
    let batch_size = config.training.batch_size as f64;
    
    // Model weights (4 bytes per parameter) + activations + gradients
    let model_memory = params * 4.0 / (1024.0 * 1024.0);
    let activation_memory = batch_size * config.architecture.hidden_dim as f64 * 4.0 / (1024.0 * 1024.0);
    let gradient_memory = model_memory; // Assuming same size as weights
    
    model_memory + activation_memory + gradient_memory
}

fn compare_metrics(metrics1: &Option<ModelMetrics>, metrics2: &Option<ModelMetrics>) -> Option<MetricsComparison> {
    match (metrics1, metrics2) {
        (Some(m1), Some(m2)) => {
            let accuracy_diff = m2.accuracy - m1.accuracy;
            let loss_diff = m2.loss - m1.loss;
            let inference_time_diff = m2.inference_time_ms - m1.inference_time_ms;
            
            let better_model = if accuracy_diff > 0.0 || (accuracy_diff == 0.0 && loss_diff < 0.0) {
                "model2".to_string()
            } else {
                "model1".to_string()
            };
            
            Some(MetricsComparison {
                accuracy_diff,
                loss_diff,
                inference_time_diff,
                better_model,
            })
        }
        _ => None,
    }
}

pub fn format_model_table(models: &[&ModelInfo]) -> String {
    let mut output = String::new();
    output.push_str("Name\t\tType\t\tStatus\t\tParameters\tMemory(MB)\tAccuracy\n");
    output.push_str("─".repeat(80).as_str());
    output.push('\n');

    for model in models {
        let accuracy = model.metrics
            .as_ref()
            .map(|m| format!("{:.3}", m.accuracy))
            .unwrap_or_else(|| "N/A".to_string());
        
        output.push_str(&format!(
            "{}\t{}\t{:?}\t{}\t\t{:.1}\t\t{}\n",
            model.name,
            model.model_type,
            model.status,
            model.parameters,
            model.memory_usage_mb,
            accuracy
        ));
    }

    output
}

pub fn create_default_config(model_type: &str) -> ModelConfig {
    let architecture = match model_type {
        "gat" => ModelArchitecture {
            hidden_dim: 256,
            num_layers: 3,
            num_heads: Some(8),
            dropout: 0.1,
            activation: "relu".to_string(),
            use_layer_norm: true,
            use_residual: true,
        },
        "base_gnn" => ModelArchitecture {
            hidden_dim: 256,
            num_layers: 3,
            num_heads: None,
            dropout: 0.1,
            activation: "relu".to_string(),
            use_layer_norm: true,
            use_residual: true,
        },
        _ => ModelArchitecture {
            hidden_dim: 128,
            num_layers: 2,
            num_heads: None,
            dropout: 0.1,
            activation: "relu".to_string(),
            use_layer_norm: false,
            use_residual: false,
        },
    };

    ModelConfig {
        model_type: model_type.to_string(),
        architecture,
        training: TrainingConfig {
            learning_rate: 0.001,
            batch_size: 32,
            epochs: 100,
            optimizer: "adam".to_string(),
            scheduler: Some("cosine".to_string()),
            early_stopping: Some(EarlyStoppingConfig {
                patience: 10,
                min_delta: 0.001,
                monitor: "val_loss".to_string(),
                mode: "min".to_string(),
            }),
        },
        dataset: DatasetConfig {
            name: "default".to_string(),
            split_ratio: (0.7, 0.2, 0.1),
            preprocessing: vec!["normalize".to_string()],
            augmentation: None,
        },
        optimization: OptimizationConfig {
            mixed_precision: true,
            gradient_clipping: Some(1.0),
            weight_decay: 0.0001,
            distributed: false,
        },
    }
} 