use gaussrdl_core::{Dataset, Database, Table};
use gaussrdl_core::Result;
use std::time::Duration;
use chrono::{DateTime, Utc};
use polars::prelude::*;

#[derive(Debug)]
pub struct UserChurnTaskProvider {
    base: crate::tasks::BaseTaskProvider,
}

impl UserChurnTaskProvider {
    pub fn new(dataset_name: &str) -> Self {
        Self {
            base: crate::tasks::BaseTaskProvider::new(
                "user-churn",
                "Predicts user churn based on historical data",
                crate::tasks::TaskType::Entity,
                dataset_name,
                vec!["AUROC".to_string(), "MAP".to_string(), "RMSE".to_string()],
            ),
        }
    }
}

impl crate::tasks::TaskProvider for UserChurnTaskProvider {
    fn name(&self) -> &str {
        self.base.name()
    }
    
    fn description(&self) -> &str {
        self.base.description()
    }
    
    fn task_type(&self) -> crate::tasks::TaskType {
        self.base.task_type()
    }
    
    fn metrics(&self) -> Vec<Box<dyn crate::metrics::Metric>> {
        self.base.metrics()
    }
    
    fn dataset_name(&self) -> &str {
        self.base.dataset_name()
    }
    
    fn load(&self, _config: &gaussrdl_core::DownloadConfig, _cache_config: &gaussrdl_core::CacheConfig) -> Result<Box<dyn crate::tasks::Task>> {
        // Create a basic user churn task
        let dataset_name = self.dataset_name();
        let dataset: Box<dyn Dataset> = match dataset_name {
            "rel-amazon" => Box::new(crate::datasets::RelAmazonDataset::new()),
            "rel-f1" => Box::new(crate::datasets::RelF1Dataset::new()),
            "rel-hm" => Box::new(crate::datasets::RelHMDataset::new()),
            _ => return Err(gaussrdl_core::Error::dataset("Unknown dataset")),
        };
        
        Ok(Box::new(UserChurnTask {
            dataset_name: self.dataset_name().to_string(),
            dataset: Some(dataset),
            timedelta: Duration::from_secs(86400), // 1 day
            num_eval_timestamps: 7,
        }))
    }
}

pub struct UserChurnTask {
    dataset_name: String,
    dataset: Option<Box<dyn Dataset>>,
    timedelta: Duration,
    num_eval_timestamps: usize,
}

impl UserChurnTask {
    pub fn new(dataset_name: &str) -> Self {
        Self {
            dataset_name: dataset_name.to_string(),
            dataset: None,
            timedelta: Duration::from_secs(86400), // 1 day
            num_eval_timestamps: 7,
        }
    }
    
    fn load_dataset(&mut self, _config: &gaussrdl_core::DownloadConfig) -> Result<()> {
        let dataset: Box<dyn Dataset> = match self.dataset_name.as_str() {
            "rel-amazon" => Box::new(crate::datasets::RelAmazonDataset::new()),
            _ => return Err(gaussrdl_core::Error::dataset(format!(
                "Unsupported dataset: {}", self.dataset_name
            ))),
        };
        
        // dataset.load(config)?;
        self.dataset = Some(dataset);
        Ok(())
    }
}

impl crate::tasks::Task for UserChurnTask {
    fn task_type(&self) -> crate::tasks::TaskType {
        crate::tasks::TaskType::Entity
    }

    fn dataset(&self) -> &dyn Dataset {
        &**self.dataset.as_ref().unwrap()
    }

    fn timedelta(&self) -> Duration {
        self.timedelta
    }

    fn num_eval_timestamps(&self) -> usize {
        self.num_eval_timestamps
    }

    fn make_table(&self, _db: &Database, _timestamps: Vec<DateTime<Utc>>) -> Result<Table> {
        // Create a mock table for user churn prediction
        let df = df! {
            "user_id" => [1u32, 2, 3, 4, 5],
            "features" => [0.1f64, 0.2, 0.3, 0.4, 0.5],
            "label" => [0i32, 1, 0, 1, 0],
        }?;
        
        Ok(Table::new(df))
    }

    fn evaluate(&self, predictions: &[f64], _metrics: Option<Vec<Box<dyn crate::metrics::Metric>>>) -> Result<Vec<f64>> {
        // Basic evaluation using AUROC and accuracy
        if predictions.is_empty() {
            return Ok(vec![0.0]);
        }
        
        // Mock ground truth labels
        let labels = vec![0.0_f64, 1.0_f64, 0.0_f64, 1.0_f64, 0.0_f64];
        let labels = &labels[..predictions.len().min(labels.len())];
        
        // Calculate simple accuracy
        let mut correct = 0;
        for (pred, label) in predictions.iter().zip(labels.iter()) {
            let pred_class: f64 = if *pred > 0.5 { 1.0 } else { 0.0 };
            let diff: f64 = pred_class - label;
            if diff.abs() < 0.1_f64 {
                correct += 1;
            }
        }
        
        let accuracy = correct as f64 / predictions.len() as f64;
        Ok(vec![accuracy])
    }
} 