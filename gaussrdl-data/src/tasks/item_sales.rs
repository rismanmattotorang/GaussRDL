use gaussrdl_core::{Dataset, Database, Table};
use gaussrdl_core::Result;
use std::time::Duration;
use chrono::{DateTime, Utc};

#[derive(Debug)]
pub struct ItemSalesTaskProvider {
    base: crate::tasks::BaseTaskProvider,
}

impl ItemSalesTaskProvider {
    pub fn new(dataset_name: &str) -> Self {
        Self {
            base: crate::tasks::BaseTaskProvider::new(
                "item-sales",
                "Predicts item sales volume",
                crate::tasks::TaskType::Entity,
                dataset_name,
                vec!["MAE".to_string(), "RMSE".to_string()],
            ),
        }
    }
}

impl crate::tasks::TaskProvider for ItemSalesTaskProvider {
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
        let mut task = ItemSalesTask::new(self.dataset_name());
        task.load_dataset(_config)?;
        Ok(Box::new(task))
    }
}

pub struct ItemSalesTask {
    dataset_name: String,
    dataset: Option<Box<dyn Dataset>>,
    timedelta: Duration,
    num_eval_timestamps: usize,
}

impl ItemSalesTask {
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
            "rel-hm" => Box::new(crate::datasets::RelHMDataset::new()),
            _ => return Err(gaussrdl_core::Error::dataset(format!(
                "Unsupported dataset: {}", self.dataset_name
            ))),
        };
        
        self.dataset = Some(dataset);
        Ok(())
    }
}

impl crate::tasks::Task for ItemSalesTask {
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
        // Create a basic table for item sales prediction
        use polars::prelude::*;
        let df = df! {
            "item_id" => [1u32, 2, 3, 4, 5],
            "features" => [0.8f64, 0.6, 0.9, 0.4, 0.7],
            "sales_volume" => [100i32, 75, 120, 50, 90],
        }?;
        
        Ok(Table::new(df))
    }

    fn evaluate(&self, predictions: &[f64], _metrics: Option<Vec<Box<dyn crate::metrics::Metric>>>) -> Result<Vec<f64>> {
        if predictions.is_empty() {
            return Ok(vec![0.0]);
        }
        
        // Mock actual sales volumes
        let actual_sales = vec![100.0, 75.0, 120.0, 50.0, 90.0];
        let actual_sales = &actual_sales[..predictions.len().min(actual_sales.len())];
        
        // Calculate Mean Absolute Error
        let mut total_error = 0.0;
        for (pred, actual) in predictions.iter().zip(actual_sales.iter()) {
            total_error += (pred - actual).abs();
        }
        
        let mae = total_error / predictions.len() as f64;
        Ok(vec![mae])
    }
} 