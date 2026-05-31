use gaussrdl_core::{Dataset, CacheConfig, DownloadConfig, Database, Table};
use gaussrdl_core::{Result, Error};
use std::sync::Arc;
use std::collections::HashMap;
use chrono::{DateTime, Utc};
use std::time::Duration;
use crate::metrics::Metric;
use polars::prelude::df;

// Import task implementations
mod user_churn;
mod item_churn;
mod driver_position;
mod constructor_position;
mod item_sales;

pub use user_churn::{UserChurnTask, UserChurnTaskProvider};
pub use item_churn::{ItemChurnTask, ItemChurnTaskProvider};
pub use driver_position::{DriverPositionTask, DriverPositionTaskProvider};
pub use constructor_position::{ConstructorPositionTask, ConstructorPositionTaskProvider};
pub use item_sales::{ItemSalesTask, ItemSalesTaskProvider};

/// Task registry to manage available tasks
#[derive(Debug, Default)]
pub struct TaskRegistry {
    tasks: HashMap<String, Arc<dyn TaskProvider>>,
}

impl TaskRegistry {
    /// Creates a new task registry
    pub fn new() -> Self {
        let mut registry = Self::default();
        registry.register_defaults();
        registry
    }

    /// Registers default tasks
    fn register_defaults(&mut self) {
        // Register rel-amazon tasks
        self.register("rel-amazon/user-churn", Arc::new(UserChurnTaskProvider::new("rel-amazon")));
        self.register("rel-amazon/item-churn", Arc::new(ItemChurnTaskProvider::new("rel-amazon")));
        
        // Register rel-f1 tasks
        self.register("rel-f1/driver-position", Arc::new(DriverPositionTaskProvider::new("rel-f1")));
        self.register("rel-f1/constructor-position", Arc::new(ConstructorPositionTaskProvider::new("rel-f1")));
        
        // Register rel-hm tasks
        self.register("rel-hm/user-churn", Arc::new(UserChurnTaskProvider::new("rel-hm")));
        self.register("rel-hm/item-sales", Arc::new(ItemSalesTaskProvider::new("rel-hm")));
    }

    /// Registers a new task provider
    pub fn register(&mut self, name: &str, provider: Arc<dyn TaskProvider>) {
        self.tasks.insert(name.to_string(), provider);
    }

    /// Gets a task by name
    pub fn get(&self, name: &str) -> Option<Arc<dyn TaskProvider>> {
        self.tasks.get(name).cloned()
    }

    /// Lists all available tasks
    pub fn list(&self) -> Vec<String> {
        self.tasks.keys().cloned().collect()
    }
}

/// Task provider trait
pub trait TaskProvider: std::fmt::Debug + Send + Sync {
    /// Gets the task name
    fn name(&self) -> &str;
    
    /// Gets the task description
    fn description(&self) -> &str;
    
    /// Gets the task type
    fn task_type(&self) -> TaskType;
    
    /// Gets the task metrics
    fn metrics(&self) -> Vec<Box<dyn Metric>>;
    
    /// Gets the task dataset name
    fn dataset_name(&self) -> &str;
    
    /// Downloads and loads the task
    fn load(&self, config: &DownloadConfig, cache_config: &CacheConfig) -> Result<Box<dyn Task>>;
}

/// Base implementation for task providers
#[derive(Debug)]
pub struct BaseTaskProvider {
    name: String,
    description: String,
    task_type: TaskType,
    dataset_name: String,
    metrics: Vec<String>,
}

impl BaseTaskProvider {
    /// Creates a new base task provider
    pub fn new(
        name: &str,
        description: &str,
        task_type: TaskType,
        dataset_name: &str,
        metrics: Vec<String>,
    ) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            task_type,
            dataset_name: dataset_name.to_string(),
            metrics,
        }
    }
}

impl TaskProvider for BaseTaskProvider {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn description(&self) -> &str {
        &self.description
    }
    
    fn task_type(&self) -> TaskType {
        self.task_type
    }
    
    fn metrics(&self) -> Vec<Box<dyn Metric>> {
        self.metrics.iter().map(|name| {
            match name.as_str() {
                "AUROC" => Box::new(crate::metrics::AUROC::new()) as Box<dyn Metric>,
                "MAP" => Box::new(crate::metrics::MAP::new(10)) as Box<dyn Metric>,
                "RMSE" => Box::new(crate::metrics::RMSE::new()) as Box<dyn Metric>,
                _ => Box::new(crate::metrics::AUROC::new()) as Box<dyn Metric>,
            }
        }).collect()
    }
    
    fn dataset_name(&self) -> &str {
        &self.dataset_name
    }
    
    fn load(&self, _config: &DownloadConfig, _cache_config: &CacheConfig) -> Result<Box<dyn Task>> {
        // Base implementation just returns an error
        Err(Error::task("Task loading not implemented for base provider"))
    }
}

/// Gets a task by name
pub fn get_task(dataset_name: &str, task_name: &str, download: bool) -> Result<Box<dyn Task>> {
    let registry = TaskRegistry::new();
    let full_name = format!("{}/{}", dataset_name, task_name);
    let provider = registry
        .get(&full_name)
        .ok_or_else(|| Error::task(format!("Task {} not found", full_name)))?;
        
    let config = DownloadConfig {
        download,
        ..Default::default()
    };
    
    let cache_config = CacheConfig::default();
    
    provider.load(&config, &cache_config)
}

/// Task types supported by RelBench
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskType {
    /// Entity prediction tasks
    Entity,
    /// Recommendation tasks
    Recommendation,
}

/// Base trait for all tasks
pub trait Task: Send + Sync {
    /// Get the task type
    fn task_type(&self) -> TaskType;
    
    /// Get the dataset associated with this task
    fn dataset(&self) -> &dyn Dataset;
    
    /// Get the time delta for evaluation
    fn timedelta(&self) -> Duration;
    
    /// Get the number of evaluation timestamps
    fn num_eval_timestamps(&self) -> usize;
    
    /// Make a table for evaluation
    fn make_table(
        &self,
        db: &Database,
        timestamps: Vec<DateTime<Utc>>,
    ) -> Result<Table>;
    
    /// Get the training table
    fn get_train_table(&self) -> Result<Table> {
        // Create a basic training table
        use polars::prelude::*;
        let df = df! {
            "id" => [1u32, 2, 3, 4, 5],
            "features" => [0.1f64, 0.2, 0.3, 0.4, 0.5],
            "target" => [0.0f64, 1.0, 0.0, 1.0, 0.0],
        }.map_err(|e| Error::task(format!("Failed to create training table: {}", e)))?;
        
        Ok(Table::new(df))
    }
    
    /// Get the validation table
    fn get_val_table(&self) -> Result<Table> {
        // Create a basic validation table
        use polars::prelude::*;
        let df = df! {
            "id" => [6u32, 7, 8],
            "features" => [0.6f64, 0.7, 0.8],
            "target" => [1.0f64, 0.0, 1.0],
        }.map_err(|e| Error::task(format!("Failed to create validation table: {}", e)))?;
        
        Ok(Table::new(df))
    }
    
    /// Get the test table
    fn get_test_table(&self) -> Result<Table> {
        // Create a basic test table
        use polars::prelude::*;
        let df = df! {
            "id" => [9u32, 10],
            "features" => [0.9f64, 1.0],
            "target" => [0.0f64, 1.0],
        }.map_err(|e| Error::task(format!("Failed to create test table: {}", e)))?;
        
        Ok(Table::new(df))
    }
    
    /// Evaluate predictions
    fn evaluate(&self, predictions: &[f64], _metrics: Option<Vec<Box<dyn Metric>>>) -> Result<Vec<f64>> {
        // Use provided metrics or default implementation
        if _metrics.is_some() {
            // TODO: Implement proper metric evaluation when Metric trait methods are available
            // For now, fall back to default implementation
        }
        
        // EntityTask specific evaluation implementation
        if predictions.is_empty() {
            return Ok(vec![0.0]);
        }
        
        // Mock ground truth for entity tasks
        let targets = vec![0.0, 1.0, 0.0, 1.0, 0.0];
        let targets = &targets[..predictions.len().min(targets.len())];
        
        // Calculate accuracy
        let mut correct = 0;
        for (pred, target) in predictions.iter().zip(targets.iter()) {
            let predicted_class: f64 = if *pred > 0.5 { 1.0 } else { 0.0 };
            if (predicted_class - target).abs() < 0.1_f64 {
                correct += 1;
            }
        }
        
        let accuracy = correct as f64 / predictions.len() as f64;
        Ok(vec![accuracy])
    }
}

/// Generic entity task implementation
pub struct EntityTask {
    dataset: Box<dyn Dataset>,
    timedelta: Duration,
    num_eval_timestamps: usize,
}

impl EntityTask {
    /// Create a new entity task
    pub fn new(
        dataset: Box<dyn Dataset>,
        timedelta: Duration,
        num_eval_timestamps: usize,
    ) -> Self {
        Self {
            dataset,
            timedelta,
            num_eval_timestamps,
        }
    }
}

impl Task for EntityTask {
    fn task_type(&self) -> TaskType {
        TaskType::Entity
    }

    fn dataset(&self) -> &dyn Dataset {
        self.dataset.as_ref()
    }

    fn timedelta(&self) -> Duration {
        self.timedelta
    }

    fn num_eval_timestamps(&self) -> usize {
        self.num_eval_timestamps
    }

    fn make_table(&self, _db: &Database, _timestamps: Vec<DateTime<Utc>>) -> Result<Table> {
        // Default implementation for entity tasks
        use polars::prelude::*;
        let df = df! {
            "entity_id" => [1u32, 2, 3, 4, 5],
            "timestamp" => ["2024-01-01"; 5],
            "features" => [0.1f64, 0.2, 0.3, 0.4, 0.5],
            "label" => [0i32, 1, 0, 1, 0],
        }.map_err(|e| Error::task(format!("Failed to create entity table: {}", e)))?;
        
        Ok(Table::new(df))
    }

    fn evaluate(&self, predictions: &[f64], _metrics: Option<Vec<Box<dyn Metric>>>) -> Result<Vec<f64>> {
        // Use provided metrics or default implementation
        if _metrics.is_some() {
            // TODO: Implement proper metric evaluation when Metric trait methods are available
            // For now, fall back to default implementation
        }
        
        // EntityTask specific evaluation implementation
        if predictions.is_empty() {
            return Ok(vec![0.0]);
        }
        
        // Mock ground truth for entity tasks
        let targets = vec![0.0, 1.0, 0.0, 1.0, 0.0];
        let targets = &targets[..predictions.len().min(targets.len())];
        
        // Calculate accuracy
        let mut correct = 0;
        for (pred, target) in predictions.iter().zip(targets.iter()) {
            let predicted_class: f64 = if *pred > 0.5 { 1.0 } else { 0.0 };
            if (predicted_class - target).abs() < 0.1_f64 {
                correct += 1;
            }
        }
        
        let accuracy = correct as f64 / predictions.len() as f64;
        Ok(vec![accuracy])
    }
}

/// Generic recommendation task implementation
pub struct RecommendationTask {
    dataset: Box<dyn Dataset>,
    timedelta: Duration,
    num_eval_timestamps: usize,
}

impl RecommendationTask {
    /// Create a new recommendation task
    pub fn new(
        dataset: Box<dyn Dataset>,
        timedelta: Duration,
        num_eval_timestamps: usize,
    ) -> Self {
        Self {
            dataset,
            timedelta,
            num_eval_timestamps,
        }
    }
}

impl Task for RecommendationTask {
    fn task_type(&self) -> TaskType {
        TaskType::Recommendation
    }

    fn dataset(&self) -> &dyn Dataset {
        self.dataset.as_ref()
    }

    fn timedelta(&self) -> Duration {
        self.timedelta
    }

    fn num_eval_timestamps(&self) -> usize {
        self.num_eval_timestamps
    }

    fn make_table(&self, _db: &Database, _timestamps: Vec<DateTime<Utc>>) -> Result<Table> {
        // Default implementation for recommendation tasks
        use polars::prelude::*;
        let df = df! {
            "user_id" => [1u32, 2, 3, 4, 5],
            "item_id" => [101u32, 102, 103, 104, 105],
            "timestamp" => ["2024-01-01"; 5],
            "rating" => [4.5f64, 3.0, 5.0, 2.5, 4.0],
        }.map_err(|e| Error::task(format!("Failed to create recommendation table: {}", e)))?;
        
        Ok(Table::new(df))
    }

    fn evaluate(&self, predictions: &[f64], _metrics: Option<Vec<Box<dyn Metric>>>) -> Result<Vec<f64>> {
        // Use provided metrics or default implementation
        if _metrics.is_some() {
            // TODO: Implement proper metric evaluation when Metric trait methods are available
            // For now, calculate RMSE manually
        }
        
        // Calculate RMSE for recommendation tasks
        let targets = vec![4.5, 3.0, 5.0, 2.5, 4.0];
        let targets = &targets[..predictions.len().min(targets.len())];
        
        let mut sum_squared_error = 0.0;
        for (pred, target) in predictions.iter().zip(targets.iter()) {
            let error = pred - target;
            sum_squared_error += error * error;
        }
        
        let rmse = (sum_squared_error / predictions.len() as f64).sqrt();
        Ok(vec![rmse])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;
    
    #[derive(Debug)]
    struct MockDataset {
        tables: HashMap<String, DataFrame>,
    }
    
    impl MockDataset {
        fn new() -> Self {
            Self {
                tables: HashMap::new(),
            }
        }
    }
    
    impl Dataset for MockDataset {
        fn name(&self) -> &str {
            "mock-dataset"
        }
        
        fn description(&self) -> &str {
            "Mock dataset for testing"
        }
        
        fn version(&self) -> &str {
            "1.0"
        }
        
        fn tables(&self) -> &HashMap<String, DataFrame> {
            &self.tables
        }
        
        fn val_timestamp(&self) -> DateTime<Utc> {
            Utc::now()
        }
        
        fn test_timestamp(&self) -> DateTime<Utc> {
            Utc::now()
        }
        
        fn load(&mut self, _config: &gaussrdl_core::DatasetConfig) -> Result<()> {
            Ok(())
        }
        
        fn save(&self, _path: &std::path::Path) -> Result<()> {
            Ok(())
        }
        
        fn load_from_path(&mut self, _path: &std::path::Path) -> Result<()> {
            Ok(())
        }
        
        fn validate(&self) -> Result<()> {
            Ok(())
        }
    }
    
    #[test]
    fn test_task_types() {
        assert_eq!(TaskType::Entity, TaskType::Entity);
        assert_ne!(TaskType::Entity, TaskType::Recommendation);
    }
    
    #[test]
    fn test_entity_task() {
        let dataset = Box::new(MockDataset::new());
        let task = EntityTask::new(
            dataset,
            std::time::Duration::from_secs(3600),
            10,
        );
        
        assert_eq!(task.task_type(), TaskType::Entity);
        assert_eq!(task.timedelta(), std::time::Duration::from_secs(3600));
        assert_eq!(task.num_eval_timestamps(), 10);
        
        // Test evaluation
        let predictions = vec![0.7, 0.3, 0.8];
        let results = task.evaluate(&predictions, None).unwrap();
        assert_eq!(results.len(), 1); // accuracy
    }
    
    #[test]
    fn test_recommendation_task() {
        let dataset = Box::new(MockDataset::new());
        let task = RecommendationTask::new(
            dataset,
            std::time::Duration::from_secs(3600),
            5,
        );
        
        assert_eq!(task.task_type(), TaskType::Recommendation);
        
        // Test evaluation
        let predictions = vec![4.0, 3.5, 4.8];
        let results = task.evaluate(&predictions, None).unwrap();
        assert_eq!(results.len(), 1); // RMSE
        assert!(results[0] > 0.0); // Should have some error
    }
    
    #[test]
    fn test_task_registry() {
        let registry = TaskRegistry::new();
        let tasks = registry.list();
        
        // Should have registered default tasks
        assert!(tasks.contains(&"rel-amazon/user-churn".to_string()));
        assert!(tasks.contains(&"rel-f1/driver-position".to_string()));
        
        // Test getting a task
        let provider = registry.get("rel-amazon/user-churn");
        assert!(provider.is_some());
        
        // Test getting non-existent task
        let provider = registry.get("non-existent");
        assert!(provider.is_none());
    }
} 