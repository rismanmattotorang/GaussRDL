use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use chrono::{DateTime, Utc};

use crate::base::{Database, Dataset, Table};
use crate::error::{Error, Result};
use crate::metrics::Metric;

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
        self.register("rel-amazon/user-churn", Arc::new(UserChurnTask::new("rel-amazon")));
        self.register("rel-amazon/item-churn", Arc::new(ItemChurnTask::new("rel-amazon")));
        
        // Register rel-f1 tasks
        self.register("rel-f1/driver-position", Arc::new(DriverPositionTask::new("rel-f1")));
        self.register("rel-f1/constructor-position", Arc::new(ConstructorPositionTask::new("rel-f1")));
        
        // Register rel-hm tasks
        self.register("rel-hm/user-churn", Arc::new(UserChurnTask::new("rel-hm")));
        self.register("rel-hm/item-sales", Arc::new(ItemSalesTask::new("rel-hm")));
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
    
    /// Validates the task
    fn validate(&self, task: &dyn Task) -> Result<()>;
}

/// Base implementation for task providers
#[derive(Debug)]
pub struct BaseTaskProvider {
    name: String,
    description: String,
    task_type: TaskType,
    dataset_name: String,
}

impl BaseTaskProvider {
    /// Creates a new base task provider
    pub fn new(
        name: &str,
        description: &str,
        task_type: TaskType,
        dataset_name: &str,
    ) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            task_type,
            dataset_name: dataset_name.to_string(),
        }
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
        let db = self.dataset().get_db(true)?;
        let val_ts = self.dataset().val_timestamp();
        let start = val_ts - chrono::Duration::from_std(self.timedelta())?;
        let end = db.min_timestamp().unwrap_or(start);
        
        let mut timestamps = Vec::new();
        let mut current = start;
        while current >= end {
            timestamps.push(current);
            current = current - chrono::Duration::from_std(self.timedelta())?;
        }
        
        self.make_table(&db, timestamps)
    }
    
    /// Get the validation table
    fn get_val_table(&self) -> Result<Table> {
        let db = self.dataset().get_db(true)?;
        let val_ts = self.dataset().val_timestamp();
        let test_ts = self.dataset().test_timestamp();
        
        let mut timestamps = Vec::new();
        let mut current = val_ts;
        let end = test_ts - chrono::Duration::from_std(self.timedelta())?;
        
        for _ in 0..self.num_eval_timestamps() {
            if current > end {
                break;
            }
            timestamps.push(current);
            current = current + chrono::Duration::from_std(self.timedelta())?;
        }
        
        self.make_table(&db, timestamps)
    }
    
    /// Get the test table
    fn get_test_table(&self) -> Result<Table> {
        let db = self.dataset().get_db(false)?;
        let test_ts = self.dataset().test_timestamp();
        let max_ts = db.max_timestamp().unwrap_or(test_ts);
        
        let mut timestamps = Vec::new();
        let mut current = test_ts;
        
        for _ in 0..self.num_eval_timestamps() {
            if current > max_ts {
                break;
            }
            timestamps.push(current);
            current = current + chrono::Duration::from_std(self.timedelta())?;
        }
        
        self.make_table(&db, timestamps)
    }
    
    /// Evaluate predictions
    fn evaluate(&self, predictions: &[f64], metrics: Option<Vec<Box<dyn Metric>>>) -> Result<Vec<f64>>;
}

/// Base implementation for entity prediction tasks
pub struct EntityTask {
    dataset: Box<dyn Dataset>,
    timedelta: Duration,
    num_eval_timestamps: usize,
}

impl EntityTask {
    /// Create a new entity prediction task
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
    
    fn make_table(&self, db: &Database, timestamps: Vec<DateTime<Utc>>) -> Result<Table> {
        // Implementation specific to entity prediction tasks
        unimplemented!()
    }
    
    fn evaluate(&self, predictions: &[f64], metrics: Option<Vec<Box<dyn Metric>>>) -> Result<Vec<f64>> {
        // Implementation specific to entity prediction tasks
        unimplemented!()
    }
}

/// Base implementation for recommendation tasks
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
    
    fn make_table(&self, db: &Database, timestamps: Vec<DateTime<Utc>>) -> Result<Table> {
        // Implementation specific to recommendation tasks
        unimplemented!()
    }
    
    fn evaluate(&self, predictions: &[f64], metrics: Option<Vec<Box<dyn Metric>>>) -> Result<Vec<f64>> {
        // Implementation specific to recommendation tasks
        unimplemented!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;
    
    struct MockDataset;
    
    impl Dataset for MockDataset {
        fn val_timestamp(&self) -> DateTime<Utc> {
            Utc.ymd(2024, 1, 1).and_hms(0, 0, 0)
        }
        
        fn test_timestamp(&self) -> DateTime<Utc> {
            Utc.ymd(2024, 2, 1).and_hms(0, 0, 0)
        }
        
        fn get_db(&self, _upto_test_timestamp: bool) -> Result<Database> {
            unimplemented!()
        }
        
        fn make_db(&self) -> Result<Database> {
            unimplemented!()
        }
        
        fn validate_and_correct_db(&self, _db: &mut Database) -> Result<()> {
            Ok(())
        }
        
        fn cache_dir(&self) -> Option<std::path::PathBuf> {
            None
        }
        
        fn set_cache_dir(&mut self, _dir: Option<std::path::PathBuf>) {}
    }
    
    #[test]
    fn test_task_types() {
        let dataset = Box::new(MockDataset);
        let timedelta = Duration::from_secs(86400); // 1 day
        let num_eval_timestamps = 7;
        
        let entity_task = EntityTask::new(dataset.clone(), timedelta, num_eval_timestamps);
        assert_eq!(entity_task.task_type(), TaskType::Entity);
        
        let rec_task = RecommendationTask::new(dataset, timedelta, num_eval_timestamps);
        assert_eq!(rec_task.task_type(), TaskType::Recommendation);
    }
} 