use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use crate::core::{Task, TaskType, Dataset, Table, Metric};
use crate::error::{Result, Error};
use crate::metrics::{Accuracy, AUROC, F1Score, MSE, RMSE, MRR, NDCG};

/// Task configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskConfig {
    /// Task name
    pub name: String,
    
    /// Task description
    pub description: String,
    
    /// Task type
    pub task_type: TaskType,
    
    /// Dataset name
    pub dataset: String,
    
    /// Target column
    pub target_column: String,
    
    /// Feature columns
    pub feature_columns: Vec<String>,
    
    /// Categorical columns
    pub categorical_columns: Vec<String>,
    
    /// Numerical columns
    pub numerical_columns: Vec<String>,
    
    /// Timestamp column
    pub timestamp_column: Option<String>,
    
    /// Group column for ranking tasks
    pub group_column: Option<String>,
    
    /// Metrics to evaluate
    pub metrics: Vec<String>,
}

/// Task registry
pub struct TaskRegistry {
    /// Available tasks
    tasks: HashMap<String, TaskConfig>,
    
    /// Dataset registry
    dataset_registry: crate::dataset::DatasetRegistry,
}

impl TaskRegistry {
    /// Create a new task registry
    pub fn new(dataset_registry: crate::dataset::DatasetRegistry) -> Self {
        Self {
            tasks: HashMap::new(),
            dataset_registry,
        }
    }
    
    /// Register a task
    pub fn register(&mut self, config: TaskConfig) {
        self.tasks.insert(config.name.clone(), config);
    }
    
    /// Get task configuration
    pub fn get_config(&self, name: &str) -> Option<&TaskConfig> {
        self.tasks.get(name)
    }
    
    /// List available tasks
    pub fn list_tasks(&self) -> Vec<&str> {
        self.tasks.keys().map(|s| s.as_str()).collect()
    }
    
    /// Get task
    pub fn get_task(&self, name: &str) -> Result<Box<dyn Task>> {
        let config = self.get_config(name)
            .ok_or_else(|| Error::task(format!("Task {} not found", name)))?;
            
        let dataset = self.dataset_registry.get_dataset(&config.dataset)?;
        
        Ok(Box::new(StandardTask {
            config: config.clone(),
            dataset,
        }))
    }
}

/// Standard task implementation
pub struct StandardTask {
    /// Task configuration
    config: TaskConfig,
    
    /// Dataset
    dataset: Box<dyn Dataset>,
}

impl Task for StandardTask {
    fn name(&self) -> &str {
        &self.config.name
    }
    
    fn description(&self) -> &str {
        &self.config.description
    }
    
    fn task_type(&self) -> TaskType {
        self.config.task_type
    }
    
    fn dataset(&self) -> &dyn Dataset {
        self.dataset.as_ref()
    }
    
    fn metrics(&self) -> Vec<Box<dyn Metric>> {
        self.config.metrics.iter()
            .map(|name| {
                match name.as_str() {
                    "accuracy" => Box::new(Accuracy::new()) as Box<dyn Metric>,
                    "auroc" => Box::new(AUROC::new()) as Box<dyn Metric>,
                    "f1" => Box::new(F1Score::new()) as Box<dyn Metric>,
                    "mse" => Box::new(MSE::new()) as Box<dyn Metric>,
                    "rmse" => Box::new(RMSE::new()) as Box<dyn Metric>,
                    "mrr" => Box::new(MRR::new()) as Box<dyn Metric>,
                    "ndcg" => Box::new(NDCG::new()) as Box<dyn Metric>,
                    _ => panic!("Unknown metric: {}", name),
                }
            })
            .collect()
    }
    
    fn get_train_data(&self) -> Result<Table> {
        let tables = self.dataset.tables()?;
        let table = tables.iter()
            .find(|t| t.name == self.config.dataset)
            .ok_or_else(|| Error::task(format!("Table {} not found", self.config.dataset)))?;
            
        let mut df = table.data.clone();
        
        // Filter by timestamp if needed
        if let Some(ts_col) = &self.config.timestamp_column {
            df = df.filter(col(ts_col).lt(lit(self.dataset.val_timestamp().timestamp())))?;
        }
        
        // Select features and target
        let mut columns = self.config.feature_columns.clone();
        columns.push(self.config.target_column.clone());
        if let Some(group_col) = &self.config.group_column {
            columns.push(group_col.clone());
        }
        df = df.select(&columns)?;
        
        Ok(Table::new(
            table.name.clone(),
            df,
            table.primary_key.clone(),
            table.foreign_keys.clone(),
            self.config.timestamp_column.clone(),
        ))
    }
    
    fn get_val_data(&self) -> Result<Table> {
        let tables = self.dataset.tables()?;
        let table = tables.iter()
            .find(|t| t.name == self.config.dataset)
            .ok_or_else(|| Error::task(format!("Table {} not found", self.config.dataset)))?;
            
        let mut df = table.data.clone();
        
        // Filter by timestamp if needed
        if let Some(ts_col) = &self.config.timestamp_column {
            df = df.filter(
                col(ts_col)
                    .gt_eq(lit(self.dataset.val_timestamp().timestamp()))
                    .and(col(ts_col).lt(lit(self.dataset.test_timestamp().timestamp())))
            )?;
        }
        
        // Select features and target
        let mut columns = self.config.feature_columns.clone();
        columns.push(self.config.target_column.clone());
        if let Some(group_col) = &self.config.group_column {
            columns.push(group_col.clone());
        }
        df = df.select(&columns)?;
        
        Ok(Table::new(
            table.name.clone(),
            df,
            table.primary_key.clone(),
            table.foreign_keys.clone(),
            self.config.timestamp_column.clone(),
        ))
    }
    
    fn get_test_data(&self) -> Result<Table> {
        let tables = self.dataset.tables()?;
        let table = tables.iter()
            .find(|t| t.name == self.config.dataset)
            .ok_or_else(|| Error::task(format!("Table {} not found", self.config.dataset)))?;
            
        let mut df = table.data.clone();
        
        // Filter by timestamp if needed
        if let Some(ts_col) = &self.config.timestamp_column {
            df = df.filter(col(ts_col).gt_eq(lit(self.dataset.test_timestamp().timestamp())))?;
        }
        
        // Select features and target
        let mut columns = self.config.feature_columns.clone();
        columns.push(self.config.target_column.clone());
        if let Some(group_col) = &self.config.group_column {
            columns.push(group_col.clone());
        }
        df = df.select(&columns)?;
        
        Ok(Table::new(
            table.name.clone(),
            df,
            table.primary_key.clone(),
            table.foreign_keys.clone(),
            self.config.timestamp_column.clone(),
        ))
    }
    
    fn evaluate(&self, predictions: &[f64], targets: &[f64]) -> Result<Vec<f64>> {
        let metrics = self.metrics();
        let mut results = Vec::new();
        
        for metric in metrics {
            let score = metric.compute(predictions, targets)?;
            results.push(score);
        }
        
        Ok(results)
    }
} 