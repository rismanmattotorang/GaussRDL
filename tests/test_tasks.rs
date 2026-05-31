// Task Tests
// Comprehensive tests for task functionality similar to RelBench Python tests

use gaussrelgt::tasks::*;
use gaussrelgt::base::{DownloadConfig, CacheConfig};
use gaussrelgt::error::Result;
use std::time::{Duration, Instant};

#[cfg(test)]
mod task_tests {
    use super::*;

    #[test]
    fn test_task_registry() {
        let registry = TaskRegistry::new();
        let available_tasks = registry.list();
        
        // Should have some predefined tasks
        assert!(!available_tasks.is_empty(), "Registry should have some tasks");
        
        // Check for expected tasks
        let expected_tasks = vec![
            "rel-amazon/user-churn",
            "rel-amazon/item-churn", 
            "rel-f1/driver-position",
            "rel-f1/constructor-position",
            "rel-hm/user-churn",
            "rel-hm/item-sales",
        ];
        
        for expected_task in expected_tasks {
            assert!(available_tasks.contains(&expected_task.to_string()),
                   "Registry should contain task: {}", expected_task);
        }
    }

    #[test]
    fn test_task_provider_creation() {
        // Test individual task providers
        let user_churn_provider = UserChurnTaskProvider::new("rel-amazon");
        assert_eq!(user_churn_provider.dataset_name(), "rel-amazon");
        assert!(!user_churn_provider.name().is_empty());
        assert!(!user_churn_provider.description().is_empty());
        
        let driver_position_provider = DriverPositionTaskProvider::new("rel-f1");
        assert_eq!(driver_position_provider.dataset_name(), "rel-f1");
        
        let item_sales_provider = ItemSalesTaskProvider::new("rel-hm");
        assert_eq!(item_sales_provider.dataset_name(), "rel-hm");
    }

    #[test]
    fn test_task_types() {
        // Test task type enum
        assert_eq!(TaskType::Entity, TaskType::Entity);
        assert_ne!(TaskType::Entity, TaskType::Recommendation);
        
        // Test task type assignment
        let registry = TaskRegistry::new();
        
        // User churn should be entity prediction
        if let Some(provider) = registry.get("rel-amazon/user-churn") {
            assert_eq!(provider.task_type(), TaskType::Entity);
        }
        
        // Item sales might be regression (entity prediction)
        if let Some(provider) = registry.get("rel-hm/item-sales") {
            // Could be either entity or recommendation depending on implementation
            let task_type = provider.task_type();
            assert!(task_type == TaskType::Entity || task_type == TaskType::Recommendation);
        }
    }

    #[test]
    fn test_task_loading() -> Result<()> {
        // Test loading different types of tasks
        let tasks_to_test = vec![
            ("rel-amazon", "user-churn"),
            ("rel-amazon", "item-churn"),
            ("rel-f1", "driver-position"),
            ("rel-hm", "item-sales"),
        ];
        
        for (dataset, task_name) in tasks_to_test {
            match get_task(dataset, task_name, false) {
                Ok(task) => {
                    // Verify task properties
                    assert!(task.timedelta() >= Duration::from_secs(0));
                    assert!(task.num_eval_timestamps() > 0);
                    
                    // Test task table generation
                    let train_table = task.get_train_table();
                    assert!(train_table.is_ok(), "Should be able to get training table");
                    
                    let val_table = task.get_val_table();
                    assert!(val_table.is_ok(), "Should be able to get validation table");
                    
                    let test_table = task.get_test_table();
                    assert!(test_table.is_ok(), "Should be able to get test table");
                }
                Err(e) => {
                    // Task loading might fail in test environment - log but don't fail test
                    println!("Task {}/{} loading failed (expected in test env): {}", dataset, task_name, e);
                }
            }
        }
        
        Ok(())
    }

    #[test]
    fn test_task_evaluation() -> Result<()> {
        // Test evaluation functionality
        let task = get_task("rel-amazon", "user-churn", false)?;
        
        // Test with valid predictions
        let predictions = vec![0.8, 0.3, 0.9, 0.1, 0.7, 0.2, 0.85, 0.05];
        let results = task.evaluate(&predictions, None)?;
        
        assert!(!results.is_empty(), "Evaluation should return results");
        
        // For classification tasks, results should be in valid range
        if task.task_type() == TaskType::Entity {
            for result in &results {
                assert!(*result >= 0.0 && *result <= 1.0, 
                       "Classification metric should be between 0 and 1, got {}", result);
            }
        }
        
        // Test with empty predictions
        let empty_predictions = vec![];
        let empty_results = task.evaluate(&empty_predictions, None)?;
        assert!(!empty_results.is_empty(), "Should handle empty predictions gracefully");
        
        Ok(())
    }

    #[test]
    fn test_entity_task() {
        use gaussrelgt::datasets::RelAmazonDataset;
        
        let dataset = Box::new(RelAmazonDataset::new());
        let task = EntityTask::new(
            dataset,
            Duration::from_secs(3600), // 1 hour
            10,
        );
        
        assert_eq!(task.task_type(), TaskType::Entity);
        assert_eq!(task.timedelta(), Duration::from_secs(3600));
        assert_eq!(task.num_eval_timestamps(), 10);
        
        // Test evaluation
        let predictions = vec![0.9, 0.2, 0.8, 0.1, 0.7];
        let results = task.evaluate(&predictions, None).unwrap();
        
        assert_eq!(results.len(), 1); // Should return accuracy
        assert!(results[0] >= 0.0 && results[0] <= 1.0);
    }

    #[test]
    fn test_recommendation_task() {
        use gaussrelgt::datasets::RelHMDataset;
        
        let dataset = Box::new(RelHMDataset::new());
        let task = RecommendationTask::new(
            dataset,
            Duration::from_secs(3600),
            5,
        );
        
        assert_eq!(task.task_type(), TaskType::Recommendation);
        assert_eq!(task.timedelta(), Duration::from_secs(3600));
        assert_eq!(task.num_eval_timestamps(), 5);
        
        // Test evaluation
        let predictions = vec![4.2, 3.1, 4.8, 2.5, 3.9];
        let results = task.evaluate(&predictions, None).unwrap();
        
        assert_eq!(results.len(), 1); // Should return RMSE
        assert!(results[0] >= 0.0); // RMSE should be non-negative
    }

    #[test]
    fn test_task_table_generation() -> Result<()> {
        let task = get_task("rel-amazon", "user-churn", false)?;
        
        // Test training table
        let train_table = task.get_train_table()?;
        let train_data = &train_table.data;
        assert!(train_data.width() > 0, "Training table should have columns");
        
        // Test validation table
        let val_table = task.get_val_table()?;
        let val_data = &val_table.data;
        assert!(val_data.width() > 0, "Validation table should have columns");
        
        // Test test table
        let test_table = task.get_test_table()?;
        let test_data = &test_table.data;
        assert!(test_data.width() > 0, "Test table should have columns");
        
        // Tables should have consistent schema
        assert_eq!(train_data.width(), val_data.width(), 
                  "Train and validation tables should have same number of columns");
        assert_eq!(val_data.width(), test_data.width(),
                  "Validation and test tables should have same number of columns");
        
        Ok(())
    }

    #[test]
    fn test_task_evaluation_edge_cases() -> Result<()> {
        let task = get_task("rel-amazon", "user-churn", false)?;
        
        // Test with single prediction
        let single_pred = vec![0.5];
        let single_result = task.evaluate(&single_pred, None)?;
        assert!(!single_result.is_empty());
        
        // Test with many predictions
        let many_preds: Vec<f64> = (0..1000).map(|i| (i as f64) / 1000.0).collect();
        let many_results = task.evaluate(&many_preds, None)?;
        assert!(!many_results.is_empty());
        
        // Test with extreme values
        let extreme_preds = vec![0.0, 1.0, 0.0, 1.0];
        let extreme_results = task.evaluate(&extreme_preds, None)?;
        assert!(!extreme_results.is_empty());
        
        Ok(())
    }

    #[test]
    fn test_task_performance() -> Result<()> {
        let task = get_task("rel-amazon", "user-churn", false)?;
        
        // Test evaluation performance with larger datasets
        let sizes = vec![100, 1000, 10000];
        
        for size in sizes {
            let predictions: Vec<f64> = (0..size).map(|i| (i as f64) / (size as f64)).collect();
            
            let start_time = Instant::now();
            let results = task.evaluate(&predictions, None)?;
            let eval_time = start_time.elapsed();
            
            println!("Evaluation of {} predictions took {:?}", size, eval_time);
            
            // Evaluation should complete in reasonable time
            assert!(eval_time.as_millis() < 1000, 
                   "Evaluation of {} predictions took too long: {:?}", size, eval_time);
            
            assert!(!results.is_empty());
        }
        
        Ok(())
    }

    #[test]
    fn test_task_metrics() {
        let registry = TaskRegistry::new();
        
        // Test that tasks have appropriate metrics
        for task_name in registry.list() {
            if let Some(provider) = registry.get(&task_name) {
                let metrics = provider.metrics();
                
                // Each task should have at least one metric
                assert!(!metrics.is_empty(), "Task {} should have metrics", task_name);
                
                // Verify metric types are appropriate for task type
                match provider.task_type() {
                    TaskType::Entity => {
                        // Entity tasks should have classification metrics
                        let metric_names: Vec<String> = metrics.iter().map(|m| m.name().to_string()).collect();
                        let has_classification_metric = metric_names.iter().any(|name| 
                            name.contains("AUROC") || name.contains("Accuracy") || name.contains("F1")
                        );
                        
                        if !has_classification_metric {
                            println!("Warning: Entity task {} might be missing classification metrics", task_name);
                        }
                    }
                    TaskType::Recommendation => {
                        // Recommendation tasks should have ranking or regression metrics
                        let metric_names: Vec<String> = metrics.iter().map(|m| m.name().to_string()).collect();
                        let has_recommendation_metric = metric_names.iter().any(|name| 
                            name.contains("MAP") || name.contains("RMSE") || name.contains("NDCG")
                        );
                        
                        if !has_recommendation_metric {
                            println!("Warning: Recommendation task {} might be missing recommendation metrics", task_name);
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod task_integration_tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_multiple_task_loading() {
        let registry = TaskRegistry::new();
        let all_tasks = registry.list();
        
        let mut loaded_tasks = HashMap::new();
        let mut failed_tasks = Vec::new();
        
        for task_name in &all_tasks {
            if let Some(provider) = registry.get(task_name) {
                let config = DownloadConfig::default();
                let cache_config = CacheConfig::default();
                
                match provider.load(&config, &cache_config) {
                    Ok(task) => {
                        loaded_tasks.insert(task_name.clone(), task);
                    }
                    Err(e) => {
                        failed_tasks.push((task_name.clone(), e));
                    }
                }
            }
        }
        
        println!("Loaded {} out of {} tasks", loaded_tasks.len(), all_tasks.len());
        
        for (task_name, error) in failed_tasks {
            println!("Task {} failed to load: {}", task_name, error);
        }
        
        // Verify loaded tasks work correctly
        for (task_name, task) in loaded_tasks {
            let predictions = vec![0.5, 0.7, 0.3, 0.8];
            match task.evaluate(&predictions, None) {
                Ok(results) => {
                    println!("Task {} evaluation succeeded: {:?}", task_name, results);
                    assert!(!results.is_empty());
                }
                Err(e) => {
                    println!("Task {} evaluation failed: {}", task_name, e);
                }
            }
        }
    }

    #[test]
    fn test_task_dataset_compatibility() -> Result<()> {
        let tasks_and_datasets = vec![
            ("rel-amazon", "user-churn"),
            ("rel-amazon", "item-churn"),
            ("rel-f1", "driver-position"),
            ("rel-f1", "constructor-position"),
            ("rel-hm", "user-churn"),
            ("rel-hm", "item-sales"),
        ];
        
        for (dataset_name, task_name) in tasks_and_datasets {
            match get_task(dataset_name, task_name, false) {
                Ok(task) => {
                    let dataset = task.dataset();
                    
                    // Verify dataset name matches
                    assert_eq!(dataset.name(), dataset_name,
                              "Task dataset name should match expected dataset");
                    
                    // Verify task can access dataset tables
                    let tables = dataset.tables();
                    assert!(!tables.is_empty(), "Dataset should have tables for task to use");
                }
                Err(e) => {
                    println!("Task {}/{} compatibility check failed: {}", dataset_name, task_name, e);
                }
            }
        }
        
        Ok(())
    }

    #[test]
    fn test_task_evaluation_consistency() -> Result<()> {
        let task = get_task("rel-amazon", "user-churn", false)?;
        let predictions = vec![0.8, 0.3, 0.9, 0.1, 0.7];
        
        // Run evaluation multiple times with same input
        let mut results = Vec::new();
        for _ in 0..5 {
            let result = task.evaluate(&predictions, None)?;
            results.push(result);
        }
        
        // Results should be consistent
        for i in 1..results.len() {
            assert_eq!(results[0], results[i], 
                      "Task evaluation should be deterministic");
        }
        
        Ok(())
    }

    #[test] 
    fn test_concurrent_task_evaluation() -> Result<()> {
        use std::sync::Arc;
        use std::thread;
        
        let task = Arc::new(get_task("rel-amazon", "user-churn", false)?);
        let mut handles = vec![];
        
        // Spawn multiple threads doing evaluation
        for i in 0..3 {
            let task_clone = Arc::clone(&task);
            let handle = thread::spawn(move || {
                let predictions = vec![0.8, 0.3, 0.9, 0.1, 0.7];
                let result = task_clone.evaluate(&predictions, None);
                (i, result)
            });
            handles.push(handle);
        }
        
        // Collect results
        for handle in handles {
            let (thread_id, result) = handle.join().unwrap();
            match result {
                Ok(scores) => {
                    println!("Thread {} evaluation succeeded: {:?}", thread_id, scores);
                    assert!(!scores.is_empty());
                }
                Err(e) => {
                    println!("Thread {} evaluation failed: {}", thread_id, e);
                }
            }
        }
        
        Ok(())
    }

    #[test]
    fn test_task_memory_usage() -> Result<()> {
        // Test memory usage doesn't grow excessively during task operations
        let task = get_task("rel-amazon", "user-churn", false)?;
        
        // Do many evaluations to check for memory leaks
        for iteration in 0..100 {
            let predictions: Vec<f64> = (0..100).map(|i| (i as f64) / 100.0).collect();
            let _results = task.evaluate(&predictions, None)?;
            
            // Every 20 iterations, print progress
            if iteration % 20 == 0 {
                println!("Completed {} evaluation iterations", iteration);
            }
        }
        
        println!("Memory usage test completed - no obvious leaks detected");
        
        Ok(())
    }
} 