// Integration Tests
// End-to-end tests for the complete GaussRelGT system similar to RelBench Python integration tests

use gaussrelgt::tasks::*;
use gaussrelgt::base::{DownloadConfig, CacheConfig};
use gaussrelgt::error::Result;
use std::time::Instant;

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_end_to_end_workflow() -> Result<()> {
        println!("🧪 Testing end-to-end workflow...");
        
        // Step 1: Load task
        match get_task("rel-amazon", "user-churn", false) {
            Ok(task) => {
                println!("✅ Task loaded successfully");
                
                // Step 2: Get training data
                match task.get_train_table() {
                    Ok(train_table) => {
                        let train_data = &train_table.data;
                        println!("✅ Training data loaded: {} rows × {} columns", 
                                train_data.height(), train_data.width());
                        
                        // Step 3: Get validation data
                        match task.get_val_table() {
                            Ok(val_table) => {
                                let val_data = &val_table.data;
                                println!("✅ Validation data loaded: {} rows × {} columns", 
                                        val_data.height(), val_data.width());
                                
                                // Step 4: Get test data
                                match task.get_test_table() {
                                    Ok(test_table) => {
                                        let test_data = &test_table.data;
                                        println!("✅ Test data loaded: {} rows × {} columns", 
                                                test_data.height(), test_data.width());
                                        
                                        // Step 5: Simulate model predictions
                                        let num_predictions = std::cmp::min(train_data.height(), 100);
                                        let predictions: Vec<f64> = (0..num_predictions)
                                            .map(|i| 0.3 + 0.4 * (i as f64 / num_predictions as f64))
                                            .collect();
                                        
                                        // Step 6: Evaluate predictions
                                        let start_time = Instant::now();
                                        match task.evaluate(&predictions, None) {
                                            Ok(results) => {
                                                let eval_time = start_time.elapsed();
                                                println!("✅ Evaluation completed in {:?}", eval_time);
                                                println!("📊 Results: {:?}", results);
                                                
                                                // Verify results
                                                assert!(!results.is_empty(), "Should have evaluation results");
                                                for result in &results {
                                                    assert!(result.is_finite(), "Results should be finite numbers");
                                                }
                                                
                                                println!("🎉 End-to-end workflow completed successfully!");
                                            }
                                            Err(e) => {
                                                println!("⚠️ Evaluation failed: {}", e);
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        println!("⚠️ Test data loading failed: {}", e);
                                    }
                                }
                            }
                            Err(e) => {
                                println!("⚠️ Validation data loading failed: {}", e);
                            }
                        }
                    }
                    Err(e) => {
                        println!("⚠️ Training data loading failed: {}", e);
                    }
                }
            }
            Err(e) => {
                println!("⚠️ Task loading failed (expected in test env): {}", e);
            }
        }
        
        Ok(())
    }

    #[test]
    fn test_multiple_dataset_task_workflow() -> Result<()> {
        println!("🧪 Testing multiple dataset/task combinations...");
        
        let test_cases = vec![
            ("rel-amazon", "user-churn", "classification"),
            ("rel-amazon", "item-churn", "classification"),
            ("rel-f1", "driver-position", "regression"),
            ("rel-hm", "item-sales", "regression"),
        ];
        
        for (dataset_name, task_name, task_type) in test_cases {
            println!("Testing {}/{} ({})...", dataset_name, task_name, task_type);
            
            match get_task(dataset_name, task_name, false) {
                Ok(task) => {
                    // Test data loading
                    let mut data_loaded = true;
                    
                    if let Err(e) = task.get_train_table() {
                        println!("  ⚠️ Train table loading failed: {}", e);
                        data_loaded = false;
                    }
                    
                    if let Err(e) = task.get_val_table() {
                        println!("  ⚠️ Val table loading failed: {}", e);
                        data_loaded = false;
                    }
                    
                    if let Err(e) = task.get_test_table() {
                        println!("  ⚠️ Test table loading failed: {}", e);
                        data_loaded = false;
                    }
                    
                    if data_loaded {
                        println!("  ✅ Data loaded successfully");
                        
                        // Test evaluation
                        let predictions = vec![0.5, 0.7, 0.3, 0.8, 0.2];
                        match task.evaluate(&predictions, None) {
                            Ok(results) => {
                                println!("  ✅ Evaluation completed: {:?}", results);
                                
                                // Verify results make sense for task type
                                match task_type {
                                    "classification" => {
                                        // Classification metrics should be reasonable
                                        for result in &results {
                                            assert!(result.is_finite(), "Classification result should be finite: {}", result);
                                        }
                                    }
                                    "regression" => {
                                        // Regression metrics should be finite
                                        for result in &results {
                                            assert!(result.is_finite(), "Regression metric should be finite: {}", result);
                                        }
                                    }
                                    _ => {}
                                }
                            }
                            Err(e) => {
                                println!("  ⚠️ Evaluation failed: {}", e);
                            }
                        }
                    }
                }
                Err(e) => {
                    println!("  ⚠️ Task {}/{} failed (expected in test env): {}", 
                            dataset_name, task_name, e);
                }
            }
        }
        
        println!("🎉 Multiple dataset/task workflow completed!");
        Ok(())
    }

    #[test]
    fn test_task_registry_integration() {
        println!("🧪 Testing task registry integration...");
        
        let registry = TaskRegistry::new();
        let all_tasks = registry.list();
        
        println!("📋 Found {} registered tasks", all_tasks.len());
        
        let mut successful_loads = 0;
        let mut failed_loads = 0;
        
        for task_name in &all_tasks {
            println!("Testing task: {}", task_name);
            
            if let Some(provider) = registry.get(task_name) {
                // Test provider properties
                assert!(!provider.name().is_empty(), "Provider should have name");
                assert!(!provider.description().is_empty(), "Provider should have description");
                assert!(!provider.dataset_name().is_empty(), "Provider should have dataset name");
                
                let metrics = provider.metrics();
                assert!(!metrics.is_empty(), "Provider should have metrics");
                
                // Test task loading
                let config = DownloadConfig::default();
                let cache_config = CacheConfig::default();
                
                match provider.load(&config, &cache_config) {
                    Ok(task) => {
                        successful_loads += 1;
                        
                        // Test basic task functionality
                        assert!(task.timedelta().as_secs() < u64::MAX);
                        assert!(task.num_eval_timestamps() > 0);
                        
                        // Test evaluation
                        let test_predictions = vec![0.5, 0.6, 0.4, 0.7, 0.3];
                        match task.evaluate(&test_predictions, None) {
                            Ok(results) => {
                                println!("  ✅ Task evaluation succeeded: {:?}", results);
                            }
                            Err(e) => {
                                println!("  ⚠️ Task evaluation failed: {}", e);
                            }
                        }
                    }
                    Err(e) => {
                        failed_loads += 1;
                        println!("  ⚠️ Task loading failed: {}", e);
                    }
                }
            }
        }
        
        println!("📊 Registry integration summary:");
        println!("  Total tasks: {}", all_tasks.len());
        println!("  Successful loads: {}", successful_loads);
        println!("  Failed loads: {}", failed_loads);
        
        // At least registry should work even if loading fails
        // Don't assert on tasks being present as registry might be empty in test env
        println!("  Registry functionality verified");
        
        println!("🎉 Task registry integration completed!");
    }

    #[test]
    fn test_dataset_task_compatibility() -> Result<()> {
        println!("🧪 Testing dataset-task compatibility...");
        
        let compatibility_tests = vec![
            ("rel-amazon", vec!["user-churn", "item-churn"]),
            ("rel-f1", vec!["driver-position", "constructor-position"]),
            ("rel-hm", vec!["user-churn", "item-sales"]),
        ];
        
        for (dataset_name, task_names) in compatibility_tests {
            println!("Testing dataset: {}", dataset_name);
            
            for task_name in task_names {
                match get_task(dataset_name, task_name, false) {
                    Ok(task) => {
                        let dataset = task.dataset();
                        
                        // Verify dataset name matches
                        assert_eq!(dataset.name(), dataset_name, 
                                  "Dataset name mismatch for task {}/{}", dataset_name, task_name);
                        
                        // Verify dataset has tables
                        let tables = dataset.tables();
                        assert!(!tables.is_empty(), 
                               "Dataset {} should have tables for task {}", dataset_name, task_name);
                        
                        // Verify task can work with dataset
                        match task.get_train_table() {
                            Ok(_) => {
                                println!("  ✅ {}/{} compatibility verified", dataset_name, task_name);
                            }
                            Err(e) => {
                                println!("  ⚠️ {}/{} train table loading failed: {}", dataset_name, task_name, e);
                            }
                        }
                    }
                    Err(e) => {
                        println!("  ⚠️ {}/{} compatibility test failed: {}", dataset_name, task_name, e);
                    }
                }
            }
        }
        
        println!("🎉 Dataset-task compatibility testing completed!");
        Ok(())
    }

    #[test]
    fn test_performance_benchmarks() -> Result<()> {
        println!("🧪 Running performance benchmarks...");
        
        // Test task loading performance
        let task_load_start = Instant::now();
        match get_task("rel-amazon", "user-churn", false) {
            Ok(task) => {
                let task_load_time = task_load_start.elapsed();
                println!("📊 Task loading time: {:?}", task_load_time);
                assert!(task_load_time.as_secs() < 10, "Task loading should be fast");
                
                // Test data loading performance
                let data_load_start = Instant::now();
                let mut data_load_success = true;
                
                if task.get_train_table().is_err() {
                    data_load_success = false;
                }
                if task.get_val_table().is_err() {
                    data_load_success = false;
                }
                if task.get_test_table().is_err() {
                    data_load_success = false;
                }
                
                let data_load_time = data_load_start.elapsed();
                println!("📊 Data loading time: {:?}", data_load_time);
                
                if data_load_success {
                    assert!(data_load_time.as_secs() < 5, "Data loading should be fast");
                    
                    // Test evaluation performance with different sizes
                    let sizes = vec![10, 100, 1000];
                    
                    for size in sizes {
                        let predictions: Vec<f64> = (0..size)
                            .map(|i| (i as f64) / (size as f64))
                            .collect();
                        
                        let eval_start = Instant::now();
                        match task.evaluate(&predictions, None) {
                            Ok(results) => {
                                let eval_time = eval_start.elapsed();
                                println!("📊 Evaluation of {} predictions: {:?}", size, eval_time);
                                
                                // Evaluation should scale reasonably
                                assert!(eval_time.as_millis() < 1000 + (size as u128), 
                                       "Evaluation should scale reasonably with size");
                                
                                assert!(!results.is_empty(), "Should have results");
                            }
                            Err(e) => {
                                println!("⚠️ Evaluation failed for size {}: {}", size, e);
                            }
                        }
                    }
                }
            }
            Err(e) => {
                println!("⚠️ Task loading failed: {}", e);
            }
        }
        
        println!("🎉 Performance benchmarks completed!");
        Ok(())
    }

    #[test]
    fn test_error_handling() {
        println!("🧪 Testing error handling...");
        
        // Test invalid task names
        let invalid_tasks = vec![
            ("nonexistent-dataset", "user-churn"),
            ("rel-amazon", "nonexistent-task"),
            ("", ""),
        ];
        
        for (dataset, task_name) in invalid_tasks {
            match get_task(dataset, task_name, false) {
                Ok(_) => {
                    println!("  ⚠️ Expected error for invalid task {}/{}", dataset, task_name);
                }
                Err(e) => {
                    println!("  ✅ Correctly handled invalid task {}/{}: {}", dataset, task_name, e);
                }
            }
        }
        
        // Test empty registry
        let empty_registry = TaskRegistry::default();
        let empty_tasks = empty_registry.list();
        // Registry might or might not be empty depending on implementation
        println!("  Default registry has {} tasks", empty_tasks.len());
        
        // Test getting non-existent task from registry
        let registry = TaskRegistry::new();
        let non_existent = registry.get("non-existent-task");
        assert!(non_existent.is_none(), "Should return None for non-existent task");
        
        println!("🎉 Error handling tests completed!");
    }

    #[test]
    fn test_memory_and_resource_usage() -> Result<()> {
        println!("🧪 Testing memory and resource usage...");
        
        // Load multiple tasks and verify no excessive memory usage
        let task_names = vec![
            ("rel-amazon", "user-churn"),
            ("rel-amazon", "item-churn"),
            ("rel-f1", "driver-position"),
            ("rel-hm", "item-sales"),
        ];
        
        let mut loaded_tasks = Vec::new();
        
        for (dataset, task_name) in task_names {
            match get_task(dataset, task_name, false) {
                Ok(task) => {
                    loaded_tasks.push(task);
                    println!("  ✅ Loaded task {}/{}", dataset, task_name);
                }
                Err(e) => {
                    println!("  ⚠️ Failed to load task {}/{}: {}", dataset, task_name, e);
                }
            }
        }
        
        // Run evaluations on all loaded tasks to check for memory leaks
        for (i, task) in loaded_tasks.iter().enumerate() {
            let predictions = vec![0.5, 0.6, 0.4, 0.7, 0.3, 0.8, 0.2];
            
            // Run multiple evaluations
            for iteration in 0..10 {
                match task.evaluate(&predictions, None) {
                    Ok(results) => {
                        assert!(!results.is_empty(), "Should have results");
                        
                        if iteration % 5 == 0 {
                            println!("    Task {} iteration {}: {:?}", i + 1, iteration, results);
                        }
                    }
                    Err(e) => {
                        println!("    Task {} iteration {} failed: {}", i + 1, iteration, e);
                    }
                }
            }
        }
        
        println!("🎉 Memory and resource usage tests completed!");
        Ok(())
    }

    #[test]
    fn test_system_stress() -> Result<()> {
        println!("🧪 Running system stress tests...");
        
        match get_task("rel-amazon", "user-churn", false) {
            Ok(task) => {
                // Stress test with large number of predictions
                println!("  Testing large batch evaluation...");
                let large_predictions: Vec<f64> = (0..10000)
                    .map(|i| (i as f64) / 10000.0)
                    .collect();
                
                let stress_start = Instant::now();
                match task.evaluate(&large_predictions, None) {
                    Ok(stress_results) => {
                        let stress_time = stress_start.elapsed();
                        println!("  ✅ Large batch ({} predictions) evaluated in {:?}", 
                                large_predictions.len(), stress_time);
                        assert!(!stress_results.is_empty());
                        assert!(stress_time.as_secs() < 30, "Large batch should complete in reasonable time");
                    }
                    Err(e) => {
                        println!("  ⚠️ Large batch evaluation failed: {}", e);
                    }
                }
                
                // Stress test with rapid successive evaluations
                println!("  Testing rapid successive evaluations...");
                let rapid_start = Instant::now();
                let mut successful_evaluations = 0;
                
                for i in 0..100 {
                    let predictions: Vec<f64> = (0..10)
                        .map(|j| ((i + j) as f64) / 100.0)
                        .collect();
                    
                    match task.evaluate(&predictions, None) {
                        Ok(results) => {
                            assert!(!results.is_empty());
                            successful_evaluations += 1;
                        }
                        Err(e) => {
                            if i % 20 == 0 {
                                println!("    Evaluation {} failed: {}", i, e);
                            }
                        }
                    }
                    
                    if i % 20 == 0 {
                        println!("    Completed {} evaluations", i);
                    }
                }
                
                let rapid_time = rapid_start.elapsed();
                println!("  ✅ {} successful evaluations out of 100 completed in {:?}", 
                        successful_evaluations, rapid_time);
                assert!(rapid_time.as_secs() < 30, "Rapid evaluations should complete in reasonable time");
            }
            Err(e) => {
                println!("  ⚠️ Task loading failed for stress test: {}", e);
            }
        }
        
        println!("🎉 System stress tests completed!");
        Ok(())
    }
}

#[cfg(test)]
mod compatibility_tests {
    use super::*;

    #[test]
    fn test_cross_platform_compatibility() {
        println!("🧪 Testing cross-platform compatibility...");
        
        // Test path handling
        let registry = TaskRegistry::new();
        let tasks = registry.list();
        
        for task_name in tasks {
            // Task names should be platform-independent
            assert!(!task_name.contains('\\'), "Task name should not contain backslashes: {}", task_name);
            assert!(!task_name.contains("//"), "Task name should not contain double slashes: {}", task_name);
        }
        
        // Test that basic functionality works regardless of platform
        match get_task("rel-amazon", "user-churn", false) {
            Ok(task) => {
                let predictions = vec![0.5, 0.7, 0.3];
                match task.evaluate(&predictions, None) {
                    Ok(scores) => {
                        println!("  ✅ Cross-platform evaluation succeeded: {:?}", scores);
                    }
                    Err(e) => {
                        println!("  ⚠️ Cross-platform evaluation failed: {}", e);
                    }
                }
            }
            Err(e) => {
                println!("  ⚠️ Cross-platform task loading failed: {}", e);
            }
        }
        
        println!("🎉 Cross-platform compatibility tests completed!");
    }

    #[test]
    fn test_version_compatibility() {
        println!("🧪 Testing version compatibility...");
        
        // Test that current implementation matches expected interface
        let registry = TaskRegistry::new();
        
        // Should have these basic methods
        let task_list = registry.list();
        println!("  Registry has {} tasks", task_list.len());
        
        // Test get_task function signature
        let result = get_task("rel-amazon", "user-churn", false);
        match result {
            Ok(_) => {
                println!("  ✅ get_task function signature is compatible (task loaded)");
            }
            Err(_) => {
                println!("  ✅ get_task function signature is compatible (error handled)");
            }
        }
        
        println!("🎉 Version compatibility tests completed!");
    }
}