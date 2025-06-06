use relbench::{get_task, TaskConfig, Result};
use std::path::PathBuf;

#[tokio::test]
async fn test_task_loading() -> Result<()> {
    // Test default loading
    let task = get_task("rel-amazon", "user-churn", false)?;
    assert_eq!(task.name(), "user-churn");
    assert_eq!(task.dataset(), "rel-amazon");
    
    Ok(())
}

#[tokio::test]
async fn test_task_config() -> Result<()> {
    let config = TaskConfig {
        split_ratio: (0.7, 0.1, 0.2),
        random_seed: Some(42),
        cache_dir: Some(PathBuf::from("./test_cache")),
        ..Default::default()
    };
    
    let task = get_task_with_config("rel-amazon", "user-churn", config)?;
    assert_eq!(task.name(), "user-churn");
    
    Ok(())
}

#[tokio::test]
async fn test_task_data_splits() -> Result<()> {
    let task = get_task("rel-amazon", "user-churn", false)?;
    
    // Get data splits
    let train = task.get_train_table()?;
    let val = task.get_val_table()?;
    let test = task.get_test_table(true)?;
    
    // Check split sizes
    let total = train.len() + val.len() + test.len();
    assert!(train.len() as f32 / total as f32 >= 0.6); // ~70%
    assert!(val.len() as f32 / total as f32 >= 0.05); // ~10%
    assert!(test.len() as f32 / total as f32 >= 0.15); // ~20%
    
    Ok(())
}

#[tokio::test]
async fn test_task_features() -> Result<()> {
    let task = get_task("rel-amazon", "user-churn", false)?;
    
    // Get training data
    let train = task.get_train_table()?;
    
    // Check feature dimensions
    let features = train.get_features(0)?;
    assert_eq!(features.len(), 64); // Assuming 64-dim features
    
    // Check label dimensions
    let labels = train.get_labels(0)?;
    assert_eq!(labels.len(), 1); // Binary classification
    
    Ok(())
}

#[tokio::test]
async fn test_task_evaluation() -> Result<()> {
    let task = get_task("rel-amazon", "user-churn", false)?;
    let test = task.get_test_table(true)?;
    
    // Make dummy predictions
    let predictions = vec![0.5; test.len()];
    
    // Evaluate predictions
    let metrics = task.evaluate(&predictions, None)?;
    
    // Check metrics
    assert!(metrics.contains_key("accuracy"));
    assert!(metrics.contains_key("auroc"));
    assert!(metrics.contains_key("f1"));
    
    // Check metric values are valid
    for (_, value) in metrics.iter() {
        assert!(*value >= 0.0 && *value <= 1.0);
    }
    
    Ok(())
}

#[tokio::test]
async fn test_task_batching() -> Result<()> {
    let task = get_task("rel-amazon", "user-churn", false)?;
    
    // Test batch iterator
    let batch_size = 32;
    let mut train_iter = task.get_train_iter(batch_size)?;
    
    let mut total_samples = 0;
    while let Some(batch) = train_iter.next()? {
        assert!(batch.len() <= batch_size);
        total_samples += batch.len();
    }
    
    let train = task.get_train_table()?;
    assert_eq!(total_samples, train.len());
    
    Ok(())
}

#[tokio::test]
async fn test_task_errors() {
    // Test invalid task name
    let result = get_task("rel-amazon", "invalid-task", false);
    assert!(result.is_err());
    
    // Test invalid dataset name
    let result = get_task("invalid-dataset", "user-churn", false);
    assert!(result.is_err());
    
    // Test invalid predictions
    let task = get_task("rel-amazon", "user-churn", false).unwrap();
    let result = task.evaluate(&vec![0.5], None);
    assert!(result.is_err()); // Wrong prediction length
}

#[tokio::test]
async fn test_task_caching() -> Result<()> {
    let config = TaskConfig {
        cache_dir: Some(PathBuf::from("./test_cache")),
        ..Default::default()
    };
    
    // First load should compute splits
    let task1 = get_task_with_config("rel-amazon", "user-churn", config.clone())?;
    let train1 = task1.get_train_table()?;
    
    // Second load should use cache
    let start = std::time::Instant::now();
    let task2 = get_task_with_config("rel-amazon", "user-churn", config)?;
    let train2 = task2.get_train_table()?;
    assert!(start.elapsed().as_secs() < 1); // Should be fast
    
    // Check splits are identical
    assert_eq!(train1.len(), train2.len());
    
    Ok(())
}

#[tokio::test]
async fn test_task_temporal() -> Result<()> {
    let task = get_task("rel-amazon", "user-churn", false)?;
    
    // Get temporal splits
    let train = task.get_train_table()?;
    let val = task.get_val_table()?;
    let test = task.get_test_table(true)?;
    
    // Get timestamps
    let train_times: Vec<i64> = train.get_column("timestamp")?;
    let val_times: Vec<i64> = val.get_column("timestamp")?;
    let test_times: Vec<i64> = test.get_column("timestamp")?;
    
    // Check temporal ordering
    let train_max = train_times.iter().max().unwrap();
    let val_min = val_times.iter().min().unwrap();
    let val_max = val_times.iter().max().unwrap();
    let test_min = test_times.iter().min().unwrap();
    
    assert!(train_max <= val_min);
    assert!(val_max <= test_min);
    
    Ok(())
}

#[tokio::test]
async fn test_task_parallel() -> Result<()> {
    use rayon::prelude::*;
    
    let task = get_task("rel-amazon", "user-churn", false)?;
    let train = task.get_train_table()?;
    
    // Test parallel feature extraction
    let features: Vec<_> = (0..train.len())
        .into_par_iter()
        .map(|i| train.get_features(i))
        .collect::<Result<_>>()?;
    
    assert_eq!(features.len(), train.len());
    assert_eq!(features[0].len(), 64); // Assuming 64-dim features
    
    Ok(())
} 