use relbench::{get_dataset, DatasetConfig, Result};
use std::path::PathBuf;

#[tokio::test]
async fn test_dataset_loading() -> Result<()> {
    // Test default loading
    let dataset = get_dataset("rel-amazon", false)?;
    assert_eq!(dataset.name(), "rel-amazon");
    assert!(dataset.tables().len() > 0);
    
    Ok(())
}

#[tokio::test]
async fn test_dataset_config() -> Result<()> {
    let config = DatasetConfig {
        cache_dir: Some(PathBuf::from("./test_cache")),
        force_download: true,
        validate_hash: true,
        ..Default::default()
    };
    
    let dataset = get_dataset_with_config("rel-amazon", config)?;
    assert_eq!(dataset.name(), "rel-amazon");
    
    Ok(())
}

#[tokio::test]
async fn test_dataset_tables() -> Result<()> {
    let dataset = get_dataset("rel-amazon", false)?;
    
    // Check table existence
    assert!(dataset.has_table("users"));
    assert!(dataset.has_table("items"));
    assert!(dataset.has_table("interactions"));
    
    // Check table schemas
    let users = dataset.get_table("users")?;
    assert!(users.has_column("user_id"));
    assert!(users.has_column("features"));
    
    let items = dataset.get_table("items")?;
    assert!(items.has_column("item_id"));
    assert!(items.has_column("features"));
    
    let interactions = dataset.get_table("interactions")?;
    assert!(interactions.has_column("user_id"));
    assert!(interactions.has_column("item_id"));
    assert!(interactions.has_column("timestamp"));
    
    Ok(())
}

#[tokio::test]
async fn test_dataset_validation() -> Result<()> {
    let dataset = get_dataset("rel-amazon", false)?;
    
    // Validate dataset integrity
    assert!(dataset.validate().is_ok());
    
    // Check primary keys
    let users = dataset.get_table("users")?;
    assert_eq!(users.primary_key(), "user_id");
    
    let items = dataset.get_table("items")?;
    assert_eq!(items.primary_key(), "item_id");
    
    // Check foreign keys
    let interactions = dataset.get_table("interactions")?;
    let foreign_keys = interactions.foreign_keys();
    assert!(foreign_keys.contains(&("user_id".into(), "users".into(), "user_id".into())));
    assert!(foreign_keys.contains(&("item_id".into(), "items".into(), "item_id".into())));
    
    Ok(())
}

#[tokio::test]
async fn test_dataset_features() -> Result<()> {
    let dataset = get_dataset("rel-amazon", false)?;
    
    // Check feature dimensions
    let users = dataset.get_table("users")?;
    let user_features = users.get_features(0)?;
    assert_eq!(user_features.len(), 64); // Assuming 64-dim features
    
    let items = dataset.get_table("items")?;
    let item_features = items.get_features(0)?;
    assert_eq!(item_features.len(), 64); // Assuming 64-dim features
    
    Ok(())
}

#[tokio::test]
async fn test_dataset_temporal() -> Result<()> {
    let dataset = get_dataset("rel-amazon", false)?;
    
    // Check temporal ordering
    let interactions = dataset.get_table("interactions")?;
    let timestamps: Vec<i64> = interactions.get_column("timestamp")?;
    
    // Verify timestamps are ordered
    for window in timestamps.windows(2) {
        assert!(window[0] <= window[1]);
    }
    
    Ok(())
}

#[tokio::test]
async fn test_dataset_errors() {
    // Test invalid dataset name
    let result = get_dataset("invalid-dataset", false);
    assert!(result.is_err());
    
    // Test invalid table name
    let dataset = get_dataset("rel-amazon", false).unwrap();
    let result = dataset.get_table("invalid-table");
    assert!(result.is_err());
    
    // Test invalid column name
    let users = dataset.get_table("users").unwrap();
    let result = users.get_column("invalid-column");
    assert!(result.is_err());
}

#[tokio::test]
async fn test_dataset_memory_mapping() -> Result<()> {
    let dataset = get_dataset("rel-amazon", false)?;
    
    // Test memory mapped table access
    let interactions = dataset.get_table_mapped("interactions")?;
    assert!(interactions.is_memory_mapped());
    
    // Test iterator
    let mut count = 0;
    for row in interactions.iter()? {
        assert!(row.get("user_id").is_some());
        assert!(row.get("item_id").is_some());
        count += 1;
        if count >= 1000 {
            break;
        }
    }
    assert_eq!(count, 1000);
    
    Ok(())
}

#[tokio::test]
async fn test_dataset_caching() -> Result<()> {
    let config = DatasetConfig {
        cache_dir: Some(PathBuf::from("./test_cache")),
        force_download: false,
        validate_hash: true,
        ..Default::default()
    };
    
    // First load should download
    let dataset1 = get_dataset_with_config("rel-amazon", config.clone())?;
    assert_eq!(dataset1.name(), "rel-amazon");
    
    // Second load should use cache
    let start = std::time::Instant::now();
    let dataset2 = get_dataset_with_config("rel-amazon", config)?;
    assert_eq!(dataset2.name(), "rel-amazon");
    assert!(start.elapsed().as_secs() < 1); // Should be fast
    
    Ok(())
}

#[tokio::test]
async fn test_dataset_parallel() -> Result<()> {
    use rayon::prelude::*;
    
    let dataset = get_dataset("rel-amazon", false)?;
    let interactions = dataset.get_table("interactions")?;
    
    // Test parallel iteration
    let count: usize = (0..interactions.len())
        .into_par_iter()
        .map(|i| {
            let row = interactions.get_row(i).unwrap();
            assert!(row.get("user_id").is_some());
            assert!(row.get("item_id").is_some());
            1
        })
        .sum();
    
    assert_eq!(count, interactions.len());
    
    Ok(())
} 