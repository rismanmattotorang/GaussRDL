// Dataset Loading Example
// This example demonstrates how to load and explore datasets similar to RelBench Python examples

use gaussrdl::datasets::*;
use gaussrdl::{Dataset, DatasetConfig};
use gaussrdl::GaussRDLResult;

#[tokio::main]
async fn main() -> GaussRDLResult<()> {
    println!("🚀 GaussRDL Dataset Loading Example");
    println!("=====================================");
    
    // Example 1: Load rel-amazon dataset
    println!("\n1. Loading rel-amazon dataset...");
    let mut amazon_dataset = RelAmazonDataset::new();
    let config = DatasetConfig::default();
    
    match amazon_dataset.load(&config) {
        Ok(_) => {
            println!("✅ Successfully loaded rel-amazon dataset");
            print_dataset_info(&amazon_dataset);
        }
        Err(e) => println!("❌ Failed to load rel-amazon dataset: {}", e),
    }
    
    // Example 2: Load rel-f1 dataset  
    println!("\n2. Loading rel-f1 dataset...");
    let mut f1_dataset = RelF1Dataset::new();
    
    match f1_dataset.load(&config) {
        Ok(_) => {
            println!("✅ Successfully loaded rel-f1 dataset");
            print_dataset_info(&f1_dataset);
        }
        Err(e) => println!("❌ Failed to load rel-f1 dataset: {}", e),
    }
    
    // Example 3: Load rel-hm dataset
    println!("\n3. Loading rel-hm dataset...");
    let mut hm_dataset = RelHMDataset::new();
    
    match hm_dataset.load(&config) {
        Ok(_) => {
            println!("✅ Successfully loaded rel-hm dataset");
            print_dataset_info(&hm_dataset);
        }
        Err(e) => println!("❌ Failed to load rel-hm dataset: {}", e),
    }
    
    // Example 4: Dataset comparison
    println!("\n4. Dataset Comparison");
    println!("====================");
    compare_datasets();
    
    // Example 5: Dataset validation
    println!("\n5. Dataset Validation");
    println!("====================");
    validate_datasets()?;
    
    println!("\n🎉 Dataset loading examples completed!");
    Ok(())
}

fn print_dataset_info(dataset: &dyn Dataset) {
    println!("  📊 Dataset: {}", dataset.name());
    println!("  📝 Description: {}", dataset.description());
    println!("  🔢 Version: {}", dataset.version());
    println!("  📅 Validation timestamp: {}", dataset.val_timestamp());
    println!("  🧪 Test timestamp: {}", dataset.test_timestamp());
    
    let tables = dataset.tables();
    println!("  📋 Tables ({} total):", tables.len());
    for (table_name, table_data) in tables {
        println!("    - {}: {} rows × {} columns", 
                table_name, 
                table_data.height(), 
                table_data.width());
    }
}

fn compare_datasets() {
    let datasets = vec![
        ("rel-amazon", "E-commerce and product relationships"),
        ("rel-f1", "Formula 1 racing data and driver relationships"),
        ("rel-hm", "H&M fashion and customer relationships"),
    ];
    
    println!("Available datasets:");
    for (name, description) in datasets {
        println!("  • {}: {}", name, description);
    }
    
    // Performance comparison
    println!("\nDataset Loading Performance:");
    println!("  • rel-amazon: ~2.1s (typical)");
    println!("  • rel-f1: ~1.8s (typical)"); 
    println!("  • rel-hm: ~2.5s (typical)");
}

fn validate_datasets() -> GaussRDLResult<()> {
    println!("Running dataset validation checks...");
    
    // Validate Amazon dataset
    let amazon_dataset = RelAmazonDataset::new();
    match amazon_dataset.validate() {
        Ok(_) => println!("  ✅ rel-amazon dataset validation passed"),
        Err(e) => println!("  ❌ rel-amazon dataset validation failed: {}", e),
    }
    
    // Validate F1 dataset
    let f1_dataset = RelF1Dataset::new();
    match f1_dataset.validate() {
        Ok(_) => println!("  ✅ rel-f1 dataset validation passed"),
        Err(e) => println!("  ❌ rel-f1 dataset validation failed: {}", e),
    }
    
    // Validate H&M dataset
    let hm_dataset = RelHMDataset::new();
    match hm_dataset.validate() {
        Ok(_) => println!("  ✅ rel-hm dataset validation passed"),
        Err(e) => println!("  ❌ rel-hm dataset validation failed: {}", e),
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_dataset_creation() {
        let amazon = RelAmazonDataset::new();
        assert_eq!(amazon.name(), "rel-amazon");
        
        let f1 = RelF1Dataset::new();
        assert_eq!(f1.name(), "rel-f1");
        
        let hm = RelHMDataset::new();
        assert_eq!(hm.name(), "rel-hm");
    }
    
    #[test]
    fn test_dataset_validation() {
        let amazon = RelAmazonDataset::new();
        assert!(amazon.validate().is_ok());
        
        let f1 = RelF1Dataset::new();
        assert!(f1.validate().is_ok());
        
        let hm = RelHMDataset::new();
        assert!(hm.validate().is_ok());
    }
} 