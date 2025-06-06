use relbench::{get_dataset, get_task, Result};

#[tokio::main]
async fn main() -> Result<()> {
    // Load dataset
    let dataset = get_dataset("rel-amazon", true)?;
    println!("Loaded dataset: {}", dataset.name());
    
    // Print available tables
    println!("Available tables:");
    for table in dataset.tables() {
        println!("- {}", table);
    }
    
    // Get task
    let task = get_task("rel-amazon", "user-churn", true)?;
    println!("\nLoaded task: {}", task.name());
    
    // Get data splits
    let train = task.get_train_table()?;
    let val = task.get_val_table()?;
    let test = task.get_test_table(true)?;
    
    println!("\nData split sizes:");
    println!("- Train: {} samples", train.len());
    println!("- Validation: {} samples", val.len());
    println!("- Test: {} samples", test.len());
    
    // Make dummy predictions (replace with actual model)
    let predictions = vec![0.5; test.len()];
    
    // Evaluate predictions
    let metrics = task.evaluate(&predictions, None)?;
    println!("\nEvaluation metrics:");
    for (name, value) in metrics.iter() {
        println!("- {}: {:.4}", name, value);
    }
    
    Ok(())
} 