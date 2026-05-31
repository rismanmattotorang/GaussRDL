// Task Evaluation Example
// This example demonstrates how to evaluate different tasks similar to RelBench Python examples

use gaussrdl::tasks::*;
use gaussrdl::GaussRDLResult;
use std::time::Instant;

#[tokio::main]
async fn main() -> GaussRDLResult<()> {
    println!("🚀 GaussRDL Task Evaluation Example");
    println!("====================================");
    
    // Example 1: User Churn Task
    println!("\n1. User Churn Task Evaluation");
    println!("=============================");
    evaluate_user_churn_task().await?;
    
    // Example 2: Item Churn Task
    println!("\n2. Item Churn Task Evaluation");
    println!("=============================");
    evaluate_item_churn_task().await?;
    
    // Example 3: Driver Position Task
    println!("\n3. Driver Position Task Evaluation");
    println!("==================================");
    evaluate_driver_position_task().await?;
    
    // Example 4: Item Sales Task
    println!("\n4. Item Sales Task Evaluation");
    println!("=============================");
    evaluate_item_sales_task().await?;
    
    // Example 5: Benchmark all tasks
    println!("\n5. Task Performance Benchmark");
    println!("=============================");
    benchmark_all_tasks().await?;
    
    println!("\n🎉 Task evaluation examples completed!");
    Ok(())
}

async fn evaluate_user_churn_task() -> GaussRDLResult<()> {
    println!("Loading user churn task for rel-amazon dataset...");
    
    let start_time = Instant::now();
    let task = get_task("rel-amazon", "user-churn", false)?;
    let load_time = start_time.elapsed();
    
    println!("✅ Task loaded in {:?}", load_time);
    println!("  📊 Task type: {:?}", task.task_type());
    println!("  ⏱️  Time delta: {:?}", task.timedelta());
    println!("  📈 Eval timestamps: {}", task.num_eval_timestamps());
    
    // Generate mock predictions
    let predictions = vec![0.8, 0.3, 0.9, 0.1, 0.7, 0.2, 0.85];
    
    let eval_start = Instant::now();
    let results = task.evaluate(&predictions, None)?;
    let eval_time = eval_start.elapsed();
    
    println!("  🎯 Evaluation results: {:?}", results);
    println!("  ⚡ Evaluation time: {:?}", eval_time);
    
    Ok(())
}

async fn evaluate_item_churn_task() -> GaussRDLResult<()> {
    println!("Loading item churn task for rel-amazon dataset...");
    
    let start_time = Instant::now();
    let task = get_task("rel-amazon", "item-churn", false)?;
    let load_time = start_time.elapsed();
    
    println!("✅ Task loaded in {:?}", load_time);
    
    // Generate mock predictions
    let predictions = vec![0.65, 0.41, 0.78, 0.23, 0.89, 0.12];
    
    let eval_start = Instant::now();
    let results = task.evaluate(&predictions, None)?;
    let eval_time = eval_start.elapsed();
    
    println!("  🎯 Evaluation results: {:?}", results);
    println!("  ⚡ Evaluation time: {:?}", eval_time);
    
    Ok(())
}

async fn evaluate_driver_position_task() -> GaussRDLResult<()> {
    println!("Loading driver position task for rel-f1 dataset...");
    
    let start_time = Instant::now();
    let task = get_task("rel-f1", "driver-position", false)?;
    let load_time = start_time.elapsed();
    
    println!("✅ Task loaded in {:?}", load_time);
    
    // Generate mock predictions (positions 1-20)
    let predictions = vec![1.2, 3.8, 2.1, 5.4, 8.7, 10.2, 15.1];
    
    let eval_start = Instant::now();
    let results = task.evaluate(&predictions, None)?;
    let eval_time = eval_start.elapsed();
    
    println!("  🎯 Evaluation results (MAE): {:?}", results);
    println!("  ⚡ Evaluation time: {:?}", eval_time);
    
    Ok(())
}

async fn evaluate_item_sales_task() -> GaussRDLResult<()> {
    println!("Loading item sales task for rel-hm dataset...");
    
    let start_time = Instant::now();
    let task = get_task("rel-hm", "item-sales", false)?;
    let load_time = start_time.elapsed();
    
    println!("✅ Task loaded in {:?}", load_time);
    
    // Generate mock predictions (sales volumes)
    let predictions = vec![95.5, 72.3, 118.7, 45.2, 88.9];
    
    let eval_start = Instant::now();
    let results = task.evaluate(&predictions, None)?;
    let eval_time = eval_start.elapsed();
    
    println!("  🎯 Evaluation results (MAE): {:?}", results);
    println!("  ⚡ Evaluation time: {:?}", eval_time);
    
    Ok(())
}

async fn benchmark_all_tasks() -> GaussRDLResult<()> {
    println!("Running performance benchmark on all tasks...");
    
    let tasks = vec![
        ("rel-amazon", "user-churn"),
        ("rel-amazon", "item-churn"),
        ("rel-f1", "driver-position"),
        ("rel-f1", "constructor-position"),
        ("rel-hm", "user-churn"),
        ("rel-hm", "item-sales"),
    ];
    
    let mut total_load_time = std::time::Duration::new(0, 0);
    let mut total_eval_time = std::time::Duration::new(0, 0);
    
    for (dataset, task_name) in tasks {
        print!("  📋 {}/{}: ", dataset, task_name);
        
        let load_start = Instant::now();
        match get_task(dataset, task_name, false) {
            Ok(task) => {
                let load_time = load_start.elapsed();
                total_load_time += load_time;
                
                // Generate mock predictions
                let predictions = vec![0.5, 0.7, 0.3, 0.8, 0.2];
                
                let eval_start = Instant::now();
                match task.evaluate(&predictions, None) {
                    Ok(results) => {
                        let eval_time = eval_start.elapsed();
                        total_eval_time += eval_time;
                        println!("✅ Load: {:?}, Eval: {:?}, Results: {:?}", 
                                load_time, eval_time, results);
                    }
                    Err(e) => println!("❌ Evaluation failed: {}", e),
                }
            }
            Err(e) => println!("❌ Load failed: {}", e),
        }
    }
    
    println!("\n📊 Benchmark Summary:");
    println!("  Total load time: {:?}", total_load_time);
    println!("  Total eval time: {:?}", total_eval_time);
    println!("  Average load time: {:?}", total_load_time / 6);
    println!("  Average eval time: {:?}", total_eval_time / 6);
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_task_loading() {
        // Test loading different tasks
        assert!(get_task("rel-amazon", "user-churn", false).is_ok());
        assert!(get_task("rel-f1", "driver-position", false).is_ok());
        assert!(get_task("rel-hm", "item-sales", false).is_ok());
    }
    
    #[test]
    fn test_task_evaluation() {
        let task = get_task("rel-amazon", "user-churn", false).unwrap();
        let predictions = vec![0.8, 0.3, 0.9, 0.1, 0.7];
        let results = task.evaluate(&predictions, None).unwrap();
        
        assert!(!results.is_empty());
        assert!(results[0] >= 0.0 && results[0] <= 1.0); // Accuracy should be between 0 and 1
    }
    
    #[test]
    fn test_empty_predictions() {
        let task = get_task("rel-amazon", "user-churn", false).unwrap();
        let predictions = vec![];
        let results = task.evaluate(&predictions, None).unwrap();
        
        assert_eq!(results, vec![0.0]);
    }
} 