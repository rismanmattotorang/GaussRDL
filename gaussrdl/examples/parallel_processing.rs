use gaussrdl::{get_dataset, get_task, GaussRDLResult};
use rayon::prelude::*;
use std::sync::Arc;
use std::time::Instant;

#[tokio::main]
async fn main() -> GaussRDLResult<()> {
    println!("⚡ GaussRDL Parallel Processing Example");
    println!("=========================================");
    
    // Load dataset and task
    println!("\n📦 Loading dataset and task...");
    let dataset = get_dataset("rel-amazon", true)?;
    let task = get_task("rel-amazon", "user-churn", true)?;
    
    println!("✅ Loaded dataset: {}", dataset.name());
    println!("✅ Loaded task: user-churn");
    
    // Wrap in Arc for shared access across threads
    let task_arc = Arc::new(task);
    
    // Demonstrate parallel batch processing
    println!("\n🔄 Parallel Batch Processing Demo");
    println!("----------------------------------");
    
    let batch_sizes = vec![50, 100, 200, 500];
    let num_batches = 8;
    
    for batch_size in batch_sizes {
        println!("\n📊 Processing {} batches of size {}", num_batches, batch_size);
        
        let start = Instant::now();
        
        // Generate batches in parallel
        let batches: Vec<Vec<f32>> = (0..num_batches)
            .into_par_iter()
            .map(|batch_id| {
                (0..batch_size)
                    .map(|i| {
                        0.3 + 0.4 * (i as f32 / batch_size as f32) + 0.1 * (batch_id as f32 / num_batches as f32)
                    })
                    .collect()
            })
            .collect();
        
        let generation_time = start.elapsed();
        
        // Process batches in parallel
        let start = Instant::now();
        let results: Vec<_> = batches
            .par_iter()
            .enumerate()
            .map(|(batch_id, predictions)| {
                let task_clone = Arc::clone(&task_arc);
                let predictions_f64: Vec<f64> = predictions.iter().map(|&x| x as f64).collect();
                let metrics = task_clone.evaluate(&predictions_f64, None).unwrap_or_default();
                (batch_id, metrics.len(), metrics.iter().sum::<f64>())
            })
            .collect();
        
        let processing_time = start.elapsed();
        
        println!("  ⏱️  Generation time: {:?}", generation_time);
        println!("  ⏱️  Processing time: {:?}", processing_time);
        println!("  📈 Results:");
        
        for (batch_id, metric_count, metric_sum) in results.iter().take(4) {
            println!("    - Batch {}: {} metrics, sum: {:.4}", batch_id, metric_count, metric_sum);
        }
    }
    
    // Demonstrate parallel model evaluation
    println!("\n🧠 Parallel Model Evaluation Demo");
    println!("----------------------------------");
    
    let model_configs = vec![
        ("MLP", 0.1),
        ("GNN", 0.2), 
        ("Transformer", 0.3),
        ("CNN", 0.4),
    ];
    
    let start = Instant::now();
    
    let model_results: Vec<_> = model_configs
        .par_iter()
        .map(|(model_name, base_score)| {
            let task_clone = Arc::clone(&task_arc);
            
            // Simulate model predictions
            let predictions: Vec<f32> = (0..100)
                .map(|i| base_score + 0.3 * (i as f32 / 100.0))
                .collect();
            
            let predictions_f64: Vec<f64> = predictions.iter().map(|&x| x as f64).collect();
            let metrics = task_clone.evaluate(&predictions_f64, None).unwrap_or_default();
            let avg_metric = if !metrics.is_empty() {
                metrics.iter().sum::<f64>() / metrics.len() as f64
            } else {
                0.0
            };
            
            (*model_name, avg_metric, metrics.len())
        })
        .collect();
    
    let parallel_eval_time = start.elapsed();
    
    println!("  ⏱️  Parallel evaluation time: {:?}", parallel_eval_time);
    println!("  📊 Model Results:");
    
    for (model_name, avg_metric, metric_count) in model_results {
        println!("    - {}: avg={:.4}, metrics={}", model_name, avg_metric, metric_count);
    }
    
    // Performance comparison
    println!("\n📈 Performance Analysis");
    println!("-----------------------");
    
    // Sequential vs Parallel comparison
    let test_size = 1000;
    let num_tests = 4;
    
    // Sequential processing
    let start = Instant::now();
    for i in 0..num_tests {
        let predictions: Vec<f32> = (0..test_size)
            .map(|j| 0.5 + 0.3 * (j as f32 / test_size as f32) + 0.1 * (i as f32))
            .collect();
        let predictions_f64: Vec<f64> = predictions.iter().map(|&x| x as f64).collect();
        let _ = task_arc.evaluate(&predictions_f64, None)?;
    }
    let sequential_time = start.elapsed();
    
    // Parallel processing
    let start = Instant::now();
    let _parallel_results: Vec<_> = (0..num_tests)
        .into_par_iter()
        .map(|i| {
            let task_clone = Arc::clone(&task_arc);
            let predictions: Vec<f32> = (0..test_size)
                .map(|j| 0.5 + 0.3 * (j as f32 / test_size as f32) + 0.1 * (i as f32))
                .collect();
            let predictions_f64: Vec<f64> = predictions.iter().map(|&x| x as f64).collect();
            task_clone.evaluate(&predictions_f64, None).unwrap_or_default()
        })
        .collect();
    let parallel_time = start.elapsed();
    
    println!("  ⏱️  Sequential time: {:?}", sequential_time);
    println!("  ⏱️  Parallel time: {:?}", parallel_time);
    
    if parallel_time.as_nanos() > 0 {
        let speedup = sequential_time.as_nanos() as f64 / parallel_time.as_nanos() as f64;
        println!("  🚀 Speedup: {:.2}x", speedup);
    }
    
    // Parallel processing features
    println!("\n💡 Parallel Processing Features:");
    println!("- ✅ Rayon for data parallelism");
    println!("- ✅ Arc for shared ownership across threads");
    println!("- ✅ Zero-cost thread safety with Rust");
    println!("- ✅ Automatic work stealing");
    println!("- ✅ CPU core utilization optimization");
    
    println!("\n✨ Parallel processing example completed!");
    Ok(())
} 