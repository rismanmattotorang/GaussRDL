use relbench::{
    get_dataset, get_task,
    utils::parallel_processing::{self, ParallelConfig},
    Result,
};
use rayon::prelude::*;
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<()> {
    // Enable parallel processing
    let config = ParallelConfig {
        num_threads: 8,
        chunk_size: 1000,
    };
    parallel_processing::init(config)?;
    
    // Load dataset and task
    let dataset = get_dataset("rel-amazon", true)?;
    let task = get_task("rel-amazon", "user-churn", true)?;
    
    // Get data
    let train = task.get_train_table()?;
    let val = task.get_val_table()?;
    let test = task.get_test_table(true)?;
    
    println!("Dataset loaded with {} training samples", train.len());
    
    // Sequential processing
    let start = Instant::now();
    let seq_result: Vec<f32> = (0..train.len())
        .map(|i| {
            // Simulate some computation
            let x = i as f32;
            (x * x).sqrt().sin()
        })
        .collect();
    let seq_time = start.elapsed();
    println!("Sequential processing took: {:?}", seq_time);
    
    // Parallel processing
    let start = Instant::now();
    let par_result: Vec<f32> = (0..train.len())
        .into_par_iter()
        .map(|i| {
            // Same computation
            let x = i as f32;
            (x * x).sqrt().sin()
        })
        .collect();
    let par_time = start.elapsed();
    println!("Parallel processing took: {:?}", par_time);
    
    // Verify results are the same
    assert_eq!(seq_result, par_result);
    println!("Results verified to be identical");
    
    // Calculate speedup
    let speedup = seq_time.as_secs_f32() / par_time.as_secs_f32();
    println!("Speedup: {:.2}x", speedup);
    
    // Parallel batch processing example
    let batch_size = 1000;
    let num_batches = (train.len() + batch_size - 1) / batch_size;
    
    let start = Instant::now();
    let results: Vec<_> = (0..num_batches)
        .into_par_iter()
        .map(|batch_idx| {
            let start_idx = batch_idx * batch_size;
            let end_idx = (start_idx + batch_size).min(train.len());
            
            // Process batch
            let batch_result = process_batch(start_idx, end_idx);
            (batch_idx, batch_result)
        })
        .collect();
    let batch_time = start.elapsed();
    
    println!("Parallel batch processing took: {:?}", batch_time);
    println!("Processed {} batches", results.len());
    
    Ok(())
}

fn process_batch(start_idx: usize, end_idx: usize) -> Vec<f32> {
    // Simulate batch processing
    (start_idx..end_idx)
        .map(|i| {
            let x = i as f32;
            (x * x).sqrt().sin()
        })
        .collect()
} 