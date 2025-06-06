use relbench::{
    get_dataset, get_task,
    utils::memory::{self, MemoryConfig},
    Result,
};
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<()> {
    // Configure memory settings
    let config = MemoryConfig {
        use_memory_mapping: true,
        cache_size: 1024 * 1024 * 1024, // 1GB
        prefetch_size: 1024 * 1024, // 1MB
    };
    memory::init(config)?;
    
    // Load dataset and task
    let dataset = get_dataset("rel-amazon", true)?;
    let task = get_task("rel-amazon", "user-churn", true)?;
    
    println!("Initial memory usage: {} MB", memory::get_usage()? / 1024 / 1024);
    
    // Memory-efficient data loading
    let start = Instant::now();
    
    // Get data using streaming iterator
    let mut train_iter = task.get_train_iter(1000)?;
    let mut val_iter = task.get_val_iter(1000)?;
    let mut test_iter = task.get_test_iter(1000)?;
    
    println!("Iterator setup took: {:?}", start.elapsed());
    println!("Memory usage after setup: {} MB", memory::get_usage()? / 1024 / 1024);
    
    // Process data in batches
    let mut total_samples = 0;
    let start = Instant::now();
    
    // Training data
    println!("\nProcessing training data:");
    while let Some(batch) = train_iter.next()? {
        total_samples += batch.len();
        if total_samples % 10000 == 0 {
            println!(
                "Processed {} samples, memory usage: {} MB",
                total_samples,
                memory::get_usage()? / 1024 / 1024
            );
        }
    }
    
    // Validation data
    println!("\nProcessing validation data:");
    while let Some(batch) = val_iter.next()? {
        total_samples += batch.len();
        if total_samples % 10000 == 0 {
            println!(
                "Processed {} samples, memory usage: {} MB",
                total_samples,
                memory::get_usage()? / 1024 / 1024
            );
        }
    }
    
    // Test data
    println!("\nProcessing test data:");
    while let Some(batch) = test_iter.next()? {
        total_samples += batch.len();
        if total_samples % 10000 == 0 {
            println!(
                "Processed {} samples, memory usage: {} MB",
                total_samples,
                memory::get_usage()? / 1024 / 1024
            );
        }
    }
    
    println!("\nProcessing completed:");
    println!("Total time: {:?}", start.elapsed());
    println!("Total samples: {}", total_samples);
    println!("Final memory usage: {} MB", memory::get_usage()? / 1024 / 1024);
    
    // Memory cleanup
    memory::cleanup()?;
    println!("Memory after cleanup: {} MB", memory::get_usage()? / 1024 / 1024);
    
    // Demonstrate memory mapping
    println!("\nTesting memory mapping:");
    let start = Instant::now();
    
    // Load large table with memory mapping
    let large_table = dataset.get_table_mapped("interactions")?;
    println!("Memory mapped table loaded in: {:?}", start.elapsed());
    println!("Memory usage with mapping: {} MB", memory::get_usage()? / 1024 / 1024);
    
    // Process memory mapped data
    let mut row_count = 0;
    for row in large_table.iter()? {
        row_count += 1;
        if row_count % 100000 == 0 {
            println!(
                "Processed {} rows, memory usage: {} MB",
                row_count,
                memory::get_usage()? / 1024 / 1024
            );
        }
    }
    
    println!("\nMemory mapping statistics:");
    println!("Total rows processed: {}", row_count);
    println!("Final memory usage: {} MB", memory::get_usage()? / 1024 / 1024);
    
    Ok(())
} 