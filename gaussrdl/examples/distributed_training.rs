// Distributed Training Example
// This example demonstrates how to set up and run distributed training

use gaussrdl::distributed::*;
use gaussrdl::GaussRDLResult;
use std::sync::Arc;
use tokio::time::{sleep, Duration};

#[tokio::main]
async fn main() -> GaussRDLResult<()> {
    println!("🚀 GaussRDL Distributed Training Example");
    println!("==========================================");
    
    // Example 1: Create distributed nodes
    println!("\n1. Creating Distributed Nodes");
    println!("=============================");
    let nodes = create_distributed_setup().await?;
    
    // Example 2: Simple distributed training
    println!("\n2. Simple Distributed Training");
    println!("==============================");
    run_simple_distributed_training(nodes.clone()).await?;
    
    // Example 3: Parameter server setup
    println!("\n3. Parameter Server Setup");
    println!("=========================");
    run_parameter_server_training().await?;
    
    // Example 4: Fault tolerance
    println!("\n4. Fault Tolerance Demo");
    println!("=======================");
    demonstrate_fault_tolerance().await?;
    
    // Example 5: Performance monitoring
    println!("\n5. Performance Monitoring");
    println!("=========================");
    monitor_training_performance().await?;
    
    println!("\n🎉 Distributed training examples completed!");
    Ok(())
}

async fn create_distributed_setup() -> GaussRDLResult<Vec<Arc<DistributedNode>>> {
    println!("Creating master and worker nodes...");
    
    // Create master node
    let master = DistributedNode::new(
        "master-001".to_string(),
        NodeRole::Master,
        "localhost".to_string(),
        8000,
    ).await?;
    
    println!("✅ Master node created: {}", master.get_info().await.id);
    
    // Create worker nodes
    let mut workers = Vec::new();
    for i in 1..=3 {
        let worker = DistributedNode::new(
            format!("worker-{:03}", i),
            NodeRole::Worker,
            "localhost".to_string(),
            8000 + i,
        ).await?;
        
        let worker_info = worker.get_info().await;
        println!("✅ Worker node created: {}", worker_info.id);
        
        // Register worker with master
        master.register_peer(worker_info).await?;
        workers.push(worker);
    }
    
    // Create parameter server
    let param_server = DistributedNode::new(
        "param-server-001".to_string(),
        NodeRole::Parameter,
        "localhost".to_string(),
        9000,
    ).await?;
    
    println!("✅ Parameter server created: {}", param_server.get_info().await.id);
    
    let mut all_nodes = vec![master];
    all_nodes.extend(workers);
    all_nodes.push(param_server);
    
    Ok(all_nodes)
}

async fn run_simple_distributed_training(nodes: Vec<Arc<DistributedNode>>) -> GaussRDLResult<()> {
    println!("Starting simple distributed training simulation...");
    
    // Create distributed trainer
    let config = InitConfig {
        batch_size: 64,
        learning_rate: 0.001,
        num_epochs: 10,
        sync_frequency: 5,
        model_config: std::collections::HashMap::new(),
    };
    
    let mut trainer = DistributedTrainer::new(config);
    
    // Add all nodes to trainer
    for node in &nodes {
        trainer.add_node(Arc::clone(node));
    }
    
    println!("  📊 Training configuration:");
    println!("    - Nodes: {}", nodes.len());
    println!("    - Batch size: 64");
    println!("    - Learning rate: 0.001");
    println!("    - Epochs: 10");
    println!("    - Sync frequency: 5");
    
    // Simulate training for a short time
    let _training_handle = tokio::spawn(async move {
        trainer.start_training().await
    });
    
    // Let it run for a few seconds
    sleep(Duration::from_secs(3)).await;
    
    // Stop training (in a real scenario, you'd wait for completion)
    println!("  ⏹️  Stopping training simulation...");
    for node in &nodes {
        node.stop().await?;
    }
    
    // Collect training statistics
    let stats = collect_training_stats(&nodes).await?;
    print_training_stats(&stats);
    
    Ok(())
}

async fn run_parameter_server_training() -> GaussRDLResult<()> {
    println!("Setting up parameter server training...");
    
    // Create parameter server
    let param_server = DistributedNode::new(
        "param-server".to_string(),
        NodeRole::Parameter,
        "localhost".to_string(),
        9001,
    ).await?;
    
    // Create workers
    let mut workers = Vec::new();
    for i in 1..=2 {
        let worker = DistributedNode::new(
            format!("ps-worker-{}", i),
            NodeRole::Worker,
            "localhost".to_string(),
            9001 + i,
        ).await?;
        workers.push(worker);
    }
    
    println!("  🗄️  Parameter server: {}", param_server.get_info().await.id);
    for worker in &workers {
        println!("  👷 Worker: {}", worker.get_info().await.id);
    }
    
    // Simulate parameter server training
    let _ps_handle = tokio::spawn({
        let param_server = Arc::clone(&param_server);
        async move {
            param_server.start().await
        }
    });
    
    // Start workers
    let mut _worker_handles = Vec::new();
    for worker in workers {
        let handle = tokio::spawn({
            let worker = Arc::clone(&worker);
            async move {
                worker.start().await
            }
        });
        _worker_handles.push(handle);
    }
    
    // Let training run briefly
    sleep(Duration::from_secs(2)).await;
    
    // Stop parameter server
    param_server.stop().await?;
    
    println!("  ✅ Parameter server training completed");
    
    Ok(())
}

async fn demonstrate_fault_tolerance() -> GaussRDLResult<()> {
    println!("Demonstrating fault tolerance...");
    
    // Create nodes
    let master = DistributedNode::new(
        "fault-master".to_string(),
        NodeRole::Master,
        "localhost".to_string(),
        10000,
    ).await?;
    
    let worker1 = DistributedNode::new(
        "fault-worker-1".to_string(),
        NodeRole::Worker,
        "localhost".to_string(),
        10001,
    ).await?;
    
    let worker2 = DistributedNode::new(
        "fault-worker-2".to_string(),
        NodeRole::Worker,
        "localhost".to_string(),
        10002,
    ).await?;
    
    // Register workers with master
    master.register_peer(worker1.get_info().await).await?;
    master.register_peer(worker2.get_info().await).await?;
    
    println!("  🔧 Starting fault tolerance test...");
    
    // Start training
    let _master_handle = tokio::spawn({
        let master = Arc::clone(&master);
        async move {
            master.start().await
        }
    });
    
    let _worker1_handle = tokio::spawn({
        let worker1 = Arc::clone(&worker1);
        async move {
            worker1.start().await
        }
    });
    
    let _worker2_handle = tokio::spawn({
        let worker2 = Arc::clone(&worker2);
        async move {
            worker2.start().await
        }
    });
    
    // Simulate worker failure after 1 second
    sleep(Duration::from_millis(1000)).await;
    println!("  ⚠️  Simulating worker-1 failure...");
    worker1.stop().await?;
    
    // Continue training with remaining worker
    sleep(Duration::from_millis(1000)).await;
    
    // Check if master is still running
    let master_stats = master.get_stats().await;
    println!("  📊 Master still running, epoch: {}", master_stats.epoch);
    
    // Stop remaining nodes
    master.stop().await?;
    worker2.stop().await?;
    
    println!("  ✅ Fault tolerance test completed");
    
    Ok(())
}

async fn monitor_training_performance() -> GaussRDLResult<()> {
    println!("Monitoring training performance...");
    
    // Create a simple setup
    let master = DistributedNode::new(
        "perf-master".to_string(),
        NodeRole::Master,
        "localhost".to_string(),
        11000,
    ).await?;
    
    let worker = DistributedNode::new(
        "perf-worker".to_string(),
        NodeRole::Worker,
        "localhost".to_string(),
        11001,
    ).await?;
    
    master.register_peer(worker.get_info().await).await?;
    
    // Start training
    let _master_handle = tokio::spawn({
        let master = Arc::clone(&master);
        async move {
            master.start().await
        }
    });
    
    let _worker_handle = tokio::spawn({
        let worker = Arc::clone(&worker);
        async move {
            worker.start().await
        }
    });
    
    // Monitor for several iterations
    for i in 1..=5 {
        sleep(Duration::from_millis(500)).await;
        
        let master_stats = master.get_stats().await;
        let worker_stats = worker.get_stats().await;
        
        println!("  📈 Iteration {}: Master loss: {:.4}, Worker loss: {:.4}", 
                i, master_stats.loss, worker_stats.loss);
        println!("    Throughput: {:.2} samples/sec, Sync time: {:.3}ms",
                master_stats.throughput, master_stats.sync_time);
    }
    
    // Stop training
    master.stop().await?;
    worker.stop().await?;
    
    println!("  ✅ Performance monitoring completed");
    
    Ok(())
}

async fn collect_training_stats(nodes: &[Arc<DistributedNode>]) -> GaussRDLResult<Vec<TrainingStats>> {
    let mut all_stats = Vec::new();
    
    for node in nodes {
        let stats = node.get_stats().await;
        all_stats.push(stats);
    }
    
    Ok(all_stats)
}

fn print_training_stats(stats: &[TrainingStats]) {
    println!("  📊 Training Statistics:");
    
    for (i, stat) in stats.iter().enumerate() {
        println!("    Node {}: epoch {}, loss: {:.4}, accuracy: {:.4}, throughput: {:.2}",
                i + 1, stat.epoch, stat.loss, stat.accuracy, stat.throughput);
    }
    
    // Calculate averages
    let avg_loss: f32 = stats.iter().map(|s| s.loss).sum::<f32>() / stats.len() as f32;
    let avg_accuracy: f32 = stats.iter().map(|s| s.accuracy).sum::<f32>() / stats.len() as f32;
    let avg_throughput: f32 = stats.iter().map(|s| s.throughput).sum::<f32>() / stats.len() as f32;
    
    println!("    📈 Averages: loss: {:.4}, accuracy: {:.4}, throughput: {:.2}",
            avg_loss, avg_accuracy, avg_throughput);
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_node_creation() {
        let node = DistributedNode::new(
            "test-node".to_string(),
            NodeRole::Master,
            "localhost".to_string(),
            8000,
        ).await.unwrap();
        
        let info = node.get_info().await;
        assert_eq!(info.id, "test-node");
        assert_eq!(info.role, NodeRole::Master);
    }
    
    #[tokio::test]
    async fn test_distributed_trainer() {
        let config = InitConfig::default();
        let trainer = DistributedTrainer::new(config);
        
        // Should start with no nodes
        let stats = trainer.get_training_stats().await.unwrap();
        assert_eq!(stats.len(), 0);
    }
    
    #[tokio::test]
    async fn test_node_stats() {
        let node = DistributedNode::new(
            "stats-test".to_string(),
            NodeRole::Worker,
            "localhost".to_string(),
            8001,
        ).await.unwrap();
        
        let stats = node.get_stats().await;
        assert_eq!(stats.epoch, 0);
        assert_eq!(stats.loss, 0.0);
        assert_eq!(stats.accuracy, 0.0);
    }
} 