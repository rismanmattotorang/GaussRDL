use std::sync::Arc;
use tokio::sync::{mpsc, RwLock, Mutex};
use gaussrdl_core::{Error, Result};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// Node role in distributed training
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NodeRole {
    Master,
    Worker,
    Parameter,
}

/// Node status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeStatus {
    Active,
    Idle,
    Failed,
}

/// Node information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    pub id: String,
    pub role: NodeRole,
    pub status: NodeStatus,
    pub address: String,
    pub port: u16,
}

/// Training message types for distributed communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Message {
    // Control messages
    Initialize(InitConfig),
    Start,
    Stop,
    Shutdown,
    
    // Data messages
    Parameters(Vec<f32>),
    Gradients(Vec<f32>),
    Metrics(HashMap<String, f32>),
    
    // Status messages
    Status(NodeStatus),
    Error(String),
    
    // Worker messages
    WorkerJoined(String),
    TrainBatch(Vec<f32>),
    
    // Heartbeat for monitoring
    Heartbeat,
}

/// Initialization configuration for distributed training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitConfig {
    pub batch_size: usize,
    pub learning_rate: f32,
    pub model_config: HashMap<String, String>,
    pub num_epochs: usize,
    pub sync_frequency: usize,
}

impl Default for InitConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            learning_rate: 0.001,
            model_config: HashMap::new(),
            num_epochs: 100,
            sync_frequency: 10,
        }
    }
}

/// Training statistics for monitoring
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct TrainingStats {
    pub epoch: usize,
    pub loss: f32,
    pub accuracy: f32,
    pub throughput: f32,
    pub sync_time: f32,
}

/// Distributed node for distributed training
pub struct DistributedNode {
    info: Arc<RwLock<NodeInfo>>,
    peers: Arc<RwLock<HashMap<String, NodeInfo>>>,
    message_sender: mpsc::Sender<Message>,
    message_receiver: Arc<Mutex<mpsc::Receiver<Message>>>,
    is_running: Arc<std::sync::atomic::AtomicBool>,
    stats: Arc<RwLock<TrainingStats>>,
}

impl DistributedNode {
    /// Create a new distributed node
    pub async fn new(
        id: String,
        role: NodeRole,
        address: String,
        port: u16,
    ) -> Result<Arc<Self>> {
        let (tx, rx) = mpsc::channel(1000); // Larger buffer for better performance
        
        let node = Arc::new(Self {
            info: Arc::new(RwLock::new(NodeInfo {
                id,
                role,
                status: NodeStatus::Idle,
                address,
                port,
            })),
            peers: Arc::new(RwLock::new(HashMap::new())),
            message_sender: tx,
            message_receiver: Arc::new(Mutex::new(rx)),
            is_running: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            stats: Arc::new(RwLock::new(TrainingStats::default())),
        });
        
        Ok(node)
    }
    
    /// Start the distributed node
    pub async fn start(self: Arc<Self>) -> Result<()> {
        self.is_running.store(true, std::sync::atomic::Ordering::SeqCst);
        
        let role = {
            let info = self.info.read().await;
            info.role
        };
        
        match role {
            NodeRole::Master => self.run_master().await,
            NodeRole::Worker => self.run_worker().await,
            NodeRole::Parameter => self.run_parameter_server().await,
        }
    }
    
    /// Stop the distributed node
    pub async fn stop(&self) -> Result<()> {
        self.is_running.store(false, std::sync::atomic::Ordering::SeqCst);
        self.send_message(Message::Shutdown).await?;
        Ok(())
    }
    
    /// Run the distributed system in master mode
    async fn run_master(self: Arc<Self>) -> Result<()> {
        let mut parameters = Vec::<f32>::new();
        let mut worker_gradients: HashMap<String, Vec<f32>> = HashMap::new();
        let mut active_workers = 0;
        
        println!("🚀 Starting master node: {}", self.info.read().await.id);
        
        while self.is_running.load(std::sync::atomic::Ordering::SeqCst) {
            if let Some(msg) = self.receive_message().await? {
                match msg {
                    Message::WorkerJoined(worker_id) => {
                        println!("👥 Worker {} joined", worker_id);
                        active_workers += 1;
                        
                        // Send initialization config to new worker
                        let init_config = InitConfig::default();
                        self.send_to_worker(Message::Initialize(init_config), &worker_id).await?;
                        
                        // Send current parameters if available
                        if !parameters.is_empty() {
                            self.send_to_worker(Message::Parameters(parameters.clone()), &worker_id).await?;
                        }
                    }
                    
                    Message::Gradients(gradients) => {
                        let worker_id = format!("worker_{}", worker_gradients.len());
                        worker_gradients.insert(worker_id, gradients);
                        
                        // Aggregate gradients when all workers have contributed
                        if worker_gradients.len() >= active_workers {
                            parameters = self.aggregate_gradients(&worker_gradients)?;
                            worker_gradients.clear();
                            
                            // Broadcast updated parameters to all workers
                            self.broadcast_to_workers(Message::Parameters(parameters.clone())).await?;
                        }
                    }
                    
                    Message::Metrics(metrics) => {
                        self.update_stats_from_metrics(metrics).await?;
                    }
                    
                    Message::Shutdown => {
                        println!("🛑 Master shutting down");
                        break;
                    }
                    
                    _ => {}
                }
            }
        }
        
        Ok(())
    }
    
    /// Run the distributed system in worker mode
    async fn run_worker(self: Arc<Self>) -> Result<()> {
        let mut local_parameters = Vec::<f32>::new();
        let mut config = InitConfig::default();
        
        println!("👷 Starting worker node: {}", self.info.read().await.id);
        
        // Notify master that worker has joined
        self.send_to_master(Message::WorkerJoined(self.info.read().await.id.clone())).await?;
        
        while self.is_running.load(std::sync::atomic::Ordering::SeqCst) {
            if let Some(msg) = self.receive_message().await? {
                match msg {
                    Message::Initialize(init_config) => {
                        config = init_config;
                        println!("⚙️ Worker {} initialized with config", self.info.read().await.id);
                    }
                    
                    Message::Parameters(params) => {
                        local_parameters = params;
                        
                        // Simulate training on a batch
                        let gradients = self.simulate_training(&local_parameters, &config).await?;
                        
                        // Send gradients back to master
                        self.send_to_master(Message::Gradients(gradients)).await?;
                    }
                    
                    Message::TrainBatch(batch) => {
                        // Process training batch
                        let gradients = self.process_batch(&batch, &local_parameters, &config).await?;
                        self.send_to_master(Message::Gradients(gradients)).await?;
                    }
                    
                    Message::Shutdown => {
                        println!("🛑 Worker {} shutting down", self.info.read().await.id);
                        break;
                    }
                    
                    _ => {}
                }
            }
        }
        
        Ok(())
    }
    
    /// Run the distributed system in parameter server mode
    async fn run_parameter_server(self: Arc<Self>) -> Result<()> {
        let mut global_parameters = Vec::<f32>::new();
        let mut gradient_buffer: Vec<Vec<f32>> = Vec::new();
        
        println!("🗄️ Starting parameter server: {}", self.info.read().await.id);
        
        while self.is_running.load(std::sync::atomic::Ordering::SeqCst) {
            if let Some(msg) = self.receive_message().await? {
                match msg {
                    Message::Initialize(config) => {
                        // Initialize global parameters
                        global_parameters = vec![0.0; config.batch_size * 128]; // Example size
                        println!("⚙️ Parameter server initialized");
                    }
                    
                    Message::Gradients(gradients) => {
                        gradient_buffer.push(gradients);
                        
                        // Update parameters when enough gradients collected
                        if gradient_buffer.len() >= 2 {
                            self.update_global_parameters(&mut global_parameters, &gradient_buffer)?;
                            gradient_buffer.clear();
                            
                            // Broadcast updated parameters
                            self.broadcast(Message::Parameters(global_parameters.clone())).await?;
                        }
                    }
                    
                    Message::Shutdown => {
                        println!("🛑 Parameter server shutting down");
                        break;
                    }
                    
                    _ => {}
                }
            }
        }
        
        Ok(())
    }
    
    /// Send message to master node
    async fn send_to_master(&self, msg: Message) -> Result<()> {
        let peers = self.peers.read().await;
        let master = peers.values()
            .find(|node| node.role == NodeRole::Master)
            .ok_or_else(|| Error::training("Master node not found"))?;
            
        self.send_to_node(msg, &master.id).await
    }
    
    /// Send message to specific worker
    async fn send_to_worker(&self, msg: Message, worker_id: &str) -> Result<()> {
        self.send_to_node(msg, worker_id).await
    }
    
    /// Send message to specific node
    async fn send_to_node(&self, msg: Message, _node_id: &str) -> Result<()> {
        // In a real implementation, this would use network communication
        // For now, we use the local message channel for testing
        self.send_message(msg).await
    }
    
    /// Broadcast message to all workers
    async fn broadcast_to_workers(&self, msg: Message) -> Result<()> {
        let peers = self.peers.read().await;
        for node in peers.values() {
            if node.role == NodeRole::Worker {
                self.send_to_node(msg.clone(), &node.id).await?;
            }
        }
        Ok(())
    }
    
    /// Broadcast message to all nodes
    async fn broadcast(&self, msg: Message) -> Result<()> {
        let peers = self.peers.read().await;
        for node in peers.values() {
            self.send_to_node(msg.clone(), &node.id).await?;
        }
        Ok(())
    }
    
    /// Send message through the channel
    async fn send_message(&self, msg: Message) -> Result<()> {
        self.message_sender.send(msg).await
            .map_err(|e| Error::training(format!("Failed to send message: {}", e)))
    }
    
    /// Receive message from the channel
    async fn receive_message(&self) -> Result<Option<Message>> {
        let mut receiver = self.message_receiver.lock().await;
        Ok(receiver.recv().await)
    }
    
    /// Register peer node
    pub async fn register_peer(&self, peer: NodeInfo) -> Result<()> {
        self.peers.write().await.insert(peer.id.clone(), peer);
        Ok(())
    }
    
    /// Get node info
    pub async fn get_info(&self) -> NodeInfo {
        self.info.read().await.clone()
    }
    
    /// Get current training statistics
    pub async fn get_stats(&self) -> TrainingStats {
        self.stats.read().await.clone()
    }
    
    /// Aggregate gradients from multiple workers
    fn aggregate_gradients(&self, gradients: &HashMap<String, Vec<f32>>) -> Result<Vec<f32>> {
        if gradients.is_empty() {
            return Ok(Vec::new());
        }
        
        let first_grad = gradients.values().next().unwrap();
        let mut aggregated = vec![0.0; first_grad.len()];
        
        // Average gradients
        for grad_vec in gradients.values() {
            for (i, &grad) in grad_vec.iter().enumerate() {
                aggregated[i] += grad;
            }
        }
        
        let num_workers = gradients.len() as f32;
        for grad in &mut aggregated {
            *grad /= num_workers;
        }
        
        Ok(aggregated)
    }
    
    /// Simulate training for worker nodes
    async fn simulate_training(&self, parameters: &[f32], config: &InitConfig) -> Result<Vec<f32>> {
        // Simulate gradient computation
        let mut gradients = Vec::with_capacity(parameters.len());
        for (i, &param) in parameters.iter().enumerate() {
            let grad = param * config.learning_rate * (0.1 + 0.01 * i as f32);
            gradients.push(grad);
        }
        
        // Simulate some processing time
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        
        Ok(gradients)
    }
    
    /// Process a training batch
    async fn process_batch(&self, batch: &[f32], parameters: &[f32], config: &InitConfig) -> Result<Vec<f32>> {
        let mut gradients = Vec::with_capacity(parameters.len().max(batch.len()));
        
        for i in 0..parameters.len().max(batch.len()) {
            let param = parameters.get(i).unwrap_or(&0.0);
            let input = batch.get(i).unwrap_or(&0.0);
            let grad = (param - input) * config.learning_rate;
            gradients.push(grad);
        }
        
        Ok(gradients)
    }
    
    /// Update global parameters in parameter server
    fn update_global_parameters(&self, parameters: &mut Vec<f32>, gradients: &[Vec<f32>]) -> Result<()> {
        if gradients.is_empty() {
            return Ok(());
        }
        
        // Average gradients and update parameters
        for grad_vec in gradients {
            for (i, &grad) in grad_vec.iter().enumerate() {
                if i < parameters.len() {
                    parameters[i] -= grad; // Simple gradient descent
                }
            }
        }
        
        Ok(())
    }
    
    /// Update statistics from received metrics
    async fn update_stats_from_metrics(&self, metrics: HashMap<String, f32>) -> Result<()> {
        let mut stats = self.stats.write().await;
        
        if let Some(&loss) = metrics.get("loss") {
            stats.loss = loss;
        }
        if let Some(&accuracy) = metrics.get("accuracy") {
            stats.accuracy = accuracy;
        }
        if let Some(&throughput) = metrics.get("throughput") {
            stats.throughput = throughput;
        }
        
        Ok(())
    }
}

/// Distributed training coordinator
pub struct DistributedTrainer {
    nodes: Vec<Arc<DistributedNode>>,
    config: InitConfig,
}

impl DistributedTrainer {
    /// Create a new distributed trainer
    pub fn new(config: InitConfig) -> Self {
        Self {
            nodes: Vec::new(),
            config,
        }
    }
    
    /// Add a node to the distributed training setup
    pub fn add_node(&mut self, node: Arc<DistributedNode>) {
        self.nodes.push(node);
    }
    
    /// Start distributed training
    pub async fn start_training(&self) -> Result<()> {
        println!("🚀 Starting distributed training with {} nodes", self.nodes.len());
        
        // Start all nodes
        let mut handles = Vec::new();
        for node in &self.nodes {
            let node_clone = Arc::clone(node);
            let handle = tokio::spawn(async move {
                node_clone.start().await
            });
            handles.push(handle);
        }
        
        // Wait for all nodes to complete
        for handle in handles {
            handle.await.map_err(|e| Error::training(format!("Node failed: {}", e)))??;
        }
        
        Ok(())
    }
    
    /// Get training statistics from all nodes
    pub async fn get_training_stats(&self) -> Result<Vec<TrainingStats>> {
        let mut stats = Vec::new();
        for node in &self.nodes {
            stats.push(node.get_stats().await);
        }
        Ok(stats)
    }
}

/// Distributed processing utilities
pub struct DistributedProcessor {
    nodes: Vec<String>,
}

impl DistributedProcessor {
    /// Create a new distributed processor
    pub fn new(nodes: Vec<String>) -> Self {
        Self { nodes }
    }
    
    /// Get number of nodes
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
    
    /// Check if distributed processing is available
    pub fn is_available(&self) -> bool {
        !self.nodes.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{timeout, Duration};
    
    #[tokio::test]
    async fn test_distributed_node_creation() {
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
    async fn test_peer_registration() {
        let master = DistributedNode::new(
            "master".to_string(),
            NodeRole::Master,
            "localhost".to_string(),
            8000,
        ).await.unwrap();
        
        let worker_info = NodeInfo {
            id: "worker".to_string(),
            role: NodeRole::Worker,
            status: NodeStatus::Active,
            address: "localhost".to_string(),
            port: 8001,
        };
        
        master.register_peer(worker_info.clone()).await.unwrap();
        
        let peers = master.peers.read().await;
        assert!(peers.contains_key("worker"));
    }
    
    #[tokio::test]
    async fn test_message_passing() {
        let node = DistributedNode::new(
            "test-node".to_string(),
            NodeRole::Worker,
            "localhost".to_string(),
            8000,
        ).await.unwrap();
        
        // Send a message
        node.send_message(Message::Heartbeat).await.unwrap();
        
        // Receive the message
        let received = timeout(Duration::from_millis(100), node.receive_message()).await;
        assert!(received.is_ok());
    }
    
    #[tokio::test]
    async fn test_gradient_aggregation() {
        let node = DistributedNode::new(
            "master".to_string(),
            NodeRole::Master,
            "localhost".to_string(),
            8000,
        ).await.unwrap();
        
        let mut gradients = HashMap::new();
        gradients.insert("worker1".to_string(), vec![1.0, 2.0, 3.0]);
        gradients.insert("worker2".to_string(), vec![2.0, 3.0, 4.0]);
        
        let aggregated = node.aggregate_gradients(&gradients).unwrap();
        assert_eq!(aggregated, vec![1.5, 2.5, 3.5]);
    }
} 