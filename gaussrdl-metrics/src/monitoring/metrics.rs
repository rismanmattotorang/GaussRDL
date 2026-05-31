use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use serde::{Serialize, Deserialize};
use tracing::{info, warn};

/// Metrics for graph operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphMetrics {
    pub num_nodes: usize,
    pub num_edges: usize,
    pub avg_degree: f64,
    pub density: f64,
    pub processing_time_ms: f64,
    pub memory_used: f64,
}

impl Default for GraphMetrics {
    fn default() -> Self {
        Self {
            num_nodes: 0,
            num_edges: 0,
            avg_degree: 0.0,
            density: 0.0,
            processing_time_ms: 0.0,
            memory_used: 0.0,
        }
    }
}

impl GraphMetrics {
    pub fn new(num_nodes: usize, num_edges: usize) -> Self {
        let avg_degree = if num_nodes > 0 {
            2.0 * num_edges as f64 / num_nodes as f64
        } else {
            0.0
        };
        
        let density = if num_nodes > 1 {
            2.0 * num_edges as f64 / (num_nodes as f64 * (num_nodes - 1) as f64)
        } else {
            0.0
        };
        
        Self {
            num_nodes,
            num_edges,
            avg_degree,
            density,
            processing_time_ms: 0.0,
            memory_used: 0.0,
        }
    }
    
    pub fn set_processing_time(&mut self, duration: Duration) {
        self.processing_time_ms = duration.as_millis() as f64;
    }
    
    pub fn set_memory_used(&mut self, bytes: u64) {
        self.memory_used = bytes as f64;
    }
}

/// Training metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    pub epoch: usize,
    pub loss: f64,
    pub accuracy: Option<f64>,
    pub learning_rate: f64,
    pub batch_time_ms: f64,
    pub memory_used: f64,
}

impl TrainingMetrics {
    pub fn new(epoch: usize, loss: f64, learning_rate: f64) -> Self {
        Self {
            epoch,
            loss,
            accuracy: None,
            learning_rate,
            batch_time_ms: 0.0,
            memory_used: 0.0,
        }
    }
}

/// Inference metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceMetrics {
    pub latency_ms: f64,
    pub throughput: f64,
    pub memory_used: f64,
    pub batch_size: usize,
}

impl InferenceMetrics {
    pub fn new(latency_ms: f64, throughput: f64, batch_size: usize) -> Self {
        Self {
            latency_ms,
            throughput,
            memory_used: 0.0,
            batch_size,
        }
    }
}

/// Performance monitor for tracking system metrics
pub struct PerformanceMonitor {
    start_time: Instant,
    metrics: HashMap<String, f64>,
}

impl PerformanceMonitor {
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            metrics: HashMap::new(),
        }
    }
    
    pub fn record_metric(&mut self, name: &str, value: f64) {
        self.metrics.insert(name.to_string(), value);
    }
    
    pub fn get_metric(&self, name: &str) -> Option<f64> {
        self.metrics.get(name).copied()
    }
    
    pub fn elapsed_ms(&self) -> f64 {
        self.start_time.elapsed().as_millis() as f64
    }
    
    pub fn reset(&mut self) {
        self.start_time = Instant::now();
        self.metrics.clear();
    }
}

impl Default for PerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

pub struct MetricsCollector {
    training_metrics: Arc<RwLock<Vec<TrainingMetrics>>>,
    inference_metrics: Arc<RwLock<Vec<InferenceMetrics>>>,
    graph_metrics: Arc<RwLock<Vec<GraphMetrics>>>,
    start_time: Instant,
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            training_metrics: Arc::new(RwLock::new(Vec::new())),
            inference_metrics: Arc::new(RwLock::new(Vec::new())),
            graph_metrics: Arc::new(RwLock::new(Vec::new())),
            start_time: Instant::now(),
        }
    }
    
    pub fn record_training_metrics(&self, metrics: TrainingMetrics) {
        // Record to time series
        if let Ok(mut guard) = self.training_metrics.write() {
            guard.push(metrics.clone());
        }
        
        // Log metrics
        info!(
            "Training metrics - Loss: {:.4}, LR: {:.6}, Batch time: {:.1}ms, Memory: {:.1}MB",
            metrics.loss,
            metrics.learning_rate,
            metrics.batch_time_ms,
            metrics.memory_used
        );
        
        // Check for anomalies
        if metrics.loss.is_nan() || metrics.loss.is_infinite() {
            warn!("Training loss is NaN/Inf at epoch {}", metrics.epoch);
        }
    }
    
    pub fn record_inference_metrics(&self, metrics: InferenceMetrics) {
        // Record to time series
        if let Ok(mut guard) = self.inference_metrics.write() {
            guard.push(metrics.clone());
        }
        
        // Log metrics
        info!(
            "Inference metrics - Latency: {:.2}ms, Throughput: {:.1} samples/sec, Memory: {:.1}MB",
            metrics.latency_ms,
            metrics.throughput,
            metrics.memory_used
        );
    }
    
    pub fn record_graph_metrics(&self, metrics: GraphMetrics) {
        // Record to time series
        if let Ok(mut guard) = self.graph_metrics.write() {
            guard.push(metrics.clone());
        }
        
        // Log metrics
        info!(
            "Graph metrics - Nodes: {}, Edges: {}, Density: {:.4}, Processing: {:.2}ms",
            metrics.num_nodes,
            metrics.num_edges,
            metrics.density,
            metrics.processing_time_ms
        );
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSummary {
    pub total_epochs: usize,
    pub current_epoch: usize,
    pub average_loss: f64,
    pub current_loss: f64,
    pub average_throughput: f64,
    pub peak_memory_mb: f64,
    pub total_time: Duration,
}

impl Default for TrainingSummary {
    fn default() -> Self {
        Self {
            total_epochs: 0,
            current_epoch: 0,
            average_loss: 0.0,
            current_loss: 0.0,
            average_throughput: 0.0,
            peak_memory_mb: 0.0,
            total_time: Duration::from_secs(0),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceSummary {
    pub total_requests: usize,
    pub average_latency: f64,
    pub p50_latency: f64,
    pub p95_latency: f64,
    pub p99_latency: f64,
    pub average_throughput: f64,
    pub peak_memory_mb: f64,
}

impl Default for InferenceSummary {
    fn default() -> Self {
        Self {
            total_requests: 0,
            average_latency: 0.0,
            p50_latency: 0.0,
            p95_latency: 0.0,
            p99_latency: 0.0,
            average_throughput: 0.0,
            peak_memory_mb: 0.0,
        }
    }
} 