use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use parking_lot::RwLock;
use serde::{Serialize, Deserialize};
use metrics::{Counter, Gauge, Histogram};
use tracing::{info, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    pub loss: f64,
    pub learning_rate: f64,
    pub gradient_norm: f64,
    pub samples_per_second: f64,
    pub memory_used: f64,
    pub gpu_utilization: Option<f64>,
    pub batch_size: usize,
    pub epoch: usize,
    pub step: usize,
    pub elapsed_time: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceMetrics {
    pub latency_ms: f64,
    pub throughput: f64,
    pub memory_used: f64,
    pub gpu_utilization: Option<f64>,
    pub batch_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphMetrics {
    pub num_nodes: usize,
    pub num_edges: usize,
    pub avg_degree: f64,
    pub density: f64,
    pub processing_time_ms: f64,
    pub memory_used: f64,
}

pub struct MetricsCollector {
    training_metrics: Arc<RwLock<Vec<TrainingMetrics>>>,
    inference_metrics: Arc<RwLock<Vec<InferenceMetrics>>>,
    graph_metrics: Arc<RwLock<Vec<GraphMetrics>>>,
    counters: HashMap<String, Counter>,
    gauges: HashMap<String, Gauge>,
    histograms: HashMap<String, Histogram>,
    start_time: Instant,
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            training_metrics: Arc::new(RwLock::new(Vec::new())),
            inference_metrics: Arc::new(RwLock::new(Vec::new())),
            graph_metrics: Arc::new(RwLock::new(Vec::new())),
            counters: HashMap::new(),
            gauges: HashMap::new(),
            histograms: HashMap::new(),
            start_time: Instant::now(),
        }
    }
    
    pub fn record_training_metrics(&self, metrics: TrainingMetrics) {
        // Record to time series
        self.training_metrics.write().push(metrics.clone());
        
        // Update prometheus-style metrics
        self.update_gauge("training_loss", metrics.loss);
        self.update_gauge("learning_rate", metrics.learning_rate);
        self.update_gauge("gradient_norm", metrics.gradient_norm);
        self.update_gauge("training_throughput", metrics.samples_per_second);
        self.update_gauge("memory_used_mb", metrics.memory_used);
        
        if let Some(gpu_util) = metrics.gpu_utilization {
            self.update_gauge("gpu_utilization", gpu_util);
        }
        
        // Log metrics
        info!(
            "Training metrics - Loss: {:.4}, LR: {:.6}, Samples/sec: {:.1}, Memory: {:.1}MB",
            metrics.loss,
            metrics.learning_rate,
            metrics.samples_per_second,
            metrics.memory_used
        );
        
        // Check for anomalies
        if metrics.loss.is_nan() || metrics.loss.is_infinite() {
            warn!("Training loss is NaN/Inf at step {}", metrics.step);
        }
        
        if metrics.gradient_norm > 100.0 {
            warn!("Large gradient norm ({:.2}) at step {}", metrics.gradient_norm, metrics.step);
        }
    }
    
    pub fn record_inference_metrics(&self, metrics: InferenceMetrics) {
        // Record to time series
        self.inference_metrics.write().push(metrics.clone());
        
        // Update prometheus-style metrics
        self.update_histogram("inference_latency_ms", metrics.latency_ms);
        self.update_gauge("inference_throughput", metrics.throughput);
        self.update_gauge("inference_memory_mb", metrics.memory_used);
        
        if let Some(gpu_util) = metrics.gpu_utilization {
            self.update_gauge("inference_gpu_utilization", gpu_util);
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
        self.graph_metrics.write().push(metrics.clone());
        
        // Update prometheus-style metrics
        self.update_gauge("graph_nodes", metrics.num_nodes as f64);
        self.update_gauge("graph_edges", metrics.num_edges as f64);
        self.update_gauge("graph_avg_degree", metrics.avg_degree);
        self.update_gauge("graph_density", metrics.density);
        self.update_histogram("graph_processing_time_ms", metrics.processing_time_ms);
        self.update_gauge("graph_memory_mb", metrics.memory_used);
        
        // Log metrics
        info!(
            "Graph metrics - Nodes: {}, Edges: {}, Processing time: {:.2}ms, Memory: {:.1}MB",
            metrics.num_nodes,
            metrics.num_edges,
            metrics.processing_time_ms,
            metrics.memory_used
        );
    }
    
    pub fn increment_counter(&self, name: &str) {
        if let Some(counter) = self.counters.get(name) {
            counter.increment(1);
        }
    }
    
    pub fn update_gauge(&self, name: &str, value: f64) {
        if let Some(gauge) = self.gauges.get(name) {
            gauge.set(value);
        }
    }
    
    pub fn update_histogram(&self, name: &str, value: f64) {
        if let Some(histogram) = self.histograms.get(name) {
            histogram.record(value);
        }
    }
    
    pub fn get_training_summary(&self) -> TrainingSummary {
        let metrics = self.training_metrics.read();
        let num_steps = metrics.len();
        
        if num_steps == 0 {
            return TrainingSummary::default();
        }
        
        let last_metrics = metrics.last().unwrap();
        let avg_loss: f64 = metrics.iter().map(|m| m.loss).sum::<f64>() / num_steps as f64;
        let avg_throughput: f64 = metrics.iter().map(|m| m.samples_per_second).sum::<f64>() / num_steps as f64;
        
        TrainingSummary {
            total_steps: num_steps,
            current_epoch: last_metrics.epoch,
            average_loss: avg_loss,
            current_loss: last_metrics.loss,
            average_throughput: avg_throughput,
            peak_memory_mb: metrics.iter().map(|m| m.memory_used).fold(0.0, f64::max),
            total_time: self.start_time.elapsed(),
        }
    }
    
    pub fn get_inference_summary(&self) -> InferenceSummary {
        let metrics = self.inference_metrics.read();
        let num_requests = metrics.len();
        
        if num_requests == 0 {
            return InferenceSummary::default();
        }
        
        let mut latencies: Vec<f64> = metrics.iter().map(|m| m.latency_ms).collect();
        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        InferenceSummary {
            total_requests: num_requests,
            average_latency: latencies.iter().sum::<f64>() / num_requests as f64,
            p50_latency: latencies[num_requests / 2],
            p95_latency: latencies[(num_requests as f64 * 0.95) as usize],
            p99_latency: latencies[(num_requests as f64 * 0.99) as usize],
            average_throughput: metrics.iter().map(|m| m.throughput).sum::<f64>() / num_requests as f64,
            peak_memory_mb: metrics.iter().map(|m| m.memory_used).fold(0.0, f64::max),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSummary {
    pub total_steps: usize,
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
            total_steps: 0,
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