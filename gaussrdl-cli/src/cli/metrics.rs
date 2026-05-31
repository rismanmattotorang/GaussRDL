// src/cli/metrics.rs
use gaussrdl_core::{Result, Error};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub latency_ms: f64,
    pub throughput_ops_per_sec: f64,
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: f64,
    pub error_rate: f64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseMetrics {
    pub connection_pool_size: u32,
    pub active_connections: u32,
    pub query_time_avg_ms: f64,
    pub slow_queries: u32,
    pub cache_hit_ratio: f64,
    pub transactions_per_sec: f64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInferenceMetrics {
    pub model_name: String,
    pub batch_size: usize,
    pub inference_time_ms: f64,
    pub preprocessing_time_ms: f64,
    pub postprocessing_time_ms: f64,
    pub memory_peak_mb: f64,
    pub accuracy: Option<f64>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingProgress {
    pub epoch: u32,
    pub step: u32,
    pub loss: f64,
    pub validation_loss: Option<f64>,
    pub accuracy: f64,
    pub validation_accuracy: Option<f64>,
    pub learning_rate: f64,
    pub batch_time_ms: f64,
    pub data_loading_time_ms: f64,
    pub memory_usage_mb: f64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSummary {
    pub metric_type: String,
    pub count: usize,
    pub avg: f64,
    pub min: f64,
    pub max: f64,
    pub std_dev: f64,
    pub percentiles: HashMap<String, f64>,
}

pub struct MetricsCollector {
    performance_metrics: Vec<PerformanceMetrics>,
    database_metrics: Vec<DatabaseMetrics>,
    model_metrics: Vec<ModelInferenceMetrics>,
    training_metrics: Vec<TrainingProgress>,
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            performance_metrics: Vec::new(),
            database_metrics: Vec::new(),
            model_metrics: Vec::new(),
            training_metrics: Vec::new(),
        }
    }

    pub fn add_performance_metric(&mut self, metric: PerformanceMetrics) {
        self.performance_metrics.push(metric);
    }

    pub fn add_database_metric(&mut self, metric: DatabaseMetrics) {
        self.database_metrics.push(metric);
    }

    pub fn add_model_metric(&mut self, metric: ModelInferenceMetrics) {
        self.model_metrics.push(metric);
    }

    pub fn add_training_metric(&mut self, metric: TrainingProgress) {
        self.training_metrics.push(metric);
    }

    pub fn get_performance_summary(&self) -> Result<MetricsSummary> {
        if self.performance_metrics.is_empty() {
            return Err(Error::other("No performance metrics available"));
        }

        let latencies: Vec<f64> = self.performance_metrics
            .iter()
            .map(|m| m.latency_ms)
            .collect();

        Ok(calculate_summary("latency_ms", &latencies))
    }

    pub fn get_training_summary(&self) -> Result<MetricsSummary> {
        if self.training_metrics.is_empty() {
            return Err(Error::other("No training metrics available"));
        }

        let losses: Vec<f64> = self.training_metrics
            .iter()
            .map(|m| m.loss)
            .collect();

        Ok(calculate_summary("loss", &losses))
    }

    pub fn export_all_metrics(&self, format: &str) -> Result<String> {
        match format.to_lowercase().as_str() {
            "json" => {
                let all_metrics = serde_json::json!({
                    "performance": self.performance_metrics,
                    "database": self.database_metrics,
                    "model": self.model_metrics,
                    "training": self.training_metrics
                });
                serde_json::to_string_pretty(&all_metrics)
                    .map_err(|e| Error::other(&format!("JSON export failed: {}", e)))
            }
            "csv" => self.export_csv(),
            _ => Err(Error::other("Unsupported export format"))
        }
    }

    fn export_csv(&self) -> Result<String> {
        let mut csv = String::new();
        
        // Performance metrics CSV
        if !self.performance_metrics.is_empty() {
            csv.push_str("Performance Metrics\n");
            csv.push_str("Timestamp,Latency(ms),Throughput(ops/s),Memory(MB),CPU(%),Error Rate\n");
            for metric in &self.performance_metrics {
                csv.push_str(&format!(
                    "{},{},{},{},{},{}\n",
                    metric.timestamp,
                    metric.latency_ms,
                    metric.throughput_ops_per_sec,
                    metric.memory_usage_mb,
                    metric.cpu_usage_percent,
                    metric.error_rate
                ));
            }
            csv.push('\n');
        }

        // Training metrics CSV
        if !self.training_metrics.is_empty() {
            csv.push_str("Training Metrics\n");
            csv.push_str("Epoch,Step,Loss,Accuracy,Learning Rate,Batch Time(ms),Memory(MB)\n");
            for metric in &self.training_metrics {
                csv.push_str(&format!(
                    "{},{},{},{},{},{},{}\n",
                    metric.epoch,
                    metric.step,
                    metric.loss,
                    metric.accuracy,
                    metric.learning_rate,
                    metric.batch_time_ms,
                    metric.memory_usage_mb
                ));
            }
        }

        Ok(csv)
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

fn calculate_summary(metric_name: &str, values: &[f64]) -> MetricsSummary {
    let count = values.len();
    let sum: f64 = values.iter().sum();
    let avg = sum / count as f64;
    
    let min = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    
    // Calculate standard deviation
    let variance: f64 = values.iter()
        .map(|x| (x - avg).powi(2))
        .sum::<f64>() / count as f64;
    let std_dev = variance.sqrt();
    
    // Calculate percentiles
    let mut sorted_values = values.to_vec();
    sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let mut percentiles = HashMap::new();
    percentiles.insert("p50".to_string(), percentile(&sorted_values, 0.5));
    percentiles.insert("p90".to_string(), percentile(&sorted_values, 0.9));
    percentiles.insert("p95".to_string(), percentile(&sorted_values, 0.95));
    percentiles.insert("p99".to_string(), percentile(&sorted_values, 0.99));

    MetricsSummary {
        metric_type: metric_name.to_string(),
        count,
        avg,
        min,
        max,
        std_dev,
        percentiles,
    }
}

fn percentile(sorted_values: &[f64], p: f64) -> f64 {
    let index = (p * (sorted_values.len() - 1) as f64) as usize;
    sorted_values[index.min(sorted_values.len() - 1)]
}

pub fn format_metrics_table(metrics: &[PerformanceMetrics]) -> String {
    let mut output = String::new();
    output.push_str("Timestamp\t\tLatency(ms)\tThroughput\tMemory(MB)\tCPU%\tError%\n");
    output.push_str("─".repeat(80).as_str());
    output.push('\n');

    for metric in metrics {
        output.push_str(&format!(
            "{}\t{:.2}\t\t{:.1}\t\t{:.1}\t\t{:.1}\t{:.2}\n",
            metric.timestamp.format("%H:%M:%S"),
            metric.latency_ms,
            metric.throughput_ops_per_sec,
            metric.memory_usage_mb,
            metric.cpu_usage_percent,
            metric.error_rate
        ));
    }

    output
}

pub fn format_training_table(metrics: &[TrainingProgress]) -> String {
    let mut output = String::new();
    output.push_str("Epoch\tStep\tLoss\t\tAccuracy\tLR\t\tBatch Time(ms)\n");
    output.push_str("─".repeat(60).as_str());
    output.push('\n');

    for metric in metrics {
        output.push_str(&format!(
            "{}\t{}\t{:.4}\t\t{:.3}\t\t{:.6}\t{:.1}\n",
            metric.epoch,
            metric.step,
            metric.loss,
            metric.accuracy,
            metric.learning_rate,
            metric.batch_time_ms
        ));
    }

    output
}

pub fn generate_metrics_report(collector: &MetricsCollector) -> Result<String> {
    let mut report = String::new();
    report.push_str("=== METRICS REPORT ===\n\n");

    // Performance metrics summary
    if let Ok(perf_summary) = collector.get_performance_summary() {
        report.push_str("Performance Metrics:\n");
        report.push_str(&format!("  Count: {}\n", perf_summary.count));
        report.push_str(&format!("  Average: {:.2} ms\n", perf_summary.avg));
        report.push_str(&format!("  Min: {:.2} ms\n", perf_summary.min));
        report.push_str(&format!("  Max: {:.2} ms\n", perf_summary.max));
        report.push_str(&format!("  Std Dev: {:.2} ms\n", perf_summary.std_dev));
        report.push_str("  Percentiles:\n");
        for (p, value) in &perf_summary.percentiles {
            report.push_str(&format!("    {}: {:.2} ms\n", p, value));
        }
        report.push('\n');
    }

    // Training metrics summary
    if let Ok(train_summary) = collector.get_training_summary() {
        report.push_str("Training Metrics:\n");
        report.push_str(&format!("  Count: {}\n", train_summary.count));
        report.push_str(&format!("  Average Loss: {:.4}\n", train_summary.avg));
        report.push_str(&format!("  Min Loss: {:.4}\n", train_summary.min));
        report.push_str(&format!("  Max Loss: {:.4}\n", train_summary.max));
        report.push('\n');
    }

    Ok(report)
} 