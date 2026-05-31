//! Metrics and evaluation utilities for GaussRDL
//! 
//! This module provides comprehensive metrics functionality for evaluating
//! model performance, data quality, and system performance.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use serde::{Deserialize, Serialize};
use crate::error::Error;

/// Result type for metric operations
pub type MetricResult<T> = std::result::Result<T, Error>;

/// Metric value types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricValue {
    /// Integer value
    Int(i64),
    /// Float value
    Float(f64),
    /// String value
    String(String),
    /// Boolean value
    Bool(bool),
    /// Vector of floats
    FloatVec(Vec<f64>),
    /// Vector of integers
    IntVec(Vec<i64>),
    /// Vector of strings
    StringVec(Vec<String>),
    /// Map of string to metric value
    Map(HashMap<String, Box<MetricValue>>),
}

impl MetricValue {
    /// Get as integer
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Self::Int(v) => Some(*v),
            Self::Float(v) => Some(*v as i64),
            _ => None,
        }
    }
    
    /// Get as float
    pub fn as_float(&self) -> Option<f64> {
        match self {
            Self::Int(v) => Some(*v as f64),
            Self::Float(v) => Some(*v),
            _ => None,
        }
    }
    
    /// Get as string
    pub fn as_string(&self) -> Option<&String> {
        match self {
            Self::String(v) => Some(v),
            _ => None,
        }
    }
    
    /// Get as boolean
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(v) => Some(*v),
            _ => None,
        }
    }
    
    /// Get as float vector
    pub fn as_float_vec(&self) -> Option<&Vec<f64>> {
        match self {
            Self::FloatVec(v) => Some(v),
            _ => None,
        }
    }
    
    /// Get as integer vector
    pub fn as_int_vec(&self) -> Option<&Vec<i64>> {
        match self {
            Self::IntVec(v) => Some(v),
            _ => None,
        }
    }
    
    /// Get as string vector
    pub fn as_string_vec(&self) -> Option<&Vec<String>> {
        match self {
            Self::StringVec(v) => Some(v),
            _ => None,
        }
    }
    
    /// Get as map
    pub fn as_map(&self) -> Option<&HashMap<String, Box<MetricValue>>> {
        match self {
            Self::Map(v) => Some(v),
            _ => None,
        }
    }
}

impl From<i64> for MetricValue {
    fn from(value: i64) -> Self {
        Self::Int(value)
    }
}

impl From<f64> for MetricValue {
    fn from(value: f64) -> Self {
        Self::Float(value)
    }
}

impl From<String> for MetricValue {
    fn from(value: String) -> Self {
        Self::String(value)
    }
}

impl From<bool> for MetricValue {
    fn from(value: bool) -> Self {
        Self::Bool(value)
    }
}

impl From<Vec<f64>> for MetricValue {
    fn from(value: Vec<f64>) -> Self {
        Self::FloatVec(value)
    }
}

impl From<Vec<i64>> for MetricValue {
    fn from(value: Vec<i64>) -> Self {
        Self::IntVec(value)
    }
}

impl From<Vec<String>> for MetricValue {
    fn from(value: Vec<String>) -> Self {
        Self::StringVec(value)
    }
}

/// Metric trait for implementing custom metrics
pub trait Metric: Send + Sync + std::fmt::Debug {
    /// Get metric name
    fn name(&self) -> &str;
    
    /// Get metric description
    fn description(&self) -> &str;
    
    /// Get metric type
    fn metric_type(&self) -> MetricType;
    
    /// Calculate metric value
    fn calculate(&self, predictions: &[f64], targets: &[f64]) -> MetricResult<MetricValue>;
    
    /// Get metric unit
    fn unit(&self) -> Option<&str> {
        None
    }
    
    /// Get metric range
    fn range(&self) -> Option<(f64, f64)> {
        None
    }
    
    /// Check if higher is better
    fn higher_is_better(&self) -> bool {
        true
    }
}

/// Metric types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricType {
    /// Classification metric
    Classification,
    /// Regression metric
    Regression,
    /// Ranking metric
    Ranking,
    /// Clustering metric
    Clustering,
    /// Custom metric
    Custom(String),
}

/// Built-in metrics
pub mod builtin {
    use super::*;
    use crate::error::Error;

    
    /// Accuracy metric for classification
    #[derive(Debug)]
    pub struct Accuracy;
    
    impl Metric for Accuracy {
        fn name(&self) -> &str {
            "accuracy"
        }
        
        fn description(&self) -> &str {
            "Classification accuracy (correct predictions / total predictions)"
        }
        
        fn metric_type(&self) -> MetricType {
            MetricType::Classification
        }
        
        fn calculate(&self, predictions: &[f64], targets: &[f64]) -> MetricResult<MetricValue> {
            if predictions.len() != targets.len() {
                return Err(Error::metrics("Predictions and targets must have the same length"));
            }
            
            if predictions.is_empty() {
                return Err(Error::metrics("Cannot calculate accuracy on empty data"));
            }
            
            let correct = predictions.iter()
                .zip(targets.iter())
                .filter(|(p, t)| (p.round() - *t).abs() < f64::EPSILON)
                .count();
            
            let accuracy = correct as f64 / predictions.len() as f64;
            Ok(MetricValue::Float(accuracy))
        }
        
        fn unit(&self) -> Option<&str> {
            Some("percentage")
        }
        
        fn range(&self) -> Option<(f64, f64)> {
            Some((0.0, 1.0))
        }
    }
    
    /// Precision metric for classification
    #[derive(Debug)]
    pub struct Precision;
    
    impl Metric for Precision {
        fn name(&self) -> &str {
            "precision"
        }
        
        fn description(&self) -> &str {
            "Classification precision (true positives / (true positives + false positives))"
        }
        
        fn metric_type(&self) -> MetricType {
            MetricType::Classification
        }
        
        fn calculate(&self, predictions: &[f64], targets: &[f64]) -> MetricResult<MetricValue> {
            if predictions.len() != targets.len() {
                return Err(Error::metrics("Predictions and targets must have the same length"));
            }
            
            if predictions.is_empty() {
                return Err(Error::metrics("Cannot calculate precision on empty data"));
            }
            
            let mut true_positives = 0;
            let mut false_positives = 0;
            
            for (p, t) in predictions.iter().zip(targets.iter()) {
                let pred_class = p.round() as i64;
                let true_class = t.round() as i64;
                
                if pred_class == 1 && true_class == 1 {
                    true_positives += 1;
                } else if pred_class == 1 && true_class == 0 {
                    false_positives += 1;
                }
            }
            
            let precision = if true_positives + false_positives == 0 {
                0.0
            } else {
                true_positives as f64 / (true_positives + false_positives) as f64
            };
            
            Ok(MetricValue::Float(precision))
        }
        
        fn unit(&self) -> Option<&str> {
            Some("percentage")
        }
        
        fn range(&self) -> Option<(f64, f64)> {
            Some((0.0, 1.0))
        }
    }
    
    /// Recall metric for classification
    #[derive(Debug)]
    pub struct Recall;
    
    impl Metric for Recall {
        fn name(&self) -> &str {
            "recall"
        }
        
        fn description(&self) -> &str {
            "Classification recall (true positives / (true positives + false negatives))"
        }
        
        fn metric_type(&self) -> MetricType {
            MetricType::Classification
        }
        
        fn calculate(&self, predictions: &[f64], targets: &[f64]) -> MetricResult<MetricValue> {
            if predictions.len() != targets.len() {
                return Err(Error::metrics("Predictions and targets must have the same length"));
            }
            
            if predictions.is_empty() {
                return Err(Error::metrics("Cannot calculate recall on empty data"));
            }
            
            let mut true_positives = 0;
            let mut false_negatives = 0;
            
            for (p, t) in predictions.iter().zip(targets.iter()) {
                let pred_class = p.round() as i64;
                let true_class = t.round() as i64;
                
                if pred_class == 1 && true_class == 1 {
                    true_positives += 1;
                } else if pred_class == 0 && true_class == 1 {
                    false_negatives += 1;
                }
            }
            
            let recall = if true_positives + false_negatives == 0 {
                0.0
            } else {
                true_positives as f64 / (true_positives + false_negatives) as f64
            };
            
            Ok(MetricValue::Float(recall))
        }
        
        fn unit(&self) -> Option<&str> {
            Some("percentage")
        }
        
        fn range(&self) -> Option<(f64, f64)> {
            Some((0.0, 1.0))
        }
    }
    
    /// F1 score metric for classification
    #[derive(Debug)]
    pub struct F1Score;
    
    impl Metric for F1Score {
        fn name(&self) -> &str {
            "f1_score"
        }
        
        fn description(&self) -> &str {
            "F1 score (harmonic mean of precision and recall)"
        }
        
        fn metric_type(&self) -> MetricType {
            MetricType::Classification
        }
        
        fn calculate(&self, predictions: &[f64], targets: &[f64]) -> MetricResult<MetricValue> {
            if predictions.len() != targets.len() {
                return Err(Error::metrics("Predictions and targets must have the same length"));
            }
            
            if predictions.is_empty() {
                return Err(Error::metrics("Cannot calculate F1 score on empty data"));
            }
            
            let precision_metric = Precision;
            let recall_metric = Recall;
            
            let precision = precision_metric.calculate(predictions, targets)?;
            let recall = recall_metric.calculate(predictions, targets)?;
            
            let precision_val = precision.as_float().unwrap_or(0.0);
            let recall_val = recall.as_float().unwrap_or(0.0);
            
            let f1_score = if precision_val + recall_val == 0.0 {
                0.0
            } else {
                2.0 * precision_val * recall_val / (precision_val + recall_val)
            };
            
            Ok(MetricValue::Float(f1_score))
        }
        
        fn unit(&self) -> Option<&str> {
            Some("percentage")
        }
        
        fn range(&self) -> Option<(f64, f64)> {
            Some((0.0, 1.0))
        }
    }
    
    /// Mean squared error metric for regression
    #[derive(Debug)]
    pub struct MeanSquaredError;
    
    impl Metric for MeanSquaredError {
        fn name(&self) -> &str {
            "mse"
        }
        
        fn description(&self) -> &str {
            "Mean squared error"
        }
        
        fn metric_type(&self) -> MetricType {
            MetricType::Regression
        }
        
        fn calculate(&self, predictions: &[f64], targets: &[f64]) -> MetricResult<MetricValue> {
            if predictions.len() != targets.len() {
                return Err(Error::metrics("Predictions and targets must have the same length"));
            }
            
            if predictions.is_empty() {
                return Err(Error::metrics("Cannot calculate MSE on empty data"));
            }
            
            let mse = predictions.iter()
                .zip(targets.iter())
                .map(|(p, t)| (p - t).powi(2))
                .sum::<f64>() / predictions.len() as f64;
            
            Ok(MetricValue::Float(mse))
        }
        
        fn higher_is_better(&self) -> bool {
            false
        }
    }
    
    /// Root mean squared error metric for regression
    #[derive(Debug)]
    pub struct RootMeanSquaredError;
    
    impl Metric for RootMeanSquaredError {
        fn name(&self) -> &str {
            "rmse"
        }
        
        fn description(&self) -> &str {
            "Root mean squared error"
        }
        
        fn metric_type(&self) -> MetricType {
            MetricType::Regression
        }
        
        fn calculate(&self, predictions: &[f64], targets: &[f64]) -> MetricResult<MetricValue> {
            if predictions.len() != targets.len() {
                return Err(Error::metrics("Predictions and targets must have the same length"));
            }
            
            if predictions.is_empty() {
                return Err(Error::metrics("Cannot calculate RMSE on empty data"));
            }
            
            let mse = predictions.iter()
                .zip(targets.iter())
                .map(|(p, t)| (p - t).powi(2))
                .sum::<f64>() / predictions.len() as f64;
            
            let rmse = mse.sqrt();
            Ok(MetricValue::Float(rmse))
        }
        
        fn higher_is_better(&self) -> bool {
            false
        }
    }
    
    /// Mean absolute error metric for regression
    #[derive(Debug)]
    pub struct MeanAbsoluteError;
    
    impl Metric for MeanAbsoluteError {
        fn name(&self) -> &str {
            "mae"
        }
        
        fn description(&self) -> &str {
            "Mean absolute error"
        }
        
        fn metric_type(&self) -> MetricType {
            MetricType::Regression
        }
        
        fn calculate(&self, predictions: &[f64], targets: &[f64]) -> MetricResult<MetricValue> {
            if predictions.len() != targets.len() {
                return Err(Error::metrics("Predictions and targets must have the same length"));
            }
            
            if predictions.is_empty() {
                return Err(Error::metrics("Cannot calculate MAE on empty data"));
            }
            
            let mae = predictions.iter()
                .zip(targets.iter())
                .map(|(p, t)| (p - t).abs())
                .sum::<f64>() / predictions.len() as f64;
            
            Ok(MetricValue::Float(mae))
        }
        
        fn higher_is_better(&self) -> bool {
            false
        }
    }
    
    /// R-squared metric for regression
    #[derive(Debug)]
    pub struct RSquared;
    
    impl Metric for RSquared {
        fn name(&self) -> &str {
            "r2"
        }
        
        fn description(&self) -> &str {
            "R-squared (coefficient of determination)"
        }
        
        fn metric_type(&self) -> MetricType {
            MetricType::Regression
        }
        
        fn calculate(&self, predictions: &[f64], targets: &[f64]) -> MetricResult<MetricValue> {
            if predictions.len() != targets.len() {
                return Err(Error::metrics("Predictions and targets must have the same length"));
            }
            
            if predictions.is_empty() {
                return Err(Error::metrics("Cannot calculate R-squared on empty data"));
            }
            
            let mean_target = targets.iter().sum::<f64>() / targets.len() as f64;
            
            let ss_res: f64 = predictions.iter()
                .zip(targets.iter())
                .map(|(p, t)| (t - p).powi(2))
                .sum();
            
            let ss_tot: f64 = targets.iter()
                .map(|t| (t - mean_target).powi(2))
                .sum();
            
            let r2 = if ss_tot == 0.0 {
                1.0
            } else {
                1.0 - (ss_res / ss_tot)
            };
            
            Ok(MetricValue::Float(r2))
        }
        
        fn range(&self) -> Option<(f64, f64)> {
            Some((0.0, 1.0))
        }
    }
    
    /// Area under ROC curve metric
    #[derive(Debug)]
    pub struct AUC;
    
    impl Metric for AUC {
        fn name(&self) -> &str {
            "auc"
        }
        
        fn description(&self) -> &str {
            "Area under ROC curve"
        }
        
        fn metric_type(&self) -> MetricType {
            MetricType::Classification
        }
        
        fn calculate(&self, predictions: &[f64], targets: &[f64]) -> MetricResult<MetricValue> {
            if predictions.len() != targets.len() {
                return Err(Error::metrics("Predictions and targets must have the same length"));
            }
            
            if predictions.is_empty() {
                return Err(Error::metrics("Cannot calculate AUC on empty data"));
            }
            
            // Simple AUC calculation using trapezoidal rule
            let mut data: Vec<(f64, f64)> = predictions.iter()
                .zip(targets.iter())
                .map(|(p, t)| (*p, *t))
                .collect();
            
            data.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            
            let mut auc = 0.0;
            let mut tp = 0.0;
            let mut fp = 0.0;
            let mut prev_score = data[0].0;
            
            let total_positives = targets.iter().filter(|&&t| t > 0.5).count() as f64;
            let total_negatives = targets.len() as f64 - total_positives;
            
            for (score, target) in data {
                if score != prev_score {
                    auc += trapezoid_area(fp / total_negatives, tp / total_positives);
                    prev_score = score;
                }
                
                if target > 0.5 {
                    tp += 1.0;
                } else {
                    fp += 1.0;
                }
            }
            
            auc += trapezoid_area(fp / total_negatives, tp / total_positives);
            
            Ok(MetricValue::Float(auc))
        }
        
        fn unit(&self) -> Option<&str> {
            Some("percentage")
        }
        
        fn range(&self) -> Option<(f64, f64)> {
            Some((0.0, 1.0))
        }
    }
    
    fn trapezoid_area(x: f64, y: f64) -> f64 {
        x * y / 2.0
    }
}

/// Metric registry for managing metrics
pub struct MetricRegistry {
    metrics: Arc<RwLock<HashMap<String, Box<dyn Metric>>>>,
}

impl MetricRegistry {
    /// Create a new metric registry
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Register a metric
    pub fn register(&self, metric: Box<dyn Metric>) -> MetricResult<()> {
        let name = metric.name().to_string();
        let mut metrics = self.metrics.write()
            .map_err(|_| Error::metrics("Failed to acquire write lock"))?;
        
        if metrics.contains_key(&name) {
            return Err(Error::metrics_with_name(
                format!("Metric '{}' already registered", name),
                name
            ));
        }
        
        metrics.insert(name, metric);
        Ok(())
    }
    


    /// Change the get method to use a closure pattern
    pub fn with_metric<F, T>(&self, name: &str, f: F) -> MetricResult<T>
    where
        F: FnOnce(&Box<dyn Metric>) -> T,
    {
        let metrics = self.metrics.read()
            .map_err(|_| Error::metrics("Failed to acquire read lock"))?;
        
        let metric = metrics.get(name)
            .ok_or_else(|| Error::metrics_with_name(
                format!("Metric '{}' not found", name),
                name.to_string()
            ))?;
        
        Ok(f(metric))
    }
    
    /// List all metric names
    pub fn list_metrics(&self) -> MetricResult<Vec<String>> {
        let metrics = self.metrics.read()
            .map_err(|_| Error::metrics("Failed to acquire read lock"))?;
        
        Ok(metrics.keys().cloned().collect())
    }
    
    /// Remove a metric
    pub fn remove(&self, name: &str) -> MetricResult<()> {
        let mut metrics = self.metrics.write()
            .map_err(|_| Error::metrics("Failed to acquire write lock"))?;
        
        if metrics.remove(name).is_none() {
            return Err(Error::metrics_with_name(
                format!("Metric '{}' not found", name),
                name.to_string()
            ));
        }
        
        Ok(())
    }
    
    /// Get registry with built-in metrics
    pub fn with_builtin_metrics() -> Self {
        let registry = Self::new();
        
        let _ = registry.register(Box::new(builtin::Accuracy));
        let _ = registry.register(Box::new(builtin::Precision));
        let _ = registry.register(Box::new(builtin::Recall));
        let _ = registry.register(Box::new(builtin::F1Score));
        let _ = registry.register(Box::new(builtin::MeanSquaredError));
        let _ = registry.register(Box::new(builtin::RootMeanSquaredError));
        let _ = registry.register(Box::new(builtin::MeanAbsoluteError));
        let _ = registry.register(Box::new(builtin::RSquared));
        let _ = registry.register(Box::new(builtin::AUC));
        
        registry
    }
}

/// Evaluator for calculating multiple metrics
pub struct Evaluator {
    registry: MetricRegistry,
}

impl Evaluator {
    /// Create a new evaluator
    pub fn new() -> Self {
        Self {
            registry: MetricRegistry::with_builtin_metrics(),
        }
    }
    
    /// Create evaluator with custom registry
    pub fn with_registry(registry: MetricRegistry) -> Self {
        Self { registry }
    }
    
    /// Evaluate predictions using specified metrics
    pub fn evaluate(
        &self,
        predictions: &[f64],
        targets: &[f64],
        metric_names: &[&str],
    ) -> MetricResult<HashMap<String, MetricValue>> {
        let mut results = HashMap::new();
        
        for metric_name in metric_names {
            let value = self.registry.with_metric(metric_name, |metric| {
                metric.calculate(predictions, targets)
            })??;
            results.insert(metric_name.to_string(), value);
        }
        
        Ok(results)
    }
    
    /// Evaluate predictions using all available metrics
    pub fn evaluate_all(
        &self,
        predictions: &[f64],
        targets: &[f64],
    ) -> MetricResult<HashMap<String, MetricValue>> {
        let metric_names: Vec<String> = self.registry.list_metrics()?;
        let metric_names: Vec<&str> = metric_names.iter().map(|s| s.as_str()).collect();
        
        self.evaluate(predictions, targets, &metric_names)
    }
    
    /// Get metric information
    pub fn get_metric_info(&self, name: &str) -> MetricResult<MetricInfo> {
        self.registry.with_metric(name, |metric| {
            MetricInfo {
                name: metric.name().to_string(),
                description: metric.description().to_string(),
                metric_type: metric.metric_type(),
                unit: metric.unit().map(|u| u.to_string()),
                range: metric.range(),
                higher_is_better: metric.higher_is_better(),
            }
        })
    }
    
    /// List all metric information
    pub fn list_metric_info(&self) -> MetricResult<Vec<MetricInfo>> {
        let metric_names = self.registry.list_metrics()?;
        let mut info = Vec::new();
        
        for name in metric_names {
            info.push(self.get_metric_info(&name)?);
        }
        
        Ok(info)
    }
}

/// Metric information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricInfo {
    /// Metric name
    pub name: String,
    /// Metric description
    pub description: String,
    /// Metric type
    pub metric_type: MetricType,
    /// Metric unit
    pub unit: Option<String>,
    /// Metric range
    pub range: Option<(f64, f64)>,
    /// Whether higher values are better
    pub higher_is_better: bool,
}

/// Evaluation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationResult {
    /// Metric values
    pub metrics: HashMap<String, MetricValue>,
    /// Evaluation metadata
    pub metadata: HashMap<String, String>,
    /// Evaluation timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl EvaluationResult {
    /// Create a new evaluation result
    pub fn new(metrics: HashMap<String, MetricValue>) -> Self {
        Self {
            metrics,
            metadata: HashMap::new(),
            timestamp: chrono::Utc::now(),
        }
    }
    
    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
    
    /// Get metric value
    pub fn get_metric(&self, name: &str) -> Option<&MetricValue> {
        self.metrics.get(name)
    }
    
    /// Get metric value as float
    pub fn get_metric_float(&self, name: &str) -> Option<f64> {
        self.metrics.get(name)?.as_float()
    }
    
    /// Get metric value as string
    pub fn get_metric_string(&self, name: &str) -> Option<&String> {
        self.metrics.get(name)?.as_string()
    }
    
    /// Get metadata value
    pub fn get_metadata(&self, key: &str) -> Option<&String> {
        self.metadata.get(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metric_value() {
        let int_val = MetricValue::Int(42);
        assert_eq!(int_val.as_int(), Some(42));
        
        let float_val = MetricValue::Float(3.14);
        assert_eq!(float_val.as_float(), Some(3.14));
        
        let string_val = MetricValue::String("test".to_string());
        assert_eq!(string_val.as_string(), Some(&"test".to_string()));
    }

    #[test]
    fn test_accuracy_metric() {
        let metric = builtin::Accuracy;
        let predictions = vec![1.0, 0.0, 1.0, 0.0];
        let targets = vec![1.0, 0.0, 0.0, 1.0];
        
        let result = metric.calculate(&predictions, &targets).unwrap();
        let accuracy = result.as_float().unwrap();
        
        assert_eq!(accuracy, 0.5); // 2 out of 4 correct
    }

    #[test]
    fn test_precision_metric() {
        let metric = builtin::Precision;
        let predictions = vec![1.0, 1.0, 0.0, 1.0];
        let targets = vec![1.0, 0.0, 0.0, 1.0];
        
        let result = metric.calculate(&predictions, &targets).unwrap();
        let precision = result.as_float().unwrap();
        
        assert_eq!(precision, 2.0 / 3.0); // 2 true positives out of 3 positive predictions
    }

    #[test]
    fn test_mse_metric() {
        let metric = builtin::MeanSquaredError;
        let predictions = vec![1.0, 2.0, 3.0];
        let targets = vec![1.5, 2.5, 3.5];
        
        let result = metric.calculate(&predictions, &targets).unwrap();
        let mse = result.as_float().unwrap();
        
        let expected_mse = (0.5_f64.powi(2) + 0.5_f64.powi(2) + 0.5_f64.powi(2)) / 3.0;
        assert!((mse - expected_mse).abs() < f64::EPSILON);
    }

    #[test]
    fn test_metric_registry() {
        let registry = MetricRegistry::with_builtin_metrics();
        let metrics = registry.list_metrics().unwrap();
        
        assert!(metrics.contains(&"accuracy".to_string()));
        assert!(metrics.contains(&"precision".to_string()));
        assert!(metrics.contains(&"mse".to_string()));
    }

    #[test]
    fn test_evaluator() {
        let evaluator = Evaluator::new();
        let predictions = vec![1.0, 0.0, 1.0, 0.0];
        let targets = vec![1.0, 0.0, 0.0, 1.0];
        
        let results = evaluator.evaluate(&predictions, &targets, &["accuracy", "precision"]).unwrap();
        
        assert!(results.contains_key("accuracy"));
        assert!(results.contains_key("precision"));
        
        let accuracy = results["accuracy"].as_float().unwrap();
        assert_eq!(accuracy, 0.5);
    }

    #[test]
    fn test_evaluation_result() {
        let mut metrics = HashMap::new();
        metrics.insert("accuracy".to_string(), MetricValue::Float(0.85));
        metrics.insert("precision".to_string(), MetricValue::Float(0.90));
        
        let result = EvaluationResult::new(metrics)
            .with_metadata("model".to_string(), "test_model".to_string())
            .with_metadata("dataset".to_string(), "test_data".to_string());
        
        assert_eq!(result.get_metric_float("accuracy"), Some(0.85));
        assert_eq!(result.get_metadata("model"), Some(&"test_model".to_string()));
    }
} 