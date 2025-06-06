// src/utils/profiler.rs
use std::collections::HashMap;
use crate::Result;

pub struct Profiler {
    enabled: bool,
    metrics: HashMap<String, f64>,
}

impl Profiler {
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            metrics: HashMap::new(),
        }
    }
    
    pub fn record_metric(&mut self, name: &str, value: f64) {
        if self.enabled {
            self.metrics.insert(name.to_string(), value);
        }
    }
    
    pub fn get_metrics(&self) -> &HashMap<String, f64> {
        &self.metrics
    }
}