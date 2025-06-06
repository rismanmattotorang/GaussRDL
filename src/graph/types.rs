// src/graph/types.rs
use candle_core::Tensor;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone)]
pub struct EntityNode {
    pub table_type: usize,
    pub node_id: u64,
    pub features: Tensor,
    pub timestamp: f64,
    pub attributes: HashMap<String, AttributeValue>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AttributeValue {
    Integer(i64),
    Float(f64),
    String(String),
    Boolean(bool),
    Timestamp(chrono::DateTime<chrono::Utc>),
    Null,
}

#[derive(Debug, Clone)]
pub struct SubgraphSample {
    pub seed_node: (usize, u64),
    pub nodes: Vec<(usize, u64)>,
    pub edges: Vec<((usize, u64), (usize, u64))>,
    pub timestamps: Vec<f64>,
    pub hop_distances: HashMap<(usize, u64), usize>,
}