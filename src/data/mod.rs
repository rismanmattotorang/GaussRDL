// src/data/mod.rs
use candle_core::Tensor;
use crate::{Result, graph::RelationalEntityGraph};

pub mod loader;
pub mod pipeline;

pub use loader::*;
pub use pipeline::*;

#[derive(Debug, Clone)]
pub struct TrainingBatch {
    pub graph: RelationalEntityGraph,
    pub seed_nodes: Vec<(usize, u64)>,
    pub subgraphs: Vec<Vec<(usize, u64)>>,
    pub timestamps: Vec<f64>,
    pub targets: Tensor,
    pub batch_size: usize,
}

#[derive(Debug, Clone)]
pub struct TestBatch {
    pub graph: RelationalEntityGraph,
    pub seed_nodes: Vec<(usize, u64)>,
    pub subgraphs: Vec<Vec<(usize, u64)>>,
    pub timestamps: Vec<f64>,
    pub targets: Tensor,
    pub batch_size: usize,
}

#[derive(Debug, Clone)]
pub struct PreparedData {
    pub train_size: usize,
    pub val_size: usize,
    pub test_size: usize,
    pub num_features: usize,
    pub num_classes: usize,
    pub task_type: String,
}
