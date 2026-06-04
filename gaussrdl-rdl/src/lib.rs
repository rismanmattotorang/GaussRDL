//! # GaussRDL v2 engine (`gaussrdl-rdl`)
//!
//! A correct, end-to-end **Relational Deep Learning** stack by Gaussian
//! Technologies, built on the [Candle](https://github.com/huggingface/candle)
//! tensor framework. Unlike a scaffold, every component here is wired together
//! and trains with real automatic differentiation:
//!
//! ```text
//! RelationalDatabase                (typed multi-table data + foreign keys)
//!   └─► HeteroGraph                  (row=node, FK=edge, temporal, leakage-free)
//!         └─► DatabaseEncoder        (PyTorch-Frame-style stype column encoders)
//!               └─► NodeEncoder      (HeteroSAGE / RGCN / GAT / RelGT)
//!                     └─► TaskHead   (classification / regression)
//!                           └─► Trainer (AdamW autodiff loop) ─► metrics
//! ```
//!
//! See [`run_experiment`] for the one-call entry point.

pub mod data;
pub mod encoder;
pub mod error;
pub mod graph;
pub mod inference;
pub mod io;
pub mod metrics;
pub mod models;
pub mod mp;
pub mod pipeline;
pub mod synthetic;
pub mod task;
pub mod train;

pub use error::{RdlError, Result};
pub use inference::InferenceReport;
pub use models::{ModelConfig, ModelKind};
pub use pipeline::{
    run_experiment, run_experiment_cb, DataSource, ExperimentConfig, ExperimentResult, SplitMetrics,
};
pub use synthetic::{SyntheticConfig, SyntheticDataset};
pub use task::TaskType;
pub use train::{EpochMetrics, LrSchedule, TrainingHistory};
