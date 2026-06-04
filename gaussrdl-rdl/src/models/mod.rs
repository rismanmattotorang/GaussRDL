//! SOTA-informed model zoo for Relational Deep Learning.
//!
//! All models are real, trainable Candle modules (gradients flow through every
//! parameter). They share the [`NodeEncoder`] interface: given the [N, in_dim]
//! node feature matrix produced by the column encoders and the graph
//! connectivity, produce [N, hidden] node representations for the task head.
//!
//! Architectures (see `MODELS.md` for citations):
//! * [`HeteroSAGE`] — the RDL baseline heterogeneous GraphSAGE.
//! * [`Rgcn`]       — Relational GCN with basis decomposition.
//! * [`Gat`]        — Graph Attention Network with edge-softmax.
//! * [`RelGt`]      — Relational Graph Transformer (multi-element tokenization
//!                    + hybrid local/global attention over learnable centroids).
//! * [`ContextGnn`] (see [`crate::rec`]) — pair-wise + two-tower recommender.

use crate::error::{RdlError, Result};
use crate::graph::GraphTensors;
use candle_core::Tensor;
use candle_nn::{Init, VarBuilder};

mod gat;
mod hetero_sage;
mod relgt;
mod rgcn;

pub use gat::Gat;
pub use hetero_sage::HeteroSAGE;
pub use relgt::RelGt;
pub use rgcn::Rgcn;

/// Which architecture to instantiate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelKind {
    HeteroSAGE,
    Rgcn,
    Gat,
    RelGt,
}

impl ModelKind {
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s.to_lowercase().replace(['-', '_'], "").as_str() {
            "heterosage" | "sage" | "rdl" | "baseline" => ModelKind::HeteroSAGE,
            "rgcn" => ModelKind::Rgcn,
            "gat" => ModelKind::Gat,
            "relgt" | "transformer" => ModelKind::RelGt,
            other => return Err(RdlError::Config(format!("unknown model `{other}`"))),
        })
    }

    pub fn name(&self) -> &'static str {
        match self {
            ModelKind::HeteroSAGE => "HeteroSAGE",
            ModelKind::Rgcn => "RGCN",
            ModelKind::Gat => "GAT",
            ModelKind::RelGt => "RelGT",
        }
    }

    pub fn all() -> &'static [ModelKind] {
        &[ModelKind::HeteroSAGE, ModelKind::Rgcn, ModelKind::Gat, ModelKind::RelGt]
    }
}

/// Shared hyperparameters.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_bases: usize,
    /// Number of learnable global centroids (RelGT).
    pub num_centroids: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 64,
            num_layers: 2,
            num_heads: 4,
            num_bases: 4,
            num_centroids: 16,
        }
    }
}

/// Common interface for node-representation models.
pub trait NodeEncoder {
    /// Map [N, in_dim] features + graph connectivity to [N, hidden].
    fn forward(&self, x: &Tensor, g: &GraphTensors) -> Result<Tensor>;
    fn out_dim(&self) -> usize;
}

/// Instantiate a model by kind.
pub fn build(
    kind: ModelKind,
    in_dim: usize,
    num_relations: usize,
    num_node_types: usize,
    cfg: &ModelConfig,
    vb: VarBuilder,
) -> Result<Box<dyn NodeEncoder>> {
    Ok(match kind {
        ModelKind::HeteroSAGE => {
            Box::new(HeteroSAGE::new(in_dim, num_relations, cfg, vb)?)
        }
        ModelKind::Rgcn => Box::new(Rgcn::new(in_dim, num_relations, cfg, vb)?),
        ModelKind::Gat => Box::new(Gat::new(in_dim, cfg, vb)?),
        ModelKind::RelGt => {
            Box::new(RelGt::new(in_dim, num_node_types, cfg, vb)?)
        }
    })
}

/// Create a randomly-initialized parameter (bare `vb.get` defaults to zeros,
/// which is a poor init for raw weight tensors, so we use a small Gaussian).
pub(crate) fn param(vb: &VarBuilder, shape: (usize, usize), name: &str) -> Result<Tensor> {
    Ok(vb.get_with_hints(shape, name, Init::Randn { mean: 0.0, stdev: 0.02 })?)
}
