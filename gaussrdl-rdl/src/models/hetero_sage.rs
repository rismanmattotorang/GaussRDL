//! Heterogeneous GraphSAGE — the RDL baseline (Fey et al., 2024).
//!
//! Per relation type `r`: `m_{j->i}^{(r)} = W_r h_j`, mean-aggregated over the
//! typed neighborhood, then combined with a self transform:
//! `h_i' = ReLU(W_self h_i + sum_r mean_{j in N_r(i)} W_r h_j)`.

use crate::error::Result;
use crate::graph::GraphTensors;
use crate::mp::{gather_rows, relation_edge_mask, scatter_mean};
use candle_nn::{linear, Linear, Module, VarBuilder};
use candle_core::Tensor;

use super::{maybe_dropout, ModelConfig, NodeEncoder};

struct SageLayer {
    self_w: Linear,
    rel_w: Vec<Linear>,
}

impl SageLayer {
    fn new(dim: usize, num_relations: usize, vb: VarBuilder) -> Result<Self> {
        let self_w = linear(dim, dim, vb.pp("self"))?;
        let mut rel_w = Vec::with_capacity(num_relations);
        for r in 0..num_relations {
            rel_w.push(linear(dim, dim, vb.pp(format!("rel{r}")))?);
        }
        Ok(Self { self_w, rel_w })
    }

    fn forward(&self, h: &Tensor, g: &GraphTensors) -> Result<Tensor> {
        let mut agg = self.self_w.forward(h)?;
        for (r, w) in self.rel_w.iter().enumerate() {
            if let Some(eidx) = relation_edge_mask(&g.etype, r as u32)? {
                let src_r = g.src.index_select(&eidx, 0)?;
                let dst_r = g.dst.index_select(&eidx, 0)?;
                let msg = w.forward(&gather_rows(h, &src_r)?)?;
                let a = scatter_mean(&msg, &dst_r, g.num_nodes)?;
                agg = (agg + a)?;
            }
        }
        Ok(agg.relu()?)
    }
}

/// Heterogeneous GraphSAGE encoder.
pub struct HeteroSAGE {
    input: Linear,
    layers: Vec<SageLayer>,
    hidden: usize,
    dropout: f32,
}

impl HeteroSAGE {
    pub fn new(
        in_dim: usize,
        num_relations: usize,
        cfg: &ModelConfig,
        vb: VarBuilder,
    ) -> Result<Self> {
        let input = linear(in_dim, cfg.hidden_dim, vb.pp("input"))?;
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for l in 0..cfg.num_layers {
            layers.push(SageLayer::new(cfg.hidden_dim, num_relations, vb.pp(format!("layer{l}")))?);
        }
        Ok(Self { input, layers, hidden: cfg.hidden_dim, dropout: cfg.dropout })
    }
}

impl NodeEncoder for HeteroSAGE {
    fn forward(&self, x: &Tensor, g: &GraphTensors, train: bool) -> Result<Tensor> {
        let mut h = self.input.forward(x)?.relu()?;
        for layer in &self.layers {
            // Residual connection for stable deep stacks.
            h = (layer.forward(&h, g)? + h)?;
            h = maybe_dropout(&h, self.dropout, train)?;
        }
        Ok(h)
    }

    fn out_dim(&self) -> usize {
        self.hidden
    }
}
