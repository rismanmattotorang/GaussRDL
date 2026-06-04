//! Graph Attention Network (Veličković et al., 2018) with multi-head
//! edge-softmax attention. Relation types are not used here (homogeneous
//! attention over the full neighborhood), which makes GAT a useful contrast to
//! the relation-aware models.

use crate::error::{RdlError, Result};
use crate::graph::GraphTensors;
use crate::mp::{edge_softmax, scatter_sum};
use candle_core::Tensor;
use candle_nn::{linear, Linear, Module, VarBuilder};

use super::{param, ModelConfig, NodeEncoder};

struct GatLayer {
    w: Linear,
    a_dst: Tensor, // [heads, head_dim]
    a_src: Tensor, // [heads, head_dim]
    heads: usize,
    head_dim: usize,
}

impl GatLayer {
    fn new(dim: usize, heads: usize, vb: VarBuilder) -> Result<Self> {
        if dim % heads != 0 {
            return Err(RdlError::Config(format!(
                "hidden_dim {dim} must be divisible by num_heads {heads}"
            )));
        }
        let head_dim = dim / heads;
        let w = linear(dim, heads * head_dim, vb.pp("w"))?;
        let a_dst = param(&vb, (heads, head_dim), "a_dst")?;
        let a_src = param(&vb, (heads, head_dim), "a_src")?;
        Ok(Self { w, a_dst, a_src, heads, head_dim })
    }

    fn forward(&self, h: &Tensor, g: &GraphTensors) -> Result<Tensor> {
        let n = h.dim(0)?;
        let wh = self.w.forward(h)?.reshape((n, self.heads, self.head_dim))?; // [N,H,hd]

        // Per-node attention contributions: sum_hd (Wh * a) -> [N, H].
        let al = wh.broadcast_mul(&self.a_dst.unsqueeze(0)?)?.sum(2)?; // [N,H]
        let ar = wh.broadcast_mul(&self.a_src.unsqueeze(0)?)?.sum(2)?; // [N,H]

        let al_dst = al.index_select(&g.dst, 0)?; // [E,H]
        let ar_src = ar.index_select(&g.src, 0)?; // [E,H]
        let e = (al_dst + ar_src)?;
        // LeakyReLU(0.2): relu(x) - 0.2*relu(-x).
        let leaky = (e.relu()? - (e.neg()?.relu()? * 0.2)?)?;
        let alpha = edge_softmax(&leaky, &g.dst, g.num_nodes)?; // [E,H]

        // Weighted messages from sources.
        let wh_flat = wh.reshape((n, self.heads * self.head_dim))?;
        let msg = wh_flat.index_select(&g.src, 0)?.reshape((g.num_edges, self.heads, self.head_dim))?;
        let weighted = msg.broadcast_mul(&alpha.unsqueeze(2)?)?; // [E,H,hd]
        let weighted = weighted.reshape((g.num_edges, self.heads * self.head_dim))?;
        let out = scatter_sum(&weighted, &g.dst, g.num_nodes)?; // [N, H*hd]
        Ok(out.relu()?)
    }
}

/// GAT encoder.
pub struct Gat {
    input: Linear,
    layers: Vec<GatLayer>,
    hidden: usize,
}

impl Gat {
    pub fn new(in_dim: usize, cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let input = linear(in_dim, cfg.hidden_dim, vb.pp("input"))?;
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for l in 0..cfg.num_layers {
            layers.push(GatLayer::new(cfg.hidden_dim, cfg.num_heads, vb.pp(format!("layer{l}")))?);
        }
        Ok(Self { input, layers, hidden: cfg.hidden_dim })
    }
}

impl NodeEncoder for Gat {
    fn forward(&self, x: &Tensor, g: &GraphTensors) -> Result<Tensor> {
        let mut h = self.input.forward(x)?.relu()?;
        for layer in &self.layers {
            h = (layer.forward(&h, g)? + h)?;
        }
        Ok(h)
    }

    fn out_dim(&self) -> usize {
        self.hidden
    }
}
