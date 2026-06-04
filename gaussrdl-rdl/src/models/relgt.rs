//! Relational Graph Transformer (Dwivedi et al., 2025), full-graph adaptation.
//!
//! Faithful to the paper's two defining ideas:
//!   1. **Multi-element tokenization** — a node's input token fuses its cell
//!      features, node *type*, *time*, and a *structural* (degree) encoding.
//!   2. **Hybrid attention** — each layer combines sparse **local attention**
//!      over graph neighbors (multi-head, edge-softmax) with **global
//!      attention to `B` learnable centroids** that summarize database-wide
//!      context. Local + global are fused, then passed through a transformer
//!      feed-forward block with residual connections and layer norm.
//!
//! The original work maintains centroids via EMA k-means; here they are learned
//! directly by backprop (a simpler, fully-differentiable variant).

use crate::error::Result;
use crate::graph::GraphTensors;
use crate::mp::{edge_softmax, scatter_sum};
use candle_core::Tensor;
use candle_nn::ops::softmax;
use candle_nn::{embedding, layer_norm, linear, Embedding, LayerNorm, Linear, Module, VarBuilder};

use super::{maybe_dropout, param, ModelConfig, NodeEncoder};

struct RelGtLayer {
    // local attention
    wq: Linear,
    wk: Linear,
    wv: Linear,
    lo: Linear,
    // global attention
    gq: Linear,
    gk: Linear,
    gv: Linear,
    go: Linear,
    centroids: Tensor, // [num_centroids, hidden]
    // fusion + ffn
    ln1: LayerNorm,
    ln2: LayerNorm,
    ff1: Linear,
    ff2: Linear,
    heads: usize,
    head_dim: usize,
    num_centroids: usize,
}

impl RelGtLayer {
    fn new(dim: usize, heads: usize, head_dim: usize, num_centroids: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            wq: linear(dim, heads * head_dim, vb.pp("wq"))?,
            wk: linear(dim, heads * head_dim, vb.pp("wk"))?,
            wv: linear(dim, heads * head_dim, vb.pp("wv"))?,
            lo: linear(heads * head_dim, dim, vb.pp("lo"))?,
            gq: linear(dim, heads * head_dim, vb.pp("gq"))?,
            gk: linear(dim, heads * head_dim, vb.pp("gk"))?,
            gv: linear(dim, heads * head_dim, vb.pp("gv"))?,
            go: linear(heads * head_dim, dim, vb.pp("go"))?,
            centroids: param(&vb, (num_centroids, dim), "centroids")?,
            ln1: layer_norm(dim, 1e-5, vb.pp("ln1"))?,
            ln2: layer_norm(dim, 1e-5, vb.pp("ln2"))?,
            ff1: linear(dim, dim * 2, vb.pp("ff1"))?,
            ff2: linear(dim * 2, dim, vb.pp("ff2"))?,
            heads,
            head_dim,
            num_centroids,
        })
    }

    fn local_attention(&self, h: &Tensor, g: &GraphTensors) -> Result<Tensor> {
        let n = h.dim(0)?;
        let (hh, hd) = (self.heads, self.head_dim);
        let scale = 1.0 / (hd as f64).sqrt();
        let q = self.wq.forward(h)?.reshape((n, hh, hd))?;
        let k = self.wk.forward(h)?.reshape((n, hh, hd))?;
        let v = self.wv.forward(h)?.reshape((n, hh * hd))?;

        let q_dst = q.index_select(&g.dst, 0)?; // [E,H,hd]
        let k_src = k.index_select(&g.src, 0)?; // [E,H,hd]
        let score = ((q_dst * k_src)?.sum(2)? * scale)?; // [E,H]
        let alpha = edge_softmax(&score, &g.dst, g.num_nodes)?; // [E,H]

        let v_src = v.index_select(&g.src, 0)?.reshape((g.num_edges, hh, hd))?;
        let weighted = v_src.broadcast_mul(&alpha.unsqueeze(2)?)?.reshape((g.num_edges, hh * hd))?;
        let out = scatter_sum(&weighted, &g.dst, g.num_nodes)?; // [N, H*hd]
        Ok(self.lo.forward(&out)?)
    }

    fn global_attention(&self, h: &Tensor) -> Result<Tensor> {
        let n = h.dim(0)?;
        let (hh, hd, c) = (self.heads, self.head_dim, self.num_centroids);
        let scale = 1.0 / (hd as f64).sqrt();
        // [H, N, hd]
        let q = self.gq.forward(h)?.reshape((n, hh, hd))?.transpose(0, 1)?.contiguous()?;
        // [H, C, hd]
        let k = self.gk.forward(&self.centroids)?.reshape((c, hh, hd))?.transpose(0, 1)?.contiguous()?;
        let v = self.gv.forward(&self.centroids)?.reshape((c, hh, hd))?.transpose(0, 1)?.contiguous()?;

        let scores = (q.matmul(&k.transpose(1, 2)?)? * scale)?; // [H, N, C]
        let attn = softmax(&scores, 2)?; // over centroids
        let ctx = attn.matmul(&v)?; // [H, N, hd]
        let ctx = ctx.transpose(0, 1)?.contiguous()?.reshape((n, hh * hd))?;
        Ok(self.go.forward(&ctx)?)
    }

    fn forward(&self, h: &Tensor, g: &GraphTensors) -> Result<Tensor> {
        let local = self.local_attention(h, g)?;
        let global = self.global_attention(h)?;
        let attn = (local + global)?;
        let h = self.ln1.forward(&(h + attn)?)?;
        let ff = self.ff2.forward(&self.ff1.forward(&h)?.relu()?)?;
        Ok(self.ln2.forward(&(h + ff)?)?)
    }
}

/// RelGT encoder.
pub struct RelGt {
    input: Linear,
    type_emb: Embedding,
    time_proj: Linear,
    deg_proj: Linear,
    layers: Vec<RelGtLayer>,
    hidden: usize,
    dropout: f32,
}

impl RelGt {
    pub fn new(
        in_dim: usize,
        num_node_types: usize,
        cfg: &ModelConfig,
        vb: VarBuilder,
    ) -> Result<Self> {
        let dim = cfg.hidden_dim;
        let head_dim = (dim / cfg.num_heads).max(1);
        let heads = cfg.num_heads;
        let input = linear(in_dim, dim, vb.pp("input"))?;
        let type_emb = embedding(num_node_types.max(1), dim, vb.pp("type_emb"))?;
        let time_proj = linear(1, dim, vb.pp("time_proj"))?;
        let deg_proj = linear(1, dim, vb.pp("deg_proj"))?;
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for l in 0..cfg.num_layers {
            layers.push(RelGtLayer::new(dim, heads, head_dim, cfg.num_centroids, vb.pp(format!("layer{l}")))?);
        }
        Ok(Self { input, type_emb, time_proj, deg_proj, layers, hidden: dim, dropout: cfg.dropout })
    }

    /// Build the multi-element token for every node.
    fn tokenize(&self, x: &Tensor, g: &GraphTensors) -> Result<Tensor> {
        let feat = self.input.forward(x)?;
        let type_e = self.type_emb.forward(&g.node_type)?;
        let time_e = self.time_proj.forward(&g.node_time.unsqueeze(1)?)?;
        // Structural (degree) encoding.
        let ones = Tensor::ones((g.num_edges, 1), candle_core::DType::F32, &g.device)?;
        let deg = scatter_sum(&ones, &g.dst, g.num_nodes)?; // [N,1]
        let deg_e = self.deg_proj.forward(&(deg * 0.1)?)?;
        Ok((((feat + type_e)? + time_e)? + deg_e)?)
    }
}

impl NodeEncoder for RelGt {
    fn forward(&self, x: &Tensor, g: &GraphTensors, train: bool) -> Result<Tensor> {
        let mut h = self.tokenize(x, g)?;
        for layer in &self.layers {
            h = layer.forward(&h, g)?;
            h = maybe_dropout(&h, self.dropout, train)?;
        }
        Ok(h)
    }

    fn out_dim(&self) -> usize {
        self.hidden
    }
}
