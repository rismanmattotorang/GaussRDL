//! Relational GCN with basis decomposition (Schlichtkrull et al., 2018).
//!
//! Each relation's weight is shared via `B` basis matrices:
//! `W_r = sum_b a_{r,b} B_b`, reducing parameters and improving generalization
//! across the many relation types in a relational schema.

use crate::error::Result;
use crate::graph::GraphTensors;
use crate::mp::{gather_rows, relation_edge_mask, scatter_mean};
use candle_core::{Tensor, D};
use candle_nn::{linear, Linear, Module, VarBuilder};

use super::{maybe_dropout, param, ModelConfig, NodeEncoder};

struct RgcnLayer {
    self_w: Linear,
    /// [num_bases, dim, dim]
    bases: Vec<Tensor>,
    /// [num_relations, num_bases]
    coeff: Tensor,
    num_relations: usize,
    num_bases: usize,
}

impl RgcnLayer {
    fn new(dim: usize, num_relations: usize, num_bases: usize, vb: VarBuilder) -> Result<Self> {
        let self_w = linear(dim, dim, vb.pp("self"))?;
        let mut bases = Vec::with_capacity(num_bases);
        for b in 0..num_bases {
            bases.push(param(&vb, (dim, dim), &format!("basis{b}"))?);
        }
        let coeff = param(&vb, (num_relations, num_bases), "coeff")?;
        Ok(Self { self_w, bases, coeff, num_relations, num_bases })
    }

    fn forward(&self, h: &Tensor, g: &GraphTensors) -> Result<Tensor> {
        let dim = h.dim(1)?;
        // Differentiable basis composition: W_all[r] = sum_b coeff[r,b] * B_b.
        // Stack bases -> [num_bases, dim*dim], then coeff @ B -> [R, dim*dim].
        let b_stack = Tensor::stack(&self.bases, 0)?; // [num_bases, dim, dim]
        let b_flat = b_stack.reshape((self.num_bases, dim * dim))?;
        let w_all = self.coeff.matmul(&b_flat)?.reshape((self.num_relations, dim, dim))?;

        let mut agg = self.self_w.forward(h)?;
        for r in 0..self.num_relations {
            let eidx = match relation_edge_mask(&g.etype, r as u32)? {
                Some(e) => e,
                None => continue,
            };
            let w_r = w_all.narrow(0, r, 1)?.reshape((dim, dim))?;
            let src_r = g.src.index_select(&eidx, 0)?;
            let dst_r = g.dst.index_select(&eidx, 0)?;
            let msg = gather_rows(h, &src_r)?.matmul(&w_r)?; // [Er, dim]
            let a = scatter_mean(&msg, &dst_r, g.num_nodes)?;
            agg = (agg + a)?;
        }
        Ok(agg.relu()?)
    }
}

/// RGCN encoder.
pub struct Rgcn {
    input: Linear,
    layers: Vec<RgcnLayer>,
    hidden: usize,
    dropout: f32,
}

impl Rgcn {
    pub fn new(
        in_dim: usize,
        num_relations: usize,
        cfg: &ModelConfig,
        vb: VarBuilder,
    ) -> Result<Self> {
        let input = linear(in_dim, cfg.hidden_dim, vb.pp("input"))?;
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for l in 0..cfg.num_layers {
            layers.push(RgcnLayer::new(
                cfg.hidden_dim,
                num_relations,
                cfg.num_bases,
                vb.pp(format!("layer{l}")),
            )?);
        }
        Ok(Self { input, layers, hidden: cfg.hidden_dim, dropout: cfg.dropout })
    }
}

impl NodeEncoder for Rgcn {
    fn forward(&self, x: &Tensor, g: &GraphTensors, train: bool) -> Result<Tensor> {
        let _ = D::Minus1;
        let mut h = self.input.forward(x)?.relu()?;
        for layer in &self.layers {
            h = (layer.forward(&h, g)? + h)?;
            h = maybe_dropout(&h, self.dropout, train)?;
        }
        Ok(h)
    }

    fn out_dim(&self) -> usize {
        self.hidden
    }
}
