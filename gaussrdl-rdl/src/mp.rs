//! Sparse message-passing primitives built on Candle tensor ops.
//!
//! These implement the gather/scatter operations that all GNN layers share,
//! using `index_select` (gather) and `scatter_add` (scatter) so that gradients
//! flow correctly through autodiff.

use crate::error::Result;
use candle_core::{DType, Tensor};

/// Gather rows of `x` ([N, D]) at integer indices `idx` ([E]) → [E, D].
pub fn gather_rows(x: &Tensor, idx: &Tensor) -> Result<Tensor> {
    Ok(x.index_select(idx, 0)?)
}

/// Scatter-sum `messages` ([E, D]) into `num_nodes` rows by destination
/// index `dst` ([E]) → [N, D].
pub fn scatter_sum(messages: &Tensor, dst: &Tensor, num_nodes: usize) -> Result<Tensor> {
    let (e, d) = messages.dims2()?;
    let idx2 = dst.unsqueeze(1)?.broadcast_as((e, d))?.contiguous()?;
    let zeros = Tensor::zeros((num_nodes, d), messages.dtype(), messages.device())?;
    Ok(zeros.scatter_add(&idx2, messages, 0)?)
}

/// Scatter-mean: like [`scatter_sum`] but divides each destination by its
/// in-degree (clamped to >= 1).
pub fn scatter_mean(messages: &Tensor, dst: &Tensor, num_nodes: usize) -> Result<Tensor> {
    let e = messages.dim(0)?;
    let sum = scatter_sum(messages, dst, num_nodes)?;
    let idx1 = dst.unsqueeze(1)?.contiguous()?;
    let ones = Tensor::ones((e, 1), messages.dtype(), messages.device())?;
    let zeros = Tensor::zeros((num_nodes, 1), messages.dtype(), messages.device())?;
    let counts = zeros.scatter_add(&idx1, &ones, 0)?.clamp(1.0, f32::INFINITY)?;
    Ok(sum.broadcast_div(&counts)?)
}

/// Edge-softmax: normalize per-edge `scores` ([E, H]) over all edges sharing a
/// destination, returning attention coefficients of the same shape.
pub fn edge_softmax(scores: &Tensor, dst: &Tensor, num_nodes: usize) -> Result<Tensor> {
    // Global max-subtraction for numerical stability.
    let gmax = scores.max_keepdim(0)?; // [1, H]
    let z = scores.broadcast_sub(&gmax)?.exp()?; // [E, H]
    let denom_nodes = scatter_sum(&z, dst, num_nodes)?; // [N, H]
    let denom_edges = denom_nodes.index_select(dst, 0)?; // [E, H]
    Ok(z.div(&(denom_edges + 1e-16)?)?)
}

/// Mask edges by relation type, returning the row indices (as a u32 tensor)
/// of edges whose `etype` equals `rel`. Returns `None` if there are none.
pub fn relation_edge_mask(etype: &Tensor, rel: u32) -> Result<Option<Tensor>> {
    let v = etype.to_vec1::<u32>()?;
    let idx: Vec<u32> = v
        .iter()
        .enumerate()
        .filter_map(|(i, &t)| if t == rel { Some(i as u32) } else { None })
        .collect();
    if idx.is_empty() {
        return Ok(None);
    }
    Ok(Some(Tensor::from_vec(idx.clone(), idx.len(), etype.device())?))
}

/// Cast an index tensor to U32 if it is not already.
pub fn as_u32(t: &Tensor) -> Result<Tensor> {
    if t.dtype() == DType::U32 {
        Ok(t.clone())
    } else {
        Ok(t.to_dtype(DType::U32)?)
    }
}
