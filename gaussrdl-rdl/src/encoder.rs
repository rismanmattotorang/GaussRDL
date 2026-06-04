//! Column ("stype") encoders that turn each table row into a node embedding.
//!
//! This mirrors PyTorch-Frame's `StypeWiseFeatureEncoder` + column-interaction
//! model: each column is embedded independently by an stype-specific module,
//! the per-column embeddings are summed, and a small residual MLP fuses them
//! into one `channels`-dimensional vector per row. Foreign-key columns are
//! *not* encoded as features — they are represented structurally as edges.

use crate::data::{ColumnData, RelationalDatabase};
use crate::error::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{embedding, linear, Embedding, Linear, Module, VarBuilder};
use std::collections::HashSet;

/// Per-column encoder variants. Each holds its (constant) prepared input plus
/// the trainable projection.
enum ColEncoder {
    Numerical { input: Tensor, proj: Linear },
    Categorical { idx: Tensor, emb: Embedding },
    Timestamp { input: Tensor, proj: Linear },
}

impl ColEncoder {
    fn forward(&self) -> Result<Tensor> {
        Ok(match self {
            ColEncoder::Numerical { input, proj } => proj.forward(input)?,
            ColEncoder::Categorical { idx, emb } => emb.forward(idx)?,
            ColEncoder::Timestamp { input, proj } => proj.forward(input)?,
        })
    }
}

/// Encoder for a single table.
struct TableEncoder {
    num_rows: usize,
    channels: usize,
    cols: Vec<ColEncoder>,
    fuse1: Linear,
    fuse2: Linear,
    device: Device,
}

impl TableEncoder {
    fn new(
        data_cols: &[(&str, &ColumnData)],
        num_rows: usize,
        channels: usize,
        device: &Device,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut cols = Vec::new();
        for (ci, (_name, data)) in data_cols.iter().enumerate() {
            let cvb = vb.pp(format!("col{ci}"));
            let enc = match data {
                ColumnData::Numerical(v) => {
                    let input = standardize(v, device)?; // [rows, 1]
                    let proj = linear(1, channels, cvb.pp("num"))?;
                    ColEncoder::Numerical { input, proj }
                }
                ColumnData::Categorical { values, cardinality } => {
                    let u: Vec<u32> = values.iter().map(|&x| x as u32).collect();
                    let idx = Tensor::from_vec(u, num_rows, device)?;
                    let emb = embedding(*cardinality, channels, cvb.pp("cat"))?;
                    ColEncoder::Categorical { idx, emb }
                }
                ColumnData::Timestamp(v) => {
                    let (input, f) = timestamp_features(v, device)?; // [rows, F]
                    let proj = linear(f, channels, cvb.pp("time"))?;
                    ColEncoder::Timestamp { input, proj }
                }
            };
            cols.push(enc);
        }
        let fuse1 = linear(channels, channels, vb.pp("fuse1"))?;
        let fuse2 = linear(channels, channels, vb.pp("fuse2"))?;
        Ok(Self {
            num_rows,
            channels,
            cols,
            fuse1,
            fuse2,
            device: device.clone(),
        })
    }

    fn forward(&self) -> Result<Tensor> {
        let mut acc = Tensor::zeros((self.num_rows, self.channels), DType::F32, &self.device)?;
        for c in &self.cols {
            acc = (acc + c.forward()?)?;
        }
        // Residual column-interaction MLP.
        let h = self.fuse1.forward(&acc)?.relu()?;
        let h = self.fuse2.forward(&h)?;
        Ok((acc + h)?)
    }
}

/// Encoder for the whole database, producing the [N, channels] node feature
/// matrix in the same node ordering that [`crate::graph::HeteroGraph`] uses.
pub struct DatabaseEncoder {
    tables: Vec<TableEncoder>,
    pub channels: usize,
}

impl DatabaseEncoder {
    pub fn new(
        db: &RelationalDatabase,
        channels: usize,
        device: &Device,
        vb: VarBuilder,
    ) -> Result<Self> {
        // Columns used as foreign keys are structural, not features.
        let mut fk_cols: HashSet<(String, String)> = HashSet::new();
        for fk in &db.foreign_keys {
            fk_cols.insert((fk.src_table.clone(), fk.column.clone()));
        }

        let mut tables = Vec::with_capacity(db.tables.len());
        for (ti, t) in db.tables.iter().enumerate() {
            let data_cols: Vec<(&str, &ColumnData)> = t
                .columns
                .iter()
                .filter(|c| !fk_cols.contains(&(t.name.clone(), c.name.clone())))
                .map(|c| (c.name.as_str(), &c.data))
                .collect();
            let enc = TableEncoder::new(
                &data_cols,
                t.num_rows,
                channels,
                device,
                vb.pp(format!("table{ti}")),
            )?;
            tables.push(enc);
        }
        Ok(Self { tables, channels })
    }

    /// Compute the full node feature matrix [N, channels].
    pub fn encode(&self) -> Result<Tensor> {
        let mut parts = Vec::with_capacity(self.tables.len());
        for t in &self.tables {
            parts.push(t.forward()?);
        }
        Ok(Tensor::cat(&parts, 0)?)
    }
}

/// Standardize a numerical column to zero mean / unit variance → [rows, 1].
fn standardize(v: &[f32], device: &Device) -> Result<Tensor> {
    let n = v.len().max(1) as f32;
    let mean = v.iter().sum::<f32>() / n;
    let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n;
    let std = var.sqrt().max(1e-6);
    let norm: Vec<f32> = v.iter().map(|x| (x - mean) / std).collect();
    Ok(Tensor::from_vec(norm, (v.len(), 1), device)?)
}

/// Cyclic + linear temporal features for a timestamp column → ([rows, F], F).
fn timestamp_features(v: &[f64], device: &Device) -> Result<(Tensor, usize)> {
    let finite: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
    let tmax = finite.iter().cloned().fold(1.0_f64, f64::max).max(1.0);
    let periods = [tmax, tmax / 4.0, tmax / 12.0];
    let f = 1 + 2 * periods.len();
    let mut data = Vec::with_capacity(v.len() * f);
    for &t in v {
        let t = if t.is_finite() { t } else { 0.0 };
        data.push((t / tmax) as f32);
        for &p in &periods {
            let ang = 2.0 * std::f64::consts::PI * t / p;
            data.push(ang.sin() as f32);
            data.push(ang.cos() as f32);
        }
    }
    Ok((Tensor::from_vec(data, (v.len(), f), device)?, f))
}
