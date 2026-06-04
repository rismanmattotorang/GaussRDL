//! Heterogeneous temporal graph construction from a relational database.
//!
//! Following Relational Deep Learning (Fey et al., 2024): every row becomes a
//! node typed by its table, every primary-key → foreign-key reference becomes a
//! (bidirectional) typed edge, and row timestamps make the graph temporal.
//! Temporal neighbor masking (`t <= seed_time`) prevents label leakage.

use crate::data::{ColumnData, RelationalDatabase};
use crate::error::{RdlError, Result};
use candle_core::{Device, Tensor};

/// A materialized heterogeneous temporal graph (CPU-side, index form).
#[derive(Debug, Clone)]
pub struct HeteroGraph {
    pub num_nodes: usize,
    pub num_relations: usize,
    /// Global node-id offset for each table.
    pub offsets: Vec<usize>,
    pub counts: Vec<usize>,
    pub table_names: Vec<String>,
    /// Table index per node.
    pub node_type: Vec<u32>,
    /// Event time per node (`NEG_INFINITY` for timeless rows).
    pub node_time: Vec<f64>,
    /// Message source node id per edge.
    pub src: Vec<u32>,
    /// Message destination node id per edge.
    pub dst: Vec<u32>,
    /// Relation type per edge.
    pub etype: Vec<u32>,
    /// Event time per edge (used for temporal masking).
    pub etime: Vec<f64>,
    pub relation_names: Vec<String>,
}

impl HeteroGraph {
    /// Build the graph from a validated relational database.
    pub fn build(db: &RelationalDatabase) -> Result<Self> {
        db.validate()?;

        let mut offsets = Vec::with_capacity(db.tables.len());
        let mut counts = Vec::with_capacity(db.tables.len());
        let mut table_names = Vec::with_capacity(db.tables.len());
        let mut node_type = Vec::new();
        let mut node_time = Vec::new();
        let mut running = 0usize;
        for (ti, t) in db.tables.iter().enumerate() {
            offsets.push(running);
            counts.push(t.num_rows);
            table_names.push(t.name.clone());
            for r in 0..t.num_rows {
                node_type.push(ti as u32);
                node_time.push(t.time[r]);
            }
            running += t.num_rows;
        }
        let num_nodes = running;

        let mut src = Vec::new();
        let mut dst = Vec::new();
        let mut etype = Vec::new();
        let mut etime = Vec::new();
        let mut relation_names = Vec::new();

        for fk in &db.foreign_keys {
            let si = db.table_index(&fk.src_table)?;
            let di = db.table_index(&fk.dst_table)?;
            let src_table = &db.tables[si];
            let col = src_table.column(&fk.column).ok_or_else(|| {
                RdlError::Schema(format!("fk column `{}` missing", fk.column))
            })?;
            let values = match &col.data {
                ColumnData::Categorical { values, .. } => values,
                _ => {
                    return Err(RdlError::Schema(format!(
                        "fk column `{}.{}` must be categorical",
                        fk.src_table, fk.column
                    )))
                }
            };
            // Forward relation: src_table -> dst_table.
            let fwd = relation_names.len() as u32;
            relation_names.push(format!("{}.{}->{}", fk.src_table, fk.column, fk.dst_table));
            // Reverse relation: dst_table -> src_table.
            let rev = relation_names.len() as u32;
            relation_names.push(format!("{}<-{}.{}", fk.dst_table, fk.src_table, fk.column));

            for (r, &v) in values.iter().enumerate() {
                let s = (offsets[si] + r) as u32;
                let d = (offsets[di] + v as usize) as u32;
                let t = src_table.time[r];
                // Message s -> d uses forward relation; d -> s reverse.
                src.push(s);
                dst.push(d);
                etype.push(fwd);
                etime.push(t);

                src.push(d);
                dst.push(s);
                etype.push(rev);
                etime.push(t);
            }
        }

        Ok(Self {
            num_nodes,
            num_relations: relation_names.len(),
            offsets,
            counts,
            table_names,
            node_type,
            node_time,
            src,
            dst,
            etype,
            etime,
            relation_names,
        })
    }

    /// Global node ids for a given table.
    pub fn table_node_ids(&self, table: &str) -> Result<std::ops::Range<usize>> {
        let i = self
            .table_names
            .iter()
            .position(|n| n == table)
            .ok_or_else(|| RdlError::Schema(format!("unknown table `{table}`")))?;
        Ok(self.offsets[i]..self.offsets[i] + self.counts[i])
    }

    /// Materialize edge tensors, keeping only edges with `etime <= seed_time`
    /// (time-consistent / leakage-free message passing).
    pub fn edge_tensors(&self, device: &Device, seed_time: f64) -> Result<GraphTensors> {
        let mut src = Vec::new();
        let mut dst = Vec::new();
        let mut etype = Vec::new();
        for i in 0..self.src.len() {
            if self.etime[i] <= seed_time {
                src.push(self.src[i]);
                dst.push(self.dst[i]);
                etype.push(self.etype[i]);
            }
        }
        let e = src.len();
        let src_t = Tensor::from_vec(src, e, device)?;
        let dst_t = Tensor::from_vec(dst, e, device)?;
        let etype_t = Tensor::from_vec(etype, e, device)?;
        let node_time_f: Vec<f32> = self
            .node_time
            .iter()
            .map(|&t| if t.is_finite() { t as f32 } else { 0.0 })
            .collect();
        let node_type_t = Tensor::from_vec(self.node_type.clone(), self.num_nodes, device)?;
        let node_time_t = Tensor::from_vec(node_time_f, self.num_nodes, device)?;
        Ok(GraphTensors {
            src: src_t,
            dst: dst_t,
            etype: etype_t,
            node_type: node_type_t,
            node_time: node_time_t,
            num_nodes: self.num_nodes,
            num_edges: e,
            num_relations: self.num_relations,
            device: device.clone(),
        })
    }
}

/// Device-resident graph connectivity used by the GNN models.
#[derive(Clone)]
pub struct GraphTensors {
    /// [E] u32, message source node per edge.
    pub src: Tensor,
    /// [E] u32, message destination node per edge.
    pub dst: Tensor,
    /// [E] u32, relation type per edge.
    pub etype: Tensor,
    /// [N] u32, table index per node.
    pub node_type: Tensor,
    /// [N] f32, normalized event time per node.
    pub node_time: Tensor,
    pub num_nodes: usize,
    pub num_edges: usize,
    pub num_relations: usize,
    pub device: Device,
}
