//! Relational database representation for RDL.
//!
//! Mirrors the RelBench / PyTorch-Frame data model: a database is a set of
//! typed tables, each row carries an optional timestamp, and primary-key →
//! foreign-key references link rows across tables. This is the lossless
//! relational structure that the rest of the pipeline turns into a
//! heterogeneous temporal graph.

use crate::error::{RdlError, Result};
use std::collections::HashMap;

/// Semantic type ("stype") of a column, following PyTorch-Frame.
///
/// The semantic type — not the physical dtype — determines which encoder is
/// used to map a cell to an embedding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stype {
    /// Continuous numerical value (encoded via per-column affine projection).
    Numerical,
    /// Categorical value with a finite vocabulary (encoded via embedding table).
    Categorical,
    /// Unix-seconds timestamp (encoded via cyclic temporal features).
    Timestamp,
}

/// Column payload. The variant determines the [`Stype`].
#[derive(Debug, Clone)]
pub enum ColumnData {
    /// One f32 per row.
    Numerical(Vec<f32>),
    /// Category index per row, plus the vocabulary cardinality.
    Categorical { values: Vec<i64>, cardinality: usize },
    /// Unix-seconds timestamp per row (f64 for range).
    Timestamp(Vec<f64>),
}

impl ColumnData {
    /// Semantic type of this column.
    pub fn stype(&self) -> Stype {
        match self {
            ColumnData::Numerical(_) => Stype::Numerical,
            ColumnData::Categorical { .. } => Stype::Categorical,
            ColumnData::Timestamp(_) => Stype::Timestamp,
        }
    }

    /// Number of rows.
    pub fn len(&self) -> usize {
        match self {
            ColumnData::Numerical(v) => v.len(),
            ColumnData::Categorical { values, .. } => values.len(),
            ColumnData::Timestamp(v) => v.len(),
        }
    }

    /// Whether the column is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A named column.
#[derive(Debug, Clone)]
pub struct Column {
    pub name: String,
    pub data: ColumnData,
}

/// A table: a set of equally-sized columns, with an optional per-row timestamp.
///
/// Row index `i` is the implicit primary key for the table.
#[derive(Debug, Clone)]
pub struct Table {
    pub name: String,
    pub num_rows: usize,
    pub columns: Vec<Column>,
    /// Optional per-row event time (Unix seconds). Rows without a meaningful
    /// time (e.g. dimension tables) use `f64::NEG_INFINITY` so they are always
    /// visible under temporal sampling.
    pub time: Vec<f64>,
}

impl Table {
    /// Build a table, validating that all columns share `num_rows`.
    pub fn new(
        name: impl Into<String>,
        num_rows: usize,
        columns: Vec<Column>,
        time: Option<Vec<f64>>,
    ) -> Result<Self> {
        for c in &columns {
            if c.data.len() != num_rows {
                return Err(RdlError::Schema(format!(
                    "column `{}` has {} rows, expected {}",
                    c.name,
                    c.data.len(),
                    num_rows
                )));
            }
        }
        let time = match time {
            Some(t) => {
                if t.len() != num_rows {
                    return Err(RdlError::Schema(format!(
                        "time column has {} rows, expected {}",
                        t.len(),
                        num_rows
                    )));
                }
                t
            }
            None => vec![f64::NEG_INFINITY; num_rows],
        };
        Ok(Self {
            name: name.into(),
            num_rows,
            columns,
            time,
        })
    }

    /// Look up a column by name.
    pub fn column(&self, name: &str) -> Option<&Column> {
        self.columns.iter().find(|c| c.name == name)
    }
}

/// A primary-key → foreign-key reference. The `column` of `src_table` holds
/// row indices into `dst_table`.
#[derive(Debug, Clone)]
pub struct ForeignKey {
    pub src_table: String,
    pub column: String,
    pub dst_table: String,
}

/// A relational database: tables plus the foreign keys linking them.
#[derive(Debug, Clone)]
pub struct RelationalDatabase {
    pub name: String,
    pub tables: Vec<Table>,
    pub foreign_keys: Vec<ForeignKey>,
}

impl RelationalDatabase {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            tables: Vec::new(),
            foreign_keys: Vec::new(),
        }
    }

    pub fn add_table(&mut self, table: Table) -> &mut Self {
        self.tables.push(table);
        self
    }

    pub fn add_foreign_key(&mut self, fk: ForeignKey) -> &mut Self {
        self.foreign_keys.push(fk);
        self
    }

    /// Index of a table by name.
    pub fn table_index(&self, name: &str) -> Result<usize> {
        self.tables
            .iter()
            .position(|t| t.name == name)
            .ok_or_else(|| RdlError::Schema(format!("unknown table `{name}`")))
    }

    pub fn table(&self, name: &str) -> Result<&Table> {
        let i = self.table_index(name)?;
        Ok(&self.tables[i])
    }

    /// Validate that every foreign key references existing tables/columns and
    /// that its indices are in-range. Returns a map from table name to index
    /// for convenience.
    pub fn validate(&self) -> Result<HashMap<String, usize>> {
        let mut idx = HashMap::new();
        for (i, t) in self.tables.iter().enumerate() {
            if idx.insert(t.name.clone(), i).is_some() {
                return Err(RdlError::Schema(format!("duplicate table `{}`", t.name)));
            }
        }
        for fk in &self.foreign_keys {
            let src = self.table(&fk.src_table)?;
            let dst_rows = self.table(&fk.dst_table)?.num_rows as i64;
            let col = src.column(&fk.column).ok_or_else(|| {
                RdlError::Schema(format!(
                    "fk column `{}.{}` not found",
                    fk.src_table, fk.column
                ))
            })?;
            match &col.data {
                ColumnData::Categorical { values, .. } => {
                    for &v in values {
                        if v < 0 || v >= dst_rows {
                            return Err(RdlError::Schema(format!(
                                "fk `{}.{}` value {} out of range for `{}` ({} rows)",
                                fk.src_table, fk.column, v, fk.dst_table, dst_rows
                            )));
                        }
                    }
                }
                _ => {
                    return Err(RdlError::Schema(format!(
                        "fk column `{}.{}` must be categorical (row indices)",
                        fk.src_table, fk.column
                    )))
                }
            }
        }
        Ok(idx)
    }
}
