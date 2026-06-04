//! Load real relational data from CSV files described by a JSON schema.
//!
//! A directory contains one CSV per table plus a `schema.json` describing each
//! column's role (numerical / categorical / timestamp feature, primary key,
//! foreign key, label, or ignored). The loader resolves foreign keys to row
//! indices in the referenced table, interns categorical strings, parses
//! timestamps, and builds a [`RelationalDatabase`] ready for graph
//! construction — plus a map of label columns from the seed table.

use crate::data::{Column, ColumnData, ForeignKey, RelationalDatabase, Stype, Table};
use crate::error::{RdlError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Role of a CSV column in the relational schema.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ColumnRole {
    Numerical,
    Categorical,
    Timestamp,
    /// Unique row identifier; used to resolve foreign keys.
    PrimaryKey,
    /// Reference to the primary key of the named table.
    ForeignKey(String),
    /// A supervised target (read out, not encoded as a feature).
    Label,
    /// Skipped entirely.
    Ignore,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnSchema {
    pub name: String,
    pub role: ColumnRole,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableSchema {
    pub name: String,
    /// CSV file name relative to the schema directory.
    pub file: String,
    /// Column used as the per-row event time (also encoded as a feature).
    pub time_column: Option<String>,
    pub columns: Vec<ColumnSchema>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseSchema {
    pub name: String,
    pub tables: Vec<TableSchema>,
    /// Table whose rows are the prediction entities (e.g. "users").
    pub seed_table: String,
}

/// Result of loading a CSV database.
pub struct CsvDatabase {
    pub db: RelationalDatabase,
    pub seed_table: String,
    /// Label columns read from the seed table (name -> per-row value).
    pub labels: HashMap<String, Vec<f32>>,
    /// All finite node event times (for picking a seed time).
    pub event_times: Vec<f64>,
}

impl CsvDatabase {
    /// A reasonable seed time: the `q`-quantile of observed event times.
    pub fn seed_time_quantile(&self, q: f64) -> f64 {
        let mut t: Vec<f64> = self.event_times.iter().copied().filter(|x| x.is_finite()).collect();
        if t.is_empty() {
            return f64::INFINITY;
        }
        t.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = ((t.len() as f64 - 1.0) * q).round() as usize;
        t[idx.min(t.len() - 1)]
    }
}

/// Parse a timestamp string: an `f64` of seconds, or a `YYYY-MM-DD` date
/// (converted to days-since-epoch × 86400). Empty/invalid → `NEG_INFINITY`.
fn parse_time(s: &str) -> f64 {
    let s = s.trim();
    if s.is_empty() {
        return f64::NEG_INFINITY;
    }
    if let Ok(v) = s.parse::<f64>() {
        return v;
    }
    // YYYY-MM-DD
    let parts: Vec<&str> = s.split(|c| c == '-' || c == 'T' || c == ' ').collect();
    if parts.len() >= 3 {
        if let (Ok(y), Ok(m), Ok(d)) =
            (parts[0].parse::<i64>(), parts[1].parse::<i64>(), parts[2].parse::<i64>())
        {
            let days = days_from_civil(y, m, d);
            return (days * 86400) as f64;
        }
    }
    f64::NEG_INFINITY
}

/// Days since 1970-01-01 (Howard Hinnant's algorithm).
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = (if y >= 0 { y } else { y - 399 }) / 400;
    let yoe = (y - era * 400) as i64;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

fn read_csv(path: &Path) -> Result<(Vec<String>, Vec<Vec<String>>)> {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)
        .map_err(|e| RdlError::Data(format!("open {}: {e}", path.display())))?;
    let headers: Vec<String> = rdr
        .headers()
        .map_err(|e| RdlError::Data(format!("headers {}: {e}", path.display())))?
        .iter()
        .map(|s| s.to_string())
        .collect();
    let mut rows = Vec::new();
    for rec in rdr.records() {
        let rec = rec.map_err(|e| RdlError::Data(format!("record {}: {e}", path.display())))?;
        rows.push(rec.iter().map(|s| s.to_string()).collect());
    }
    Ok((headers, rows))
}

/// Load a CSV database from `dir`, reading `dir/schema.json`.
pub fn load_database_csv(dir: impl AsRef<Path>) -> Result<CsvDatabase> {
    let dir = dir.as_ref();
    let schema_path = dir.join("schema.json");
    let schema_str = std::fs::read_to_string(&schema_path)
        .map_err(|e| RdlError::Data(format!("read {}: {e}", schema_path.display())))?;
    let schema: DatabaseSchema = serde_json::from_str(&schema_str)
        .map_err(|e| RdlError::Data(format!("parse schema.json: {e}")))?;

    // Pass 1: read every CSV and build primary-key value -> row-index maps.
    let mut raw: HashMap<String, (Vec<String>, Vec<Vec<String>>)> = HashMap::new();
    let mut pk_maps: HashMap<String, HashMap<String, i64>> = HashMap::new();
    for t in &schema.tables {
        let (headers, rows) = read_csv(&dir.join(&t.file))?;
        let col_idx = |name: &str| headers.iter().position(|h| h == name);
        if let Some(pk_col) = t.columns.iter().find(|c| c.role == ColumnRole::PrimaryKey) {
            let ci = col_idx(&pk_col.name).ok_or_else(|| {
                RdlError::Schema(format!("pk column `{}` missing in {}", pk_col.name, t.file))
            })?;
            let mut map = HashMap::new();
            for (r, row) in rows.iter().enumerate() {
                map.insert(row[ci].clone(), r as i64);
            }
            pk_maps.insert(t.name.clone(), map);
        }
        raw.insert(t.name.clone(), (headers, rows));
    }

    // Pass 2: build tables, foreign keys, and label vectors.
    let mut db = RelationalDatabase::new(schema.name.clone());
    let mut labels: HashMap<String, Vec<f32>> = HashMap::new();
    let mut event_times = Vec::new();
    let mut fkeys = Vec::new();

    for t in &schema.tables {
        let (headers, rows) = &raw[&t.name];
        let num_rows = rows.len();
        let col_idx = |name: &str| headers.iter().position(|h| h == name);
        let mut columns = Vec::new();

        for c in &t.columns {
            let ci = match col_idx(&c.name) {
                Some(i) => i,
                None => return Err(RdlError::Schema(format!("column `{}` missing in {}", c.name, t.file))),
            };
            let cells: Vec<&str> = rows.iter().map(|r| r[ci].as_str()).collect();
            match &c.role {
                ColumnRole::Numerical => {
                    let v: Vec<f32> = cells.iter().map(|s| s.trim().parse::<f32>().unwrap_or(0.0)).collect();
                    columns.push(Column { name: c.name.clone(), data: ColumnData::Numerical(v) });
                }
                ColumnRole::Categorical => {
                    let (values, cardinality) = intern(&cells);
                    columns.push(Column {
                        name: c.name.clone(),
                        data: ColumnData::Categorical { values, cardinality },
                    });
                }
                ColumnRole::Timestamp => {
                    let v: Vec<f64> = cells.iter().map(|s| parse_time(s)).collect();
                    columns.push(Column { name: c.name.clone(), data: ColumnData::Timestamp(v) });
                }
                ColumnRole::ForeignKey(target) => {
                    let map = pk_maps.get(target).ok_or_else(|| {
                        RdlError::Schema(format!("fk `{}.{}` targets `{target}` which has no primary key", t.name, c.name))
                    })?;
                    let target_rows = raw[target].1.len() as i64;
                    let mut values = Vec::with_capacity(num_rows);
                    for s in &cells {
                        let idx = *map.get(*s).ok_or_else(|| {
                            RdlError::Data(format!("fk `{}.{}` value `{s}` not found in `{target}`", t.name, c.name))
                        })?;
                        values.push(idx);
                    }
                    columns.push(Column {
                        name: c.name.clone(),
                        data: ColumnData::Categorical { values, cardinality: target_rows as usize },
                    });
                    fkeys.push(ForeignKey {
                        src_table: t.name.clone(),
                        column: c.name.clone(),
                        dst_table: target.clone(),
                    });
                }
                ColumnRole::Label => {
                    if t.name == schema.seed_table {
                        let v: Vec<f32> = cells.iter().map(|s| s.trim().parse::<f32>().unwrap_or(0.0)).collect();
                        labels.insert(c.name.clone(), v);
                    }
                }
                ColumnRole::PrimaryKey | ColumnRole::Ignore => {}
            }
        }

        let time = t.time_column.as_ref().and_then(|tc| col_idx(tc)).map(|ci| {
            rows.iter().map(|r| parse_time(&r[ci])).collect::<Vec<f64>>()
        });
        if let Some(tv) = &time {
            event_times.extend(tv.iter().copied());
        }
        db.add_table(Table::new(t.name.clone(), num_rows, columns, time)?);
    }

    for fk in fkeys {
        db.add_foreign_key(fk);
    }
    db.validate()?;

    Ok(CsvDatabase { db, seed_table: schema.seed_table, labels, event_times })
}

/// Write a relational database to a directory of CSVs plus `schema.json`,
/// optionally attaching label columns to the seed table. Useful for exporting
/// and for round-trip testing.
pub fn write_database_csv(
    dir: impl AsRef<Path>,
    db: &RelationalDatabase,
    seed_table: &str,
    labels: &HashMap<String, Vec<f32>>,
) -> Result<()> {
    let dir = dir.as_ref();
    std::fs::create_dir_all(dir).map_err(|e| RdlError::Data(e.to_string()))?;

    // Map foreign keys for role tagging.
    let mut fk_of: HashMap<(String, String), String> = HashMap::new();
    for fk in &db.foreign_keys {
        fk_of.insert((fk.src_table.clone(), fk.column.clone()), fk.dst_table.clone());
    }

    let mut table_schemas = Vec::new();
    for t in &db.tables {
        let file = format!("{}.csv", t.name);
        let mut wtr = csv::Writer::from_path(dir.join(&file))
            .map_err(|e| RdlError::Data(e.to_string()))?;

        let mut col_names: Vec<String> = vec!["__id".to_string()];
        let mut col_schemas = vec![ColumnSchema { name: "__id".into(), role: ColumnRole::PrimaryKey }];
        for c in &t.columns {
            col_names.push(c.name.clone());
            let role = if let Some(dst) = fk_of.get(&(t.name.clone(), c.name.clone())) {
                ColumnRole::ForeignKey(dst.clone())
            } else {
                match c.data.stype() {
                    Stype::Numerical => ColumnRole::Numerical,
                    Stype::Categorical => ColumnRole::Categorical,
                    Stype::Timestamp => ColumnRole::Timestamp,
                }
            };
            col_schemas.push(ColumnSchema { name: c.name.clone(), role });
        }
        let has_time = t.time.iter().any(|x| x.is_finite());
        if has_time {
            col_names.push("__time".to_string());
            col_schemas.push(ColumnSchema { name: "__time".into(), role: ColumnRole::Timestamp });
        }
        let seed_labels: Vec<(&String, &Vec<f32>)> =
            if t.name == seed_table { labels.iter().collect() } else { Vec::new() };
        for (ln, _) in &seed_labels {
            col_names.push((*ln).clone());
            col_schemas.push(ColumnSchema { name: (*ln).clone(), role: ColumnRole::Label });
        }

        wtr.write_record(&col_names).map_err(|e| RdlError::Data(e.to_string()))?;
        for r in 0..t.num_rows {
            let mut rec: Vec<String> = vec![r.to_string()];
            for c in &t.columns {
                rec.push(match &c.data {
                    ColumnData::Numerical(v) => format!("{}", v[r]),
                    ColumnData::Categorical { values, .. } => format!("{}", values[r]),
                    ColumnData::Timestamp(v) => format!("{}", v[r]),
                });
            }
            if has_time {
                rec.push(format!("{}", t.time[r]));
            }
            for (_, lv) in &seed_labels {
                rec.push(format!("{}", lv[r]));
            }
            wtr.write_record(&rec).map_err(|e| RdlError::Data(e.to_string()))?;
        }
        wtr.flush().map_err(|e| RdlError::Data(e.to_string()))?;

        table_schemas.push(TableSchema {
            name: t.name.clone(),
            file,
            time_column: if has_time { Some("__time".to_string()) } else { None },
            columns: col_schemas,
        });
    }

    let schema = DatabaseSchema { name: db.name.clone(), tables: table_schemas, seed_table: seed_table.to_string() };
    std::fs::write(dir.join("schema.json"), serde_json::to_string_pretty(&schema).unwrap())
        .map_err(|e| RdlError::Data(e.to_string()))?;
    Ok(())
}

/// Intern categorical strings into contiguous indices.
fn intern(cells: &[&str]) -> (Vec<i64>, usize) {
    let mut map: HashMap<String, i64> = HashMap::new();
    let mut values = Vec::with_capacity(cells.len());
    for s in cells {
        let next = map.len() as i64;
        let id = *map.entry(s.to_string()).or_insert(next);
        values.push(id);
    }
    (values, map.len().max(1))
}
