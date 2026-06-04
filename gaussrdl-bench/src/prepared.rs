//! Materialize prepared (offline) benchmark datasets to a CSV directory.

use crate::registry::PreparedScale;
use gaussrdl_rdl::io::write_database_csv;
use gaussrdl_rdl::synthetic::{SyntheticConfig, SyntheticDataset};
use gaussrdl_rdl::Result;
use std::collections::HashMap;
use std::path::Path;

/// Generate a prepared dataset and write it to `dir` as CSVs + `schema.json`,
/// attaching both `churn` and `ltv` label columns to the seed table.
pub fn materialize(scale: PreparedScale, dir: impl AsRef<Path>) -> Result<()> {
    let cfg = SyntheticConfig {
        num_users: scale.num_users,
        num_items: scale.num_items,
        avg_trans_per_user: scale.avg_trans_per_user,
        ..Default::default()
    };
    let ds = SyntheticDataset::generate(cfg);
    let mut labels = HashMap::new();
    labels.insert("churn".to_string(), ds.churn_labels.clone());
    labels.insert("ltv".to_string(), ds.ltv_labels.clone());
    write_database_csv(dir, &ds.db, "users", &labels)
}
