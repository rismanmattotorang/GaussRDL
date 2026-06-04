//! # GaussRDL benchmark suite (`gaussrdl-bench`)
//!
//! Dataset registry, downloader/manager, and a paper-faithful benchmark runner
//! for the GaussRDL v2 engine. Lets researchers browse benchmark datasets,
//! fetch and manage them locally, then train, evaluate, and report results.
//!
//! * [`registry`] — catalog of datasets (official RelBench + offline prepared).
//! * [`manager::DatasetManager`] — local cache: status, download/materialize,
//!   delete, load.
//! * [`bench::run_benchmark`] — train + evaluate + report (mean ± std / seeds).

pub mod bench;
pub mod download;
pub mod manager;
pub mod prepared;
pub mod registry;

use serde::{Deserialize, Serialize};

pub use bench::{run_benchmark, BenchmarkReport, SeedScore};
pub use manager::{DatasetManager, DatasetStatus};
pub use registry::{catalog, find, DatasetInfo, DatasetSource, PreparedScale, TaskInfo, TaskKind};

/// A progress update emitted during download/materialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Progress {
    pub message: String,
    /// Completion fraction in [0, 1] when known.
    pub fraction: Option<f32>,
}

impl Progress {
    pub fn msg(m: impl Into<String>) -> Self {
        Self { message: m.into(), fraction: None }
    }
    pub fn done(m: impl Into<String>) -> Self {
        Self { message: m.into(), fraction: Some(1.0) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gaussrdl_rdl::models::{ModelConfig, ModelKind};

    fn temp_root() -> std::path::PathBuf {
        std::env::temp_dir().join(format!("gaussrdl_bench_test_{}", std::process::id()))
    }

    #[test]
    fn catalog_is_well_formed() {
        let cat = catalog();
        assert!(cat.len() >= 5);
        // Every prepared dataset is offline-capable and has at least one task.
        for d in &cat {
            assert!(!d.tasks.is_empty(), "{} has no tasks", d.id);
            if !d.official {
                assert!(d.offline_capable());
            }
        }
        assert!(find("gauss-ecom-small").is_some());
        assert!(find("rel-f1").is_some());
    }

    #[test]
    fn manage_prepared_dataset_lifecycle() {
        let root = temp_root().join("lifecycle");
        let mgr = DatasetManager::new(&root);
        let info = find("gauss-ecom-small").unwrap();

        assert_eq!(mgr.status(&info), DatasetStatus::NotDownloaded);
        let mut events = 0usize;
        mgr.ensure(&info, &mut |_p| events += 1).unwrap();
        assert!(events > 0);
        assert_eq!(mgr.status(&info), DatasetStatus::Ready);
        assert!(mgr.size_bytes("gauss-ecom-small") > 0);

        // Loadable into a relational database.
        let csv = mgr.load(&info).unwrap();
        assert_eq!(csv.db.tables.len(), 3);
        assert!(csv.labels.contains_key("churn"));

        // Idempotent.
        mgr.ensure(&info, &mut |_p| {}).unwrap();
        assert_eq!(mgr.status(&info), DatasetStatus::Ready);

        mgr.delete("gauss-ecom-small").unwrap();
        assert_eq!(mgr.status(&info), DatasetStatus::NotDownloaded);
    }

    #[test]
    fn benchmark_runs_and_reports() {
        let root = temp_root().join("bench");
        let mgr = DatasetManager::new(&root);
        let info = find("gauss-ecom-small").unwrap();
        mgr.ensure(&info, &mut |_p| {}).unwrap();

        let report = run_benchmark(
            &mgr,
            &info,
            "user-churn",
            ModelKind::HeteroSAGE,
            ModelConfig { hidden_dim: 32, num_layers: 2, ..Default::default() },
            32,
            40,
            &[0, 1],
            &mut |_em| {},
        )
        .unwrap();

        assert_eq!(report.metric_name, "ROC-AUC");
        assert!(report.higher_is_better);
        assert_eq!(report.seeds.len(), 2);
        assert!(report.test_mean > 0.6, "weak benchmark AUROC {}", report.test_mean);
        assert!(report.test_std >= 0.0);
        assert!(!report.history.epochs.is_empty());
        let _ = mgr.delete("gauss-ecom-small");
    }

    #[test]
    fn official_dataset_not_runnable_offline() {
        let root = temp_root().join("official");
        let mgr = DatasetManager::new(&root);
        let info = find("rel-f1").unwrap();
        // No label column materialized; running must error clearly.
        let err = run_benchmark(
            &mgr,
            &info,
            "driver-dnf",
            ModelKind::HeteroSAGE,
            ModelConfig::default(),
            32,
            5,
            &[0],
            &mut |_| {},
        );
        assert!(err.is_err());
    }
}
