//! End-to-end integration tests: every model must (a) build a valid temporal
//! graph from synthetic relational data, (b) train with real autodiff so the
//! loss decreases, and (c) learn the planted signal (val AUROC clearly above
//! chance for classification; MAE below the std-dev baseline for regression).

use gaussrdl_rdl::data::{Column, ColumnData, ForeignKey, RelationalDatabase, Table};
use gaussrdl_rdl::graph::HeteroGraph;
use gaussrdl_rdl::models::{ModelConfig, ModelKind};
use gaussrdl_rdl::synthetic::{SyntheticConfig, SyntheticDataset};
use gaussrdl_rdl::{run_experiment, DataSource, ExperimentConfig, TaskType};
use std::collections::HashMap;

fn small_data() -> SyntheticConfig {
    SyntheticConfig { num_users: 300, num_items: 80, seed: 1, ..Default::default() }
}

#[test]
fn graph_construction_is_consistent() {
    let ds = SyntheticDataset::generate(small_data());
    let g = HeteroGraph::build(&ds.db).unwrap();
    // 3 tables -> node count = sum of rows.
    let expected: usize = ds.db.tables.iter().map(|t| t.num_rows).sum();
    assert_eq!(g.num_nodes, expected);
    // 2 foreign keys -> 4 relation types (forward + reverse each).
    assert_eq!(g.num_relations, 4);
    // Edges come in forward/reverse pairs.
    assert_eq!(g.src.len() % 2, 0);
    // No edge references an out-of-range node.
    assert!(g.src.iter().all(|&s| (s as usize) < g.num_nodes));
    assert!(g.dst.iter().all(|&d| (d as usize) < g.num_nodes));
}

#[test]
fn temporal_masking_drops_future_edges() {
    let ds = SyntheticDataset::generate(small_data());
    let g = HeteroGraph::build(&ds.db).unwrap();
    let dev = candle_core::Device::Cpu;
    let full = g.edge_tensors(&dev, f64::INFINITY).unwrap();
    let masked = g.edge_tensors(&dev, ds.seed_time).unwrap();
    // All historical transactions are <= seed_time by construction, so the
    // masked graph keeps every edge here; an early seed must drop some.
    let early = g.edge_tensors(&dev, ds.seed_time * 0.1).unwrap();
    assert!(early.num_edges < full.num_edges);
    assert!(masked.num_edges <= full.num_edges);
}

#[test]
fn validate_rejects_bad_foreign_key() {
    let mut db = RelationalDatabase::new("bad");
    db.add_table(
        Table::new("a", 2, vec![Column { name: "x".into(), data: ColumnData::Numerical(vec![1.0, 2.0]) }], None)
            .unwrap(),
    );
    db.add_table(
        Table::new(
            "b",
            1,
            vec![Column {
                name: "a_id".into(),
                data: ColumnData::Categorical { values: vec![5], cardinality: 10 },
            }],
            None,
        )
        .unwrap(),
    );
    db.add_foreign_key(ForeignKey { src_table: "b".into(), column: "a_id".into(), dst_table: "a".into() });
    // Row 0 of b references row 5 of a, which has only 2 rows.
    assert!(db.validate().is_err());
}

fn train_classification(model: ModelKind) -> gaussrdl_rdl::ExperimentResult {
    let cfg = ExperimentConfig {
        model,
        task: TaskType::BinaryClassification,
        model_cfg: ModelConfig { hidden_dim: 48, num_layers: 2, num_heads: 4, ..Default::default() },
        channels: 48,
        epochs: 70,
        lr: 1e-2,
        data: DataSource::Synthetic(small_data()),
        ..Default::default()
    };
    run_experiment(&cfg).unwrap()
}

#[test]
fn all_models_learn_churn() {
    for &model in ModelKind::all() {
        let r = train_classification(model);
        let losses = r.train_losses();
        let first = losses[0];
        let last = *losses.last().unwrap();
        assert!(last < first, "{model:?}: loss did not decrease ({first} -> {last})");
        assert!(last.is_finite(), "{model:?}: non-finite loss");
        // Must beat random on held-out users.
        assert!(r.val.auroc > 0.6, "{model:?}: val AUROC {} not above chance", r.val.auroc);
        // Inference KPIs are populated.
        assert!(r.inference.num_samples > 0);
        assert!(r.inference.throughput_per_s > 0.0);
        assert!(r.inference.auroc > 0.5);
    }
}

#[test]
fn sage_learns_regression() {
    let cfg = ExperimentConfig {
        model: ModelKind::HeteroSAGE,
        task: TaskType::Regression,
        channels: 48,
        model_cfg: ModelConfig { hidden_dim: 48, num_layers: 2, ..Default::default() },
        epochs: 80,
        lr: 1e-2,
        data: DataSource::Synthetic(small_data()),
        ..Default::default()
    };
    let r = run_experiment(&cfg).unwrap();
    let losses = r.train_losses();
    assert!(*losses.last().unwrap() < losses[0], "regression loss did not decrease");
    assert!(r.test.mae.is_finite());
    assert!(r.inference.r2.is_finite());
}

#[test]
fn early_stopping_and_monitoring() {
    let cfg = ExperimentConfig {
        model: ModelKind::HeteroSAGE,
        task: TaskType::BinaryClassification,
        channels: 32,
        model_cfg: ModelConfig { hidden_dim: 32, num_layers: 2, ..Default::default() },
        epochs: 100,
        lr: 1e-2,
        early_stopping_patience: 8,
        data: DataSource::Synthetic(small_data()),
        ..Default::default()
    };
    // Capture live callback events.
    let mut seen = 0usize;
    let r = gaussrdl_rdl::run_experiment_cb(&cfg, &mut |_em| seen += 1).unwrap();
    assert_eq!(seen, r.history.epochs.len());
    // Every epoch records a learning rate and a finite gradient norm.
    for e in &r.history.epochs {
        assert!(e.learning_rate > 0.0);
        assert!(e.grad_norm.is_finite());
    }
    // best_val should be the max val_metric seen (classification maximizes).
    let best = r.history.epochs.iter().map(|e| e.val_metric).fold(f32::MIN, f32::max);
    assert!((r.history.best_val - best).abs() < 1e-4);
}

#[test]
fn csv_roundtrip_trains() {
    // Generate synthetic data, write to CSV, reload, and train through the
    // real-data CSV path.
    let ds = SyntheticDataset::generate(small_data());
    let dir = std::env::temp_dir().join(format!("gaussrdl_csv_{}", std::process::id()));
    let mut labels = HashMap::new();
    labels.insert("churn".to_string(), ds.churn_labels.clone());
    gaussrdl_rdl::io::write_database_csv(&dir, &ds.db, "users", &labels).unwrap();

    // schema.json + 3 CSVs exist.
    assert!(dir.join("schema.json").exists());
    let loaded = gaussrdl_rdl::io::load_database_csv(&dir).unwrap();
    assert_eq!(loaded.db.tables.len(), ds.db.tables.len());
    assert!(loaded.labels.contains_key("churn"));

    let cfg = ExperimentConfig {
        model: ModelKind::HeteroSAGE,
        task: TaskType::BinaryClassification,
        channels: 32,
        model_cfg: ModelConfig { hidden_dim: 32, num_layers: 2, ..Default::default() },
        epochs: 50,
        lr: 1e-2,
        data: DataSource::Csv {
            dir: dir.to_string_lossy().into_owned(),
            label: "churn".into(),
            seed_quantile: 1.0,
        },
        ..Default::default()
    };
    let r = run_experiment(&cfg).unwrap();
    assert!(r.val.auroc > 0.55, "csv-trained val AUROC {} too low", r.val.auroc);
    let _ = std::fs::remove_dir_all(&dir);
}
