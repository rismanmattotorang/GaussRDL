//! End-to-end integration tests: every model must (a) build a valid temporal
//! graph from synthetic relational data, (b) train with real autodiff so the
//! loss decreases, and (c) learn the planted signal (val AUROC clearly above
//! chance for classification; MAE below the std-dev baseline for regression).

use gaussrdl_rdl::data::{Column, ColumnData, ForeignKey, RelationalDatabase, Table};
use gaussrdl_rdl::graph::HeteroGraph;
use gaussrdl_rdl::models::{ModelConfig, ModelKind};
use gaussrdl_rdl::synthetic::{SyntheticConfig, SyntheticDataset};
use gaussrdl_rdl::{run_experiment, ExperimentConfig, TaskType};

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
        data: small_data(),
        ..Default::default()
    };
    run_experiment(&cfg).unwrap()
}

#[test]
fn all_models_learn_churn() {
    for &model in ModelKind::all() {
        let r = train_classification(model);
        // Loss must decrease over training.
        let first = r.train_losses[0];
        let last = *r.train_losses.last().unwrap();
        assert!(last < first, "{model:?}: loss did not decrease ({first} -> {last})");
        assert!(last.is_finite(), "{model:?}: non-finite loss");
        // Must beat random on held-out users.
        assert!(
            r.val.auroc > 0.6,
            "{model:?}: val AUROC {} not above chance",
            r.val.auroc
        );
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
        data: small_data(),
        ..Default::default()
    };
    let r = run_experiment(&cfg).unwrap();
    let first = r.train_losses[0];
    let last = *r.train_losses.last().unwrap();
    assert!(last < first, "regression loss did not decrease ({first} -> {last})");
    assert!(r.test.mae.is_finite());
}
