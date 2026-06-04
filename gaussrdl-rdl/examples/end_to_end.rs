//! End-to-end demonstration: generate a synthetic relational database, build
//! the temporal graph, and train each SOTA model on the user-churn task.

use gaussrdl_rdl::models::{ModelConfig, ModelKind};
use gaussrdl_rdl::synthetic::SyntheticConfig;
use gaussrdl_rdl::{run_experiment, DataSource, ExperimentConfig, TaskType};

fn main() -> gaussrdl_rdl::Result<()> {
    println!("GaussRDL v2 — end-to-end Relational Deep Learning demo\n");

    for &model in ModelKind::all() {
        let cfg = ExperimentConfig {
            model,
            task: TaskType::BinaryClassification,
            model_cfg: ModelConfig { hidden_dim: 64, num_layers: 2, ..Default::default() },
            channels: 64,
            epochs: 80,
            lr: 1e-2,
            data: DataSource::Synthetic(SyntheticConfig::default()),
            ..Default::default()
        };
        let r = run_experiment(&cfg)?;
        println!(
            "{:<10} | nodes={:<5} edges={:<6} rels={} | val AUROC={:.4} ACC={:.4} | test AUROC={:.4}",
            r.model, r.num_nodes, r.num_edges, r.num_relations,
            r.val.auroc, r.val.accuracy, r.test.auroc,
        );
    }
    Ok(())
}
