//! Minimal CLI for the GaussRDL v2 engine.
//!
//! Usage:
//!   gaussrdl-rdl <model> <task> [epochs]
//!   gaussrdl-rdl benchmark            # run every model on both tasks
//!
//! model: sage | rgcn | gat | relgt
//! task : churn | ltv

use gaussrdl_rdl::models::ModelKind;
use gaussrdl_rdl::{run_experiment, ExperimentConfig, TaskType};

fn run_one(model: ModelKind, task: TaskType, epochs: usize, verbose: bool) {
    let cfg = ExperimentConfig {
        model,
        task,
        epochs,
        verbose,
        ..Default::default()
    };
    match run_experiment(&cfg) {
        Ok(r) => {
            let head = match task {
                TaskType::BinaryClassification => {
                    format!("AUROC={:.4} ACC={:.4}", r.val.auroc, r.val.accuracy)
                }
                TaskType::Regression => format!("MAE={:.3} RMSE={:.3}", r.val.mae, r.val.rmse),
            };
            let test = match task {
                TaskType::BinaryClassification => {
                    format!("AUROC={:.4}", r.test.auroc)
                }
                TaskType::Regression => format!("MAE={:.3}", r.test.mae),
            };
            println!(
                "{:<10} {:<14} | graph: {} nodes / {} edges / {} rels | val {} | test {}",
                r.model,
                format!("{:?}", task),
                r.num_nodes,
                r.num_edges,
                r.num_relations,
                head,
                test
            );
        }
        Err(e) => eprintln!("error running {model:?}/{task:?}: {e}"),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() >= 2 && args[1] == "benchmark" {
        println!("GaussRDL v2 — full benchmark on synthetic relational data\n");
        for &task in &[TaskType::BinaryClassification, TaskType::Regression] {
            for &model in ModelKind::all() {
                run_one(model, task, 60, false);
            }
            println!();
        }
        return;
    }

    if args.len() < 3 {
        eprintln!("usage: gaussrdl-rdl <model> <task> [epochs]");
        eprintln!("       gaussrdl-rdl benchmark");
        eprintln!("model: sage | rgcn | gat | relgt    task: churn | ltv");
        std::process::exit(2);
    }
    let model = ModelKind::parse(&args[1]).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(2);
    });
    let task = TaskType::parse(&args[2]).unwrap_or_else(|| {
        eprintln!("unknown task `{}` (churn|ltv)", args[2]);
        std::process::exit(2);
    });
    let epochs = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(80);
    run_one(model, task, epochs, true);
}
