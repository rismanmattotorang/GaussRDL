//! Minimal CLI for the GaussRDL v2 engine.
//!
//! Usage:
//!   gaussrdl-rdl <model> <task> [epochs]
//!   gaussrdl-rdl benchmark            # run every model on both tasks
//!
//! model: sage | rgcn | gat | relgt
//! task : churn | ltv

use gaussrdl_rdl::models::ModelKind;
use gaussrdl_rdl::synthetic::{SyntheticConfig, SyntheticDataset};
use gaussrdl_rdl::{run_experiment, DataSource, ExperimentConfig, TaskType};
use std::collections::HashMap;

fn run_one(model: ModelKind, task: TaskType, epochs: usize, verbose: bool) {
    run_cfg(ExperimentConfig { model, task, epochs, verbose, ..Default::default() }, task);
}

fn run_cfg(cfg: ExperimentConfig, task: TaskType) {
    let model_name = cfg.model.name().to_string();
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
        Err(e) => eprintln!("error running {model_name}/{task:?}: {e}"),
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

    // Generate a real CSV database from the synthetic generator.
    if args.len() >= 2 && args[1] == "make-sample" {
        let dir = args.get(2).cloned().unwrap_or_else(|| "data/sample".to_string());
        let ds = SyntheticDataset::generate(SyntheticConfig::default());
        let mut labels = HashMap::new();
        labels.insert("churn".to_string(), ds.churn_labels.clone());
        labels.insert("ltv".to_string(), ds.ltv_labels.clone());
        match gaussrdl_rdl::io::write_database_csv(&dir, &ds.db, "users", &labels) {
            Ok(()) => println!("wrote sample CSV database + schema.json to {dir}"),
            Err(e) => {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        }
        return;
    }

    // Train on a real CSV database: gaussrdl-rdl csv <dir> <label> <model> <task> [epochs]
    if args.len() >= 2 && args[1] == "csv" {
        if args.len() < 6 {
            eprintln!("usage: gaussrdl-rdl csv <dir> <label> <model> <task> [epochs]");
            std::process::exit(2);
        }
        let dir = args[2].clone();
        let label = args[3].clone();
        let model = ModelKind::parse(&args[4]).unwrap_or_else(|e| { eprintln!("{e}"); std::process::exit(2) });
        let task = TaskType::parse(&args[5]).unwrap_or_else(|| { eprintln!("bad task"); std::process::exit(2) });
        let epochs = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(80);
        let cfg = ExperimentConfig {
            model,
            task,
            epochs,
            verbose: true,
            data: DataSource::Csv { dir, label, seed_quantile: 1.0 },
            ..Default::default()
        };
        run_cfg(cfg, task);
        return;
    }

    if args.len() < 3 {
        eprintln!("usage: gaussrdl-rdl <model> <task> [epochs]");
        eprintln!("       gaussrdl-rdl benchmark");
        eprintln!("       gaussrdl-rdl make-sample <dir>");
        eprintln!("       gaussrdl-rdl csv <dir> <label> <model> <task> [epochs]");
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
