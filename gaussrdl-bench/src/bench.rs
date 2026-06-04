//! Paper-faithful benchmark runner.
//!
//! Follows RelBench best practice: the correct headline metric per task type
//! (ROC-AUC for classification, MAE for regression), temporal leakage-free
//! splits (inherited from the engine), validation-based model selection, and
//! **mean ± std over multiple seeds** so reported numbers are robust.

use crate::manager::{DatasetManager, DatasetStatus};
use crate::registry::{DatasetInfo, TaskKind};
use gaussrdl_rdl::models::ModelKind;
use gaussrdl_rdl::{
    run_experiment_cb, DataSource, EpochMetrics, ExperimentConfig, InferenceReport, LrSchedule,
    ModelConfig, RdlError, Result, TaskType, TrainingHistory,
};
use serde::{Deserialize, Serialize};

/// Per-seed validation/test scores of the headline metric.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeedScore {
    pub seed: u64,
    pub val: f32,
    pub test: f32,
}

/// A benchmark report suitable for a results table in a paper.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkReport {
    pub dataset: String,
    pub task: String,
    pub model: String,
    pub metric_name: String,
    pub higher_is_better: bool,
    pub test_mean: f32,
    pub test_std: f32,
    pub val_mean: f32,
    pub seeds: Vec<SeedScore>,
    pub num_nodes: usize,
    pub num_edges: usize,
    pub num_relations: usize,
    /// Inference KPIs from the first seed.
    pub inference: InferenceReport,
    /// Training history from the first seed (for plotting).
    pub history: TrainingHistory,
}

impl BenchmarkReport {
    /// `metric mean ± std` formatted like a paper table cell.
    pub fn headline(&self) -> String {
        format!("{:.4} ± {:.4} {}", self.test_mean, self.test_std, self.metric_name)
    }
}

fn mean_std(xs: &[f32]) -> (f32, f32) {
    let n = xs.len().max(1) as f32;
    let m = xs.iter().sum::<f32>() / n;
    let v = xs.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / n;
    (m, v.sqrt())
}

/// Run a benchmark for `model` on `task_name` of `info`, over `seeds`.
///
/// The dataset must already be present locally (`manager.ensure`). The `cb`
/// receives epoch metrics for the first seed (for live monitoring).
#[allow(clippy::too_many_arguments)]
pub fn run_benchmark(
    manager: &DatasetManager,
    info: &DatasetInfo,
    task_name: &str,
    model: ModelKind,
    model_cfg: ModelConfig,
    channels: usize,
    epochs: usize,
    seeds: &[u64],
    cb: &mut dyn FnMut(&EpochMetrics),
) -> Result<BenchmarkReport> {
    if manager.status(info) != DatasetStatus::Ready {
        return Err(RdlError::Data(format!(
            "dataset `{}` is not available locally; download it first",
            info.id
        )));
    }
    let task = info
        .task(task_name)
        .ok_or_else(|| RdlError::Config(format!("task `{task_name}` not in dataset `{}`", info.id)))?;
    if task.label.is_empty() {
        return Err(RdlError::Config(format!(
            "task `{task_name}` has no materialized label column (official RelBench tasks require the upstream data)"
        )));
    }
    let task_type = match task.kind {
        TaskKind::Classification => TaskType::BinaryClassification,
        TaskKind::Regression => TaskType::Regression,
    };
    let dir = manager.dir(&info.id).to_string_lossy().into_owned();
    let seeds = if seeds.is_empty() { &[0u64][..] } else { seeds };

    let mut per_seed = Vec::new();
    let mut val_scores = Vec::new();
    let mut test_scores = Vec::new();
    let mut first: Option<(InferenceReport, TrainingHistory, usize, usize, usize)> = None;

    for (i, &seed) in seeds.iter().enumerate() {
        let cfg = ExperimentConfig {
            model,
            task: task_type,
            model_cfg: model_cfg.clone(),
            channels,
            epochs,
            lr: 1e-2,
            schedule: LrSchedule::Cosine,
            early_stopping_patience: 0,
            seed,
            data: DataSource::Csv { dir: dir.clone(), label: task.label.clone(), seed_quantile: 1.0 },
            ..Default::default()
        };
        // Forward epoch metrics only for the first seed.
        let r = if i == 0 {
            run_experiment_cb(&cfg, cb)?
        } else {
            run_experiment_cb(&cfg, &mut |_| {})?
        };
        let (val, test) = match task.kind {
            TaskKind::Classification => (r.val.auroc, r.test.auroc),
            TaskKind::Regression => (r.val.mae, r.test.mae),
        };
        per_seed.push(SeedScore { seed, val, test });
        val_scores.push(val);
        test_scores.push(test);
        if first.is_none() {
            first = Some((r.inference.clone(), r.history.clone(), r.num_nodes, r.num_edges, r.num_relations));
        }
    }

    let (test_mean, test_std) = mean_std(&test_scores);
    let (val_mean, _) = mean_std(&val_scores);
    let (inference, history, num_nodes, num_edges, num_relations) = first.unwrap();

    Ok(BenchmarkReport {
        dataset: info.id.clone(),
        task: task_name.to_string(),
        model: model.name().to_string(),
        metric_name: task.kind.metric_name().to_string(),
        higher_is_better: task.kind.higher_is_better(),
        test_mean,
        test_std,
        val_mean,
        seeds: per_seed,
        num_nodes,
        num_edges,
        num_relations,
        inference,
        history,
    })
}
