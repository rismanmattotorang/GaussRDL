//! End-to-end RDL experiment: relational DB → hetero temporal graph → column
//! encoders → SOTA model → task head → monitored autodiff training → eval.
//!
//! Supports synthetic data and real CSV databases, full training monitoring
//! (per-epoch metrics, LR schedule, early stopping, checkpoints), and inference
//! KPIs on the held-out test set.

use crate::encoder::DatabaseEncoder;
use crate::error::{RdlError, Result};
use crate::graph::HeteroGraph;
use crate::inference::InferenceReport;
use crate::io::load_database_csv;
use crate::metrics::{accuracy_from_logits, mae, rmse, roc_auc};
use crate::models::{self, ModelConfig, ModelKind};
use crate::synthetic::{SyntheticConfig, SyntheticDataset};
use crate::task::{loss, TaskHead, TaskType};
use crate::train::{grad_norm, EarlyStopper, EpochMetrics, LrSchedule, TrainingHistory};
use candle_core::{DType, Device, Tensor};
use candle_nn::ops::sigmoid;
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Where the relational data comes from.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataSource {
    /// Built-in synthetic generator.
    Synthetic(SyntheticConfig),
    /// A directory of CSVs + `schema.json`; `label` is the seed-table column to
    /// predict; the seed time is the `seed_quantile` of observed event times.
    Csv { dir: String, label: String, seed_quantile: f64 },
}

impl Default for DataSource {
    fn default() -> Self {
        DataSource::Synthetic(SyntheticConfig::default())
    }
}

/// Configuration for a single experiment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentConfig {
    pub model: ModelKind,
    pub task: TaskType,
    pub model_cfg: ModelConfig,
    pub channels: usize,
    pub epochs: usize,
    pub lr: f64,
    pub weight_decay: f64,
    pub schedule: LrSchedule,
    /// Stop if val metric does not improve for this many epochs (0 = disabled).
    pub early_stopping_patience: usize,
    pub seed: u64,
    pub data: DataSource,
    /// Optional path to persist the best checkpoint.
    pub checkpoint: Option<String>,
    pub verbose: bool,
}

impl Default for ExperimentConfig {
    fn default() -> Self {
        Self {
            model: ModelKind::HeteroSAGE,
            task: TaskType::BinaryClassification,
            model_cfg: ModelConfig::default(),
            channels: 64,
            epochs: 60,
            lr: 1e-2,
            weight_decay: 1e-5,
            schedule: LrSchedule::Cosine,
            early_stopping_patience: 0,
            seed: 0,
            data: DataSource::default(),
            checkpoint: None,
            verbose: false,
        }
    }
}

/// Metrics for one data split.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SplitMetrics {
    pub loss: f32,
    pub auroc: f32,
    pub accuracy: f32,
    pub mae: f32,
    pub rmse: f32,
}

/// Result of running an experiment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentResult {
    pub model: String,
    pub task: TaskType,
    pub num_nodes: usize,
    pub num_edges: usize,
    pub num_relations: usize,
    pub history: TrainingHistory,
    pub val: SplitMetrics,
    pub test: SplitMetrics,
    pub inference: InferenceReport,
}

impl ExperimentResult {
    /// Convenience accessor for the per-epoch training loss curve.
    pub fn train_losses(&self) -> Vec<f32> {
        self.history.epochs.iter().map(|e| e.train_loss).collect()
    }
}

fn split_indices(n: usize, seed: u64) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    use rand::rngs::StdRng;
    use rand::seq::SliceRandom;
    use rand::SeedableRng;
    let mut idx: Vec<u32> = (0..n as u32).collect();
    let mut rng = StdRng::seed_from_u64(seed);
    idx.shuffle(&mut rng);
    let n_train = (n as f64 * 0.6) as usize;
    let n_val = (n as f64 * 0.2) as usize;
    (idx[..n_train].to_vec(), idx[n_train..n_train + n_val].to_vec(), idx[n_train + n_val..].to_vec())
}

/// Run an experiment with no live callback.
pub fn run_experiment(cfg: &ExperimentConfig) -> Result<ExperimentResult> {
    run_experiment_cb(cfg, &mut |_| {})
}

/// Run an experiment, invoking `cb` after every epoch with the latest metrics
/// (used by the TUI and web UI for live monitoring).
pub fn run_experiment_cb(
    cfg: &ExperimentConfig,
    cb: &mut dyn FnMut(&EpochMetrics),
) -> Result<ExperimentResult> {
    let device = Device::Cpu;

    // --- Resolve data source -> (db, seed_table, labels, seed_time) ---
    let (db, seed_table, raw_labels, seed_time) = match &cfg.data {
        DataSource::Synthetic(scfg) => {
            let ds = SyntheticDataset::generate(scfg.clone());
            let labels = match cfg.task {
                TaskType::BinaryClassification => ds.churn_labels.clone(),
                TaskType::Regression => ds.ltv_labels.clone(),
            };
            (ds.db, "users".to_string(), labels, ds.seed_time)
        }
        DataSource::Csv { dir, label, seed_quantile } => {
            let loaded = load_database_csv(dir)?;
            let labels = loaded
                .labels
                .get(label)
                .ok_or_else(|| RdlError::Config(format!("label column `{label}` not found in seed table")))?
                .clone();
            let st = loaded.seed_time_quantile(*seed_quantile);
            (loaded.db, loaded.seed_table.clone(), labels, st)
        }
    };

    let graph = HeteroGraph::build(&db)?;
    let g = graph.edge_tensors(&device, seed_time)?;
    let seed_range = graph.table_node_ids(&seed_table)?;
    let seed_offset = seed_range.start;
    let num_seeds = seed_range.len();
    if raw_labels.len() != num_seeds {
        return Err(RdlError::Config(format!(
            "{} labels for {} seed rows",
            raw_labels.len(),
            num_seeds
        )));
    }
    let num_node_types = graph.table_names.len();

    // --- Parameters ---
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let encoder = DatabaseEncoder::new(&db, cfg.channels, &device, vb.pp("encoder"))?;
    let model = models::build(
        cfg.model,
        cfg.channels,
        graph.num_relations,
        num_node_types,
        &cfg.model_cfg,
        vb.pp("model"),
    )?;
    let head = TaskHead::new(model.out_dim(), vb.pp("task"))?;

    // --- Labels + standardization (regression) ---
    let (train_i, val_i, test_i) = split_indices(num_seeds, cfg.seed.wrapping_add(7));
    let (lmean, lstd) = match cfg.task {
        TaskType::Regression => {
            let tr: Vec<f32> = train_i.iter().map(|&i| raw_labels[i as usize]).collect();
            let m = tr.iter().sum::<f32>() / tr.len().max(1) as f32;
            let v = tr.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / tr.len().max(1) as f32;
            (m, v.sqrt().max(1e-6))
        }
        TaskType::BinaryClassification => (0.0, 1.0),
    };
    let norm_labels: Vec<f32> = raw_labels.iter().map(|&y| (y - lmean) / lstd).collect();

    let idx_t = |v: &[u32]| -> Result<Tensor> { Ok(Tensor::from_vec(v.to_vec(), v.len(), &device)?) };
    let tgt_t = |v: &[u32]| -> Result<Tensor> {
        let t: Vec<f32> = v.iter().map(|&i| norm_labels[i as usize]).collect();
        Ok(Tensor::from_vec(t.clone(), t.len(), &device)?)
    };
    let train_idx = idx_t(&train_i)?;
    let train_tgt = tgt_t(&train_i)?;
    let val_idx = idx_t(&val_i)?;

    let forward = |train: bool| -> Result<Tensor> {
        let x = encoder.encode()?;
        let z = model.forward(&x, &g, train)?;
        let seed_z = z.narrow(0, seed_offset, num_seeds)?;
        head.forward(&seed_z)
    };

    // Headline metric per split: AUROC (clf, maximize) / MAE (reg, minimize).
    let split_metric = |preds_all: &Tensor, split: &[u32]| -> Result<f32> {
        let p = preds_all.index_select(&idx_t(split)?, 0)?;
        let labels: Vec<f32> = split.iter().map(|&i| raw_labels[i as usize]).collect();
        Ok(match cfg.task {
            TaskType::BinaryClassification => {
                let probs = sigmoid(&p)?.to_vec1::<f32>()?;
                roc_auc(&probs, &labels)
            }
            TaskType::Regression => {
                let preds: Vec<f32> = p.to_vec1::<f32>()?.iter().map(|&x| x * lstd + lmean).collect();
                mae(&preds, &labels)
            }
        })
    };

    let vars = varmap.all_vars();
    let mut opt = AdamW::new(
        vars.clone(),
        ParamsAdamW { lr: cfg.lr, weight_decay: cfg.weight_decay, ..Default::default() },
    )?;

    let maximize = matches!(cfg.task, TaskType::BinaryClassification);
    let mut stopper = EarlyStopper::new(cfg.early_stopping_patience, maximize);
    let ckpt_path = cfg.checkpoint.clone().unwrap_or_else(|| {
        // Unique per experiment so concurrent runs never clobber each other.
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir()
            .join(format!("gaussrdl_ckpt_{}_{}_{}.safetensors", std::process::id(), n, nanos))
            .to_string_lossy()
            .into_owned()
    });

    let mut history = TrainingHistory::default();
    history.best_val = if maximize { f32::NEG_INFINITY } else { f32::INFINITY };

    for epoch in 0..cfg.epochs {
        let t0 = Instant::now();
        let lr = cfg.schedule.at(cfg.lr, epoch, cfg.epochs);
        opt.set_learning_rate(lr);

        let preds_all = forward(true)?;
        let preds = preds_all.index_select(&train_idx, 0)?;
        let l = loss(cfg.task, &preds, &train_tgt)?;
        let grads = l.backward()?;
        let gn = grad_norm(&grads, &vars)?;
        opt.step(&grads)?;
        let train_loss = l.to_scalar::<f32>()?;

        // Validation (eval mode, no dropout).
        let eval_preds = forward(false)?;
        let val_metric = split_metric(&eval_preds, &val_i)?;
        let val_p = eval_preds.index_select(&val_idx, 0)?;
        let val_tgt = tgt_t(&val_i)?;
        let val_loss = loss(cfg.task, &val_p, &val_tgt)?.to_scalar::<f32>()?;

        let em = EpochMetrics {
            epoch,
            train_loss,
            val_metric,
            val_loss,
            learning_rate: lr as f32,
            grad_norm: gn,
            seconds: t0.elapsed().as_secs_f32(),
        };

        if stopper.update(epoch, val_metric) {
            history.best_val = val_metric;
            history.best_epoch = epoch;
            let _ = varmap.save(&ckpt_path); // best-effort checkpoint
        }
        if cfg.verbose && (epoch % 10 == 0 || epoch + 1 == cfg.epochs) {
            println!(
                "epoch {epoch:>3}  loss={train_loss:.4}  val={val_metric:.4}  lr={lr:.5}  |g|={gn:.3}"
            );
        }
        cb(&em);
        history.epochs.push(em);

        if stopper.should_stop() {
            history.stopped_early = true;
            if cfg.verbose {
                println!("early stopping at epoch {epoch} (best epoch {})", stopper.best_epoch);
            }
            break;
        }
    }

    // Restore best checkpoint for final evaluation.
    let _ = varmap.load(&ckpt_path);
    if cfg.checkpoint.is_none() {
        let _ = std::fs::remove_file(&ckpt_path);
    }

    // --- Final eval + inference KPIs ---
    let t0 = Instant::now();
    let preds_all = forward(false)?;
    let latency_ms = t0.elapsed().as_secs_f32() * 1000.0;

    let eval_split = |split: &[u32]| -> Result<SplitMetrics> {
        let p = preds_all.index_select(&idx_t(split)?, 0)?;
        let labels: Vec<f32> = split.iter().map(|&i| raw_labels[i as usize]).collect();
        let mut m = SplitMetrics::default();
        match cfg.task {
            TaskType::BinaryClassification => {
                let logits = p.to_vec1::<f32>()?;
                let probs = sigmoid(&p)?.to_vec1::<f32>()?;
                m.auroc = roc_auc(&probs, &labels);
                m.accuracy = accuracy_from_logits(&logits, &labels);
                let tgt = Tensor::from_vec(labels.clone(), labels.len(), &device)?;
                m.loss = loss(cfg.task, &p, &tgt)?.to_scalar::<f32>()?;
            }
            TaskType::Regression => {
                let preds: Vec<f32> = p.to_vec1::<f32>()?.iter().map(|&x| x * lstd + lmean).collect();
                m.mae = mae(&preds, &labels);
                m.rmse = rmse(&preds, &labels);
                m.loss = m.rmse;
            }
        }
        Ok(m)
    };

    // Inference report on the test split.
    let test_p = preds_all.index_select(&idx_t(&test_i)?, 0)?;
    let test_labels: Vec<f32> = test_i.iter().map(|&i| raw_labels[i as usize]).collect();
    let inference = match cfg.task {
        TaskType::BinaryClassification => {
            let logits = test_p.to_vec1::<f32>()?;
            let probs = sigmoid(&test_p)?.to_vec1::<f32>()?;
            InferenceReport::build(cfg.task, &probs, &logits, &test_labels, latency_ms)
        }
        TaskType::Regression => {
            let preds: Vec<f32> = test_p.to_vec1::<f32>()?.iter().map(|&x| x * lstd + lmean).collect();
            InferenceReport::build(cfg.task, &preds, &preds, &test_labels, latency_ms)
        }
    };

    Ok(ExperimentResult {
        model: cfg.model.name().to_string(),
        task: cfg.task,
        num_nodes: graph.num_nodes,
        num_edges: g.num_edges,
        num_relations: graph.num_relations,
        history,
        val: eval_split(&val_i)?,
        test: eval_split(&test_i)?,
        inference,
    })
}
