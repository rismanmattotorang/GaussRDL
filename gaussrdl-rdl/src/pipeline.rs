//! End-to-end RDL experiment: synthetic relational DB → hetero temporal graph
//! → column encoders → SOTA model → task head → real autodiff training → eval.

use crate::encoder::DatabaseEncoder;
use crate::error::Result;
use crate::graph::HeteroGraph;
use crate::metrics::{accuracy_from_logits, mae, rmse, roc_auc};
use crate::models::{self, ModelConfig, ModelKind};
use crate::synthetic::{SyntheticConfig, SyntheticDataset};
use crate::task::{loss, TaskHead, TaskType};
use candle_core::{DType, Device, Tensor};
use candle_nn::ops::sigmoid;
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};

/// Configuration for a single experiment.
#[derive(Debug, Clone)]
pub struct ExperimentConfig {
    pub model: ModelKind,
    pub task: TaskType,
    pub model_cfg: ModelConfig,
    /// Width of the column-encoder node embeddings.
    pub channels: usize,
    pub epochs: usize,
    pub lr: f64,
    pub weight_decay: f64,
    pub seed: u64,
    pub data: SyntheticConfig,
    /// Print per-epoch progress.
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
            seed: 0,
            data: SyntheticConfig::default(),
            verbose: false,
        }
    }
}

/// Metrics for one data split.
#[derive(Debug, Clone, Default)]
pub struct SplitMetrics {
    pub loss: f32,
    pub auroc: f32,
    pub accuracy: f32,
    pub mae: f32,
    pub rmse: f32,
}

/// Result of running an experiment.
#[derive(Debug, Clone)]
pub struct ExperimentResult {
    pub model: String,
    pub task: TaskType,
    pub num_nodes: usize,
    pub num_edges: usize,
    pub num_relations: usize,
    pub train_losses: Vec<f32>,
    pub val: SplitMetrics,
    pub test: SplitMetrics,
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
    let train = idx[..n_train].to_vec();
    let val = idx[n_train..n_train + n_val].to_vec();
    let test = idx[n_train + n_val..].to_vec();
    (train, val, test)
}

/// Run a complete experiment and return its metrics.
pub fn run_experiment(cfg: &ExperimentConfig) -> Result<ExperimentResult> {
    let device = Device::Cpu;
    let ds = SyntheticDataset::generate(cfg.data.clone());
    let graph = HeteroGraph::build(&ds.db)?;
    let g = graph.edge_tensors(&device, ds.seed_time)?;

    let users_range = graph.table_node_ids("users")?;
    let user_offset = users_range.start;
    let num_users = users_range.len();
    let num_node_types = graph.table_names.len();

    // Shared parameter store.
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let encoder = DatabaseEncoder::new(&ds.db, cfg.channels, &device, vb.pp("encoder"))?;
    let model = models::build(
        cfg.model,
        cfg.channels,
        graph.num_relations,
        num_node_types,
        &cfg.model_cfg,
        vb.pp("model"),
    )?;
    let head = TaskHead::new(model.out_dim(), vb.pp("task"))?;

    // Labels + (for regression) standardization.
    let raw_labels: Vec<f32> = match cfg.task {
        TaskType::BinaryClassification => ds.churn_labels.clone(),
        TaskType::Regression => ds.ltv_labels.clone(),
    };
    let (train_i, val_i, test_i) = split_indices(num_users, cfg.seed.wrapping_add(7));
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

    let idx_tensor = |v: &[u32]| -> Result<Tensor> { Ok(Tensor::from_vec(v.to_vec(), v.len(), &device)?) };
    let target_tensor = |v: &[u32]| -> Result<Tensor> {
        let t: Vec<f32> = v.iter().map(|&i| norm_labels[i as usize]).collect();
        Ok(Tensor::from_vec(t.clone(), t.len(), &device)?)
    };
    let train_idx_t = idx_tensor(&train_i)?;
    let train_tgt_t = target_tensor(&train_i)?;

    // Forward producing the per-user prediction vector.
    let forward = |encoder: &DatabaseEncoder,
                   model: &Box<dyn crate::models::NodeEncoder>,
                   head: &TaskHead|
     -> Result<Tensor> {
        let x = encoder.encode()?;
        let z = model.forward(&x, &g)?;
        let user_z = z.narrow(0, user_offset, num_users)?;
        head.forward(&user_z)
    };

    let mut opt = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW {
            lr: cfg.lr,
            weight_decay: cfg.weight_decay,
            ..Default::default()
        },
    )?;

    let mut train_losses = Vec::with_capacity(cfg.epochs);
    for epoch in 0..cfg.epochs {
        let preds_all = forward(&encoder, &model, &head)?;
        let preds = preds_all.index_select(&train_idx_t, 0)?;
        let l = loss(cfg.task, &preds, &train_tgt_t)?;
        opt.backward_step(&l)?;
        let lv = l.to_scalar::<f32>()?;
        train_losses.push(lv);
        if cfg.verbose && (epoch % 10 == 0 || epoch + 1 == cfg.epochs) {
            println!("epoch {epoch:>3}  train_loss = {lv:.4}");
        }
    }

    // Evaluation (no grad needed; just read predictions).
    let preds_all = forward(&encoder, &model, &head)?;
    let eval_split = |split: &[u32]| -> Result<SplitMetrics> {
        let idx = idx_tensor(split)?;
        let p = preds_all.index_select(&idx, 0)?;
        let mut m = SplitMetrics::default();
        let labels: Vec<f32> = split.iter().map(|&i| raw_labels[i as usize]).collect();
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
                // De-standardize predictions to original units.
                let pn = p.to_vec1::<f32>()?;
                let preds: Vec<f32> = pn.iter().map(|&x| x * lstd + lmean).collect();
                m.mae = mae(&preds, &labels);
                m.rmse = rmse(&preds, &labels);
                m.loss = m.rmse;
            }
        }
        Ok(m)
    };

    Ok(ExperimentResult {
        model: cfg.model.name().to_string(),
        task: cfg.task,
        num_nodes: graph.num_nodes,
        num_edges: g.num_edges,
        num_relations: graph.num_relations,
        train_losses,
        val: eval_split(&val_i)?,
        test: eval_split(&test_i)?,
    })
}
