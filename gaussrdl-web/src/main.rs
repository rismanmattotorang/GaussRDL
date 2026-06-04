//! Web UI for GaussRDL — Gaussian Technologies' Relational Deep Learning suite.
//!
//! Two workspaces in one app:
//!   * **Datasets** — browse the benchmark catalog, download/manage datasets.
//!   * **Benchmark** — train a model on a dataset/task, evaluate, and report
//!     paper-style results (mean ± std over seeds) with live curves and KPIs.
//!
//! Run: `cargo run -p gaussrdl-web` then open http://127.0.0.1:8080

use axum::{routing::get, routing::post, Json, Router};
use gaussrdl_bench::{catalog, find, run_benchmark, DatasetManager, DatasetStatus};
use gaussrdl_rdl::models::{ModelConfig, ModelKind};
use gaussrdl_rdl::synthetic::SyntheticConfig;
use gaussrdl_rdl::{run_experiment, DataSource, ExperimentConfig, LrSchedule, TaskType};
use serde::Deserialize;
use serde_json::{json, Value};

// ---------------------------------------------------------------------------
// Quick (synthetic) training request — kept for the lightweight playground.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct TrainRequest {
    pub model: String,
    pub task: String,
    #[serde(default = "d_epochs")] pub epochs: usize,
    #[serde(default = "d_hidden")] pub hidden_dim: usize,
    #[serde(default = "d_layers")] pub num_layers: usize,
    #[serde(default = "d_heads")] pub num_heads: usize,
    #[serde(default = "d_lr")] pub lr: f64,
    #[serde(default = "d_dropout")] pub dropout: f32,
    #[serde(default = "d_users")] pub num_users: usize,
    #[serde(default)] pub early_stopping_patience: usize,
}

fn d_epochs() -> usize { 50 }
fn d_hidden() -> usize { 64 }
fn d_layers() -> usize { 2 }
fn d_heads() -> usize { 4 }
fn d_lr() -> f64 { 0.01 }
fn d_dropout() -> f32 { 0.1 }
fn d_users() -> usize { 300 }

/// Map a quick-train request to a validated config (pure; unit-tested).
pub fn build_config(req: &TrainRequest) -> Result<ExperimentConfig, String> {
    let model = ModelKind::parse(&req.model).map_err(|e| e.to_string())?;
    let task = TaskType::parse(&req.task).ok_or_else(|| format!("unknown task `{}`", req.task))?;
    if req.hidden_dim % req.num_heads != 0 {
        return Err(format!("hidden_dim ({}) must be divisible by num_heads ({})", req.hidden_dim, req.num_heads));
    }
    Ok(ExperimentConfig {
        model,
        task,
        model_cfg: ModelConfig {
            hidden_dim: req.hidden_dim,
            num_layers: req.num_layers,
            num_heads: req.num_heads,
            dropout: req.dropout,
            ..Default::default()
        },
        channels: req.hidden_dim,
        epochs: req.epochs.clamp(1, 500),
        lr: req.lr,
        schedule: LrSchedule::Cosine,
        early_stopping_patience: req.early_stopping_patience,
        data: DataSource::Synthetic(SyntheticConfig { num_users: req.num_users.clamp(50, 5000), ..Default::default() }),
        ..Default::default()
    })
}

async fn train(Json(req): Json<TrainRequest>) -> Json<Value> {
    let cfg = match build_config(&req) {
        Ok(c) => c,
        Err(e) => return Json(json!({ "error": e })),
    };
    match tokio::task::spawn_blocking(move || run_experiment(&cfg)).await {
        Ok(Ok(r)) => Json(serde_json::to_value(&r).unwrap()),
        Ok(Err(e)) => Json(json!({ "error": e.to_string() })),
        Err(e) => Json(json!({ "error": format!("join error: {e}") })),
    }
}

// ---------------------------------------------------------------------------
// Dataset management
// ---------------------------------------------------------------------------

fn status_str(s: &DatasetStatus) -> &'static str {
    match s {
        DatasetStatus::Ready => "ready",
        DatasetStatus::NotDownloaded => "not_downloaded",
        DatasetStatus::Error(_) => "error",
    }
}

async fn datasets() -> Json<Value> {
    let mgr = DatasetManager::with_default_root();
    let items: Vec<Value> = catalog()
        .iter()
        .map(|d| {
            let tasks: Vec<Value> = d
                .tasks
                .iter()
                .map(|t| json!({ "name": t.name, "metric": t.kind.metric_name(), "description": t.description, "runnable": !t.label.is_empty() }))
                .collect();
            json!({
                "id": d.id, "display_name": d.display_name, "domain": d.domain,
                "description": d.description, "official": d.official, "approx_size": d.approx_size,
                "citation": d.citation, "offline_capable": d.offline_capable(),
                "status": status_str(&mgr.status(d)), "size_bytes": mgr.size_bytes(&d.id),
                "tasks": tasks,
            })
        })
        .collect();
    Json(json!({ "datasets": items, "cache_dir": mgr.root().to_string_lossy() }))
}

#[derive(Deserialize)]
struct IdReq { id: String }

async fn dataset_download(Json(req): Json<IdReq>) -> Json<Value> {
    let info = match find(&req.id) { Some(i) => i, None => return Json(json!({"error":"unknown dataset"})) };
    let id = req.id.clone();
    let res = tokio::task::spawn_blocking(move || {
        let mgr = DatasetManager::with_default_root();
        mgr.ensure(&info, &mut |_p| {}).map(|_| ())
    })
    .await;
    match res {
        Ok(Ok(())) => Json(json!({ "id": id, "status": "ready" })),
        Ok(Err(e)) => Json(json!({ "id": id, "status": "error", "error": e.to_string() })),
        Err(e) => Json(json!({ "error": format!("join error: {e}") })),
    }
}

async fn dataset_delete(Json(req): Json<IdReq>) -> Json<Value> {
    let mgr = DatasetManager::with_default_root();
    match mgr.delete(&req.id) {
        Ok(()) => Json(json!({ "id": req.id, "status": "not_downloaded" })),
        Err(e) => Json(json!({ "error": e.to_string() })),
    }
}

// ---------------------------------------------------------------------------
// Benchmark
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct BenchRequest {
    pub dataset: String,
    pub task: String,
    pub model: String,
    #[serde(default = "d_hidden")] pub hidden_dim: usize,
    #[serde(default = "d_layers")] pub num_layers: usize,
    #[serde(default = "d_heads")] pub num_heads: usize,
    #[serde(default = "d_dropout")] pub dropout: f32,
    #[serde(default = "d_epochs")] pub epochs: usize,
    #[serde(default = "d_seeds")] pub seeds: usize,
}
fn d_seeds() -> usize { 3 }

/// Validate a benchmark request into its components (pure; unit-tested).
pub fn validate_bench(req: &BenchRequest) -> Result<(ModelKind, ModelConfig, Vec<u64>), String> {
    let model = ModelKind::parse(&req.model).map_err(|e| e.to_string())?;
    let info = find(&req.dataset).ok_or_else(|| format!("unknown dataset `{}`", req.dataset))?;
    if info.task(&req.task).is_none() {
        return Err(format!("dataset `{}` has no task `{}`", req.dataset, req.task));
    }
    if req.hidden_dim % req.num_heads != 0 {
        return Err(format!("hidden_dim ({}) must be divisible by num_heads ({})", req.hidden_dim, req.num_heads));
    }
    let cfg = ModelConfig {
        hidden_dim: req.hidden_dim,
        num_layers: req.num_layers,
        num_heads: req.num_heads,
        dropout: req.dropout,
        ..Default::default()
    };
    let seeds: Vec<u64> = (0..req.seeds.clamp(1, 5) as u64).collect();
    Ok((model, cfg, seeds))
}

async fn benchmark(Json(req): Json<BenchRequest>) -> Json<Value> {
    let (model, cfg, seeds) = match validate_bench(&req) {
        Ok(x) => x,
        Err(e) => return Json(json!({ "error": e })),
    };
    let channels = req.hidden_dim;
    let epochs = req.epochs.clamp(1, 500);
    let task = req.task.clone();
    let id = req.dataset.clone();
    let res = tokio::task::spawn_blocking(move || {
        let mgr = DatasetManager::with_default_root();
        let info = find(&id).unwrap();
        mgr.ensure(&info, &mut |_| {})?;
        run_benchmark(&mgr, &info, &task, model, cfg, channels, epochs, &seeds, &mut |_| {})
    })
    .await;
    match res {
        Ok(Ok(report)) => Json(serde_json::to_value(&report).unwrap()),
        Ok(Err(e)) => Json(json!({ "error": e.to_string() })),
        Err(e) => Json(json!({ "error": format!("join error: {e}") })),
    }
}

async fn index() -> axum::response::Html<&'static str> {
    axum::response::Html(INDEX_HTML)
}

#[tokio::main]
async fn main() {
    let app = Router::new()
        .route("/", get(index))
        .route("/api/datasets", get(datasets))
        .route("/api/datasets/download", post(dataset_download))
        .route("/api/datasets/delete", post(dataset_delete))
        .route("/api/benchmark", post(benchmark))
        .route("/api/train", post(train));
    let port: u16 = std::env::var("PORT").ok().and_then(|p| p.parse().ok()).unwrap_or(8080);
    let addr = format!("127.0.0.1:{port}");
    println!("GaussRDL web UI on http://{addr}");
    let listener = tokio::net::TcpListener::bind(&addr).await.expect("bind");
    axum::serve(listener, app).await.expect("serve");
}

const INDEX_HTML: &str = include_str!("index.html");

#[cfg(test)]
mod tests {
    use super::*;

    fn treq() -> TrainRequest {
        TrainRequest { model: "relgt".into(), task: "churn".into(), epochs: 10, hidden_dim: 32, num_layers: 2, num_heads: 4, lr: 0.01, dropout: 0.1, num_users: 100, early_stopping_patience: 0 }
    }

    #[test]
    fn build_config_ok() {
        let c = build_config(&treq()).unwrap();
        assert_eq!(c.model, ModelKind::RelGt);
        assert_eq!(c.task, TaskType::BinaryClassification);
    }

    #[test]
    fn build_config_rejects_indivisible_heads() {
        let mut r = treq();
        r.hidden_dim = 30;
        assert!(build_config(&r).is_err());
    }

    #[test]
    fn validate_bench_ok() {
        let r = BenchRequest { dataset: "gauss-ecom-small".into(), task: "user-churn".into(), model: "relgt".into(), hidden_dim: 64, num_layers: 2, num_heads: 4, dropout: 0.1, epochs: 20, seeds: 3 };
        let (m, _cfg, seeds) = validate_bench(&r).unwrap();
        assert_eq!(m, ModelKind::RelGt);
        assert_eq!(seeds.len(), 3);
    }

    #[test]
    fn validate_bench_rejects_unknown_dataset_and_task() {
        let mut r = BenchRequest { dataset: "nope".into(), task: "user-churn".into(), model: "gat".into(), hidden_dim: 64, num_layers: 2, num_heads: 4, dropout: 0.1, epochs: 20, seeds: 1 };
        assert!(validate_bench(&r).is_err());
        r.dataset = "gauss-ecom-small".into();
        r.task = "no-such-task".into();
        assert!(validate_bench(&r).is_err());
    }
}
