//! Web UI for GaussRDL: configure a model, launch training, and watch the
//! loss / validation curves and inference KPIs from the browser.
//!
//! Run: `cargo run -p gaussrdl-web` then open http://127.0.0.1:8080

use axum::{routing::get, routing::post, Json, Router};
use gaussrdl_rdl::models::{ModelConfig, ModelKind};
use gaussrdl_rdl::synthetic::SyntheticConfig;
use gaussrdl_rdl::{run_experiment, DataSource, ExperimentConfig, LrSchedule, TaskType};
use serde::Deserialize;

/// Training request sent by the browser form.
#[derive(Debug, Clone, Deserialize)]
pub struct TrainRequest {
    pub model: String,
    pub task: String,
    #[serde(default = "d_epochs")]
    pub epochs: usize,
    #[serde(default = "d_hidden")]
    pub hidden_dim: usize,
    #[serde(default = "d_layers")]
    pub num_layers: usize,
    #[serde(default = "d_heads")]
    pub num_heads: usize,
    #[serde(default = "d_lr")]
    pub lr: f64,
    #[serde(default = "d_dropout")]
    pub dropout: f32,
    #[serde(default = "d_users")]
    pub num_users: usize,
    #[serde(default)]
    pub early_stopping_patience: usize,
}

fn d_epochs() -> usize { 50 }
fn d_hidden() -> usize { 64 }
fn d_layers() -> usize { 2 }
fn d_heads() -> usize { 4 }
fn d_lr() -> f64 { 0.01 }
fn d_dropout() -> f32 { 0.1 }
fn d_users() -> usize { 300 }

/// Map a request to a validated experiment config (pure; unit-tested).
pub fn build_config(req: &TrainRequest) -> Result<ExperimentConfig, String> {
    let model = ModelKind::parse(&req.model).map_err(|e| e.to_string())?;
    let task = TaskType::parse(&req.task).ok_or_else(|| format!("unknown task `{}`", req.task))?;
    if req.hidden_dim % req.num_heads != 0 {
        return Err(format!(
            "hidden_dim ({}) must be divisible by num_heads ({})",
            req.hidden_dim, req.num_heads
        ));
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
        data: DataSource::Synthetic(SyntheticConfig {
            num_users: req.num_users.clamp(50, 5000),
            ..Default::default()
        }),
        ..Default::default()
    })
}

async fn train(Json(req): Json<TrainRequest>) -> Json<serde_json::Value> {
    let cfg = match build_config(&req) {
        Ok(c) => c,
        Err(e) => return Json(serde_json::json!({ "error": e })),
    };
    let result = tokio::task::spawn_blocking(move || run_experiment(&cfg)).await;
    match result {
        Ok(Ok(r)) => Json(serde_json::to_value(&r).unwrap()),
        Ok(Err(e)) => Json(serde_json::json!({ "error": e.to_string() })),
        Err(e) => Json(serde_json::json!({ "error": format!("join error: {e}") })),
    }
}

async fn models() -> Json<serde_json::Value> {
    let names: Vec<&str> = ModelKind::all().iter().map(|m| m.name()).collect();
    Json(serde_json::json!({ "models": names, "tasks": ["churn", "ltv"] }))
}

async fn index() -> axum::response::Html<&'static str> {
    axum::response::Html(INDEX_HTML)
}

#[tokio::main]
async fn main() {
    let app = Router::new()
        .route("/", get(index))
        .route("/api/models", get(models))
        .route("/api/train", post(train));
    let port: u16 = std::env::var("PORT").ok().and_then(|p| p.parse().ok()).unwrap_or(8080);
    let addr = format!("127.0.0.1:{port}");
    println!("GaussRDL web UI on http://{addr}");
    let listener = tokio::net::TcpListener::bind(&addr).await.expect("bind");
    axum::serve(listener, app).await.expect("serve");
}

const INDEX_HTML: &str = r#"<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>GaussRDL — Gaussian Technologies</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
 :root{--bg:#0b1020;--card:#151c33;--fg:#e8ecf6;--mut:#93a0c0;--acc:#5b8cff;--ok:#37d39b}
 *{box-sizing:border-box} body{margin:0;font-family:ui-sans-serif,system-ui,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--fg)}
 header{padding:18px 26px;border-bottom:1px solid #222c4d;display:flex;align-items:baseline;gap:12px}
 h1{font-size:20px;margin:0} .tag{color:var(--mut);font-size:13px}
 .wrap{display:grid;grid-template-columns:330px 1fr;gap:18px;padding:18px 26px}
 .card{background:var(--card);border:1px solid #222c4d;border-radius:12px;padding:16px}
 label{display:block;font-size:12px;color:var(--mut);margin:10px 0 4px}
 select,input{width:100%;padding:8px;border-radius:8px;border:1px solid #2b3760;background:#0e1430;color:var(--fg)}
 button{margin-top:16px;width:100%;padding:10px;border:0;border-radius:8px;background:var(--acc);color:#fff;font-weight:600;cursor:pointer}
 button:disabled{opacity:.5;cursor:wait}
 canvas{width:100%;height:230px;background:#0e1430;border-radius:8px;border:1px solid #2b3760}
 .kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:14px}
 .kpi{background:#0e1430;border:1px solid #2b3760;border-radius:8px;padding:10px}
 .kpi .v{font-size:20px;font-weight:700;color:var(--ok)} .kpi .k{font-size:11px;color:var(--mut)}
 .row{display:grid;grid-template-columns:1fr 1fr;gap:10px}
 #status{font-size:13px;color:var(--mut);margin-top:8px;min-height:18px}
</style></head>
<body>
<header><h1>GaussRDL</h1><span class="tag">Relational Deep Learning · Gaussian Technologies</span></header>
<div class="wrap">
 <div class="card">
  <label>Model</label>
  <select id="model"><option>HeteroSAGE</option><option>RGCN</option><option>GAT</option><option selected>RelGT</option></select>
  <label>Task</label>
  <select id="task"><option value="churn">User churn (classification)</option><option value="ltv">User LTV (regression)</option></select>
  <div class="row">
   <div><label>Hidden dim</label><input id="hidden_dim" type="number" value="64" step="8"></div>
   <div><label>Heads</label><input id="num_heads" type="number" value="4"></div>
  </div>
  <div class="row">
   <div><label>Layers</label><input id="num_layers" type="number" value="2"></div>
   <div><label>Dropout</label><input id="dropout" type="number" value="0.1" step="0.05"></div>
  </div>
  <div class="row">
   <div><label>Epochs</label><input id="epochs" type="number" value="50"></div>
   <div><label>LR</label><input id="lr" type="number" value="0.01" step="0.001"></div>
  </div>
  <div class="row">
   <div><label># Users</label><input id="num_users" type="number" value="300" step="50"></div>
   <div><label>Early stop</label><input id="early_stopping_patience" type="number" value="0"></div>
  </div>
  <button id="go" onclick="train()">Train &amp; monitor</button>
  <div id="status"></div>
 </div>
 <div>
  <div class="card"><b>Training curves</b> &nbsp;<span class="tag">loss (blue) · val metric (green)</span>
   <canvas id="chart" width="900" height="230"></canvas></div>
  <div class="card" style="margin-top:18px"><b>Final metrics &amp; inference KPIs</b>
   <div class="kpis" id="kpis"></div></div>
 </div>
</div>
<script>
function num(id){return parseFloat(document.getElementById(id).value)}
function int(id){return parseInt(document.getElementById(id).value)}
function drawLine(ctx,W,H,arr,color,min,max,pad){
  if(arr.length<2)return; ctx.strokeStyle=color; ctx.lineWidth=2; ctx.beginPath();
  for(let i=0;i<arr.length;i++){const x=pad+(W-2*pad)*i/(arr.length-1);
    const y=H-pad-(H-2*pad)*(arr[i]-min)/((max-min)||1); i?ctx.lineTo(x,y):ctx.moveTo(x,y);} ctx.stroke();
}
function chart(hist){
  const c=document.getElementById('chart'),ctx=c.getContext('2d'),W=c.width,H=c.height,pad=24;
  ctx.clearRect(0,0,W,H);
  const loss=hist.map(e=>e.train_loss), val=hist.map(e=>e.val_metric);
  const lmin=Math.min(...loss),lmax=Math.max(...loss);
  const vmin=Math.min(...val),vmax=Math.max(...val);
  drawLine(ctx,W,H,loss,'#5b8cff',lmin,lmax,pad);
  drawLine(ctx,W,H,val,'#37d39b',vmin,vmax,pad);
}
function kpi(k,v){return `<div class="kpi"><div class="v">${v}</div><div class="k">${k}</div></div>`}
async function train(){
  const btn=document.getElementById('go'); btn.disabled=true;
  document.getElementById('status').textContent='Training…';
  const body={model:document.getElementById('model').value,task:document.getElementById('task').value,
    hidden_dim:int('hidden_dim'),num_heads:int('num_heads'),num_layers:int('num_layers'),
    dropout:num('dropout'),epochs:int('epochs'),lr:num('lr'),num_users:int('num_users'),
    early_stopping_patience:int('early_stopping_patience')};
  try{
    const r=await fetch('/api/train',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const d=await r.json();
    if(d.error){document.getElementById('status').textContent='Error: '+d.error; btn.disabled=false; return;}
    chart(d.history.epochs);
    const inf=d.inference; let html='';
    if(d.task==='BinaryClassification'){
      html+=kpi('val ROC-AUC',d.val.auroc.toFixed(4));
      html+=kpi('test ROC-AUC',d.test.auroc.toFixed(4));
      html+=kpi('accuracy',d.test.accuracy.toFixed(4));
      html+=kpi('Brier',inf.brier.toFixed(4));
    }else{
      html+=kpi('val MAE',d.val.mae.toFixed(2));
      html+=kpi('test MAE',d.test.mae.toFixed(2));
      html+=kpi('RMSE',d.test.rmse.toFixed(2));
      html+=kpi('R²',inf.r2.toFixed(3));
    }
    html+=kpi('throughput/s',Math.round(inf.throughput_per_s));
    html+=kpi('latency ms',inf.latency_ms.toFixed(1));
    html+=kpi('epochs',d.history.epochs.length);
    html+=kpi('best epoch',d.history.best_epoch);
    document.getElementById('kpis').innerHTML=html;
    document.getElementById('status').textContent=
      `done · ${d.num_nodes} nodes / ${d.num_edges} edges / ${d.num_relations} relations`;
  }catch(e){document.getElementById('status').textContent='Error: '+e}
  btn.disabled=false;
}
</script>
</body></html>"#;

#[cfg(test)]
mod tests {
    use super::*;

    fn req() -> TrainRequest {
        TrainRequest {
            model: "relgt".into(),
            task: "churn".into(),
            epochs: 10,
            hidden_dim: 32,
            num_layers: 2,
            num_heads: 4,
            lr: 0.01,
            dropout: 0.1,
            num_users: 100,
            early_stopping_patience: 0,
        }
    }

    #[test]
    fn build_config_ok() {
        let c = build_config(&req()).unwrap();
        assert_eq!(c.model, ModelKind::RelGt);
        assert_eq!(c.task, TaskType::BinaryClassification);
        assert_eq!(c.model_cfg.hidden_dim, 32);
    }

    #[test]
    fn build_config_rejects_indivisible_heads() {
        let mut r = req();
        r.hidden_dim = 30; // not divisible by 4
        assert!(build_config(&r).is_err());
    }

    #[test]
    fn build_config_rejects_unknown_model() {
        let mut r = req();
        r.model = "nope".into();
        assert!(build_config(&r).is_err());
    }
}
