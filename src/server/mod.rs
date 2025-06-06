use std::net::SocketAddr;
use std::sync::Arc;
use axum::{
    routing::{get, post},
    Router, Json,
    extract::{State, Path},
};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tower_http::trace::TraceLayer;

use crate::error::Result;
use crate::{get_dataset, get_task};
use crate::tasks::TaskRegistry;

/// Server state
#[derive(Clone)]
pub struct ServerState {
    registry: Arc<TaskRegistry>,
    models: Arc<RwLock<Models>>,
}

/// Model storage
#[derive(Default)]
struct Models {
    loaded: std::collections::HashMap<String, Arc<crate::models::Model>>,
}

/// Server implementation
pub struct Server {
    addr: SocketAddr,
    enable_docs: bool,
}

impl Server {
    /// Create a new server
    pub fn new(host: &str, port: u16, enable_docs: bool) -> Result<Self> {
        let addr = format!("{}:{}", host, port).parse()?;
        Ok(Self { addr, enable_docs })
    }
    
    /// Run the server
    pub async fn run(&self) -> Result<()> {
        let state = ServerState {
            registry: Arc::new(TaskRegistry::new()),
            models: Arc::new(RwLock::new(Models::default())),
        };
        
        let app = Router::new()
            .route("/", get(health_check))
            .route("/tasks", get(list_tasks))
            .route("/tasks/:dataset/:task", get(get_task_info))
            .route("/datasets/:name", get(get_dataset_info))
            .route("/train", post(train_model))
            .route("/predict", post(predict))
            .route("/evaluate", post(evaluate))
            .layer(TraceLayer::new_for_http())
            .with_state(state);
            
        if self.enable_docs {
            // Add OpenAPI documentation
            use utoipa::OpenApi;
            use utoipa_swagger_ui::SwaggerUi;
            
            #[derive(OpenApi)]
            #[openapi(
                paths(
                    health_check,
                    list_tasks,
                    get_task_info,
                    get_dataset_info,
                    train_model,
                    predict,
                    evaluate
                ),
                components(
                    schemas(TaskInfo, DatasetInfo, TrainRequest, PredictRequest, EvaluateRequest)
                ),
                tags(
                    (name = "relbench", description = "RelBench API")
                )
            )]
            struct ApiDoc;
            
            let app = app.merge(SwaggerUi::new("/docs").url("/api-docs/openapi.json", ApiDoc::openapi()));
        }
        
        println!("Server running on http://{}", self.addr);
        axum::Server::bind(&self.addr)
            .serve(app.into_make_service())
            .await?;
            
        Ok(())
    }
}

/// Health check endpoint
#[utoipa::path(
    get,
    path = "/",
    responses(
        (status = 200, description = "Server is healthy", body = String)
    )
)]
async fn health_check() -> &'static str {
    "OK"
}

/// Task information
#[derive(Debug, Serialize)]
struct TaskInfo {
    name: String,
    description: String,
    task_type: String,
    dataset: String,
    metrics: Vec<String>,
}

/// List tasks endpoint
#[utoipa::path(
    get,
    path = "/tasks",
    responses(
        (status = 200, description = "List of available tasks", body = Vec<TaskInfo>)
    )
)]
async fn list_tasks(
    State(state): State<ServerState>
) -> Json<Vec<TaskInfo>> {
    let tasks = state.registry.list();
    let task_infos = tasks.into_iter()
        .filter_map(|name| {
            let provider = state.registry.get(&name)?;
            Some(TaskInfo {
                name: name.clone(),
                description: provider.description().to_string(),
                task_type: format!("{:?}", provider.task_type()),
                dataset: provider.dataset_name().to_string(),
                metrics: provider.metrics()
                    .into_iter()
                    .map(|m| m.name().to_string())
                    .collect(),
            })
        })
        .collect();
    Json(task_infos)
}

/// Dataset information
#[derive(Debug, Serialize)]
struct DatasetInfo {
    name: String,
    tables: Vec<String>,
    num_rows: usize,
    val_timestamp: String,
    test_timestamp: String,
}

/// Get dataset info endpoint
#[utoipa::path(
    get,
    path = "/datasets/{name}",
    responses(
        (status = 200, description = "Dataset information", body = DatasetInfo),
        (status = 404, description = "Dataset not found")
    )
)]
async fn get_dataset_info(
    Path(name): Path<String>
) -> Result<Json<DatasetInfo>> {
    let dataset = get_dataset(&name, false)?;
    let db = dataset.get_db(true)?;
    
    Ok(Json(DatasetInfo {
        name,
        tables: db.table_names(),
        num_rows: db.get_table("review")
            .map(|t| t.len())
            .unwrap_or_default(),
        val_timestamp: dataset.val_timestamp().to_rfc3339(),
        test_timestamp: dataset.test_timestamp().to_rfc3339(),
    }))
}

/// Get task info endpoint
#[utoipa::path(
    get,
    path = "/tasks/{dataset}/{task}",
    responses(
        (status = 200, description = "Task information", body = TaskInfo),
        (status = 404, description = "Task not found")
    )
)]
async fn get_task_info(
    Path((dataset, task)): Path<(String, String)>,
    State(state): State<ServerState>,
) -> Result<Json<TaskInfo>> {
    let name = format!("{}/{}", dataset, task);
    let provider = state.registry.get(&name)
        .ok_or_else(|| crate::error::Error::task(format!("Task {} not found", name)))?;
        
    Ok(Json(TaskInfo {
        name,
        description: provider.description().to_string(),
        task_type: format!("{:?}", provider.task_type()),
        dataset: provider.dataset_name().to_string(),
        metrics: provider.metrics()
            .into_iter()
            .map(|m| m.name().to_string())
            .collect(),
    }))
}

/// Training request
#[derive(Debug, Deserialize)]
struct TrainRequest {
    dataset: String,
    task: String,
    model_config: Option<serde_yaml::Value>,
    epochs: Option<usize>,
    batch_size: Option<usize>,
    learning_rate: Option<f32>,
}

/// Train model endpoint
#[utoipa::path(
    post,
    path = "/train",
    request_body = TrainRequest,
    responses(
        (status = 200, description = "Training started successfully"),
        (status = 400, description = "Invalid request"),
        (status = 404, description = "Task not found")
    )
)]
async fn train_model(
    State(state): State<ServerState>,
    Json(req): Json<TrainRequest>,
) -> Result<&'static str> {
    let task = get_task(&req.dataset, &req.task, true)?;
    
    let model_config = req.model_config.unwrap_or_default();
    
    let train_config = crate::training::TrainConfig {
        epochs: req.epochs.unwrap_or(100),
        batch_size: req.batch_size.unwrap_or(32),
        learning_rate: req.learning_rate.unwrap_or(0.001),
        ..Default::default()
    };
    
    let trainer = crate::training::Trainer::new(task, model_config, train_config)?;
    
    // Start training in background
    tokio::spawn(async move {
        if let Err(e) = trainer.train() {
            eprintln!("Training error: {}", e);
        }
    });
    
    Ok("Training started")
}

/// Prediction request
#[derive(Debug, Deserialize)]
struct PredictRequest {
    model_id: String,
    inputs: Vec<serde_json::Value>,
}

/// Make predictions endpoint
#[utoipa::path(
    post,
    path = "/predict",
    request_body = PredictRequest,
    responses(
        (status = 200, description = "Predictions", body = Vec<f64>),
        (status = 400, description = "Invalid request"),
        (status = 404, description = "Model not found")
    )
)]
async fn predict(
    State(state): State<ServerState>,
    Json(req): Json<PredictRequest>,
) -> Result<Json<Vec<f64>>> {
    let models = state.models.read().await;
    let model = models.loaded.get(&req.model_id)
        .ok_or_else(|| crate::error::Error::model("Model not found"))?;
        
    let predictions = model.predict_json(&req.inputs)?;
    Ok(Json(predictions))
}

/// Evaluation request
#[derive(Debug, Deserialize)]
struct EvaluateRequest {
    dataset: String,
    task: String,
    model_id: String,
}

/// Evaluate model endpoint
#[utoipa::path(
    post,
    path = "/evaluate",
    request_body = EvaluateRequest,
    responses(
        (status = 200, description = "Evaluation metrics", body = Vec<f64>),
        (status = 400, description = "Invalid request"),
        (status = 404, description = "Task or model not found")
    )
)]
async fn evaluate(
    State(state): State<ServerState>,
    Json(req): Json<EvaluateRequest>,
) -> Result<Json<Vec<f64>>> {
    let task = get_task(&req.dataset, &req.task, false)?;
    
    let models = state.models.read().await;
    let model = models.loaded.get(&req.model_id)
        .ok_or_else(|| crate::error::Error::model("Model not found"))?;
        
    let test_table = task.get_test_table()?;
    let predictions = model.predict(&test_table)?;
    let metrics = task.evaluate(&predictions, None)?;
    
    Ok(Json(metrics))
} 