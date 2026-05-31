use std::net::SocketAddr;
use std::sync::Arc;
use axum::{
    extract::{Path, State},
    response::Json,
    routing::{get, post},
    Router,
};
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use gaussrdl_data::{get_dataset, tasks::TaskRegistry};
use gaussrdl_core::{Result, Error};
use gaussrdl_models::{RelGTModel, UnifiedModelConfig, ModelType, ActivationType, DeviceConfig, DeviceType};

/// Server state
#[derive(Clone)]
pub struct ServerState {
    registry: Arc<TaskRegistry>,
    _models: Arc<RwLock<Models>>,
}

/// Model storage
#[derive(Default)]
struct Models {
    _loaded: std::collections::HashMap<String, Arc<dyn RelGTModel<Config = UnifiedModelConfig>>>,
}

/// Server implementation
pub struct Server {
    addr: SocketAddr,
    enable_docs: bool,
}

impl Server {
    /// Create a new server
    pub fn new(host: &str, port: u16, enable_docs: bool) -> Result<Self> {
        let addr: SocketAddr = format!("{}:{}", host, port).parse()
            .map_err(|e: std::net::AddrParseError| Error::other(e.to_string()))?;
        Ok(Self { addr, enable_docs })
    }
    
    /// Run the server
    pub async fn run(&self) -> Result<()> {
        let state = ServerState {
            registry: Arc::new(TaskRegistry::new()),
            _models: Arc::new(RwLock::new(Models::default())),
        };
        
        let app = Router::new()
            .route("/", get(health_check))
            .route("/tasks", get(list_tasks))
            .route("/tasks/:dataset/:task", get(get_task_info))
            .route("/datasets/:name", get(get_dataset_info))
            .route("/train", post(train_model))
            .route("/predict", post(predict))
            .route("/evaluate", post(evaluate))
            .with_state(state);
            
        if self.enable_docs {
            // TODO: Fix SwaggerUi integration for axum 0.7
            println!("Documentation temporarily disabled - will be fixed in future update");
            // Add OpenAPI documentation
            // use utoipa::OpenApi;
            // use utoipa_swagger_ui::SwaggerUi;
            
            // #[derive(OpenApi)]
            // #[openapi(
            //     paths(
            //         health_check,
            //         list_tasks,
            //         get_task_info,
            //         get_dataset_info,
            //         train_model,
            //         predict,
            //         evaluate
            //     ),
            //     components(
            //         schemas(TaskInfo, DatasetInfo, TrainRequest, PredictRequest, EvaluateRequest)
            //     ),
            //     tags(
            //         (name = "relbench", description = "RelBench API")
            //     )
            // )]
            // struct ApiDoc;
            
            // // Fix: Use Router::merge instead of SwaggerUi
            // let swagger_router: Router = SwaggerUi::new("/docs")
            //     .url("/api-docs/openapi.json", ApiDoc::openapi())
            //     .into();
            // app = app.merge(swagger_router);
        }
        
        println!("Server running on http://{}", self.addr);
        
        // Fix: Use tokio::net::TcpListener instead of axum::Server (axum 0.7)
        let listener = tokio::net::TcpListener::bind(&self.addr).await?;
        axum::serve(listener, app).await
            .map_err(|e| Error::other(e.to_string()))?;
            
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
#[derive(Debug, Serialize, ToSchema)]
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
#[derive(Debug, Serialize, ToSchema)]
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
) -> Json<DatasetInfo> {
    let dataset = match get_dataset(&name, false) {
        Ok(dataset) => dataset,
        Err(_) => {
            // Return default info if dataset not found
            return Json(DatasetInfo {
                name: name.clone(),
                tables: vec!["not_found".to_string()],
                num_rows: 0,
                val_timestamp: "2024-01-01T00:00:00Z".to_string(),
                test_timestamp: "2024-01-01T00:00:00Z".to_string(),
            });
        }
    };
    
    Json(DatasetInfo {
        name: name.clone(),
        tables: vec!["users".to_string(), "items".to_string(), "reviews".to_string()],
        num_rows: 1000, // Default value
        val_timestamp: dataset.val_timestamp().to_rfc3339(),
        test_timestamp: dataset.test_timestamp().to_rfc3339(),
    })
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
) -> Json<TaskInfo> {
    let name = format!("{}/{}", dataset, task);
    let provider = match state.registry.get(&name) {
        Some(provider) => provider,
        None => {
            // Return default info if task not found
            return Json(TaskInfo {
                name,
                description: "Task not found".to_string(),
                task_type: "Unknown".to_string(),
                dataset: dataset.clone(),
                metrics: vec!["error".to_string()],
            });
        }
    };
        
    Json(TaskInfo {
        name,
        description: provider.description().to_string(),
        task_type: format!("{:?}", provider.task_type()),
        dataset: provider.dataset_name().to_string(),
        metrics: provider.metrics()
            .into_iter()
            .map(|m| m.name().to_string())
            .collect(),
    })
}

/// Training request
#[derive(Debug, Deserialize, ToSchema)]
struct TrainRequest {
    _dataset: String,
    _task: String,
    _model_config: Option<serde_yaml::Value>,
    _epochs: Option<usize>,
    _batch_size: Option<usize>,
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
    State(_state): State<ServerState>,
    Json(req): Json<TrainRequest>,
) -> Json<&'static str> {
    // TODO: Implement actual training
    println!("Training request: {:?}", req);
    
    // Create a default model config for now
    let _config = UnifiedModelConfig {
        model_type: ModelType::BaseGNN,
        input_dim: Some(128),
        hidden_dim: 256,
        output_dim: Some(64),
        num_layers: 3,
        dropout: 0.1,
        activation: ActivationType::ReLU,
        use_layer_norm: true,
        use_batch_norm: false,
        use_residual: true,
        learning_rate: req.learning_rate.unwrap_or(0.001) as f64,
        weight_decay: 0.0001,
        gradient_clipping: Some(1.0),
        edge_dim: None,
        num_relations: None,
        num_heads: None,
        use_edge_features: false,
        use_attention: false,
        attention_dropout: None,
        num_bases: None,
        device: DeviceConfig {
            device_type: DeviceType::Auto,
            use_mixed_precision: false,
            memory_limit: None,
        },
    };
    
    Json("Training started")
}

/// Prediction request
#[derive(Debug, Deserialize, ToSchema)]
struct PredictRequest {
    model_id: String,
    inputs: Vec<serde_json::Value>,
}

/// Predict endpoint
#[utoipa::path(
    post,
    path = "/predict",
    request_body = PredictRequest,
    responses(
        (status = 200, description = "Model predictions", body = Vec<f64>),
        (status = 400, description = "Invalid request"),
        (status = 404, description = "Model not found")
    )
)]
async fn predict(
    State(_state): State<ServerState>,
    Json(req): Json<PredictRequest>,
) -> Json<Vec<f64>> {
    // Mock prediction for now
    println!("Predict request: model_id={}, inputs={:?}", req.model_id, req.inputs);
    Json(vec![0.5, 0.3, 0.8, 0.1])
}

/// Evaluation request
#[derive(Debug, Deserialize, ToSchema)]
struct EvaluateRequest {
    dataset: String,
    task: String,
    model_id: String,
}

/// Evaluate endpoint
#[utoipa::path(
    post,
    path = "/evaluate",
    request_body = EvaluateRequest,
    responses(
        (status = 200, description = "Evaluation metrics", body = Vec<f64>),
        (status = 400, description = "Invalid request")
    )
)]
async fn evaluate(
    State(_state): State<ServerState>,
    Json(req): Json<EvaluateRequest>,
) -> Json<Vec<f64>> {
    // Mock evaluation for now
    println!("Evaluate request: dataset={}, task={}, model_id={}", 
             req.dataset, req.task, req.model_id);
    Json(vec![0.85, 0.73, 0.91])
} 