// GaussRDL: Gaussian Relational Deep Learning Toolkit
//
// Unified API surface for all GaussRDL features.
//
// # Features
// - Dataset management, task registry, and evaluation
// - Graph construction, algorithms, and sampling
// - Model configs, training, distributed, and checkpointing
// - Metrics, monitoring, and performance profiling
// - Database connectivity and schema
// - CLI, server, and utility modules

// --- Core ---
pub use gaussrdl_core::*;

// --- Data ---
pub use gaussrdl_data::*;
pub use gaussrdl_data::datasets::*;
pub use gaussrdl_data::data::*;
pub use gaussrdl_data::tasks::*;
pub use gaussrdl_data::task::*;
pub use gaussrdl_data::dataset::*;

// --- Graph ---
pub use gaussrdl_graph::*;
pub use gaussrdl_graph::graph::*;
pub use gaussrdl_graph::algorithms::*;
pub use gaussrdl_graph::compression::*;
pub use gaussrdl_graph::visualization::*;
pub use gaussrdl_graph::monitoring::*;
pub use gaussrdl_graph::parallel::*;

// --- Models ---
pub use gaussrdl_models::*;

// --- Training ---
pub use gaussrdl_training::*;
pub use gaussrdl_training::training::*;

// --- Metrics ---
pub use gaussrdl_metrics::metrics::*;
pub use gaussrdl_metrics::monitoring::*;

// --- Database ---
pub use gaussrdl_database::*;
pub use gaussrdl_database::database::*;

// --- Server ---
pub use gaussrdl_server::*;
pub use gaussrdl_server::server::*;

// --- CLI ---
pub use gaussrdl_cli::cli as cli;

// --- Utils ---
pub use gaussrdl_utils::*;
pub use gaussrdl_utils::utils::*;
pub use gaussrdl_utils::memory::*;
pub use gaussrdl_utils::parallel::*;
pub use gaussrdl_utils::io::*;
pub use gaussrdl_utils::gpu::*;
pub use gaussrdl_utils::distributed::*;
pub use gaussrdl_utils::di::*;

// --- Type Aliases for Backward Compatibility ---
pub use gaussrdl_core::{Result as GaussRDLResult, Error as GaussRDLError}; 
 