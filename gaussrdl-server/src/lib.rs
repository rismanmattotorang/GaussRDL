//! HTTP server and API for GaussRDL

pub mod server;

// Re-export common types for convenience
pub use gaussrdl_core::{Result, Error};
pub use gaussrdl_data::{get_dataset, tasks::{get_task, TaskRegistry}};
pub use gaussrdl_models::{RelGTModel, UnifiedModelConfig, ModelType, ActivationType, DeviceConfig, DeviceType};

// Re-export server types
pub use server::{Server, ServerState};
