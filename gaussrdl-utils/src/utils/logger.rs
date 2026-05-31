// src/utils/logger.rs
use tracing::{info, warn, error, debug, trace};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use gaussrdl_core::Result;

/// Initialize logging with default configuration
pub fn init_logging() -> Result<()> {
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into()),
        ))
        .with(tracing_subscriber::fmt::layer())
        .init();
    
    Ok(())
}

/// Initialize logging with custom configuration
pub fn init_logging_with_config(level: &str) -> Result<()> {
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(level))
        .with(tracing_subscriber::fmt::layer())
        .init();
    
    Ok(())
}

/// Log performance metrics
pub fn log_performance_metrics(metrics: &str) {
    info!("Performance metrics: {}", metrics);
}

/// Log error with context
pub fn log_error_with_context(error: &str, context: &str) {
    error!("Error: {} | Context: {}", error, context);
}

/// Log warning with context
pub fn log_warning_with_context(warning: &str, context: &str) {
    warn!("Warning: {} | Context: {}", warning, context);
}

/// Log debug information
pub fn log_debug_info(info: &str) {
    debug!("Debug: {}", info);
}

/// Log trace information
pub fn log_trace_info(info: &str) {
    trace!("Trace: {}", info);
}