// src/utils/logger.rs
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};
use crate::Result;

pub fn init_logging(log_level: &str) -> Result<()> {
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(format!("gaussrelgt={}", log_level)));
    
    tracing_subscriber::registry()
        .with(env_filter)
        .with(fmt::layer().pretty())
        .init();
    
    Ok(())
}