//! Utility functions and helpers

use std::path::PathBuf;
use gaussrdl_core::{Error, Result};
use std::collections::HashMap;
use tempfile;
use sysinfo::SystemExt;

/// Logging utilities
pub mod logger;
/// Random number generation utilities
pub mod random;
/// Checkpoint utilities
pub mod checkpoint;

/// Common utility functions
pub fn get_temp_dir() -> Result<PathBuf> {
    tempfile::tempdir()
        .map(|dir| dir.path().to_path_buf())
        .map_err(|e| Error::Io {
            message: "Failed to create temp directory".to_string(),
            source: e,
            backtrace: std::backtrace::Backtrace::capture(),
        })
}

/// Get system information
pub fn get_system_info() -> HashMap<String, String> {
    let mut info = HashMap::new();
    info.insert("cpu_count".to_string(), num_cpus::get().to_string());
    info.insert("memory_total".to_string(), sysinfo::System::new_all().total_memory().to_string());
    info
}