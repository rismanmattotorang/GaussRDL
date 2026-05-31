//! Async I/O utilities for GaussRDL

use std::path::Path;
use gaussrdl_core::Result;

/// Async IO utilities
pub mod async_io;

/// File utilities
pub struct FileUtils;

impl FileUtils {
    /// Read file as string
    pub async fn read_file_as_string(path: &Path) -> Result<String> {
        let content = tokio::fs::read_to_string(path).await
            .map_err(|e| gaussrdl_core::Error::Io {
                message: format!("Failed to read file: {}", path.display()),
                source: e,
                backtrace: std::backtrace::Backtrace::capture(),
            })?;
        Ok(content)
    }
    
    /// Write string to file
    pub async fn write_string_to_file(path: &Path, content: &str) -> Result<()> {
        tokio::fs::write(path, content).await
            .map_err(|e| gaussrdl_core::Error::Io {
                message: format!("Failed to write file: {}", path.display()),
                source: e,
                backtrace: std::backtrace::Backtrace::capture(),
            })?;
        Ok(())
    }
    
    /// Check if file exists
    pub async fn file_exists(path: &Path) -> bool {
        tokio::fs::metadata(path).await.is_ok()
    }
    
    /// Get file size
    pub async fn get_file_size(path: &Path) -> Result<u64> {
        let metadata = tokio::fs::metadata(path).await
            .map_err(|e| gaussrdl_core::Error::Io {
                message: format!("Failed to get file metadata: {}", path.display()),
                source: e,
                backtrace: std::backtrace::Backtrace::capture(),
            })?;
        Ok(metadata.len())
    }
} 