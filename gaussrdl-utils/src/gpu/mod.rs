//! GPU utilities for GaussRDL

pub mod memory;

/// GPU configuration
#[derive(Debug, Clone)]
pub struct GpuConfig {
    /// Device to use
    pub device: candle_core::Device,
    /// Memory limit in bytes
    pub memory_limit: usize,
    /// Whether to use mixed precision
    pub use_mixed_precision: bool,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            device: candle_core::Device::Cpu,
            memory_limit: 1024 * 1024 * 1024, // 1GB
            use_mixed_precision: false,
        }
    }
}

/// GPU manager
pub struct GpuManager {
    config: GpuConfig,
}

impl GpuManager {
    /// Create a new GPU manager
    pub fn new(config: GpuConfig) -> Self {
        Self { config }
    }
    
    /// Get the device
    pub fn device(&self) -> &candle_core::Device {
        &self.config.device
    }
    
    /// Check if GPU is available
    pub fn is_gpu_available(&self) -> bool {
        match &self.config.device {
            candle_core::Device::Cpu => false,
            candle_core::Device::Cuda(_) => true,
            candle_core::Device::Metal(_) => true,
        }
    }
} 