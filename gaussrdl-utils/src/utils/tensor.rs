use candle_core::{DType, Result as CandleResult, Tensor, Device};
use half::{bf16, f16};

pub trait HalfPrecisionExt {
    fn to_half_precision(&self, dtype: DType) -> CandleResult<Tensor>;
    fn from_half_precision(&self) -> CandleResult<Tensor>;
}

impl HalfPrecisionExt for Tensor {
    fn to_half_precision(&self, dtype: DType) -> CandleResult<Tensor> {
        match dtype {
            DType::BF16 | DType::F16 => {
                // Convert to f32 first
                let f32_tensor = self.to_dtype(DType::F32)?;
                // Then convert to target dtype
                f32_tensor.to_dtype(dtype)
            }
            _ => Ok(self.clone()),
        }
    }
    
    fn from_half_precision(&self) -> CandleResult<Tensor> {
        match self.dtype() {
            DType::BF16 | DType::F16 => self.to_dtype(DType::F32),
            _ => Ok(self.clone()),
        }
    }
}

pub fn random_uniform(
    min: f32,
    max: f32,
    shape: &[usize],
    dtype: DType,
    device: &Device,
) -> CandleResult<Tensor> {
    // Generate random values in f32
    let f32_tensor = Tensor::rand(min, max, shape, device)?;
    
    // Convert to target dtype if needed
    f32_tensor.to_half_precision(dtype)
}

pub fn random_normal(
    mean: f32,
    std: f32,
    shape: &[usize],
    dtype: DType,
    device: &Device,
) -> CandleResult<Tensor> {
    // Generate random values in f32
    let f32_tensor = Tensor::randn(mean, std, shape, device)?;
    
    // Convert to target dtype if needed
    f32_tensor.to_half_precision(dtype)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_half_precision_conversion() {
        let device = Device::Cpu;
        let tensor = Tensor::new(&[1.0f32, 2.0, 3.0], &device).unwrap();
        
        // Test BF16 conversion
        let bf16_tensor = tensor.to_half_precision(DType::BF16).unwrap();
        assert_eq!(bf16_tensor.dtype(), DType::BF16);
        
        // Test F16 conversion
        let f16_tensor = tensor.to_half_precision(DType::F16).unwrap();
        assert_eq!(f16_tensor.dtype(), DType::F16);
        
        // Test back conversion
        let f32_tensor = bf16_tensor.from_half_precision().unwrap();
        assert_eq!(f32_tensor.dtype(), DType::F32);
    }
    
    #[test]
    fn test_random_generation() {
        let device = Device::Cpu;
        let shape = &[2, 3];
        
        // Test uniform random generation
        let tensor = random_uniform(-1.0, 1.0, shape, DType::BF16, &device).unwrap();
        assert_eq!(tensor.shape(), shape);
        assert_eq!(tensor.dtype(), DType::BF16);
        
        // Test normal random generation
        let tensor = random_normal(0.0, 1.0, shape, DType::F16, &device).unwrap();
        assert_eq!(tensor.shape(), shape);
        assert_eq!(tensor.dtype(), DType::F16);
    }
} 