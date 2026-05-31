use candle_core::{DType, Device, Tensor};
use half::{bf16, f16};
use rand::{Rng, SeedableRng, thread_rng};
use rand::rngs::{StdRng, ThreadRng};
use rand::seq::SliceRandom;
use gaussrdl_core::{Error, Result};
use rand_distr::{Distribution, Normal, Uniform};

pub fn random_normal(mean: f32, std: f32, shape: &[usize], dtype: DType, device: &Device) -> Result<Tensor> {
    let total_elements = shape.iter().product::<usize>();
    let mut rng = rand::thread_rng();
    let dist = Normal::new(mean, std).map_err(|e| Error::other(e.to_string()))?;
    
    match dtype {
        DType::F32 => {
            let data: Vec<f32> = (0..total_elements)
                .map(|_| dist.sample(&mut rng))
                .collect();
            Tensor::from_vec(data, shape, device).map_err(Error::from)
        },
        DType::F16 => {
            let data: Vec<f16> = (0..total_elements)
                .map(|_| f16::from_f32(dist.sample(&mut rng)))
                .collect();
            Tensor::from_vec(data, shape, device).map_err(Error::from)
        },
        DType::BF16 => {
            let data: Vec<bf16> = (0..total_elements)
                .map(|_| bf16::from_f32(dist.sample(&mut rng)))
                .collect();
            Tensor::from_vec(data, shape, device).map_err(Error::from)
        },
        _ => Err(Error::other(format!("Unsupported dtype for random_normal: {:?}", dtype))),
    }
}

pub fn random_uniform(min: f32, max: f32, shape: &[usize], dtype: DType, device: &Device) -> Result<Tensor> {
    let total_elements = shape.iter().product::<usize>();
    let mut rng = rand::thread_rng();
    let dist = Uniform::new(min, max);
    
    match dtype {
        DType::F32 => {
            let data: Vec<f32> = (0..total_elements)
                .map(|_| dist.sample(&mut rng))
                .collect();
            Tensor::from_vec(data, shape, device).map_err(Error::from)
        },
        DType::F16 => {
            let data: Vec<f16> = (0..total_elements)
                .map(|_| f16::from_f32(dist.sample(&mut rng)))
                .collect();
            Tensor::from_vec(data, shape, device).map_err(Error::from)
        },
        DType::BF16 => {
            let data: Vec<bf16> = (0..total_elements)
                .map(|_| bf16::from_f32(dist.sample(&mut rng)))
                .collect();
            Tensor::from_vec(data, shape, device).map_err(Error::from)
        },
        _ => Err(Error::other(format!("Unsupported dtype for random_uniform: {:?}", dtype))),
    }
}

pub fn rand_bf16(min: f32, max: f32, size: &[usize]) -> Result<Tensor> {
    let mut rng = rand::thread_rng();
    let dist = rand::distributions::Uniform::new(min, max);
    let data: Vec<bf16> = (0..size.iter().product())
        .map(|_| bf16::from_f32(dist.sample(&mut rng)))
        .collect();
    Tensor::from_vec(data, size, &Device::Cpu).map_err(Error::from)
}

pub fn rand_f16(min: f32, max: f32, size: &[usize]) -> Result<Tensor> {
    let mut rng = rand::thread_rng();
    let dist = rand::distributions::Uniform::new(min, max);
    let data: Vec<f16> = (0..size.iter().product())
        .map(|_| f16::from_f32(dist.sample(&mut rng)))
        .collect();
    Tensor::from_vec(data, size, &Device::Cpu).map_err(Error::from)
}

pub fn randn_bf16(mean: f32, std: f32, size: &[usize]) -> Result<Tensor> {
    let mut rng = rand::thread_rng();
    let dist = rand_distr::Normal::new(mean, std).unwrap();
    let data: Vec<bf16> = (0..size.iter().product())
        .map(|_| bf16::from_f32(dist.sample(&mut rng)))
        .collect();
    Tensor::from_vec(data, size, &Device::Cpu).map_err(Error::from)
}

pub fn randn_f16(mean: f32, std: f32, size: &[usize]) -> Result<Tensor> {
    let mut rng = rand::thread_rng();
    let dist = rand_distr::Normal::new(mean, std).unwrap();
    let data: Vec<f16> = (0..size.iter().product())
        .map(|_| f16::from_f32(dist.sample(&mut rng)))
        .collect();
    Tensor::from_vec(data, size, &Device::Cpu).map_err(Error::from)
} 

/// Generate a random seed
pub fn generate_seed() -> u64 {
    thread_rng().gen()
}

/// Create a seeded random number generator
pub fn create_seeded_rng(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

/// Generate random bytes
pub fn generate_random_bytes(length: usize) -> Result<Vec<u8>> {
    let mut rng = thread_rng();
    let bytes: Vec<u8> = (0..length).map(|_| rng.gen()).collect();
    Ok(bytes)
}

/// Generate random string
pub fn generate_random_string(length: usize) -> Result<String> {
    const CHARSET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ\
                            abcdefghijklmnopqrstuvwxyz\
                            0123456789)(*&^%$#@!~";
    
    let mut rng = thread_rng();
    let string: String = (0..length)
        .map(|_| {
            let idx = rng.gen_range(0..CHARSET.len());
            CHARSET[idx] as char
        })
        .collect();
    
    Ok(string)
}

/// Generate random integer in range
pub fn random_int(min: i64, max: i64) -> Result<i64> {
    if min >= max {
        return Err(Error::validation("Invalid range: min must be less than max"));
    }
    
    let mut rng = thread_rng();
    Ok(rng.gen_range(min..=max))
}

/// Generate random float in range
pub fn random_float(min: f64, max: f64) -> Result<f64> {
    if min >= max {
        return Err(Error::validation("Invalid range: min must be less than max"));
    }
    
    let mut rng = thread_rng();
    Ok(rng.gen_range(min..=max))
}

/// Shuffle a vector in place
pub fn shuffle_vec<T>(rng: &mut ThreadRng, vec: &mut Vec<T>) {
    vec.shuffle(rng);
}

/// Shuffle indices
pub fn shuffle_indices(rng: &mut ThreadRng, indices: &mut Vec<usize>) {
    indices.shuffle(rng);
} 