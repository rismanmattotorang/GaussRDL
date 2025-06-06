use std::sync::Arc;
use rayon::prelude::*;
use crate::error::{Result, Error};

/// Parallel processing configuration
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Number of threads
    pub num_threads: usize,
    
    /// Chunk size for parallel operations
    pub chunk_size: usize,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            num_threads: num_cpus::get(),
            chunk_size: 1000,
        }
    }
}

/// Parallel executor
pub struct ParallelExecutor {
    /// Configuration
    config: ParallelConfig,
    
    /// Thread pool
    pool: rayon::ThreadPool,
}

impl ParallelExecutor {
    /// Create a new parallel executor
    pub fn new(config: ParallelConfig) -> Result<Self> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(config.num_threads)
            .build()
            .map_err(|e| Error::other(e.to_string()))?;
            
        Ok(Self { config, pool })
    }
    
    /// Execute a function in parallel
    pub fn execute<F, T>(&self, data: Vec<T>, f: F) -> Result<Vec<T::Output>>
    where
        F: Fn(T) + Send + Sync,
        T: Send,
        T::Output: Send,
    {
        let f = Arc::new(f);
        let chunk_size = self.config.chunk_size;
        
        let results = self.pool.install(|| {
            data.into_par_iter()
                .chunks(chunk_size)
                .flat_map(|chunk| {
                    chunk.into_iter()
                        .map(|item| f(item))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        });
        
        Ok(results)
    }
    
    /// Execute a function in parallel with index
    pub fn execute_indexed<F, T>(&self, data: Vec<T>, f: F) -> Result<Vec<T::Output>>
    where
        F: Fn(usize, T) + Send + Sync,
        T: Send,
        T::Output: Send,
    {
        let f = Arc::new(f);
        let chunk_size = self.config.chunk_size;
        
        let results = self.pool.install(|| {
            data.into_par_iter()
                .enumerate()
                .chunks(chunk_size)
                .flat_map(|chunk| {
                    chunk.into_iter()
                        .map(|(i, item)| f(i, item))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        });
        
        Ok(results)
    }
    
    /// Execute a function in parallel with shared state
    pub fn execute_with_state<F, T, S>(&self, data: Vec<T>, state: S, f: F) -> Result<Vec<T::Output>>
    where
        F: Fn(&S, T) + Send + Sync,
        T: Send,
        T::Output: Send,
        S: Send + Sync,
    {
        let f = Arc::new(f);
        let state = Arc::new(state);
        let chunk_size = self.config.chunk_size;
        
        let results = self.pool.install(|| {
            data.into_par_iter()
                .chunks(chunk_size)
                .flat_map(|chunk| {
                    let state = Arc::clone(&state);
                    chunk.into_iter()
                        .map(|item| f(&state, item))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        });
        
        Ok(results)
    }
    
    /// Execute a function in parallel with mutable state
    pub fn execute_with_mut_state<F, T, S>(&self, data: Vec<T>, state: S, f: F) -> Result<Vec<T::Output>>
    where
        F: Fn(&parking_lot::Mutex<S>, T) + Send + Sync,
        T: Send,
        T::Output: Send,
        S: Send + Sync,
    {
        let f = Arc::new(f);
        let state = Arc::new(parking_lot::Mutex::new(state));
        let chunk_size = self.config.chunk_size;
        
        let results = self.pool.install(|| {
            data.into_par_iter()
                .chunks(chunk_size)
                .flat_map(|chunk| {
                    let state = Arc::clone(&state);
                    chunk.into_iter()
                        .map(|item| f(&state, item))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        });
        
        Ok(results)
    }
    
    /// Get the number of threads
    pub fn num_threads(&self) -> usize {
        self.config.num_threads
    }
    
    /// Get the chunk size
    pub fn chunk_size(&self) -> usize {
        self.config.chunk_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_parallel_execution() {
        let config = ParallelConfig::default();
        let executor = ParallelExecutor::new(config).unwrap();
        
        let data: Vec<i32> = (0..1000).collect();
        let results = executor.execute(data, |x| x * 2).unwrap();
        
        assert_eq!(results.len(), 1000);
        for (i, result) in results.iter().enumerate() {
            assert_eq!(*result, i as i32 * 2);
        }
    }
    
    #[test]
    fn test_parallel_execution_with_index() {
        let config = ParallelConfig::default();
        let executor = ParallelExecutor::new(config).unwrap();
        
        let data: Vec<i32> = (0..1000).collect();
        let results = executor.execute_indexed(data, |i, x| i as i32 + x).unwrap();
        
        assert_eq!(results.len(), 1000);
        for (i, result) in results.iter().enumerate() {
            assert_eq!(*result, i as i32 * 2);
        }
    }
    
    #[test]
    fn test_parallel_execution_with_state() {
        let config = ParallelConfig::default();
        let executor = ParallelExecutor::new(config).unwrap();
        
        let data: Vec<i32> = (0..1000).collect();
        let state = 2;
        let results = executor.execute_with_state(data, state, |s, x| x * s).unwrap();
        
        assert_eq!(results.len(), 1000);
        for (i, result) in results.iter().enumerate() {
            assert_eq!(*result, i as i32 * 2);
        }
    }
    
    #[test]
    fn test_parallel_execution_with_mut_state() {
        let config = ParallelConfig::default();
        let executor = ParallelExecutor::new(config).unwrap();
        
        let data: Vec<i32> = (0..1000).collect();
        let state = vec![0; 1000];
        let results = executor.execute_with_mut_state(data, state, |s, x| {
            let mut state = s.lock();
            state[x as usize] = x;
            x
        }).unwrap();
        
        assert_eq!(results.len(), 1000);
        for (i, result) in results.iter().enumerate() {
            assert_eq!(*result, i as i32);
        }
    }
} 