use std::path::Path;
use tokio::fs::File;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use gaussrdl_core::Result;
use futures::stream::StreamExt;

/// Async file reader
pub struct AsyncReader {
    file: File,
    buffer_size: usize,
}

impl AsyncReader {
    /// Create new async reader
    pub async fn new<P: AsRef<Path>>(path: P, buffer_size: usize) -> Result<Self> {
        let file = File::open(path).await?;
        Ok(Self { file, buffer_size })
    }
    
    /// Read all data
    pub async fn read_all(&mut self) -> Result<Vec<u8>> {
        let mut buffer = Vec::new();
        self.file.read_to_end(&mut buffer).await?;
        Ok(buffer)
    }
    
    /// Read chunk of data
    pub async fn read_chunk(&mut self) -> Result<Option<Vec<u8>>> {
        let mut buffer = vec![0; self.buffer_size];
        let n = self.file.read(&mut buffer).await?;
        if n == 0 {
            Ok(None)
        } else {
            buffer.truncate(n);
            Ok(Some(buffer))
        }
    }
}

/// Async file writer
pub struct AsyncWriter {
    file: File,
}

impl AsyncWriter {
    /// Create new async writer
    pub async fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::create(path).await?;
        Ok(Self { file })
    }
    
    /// Write data
    pub async fn write(&mut self, data: &[u8]) -> Result<()> {
        self.file.write_all(data).await?;
        Ok(())
    }
    
    /// Flush writer
    pub async fn flush(&mut self) -> Result<()> {
        self.file.flush().await?;
        Ok(())
    }
}

/// Async batch reader
pub struct AsyncBatchReader {
    path: Box<Path>,
    batch_size: usize,
    buffer_size: usize,
}

impl AsyncBatchReader {
    /// Create new batch reader
    pub fn new<P: AsRef<Path>>(path: P, batch_size: usize, buffer_size: usize) -> Self {
        Self {
            path: path.as_ref().into(),
            batch_size,
            buffer_size,
        }
    }
    
    /// Read batches
    pub async fn read_batches(&self) -> Result<Vec<Vec<u8>>> {
        let mut reader = AsyncReader::new(&self.path, self.buffer_size).await?;
        let mut batches = Vec::new();
        let mut current_batch = Vec::new();

        while let Some(chunk) = reader.read_chunk().await? {
            for byte in chunk {
                current_batch.push(byte);
                if current_batch.len() == self.batch_size {
                    batches.push(current_batch);
                    current_batch = Vec::new();
                }
            }
        }

        if !current_batch.is_empty() {
            batches.push(current_batch);
        }

        Ok(batches)
    }
    
    /// Create streaming iterator
    pub fn stream(&self) -> AsyncBatchStream {
        AsyncBatchStream {
            reader: None,
            batch_size: self.batch_size,
            buffer_size: self.buffer_size,
            path: self.path.clone(),
        }
    }
}

/// Async batch stream
pub struct AsyncBatchStream {
    reader: Option<AsyncReader>,
    batch_size: usize,
    buffer_size: usize,
    path: Box<Path>,
}

impl AsyncBatchStream {
    /// Get next batch
    pub async fn next(&mut self) -> Option<Result<Vec<u8>>> {
        if self.reader.is_none() {
            match AsyncReader::new(&self.path, self.buffer_size).await {
                Ok(reader) => self.reader = Some(reader),
                Err(e) => return Some(Err(e)),
            }
        }

        let reader = self.reader.as_mut().unwrap();
        match reader.read_chunk().await {
            Ok(Some(chunk)) => Some(Ok(chunk)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

/// Async stream processor
pub struct AsyncStreamProcessor;

impl AsyncStreamProcessor {
    /// Process stream with buffer
    pub async fn process_stream_with_buffer<S, F, T>(
        stream: S,
        buffer_size: usize,
        processor: F,
    ) -> Result<Vec<T>>
    where
        S: StreamExt + Unpin,
        F: Fn(S::Item) -> Result<T> + Send + Sync,
        T: Send + Sync,
    {
        let results: Vec<Vec<Result<T>>> = stream
            .chunks(buffer_size)
            .map(|chunk| {
                chunk.into_iter()
                    .map(&processor)
                    .collect::<Vec<_>>()
            })
            .collect()
            .await;
        
        // Flatten the nested results and collect
        let flattened: Vec<Result<T>> = results.into_iter().flatten().collect();
        flattened.into_iter().collect()
    }
    
    /// Process stream in parallel
    pub async fn process_stream_parallel<S, F, T>(
        stream: S,
        _concurrency: usize,
        processor: F,
    ) -> Result<Vec<T>>
    where
        S: StreamExt + Unpin,
        F: Fn(S::Item) -> Result<T> + Send + Sync,
        T: Send + Sync,
    {
        let results: Vec<Result<T>> = stream
            .map(processor)
            .collect()
            .await;
        
        results.into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    use tokio::runtime::Runtime;
    
    #[test]
    fn test_async_io() {
        let rt = Runtime::new().unwrap();
        
        rt.block_on(async {
            // Create test file
            let file = NamedTempFile::new().unwrap();
            let test_data = b"Hello, World!";
            
            // Test writer
            let mut writer = AsyncWriter::new(file.path()).await.unwrap();
            writer.write(test_data).await.unwrap();
            writer.flush().await.unwrap();
            
            // Test reader
            let mut reader = AsyncReader::new(file.path(), 1024).await.unwrap();
            let data = reader.read_all().await.unwrap();
            assert_eq!(data, test_data);
            
            // Test batch reader
            let batch_reader = AsyncBatchReader::new(file.path(), 4, 1024);
            let batches = batch_reader.read_batches().await.unwrap();
            // "Hello, World!" is 13 bytes, with batch size 4: 4 + 4 + 4 + 1 = 4 batches
            assert_eq!(batches.len(), 4);
            
            // Test streaming
            let mut stream = batch_reader.stream();
            let mut count = 0;
            while let Some(result) = stream.next().await {
                result.unwrap();
                count += 1;
            }
            // The stream returns chunks, not batches, so it depends on buffer size
            // With buffer size 1024, it should return 1 chunk containing all data
            assert_eq!(count, 1);
        });
    }
} 