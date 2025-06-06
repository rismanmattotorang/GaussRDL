use polars::prelude::*;
use std::path::PathBuf;
use crate::error::Result;
use crate::base::{Database, Table};

/// Configuration for data loading
#[derive(Debug, Clone)]
pub struct DataLoaderConfig {
    pub batch_size: usize,
    pub num_threads: usize,
    pub cache_size: usize,
    pub use_memory_map: bool,
}

impl Default for DataLoaderConfig {
    fn default() -> Self {
        Self {
            batch_size: 10000,
            num_threads: num_cpus::get(),
            cache_size: 1024 * 1024 * 1024, // 1GB
            use_memory_map: true,
        }
    }
}

/// Efficient data loader using polars
pub struct DataLoader {
    config: DataLoaderConfig,
}

impl DataLoader {
    /// Creates a new data loader
    pub fn new(config: DataLoaderConfig) -> Self {
        Self { config }
    }

    /// Loads a parquet file efficiently
    pub async fn load_parquet(&self, path: &PathBuf) -> Result<DataFrame> {
        let mut reader = ParquetReader::new(std::fs::File::open(path)?)
            .with_n_rows(self.config.batch_size)
            .with_parallel(self.config.num_threads)
            .with_cache_size(self.config.cache_size);

        if self.config.use_memory_map {
            reader = reader.with_memory_map(true);
        }

        Ok(reader.finish()?)
    }

    /// Loads a CSV file efficiently
    pub async fn load_csv(&self, path: &PathBuf) -> Result<DataFrame> {
        let mut reader = CsvReader::from_path(path)?
            .with_chunk_size(self.config.batch_size)
            .with_n_threads(Some(self.config.num_threads));

        if self.config.use_memory_map {
            reader = reader.with_memory_map(true);
        }

        Ok(reader.finish()?)
    }

    /// Loads a JSON file efficiently
    pub async fn load_json(&self, path: &PathBuf) -> Result<DataFrame> {
        let mut reader = JsonReader::new(std::fs::File::open(path)?)
            .with_batch_size(self.config.batch_size)
            .with_json_format(JsonFormat::JsonLines);

        if self.config.use_memory_map {
            reader = reader.with_memory_map(true);
        }

        Ok(reader.finish()?)
    }

    /// Converts a polars DataFrame to a RelBench Table
    pub fn to_table(&self, df: DataFrame, config: TableConfig) -> Result<Table> {
        Table::from_polars(df, config)
    }
}

/// Configuration for table conversion
#[derive(Debug, Clone)]
pub struct TableConfig {
    pub pkey_col: String,
    pub time_col: Option<String>,
    pub fkey_cols: Vec<(String, String)>, // (column, referenced table)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_parquet_loading() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.parquet");
        
        // Create test data
        let df = DataFrame::new(vec![
            Series::new("id", vec![1, 2, 3]),
            Series::new("value", vec![10, 20, 30]),
        ]).unwrap();
        
        // Save test data
        let mut writer = ParquetWriter::new(std::fs::File::create(&path).unwrap());
        writer.finish(&df).unwrap();
        
        // Test loading
        let loader = DataLoader::new(DataLoaderConfig::default());
        let loaded_df = loader.load_parquet(&path).await.unwrap();
        
        assert_eq!(loaded_df.shape(), (3, 2));
    }
} 