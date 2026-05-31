use polars::prelude::*;
use std::path::PathBuf;
use std::collections::HashMap;
use candle_core::{Device, Tensor};
use serde::{Serialize, Deserialize};
use crate::error::Result;
use crate::base::{Database, Table};
use crate::graph::*;
use crate::models::ModelInput;

/// Enhanced configuration for data loading with graph learning support
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataLoaderConfig {
    pub batch_size: usize,
    pub num_threads: usize,
    pub cache_size: usize,
    pub use_memory_map: bool,
    pub enable_compression: bool,
    pub chunk_size: Option<usize>,
    pub max_memory_usage_mb: Option<usize>,
    pub graph_config: GraphDataConfig,
}

/// Configuration for graph data loading
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphDataConfig {
    pub node_sampling_strategy: NodeSamplingStrategy,
    pub edge_sampling_strategy: EdgeSamplingStrategy,
    pub max_nodes_per_graph: usize,
    pub max_edges_per_graph: usize,
    pub preserve_node_order: bool,
    pub include_node_features: bool,
    pub include_edge_features: bool,
    pub temporal_encoding: bool,
}

/// Node sampling strategies for graph construction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeSamplingStrategy {
    All,
    Random { fraction: f64 },
    TopK { k: usize, metric: String },
    Temporal { window_hours: u64 },
    Stratified { by_type: bool },
}

/// Edge sampling strategies for graph construction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EdgeSamplingStrategy {
    All,
    Random { fraction: f64 },
    WeightBased { threshold: f64 },
    Temporal { window_hours: u64 },
    KNearest { k: usize },
}

impl Default for DataLoaderConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            num_threads: num_cpus::get(),
            cache_size: 1024 * 1024 * 1024, // 1GB
            use_memory_map: true,
            enable_compression: true,
            chunk_size: Some(10000),
            max_memory_usage_mb: Some(4096), // 4GB
            graph_config: GraphDataConfig::default(),
        }
    }
}

impl Default for GraphDataConfig {
    fn default() -> Self {
        Self {
            node_sampling_strategy: NodeSamplingStrategy::All,
            edge_sampling_strategy: EdgeSamplingStrategy::All,
            max_nodes_per_graph: 10000,
            max_edges_per_graph: 50000,
            preserve_node_order: true,
            include_node_features: true,
            include_edge_features: true,
            temporal_encoding: true,
        }
    }
}

/// Enhanced data loader for graph learning with comprehensive functionality
pub struct GraphDataLoader {
    config: DataLoaderConfig,
    device: Device,
    schema_cache: HashMap<String, GraphSchema>,
    statistics: LoaderStatistics,
}

/// Data loading statistics for monitoring and optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoaderStatistics {
    pub total_files_loaded: u64,
    pub total_rows_processed: u64,
    pub total_graphs_created: u64,
    pub avg_load_time_ms: f64,
    pub cache_hit_rate: f64,
    pub memory_usage_mb: f64,
    pub error_count: u64,
}

impl Default for LoaderStatistics {
    fn default() -> Self {
        Self {
            total_files_loaded: 0,
            total_rows_processed: 0,
            total_graphs_created: 0,
            avg_load_time_ms: 0.0,
            cache_hit_rate: 0.0,
            memory_usage_mb: 0.0,
            error_count: 0,
        }
    }
}

impl GraphDataLoader {
    /// Creates a new enhanced data loader
    pub fn new(config: DataLoaderConfig, device: Device) -> Self {
        Self { 
            config,
            device,
            schema_cache: HashMap::new(),
            statistics: LoaderStatistics::default(),
        }
    }

    /// Loads a parquet file efficiently with memory management
    pub async fn load_parquet(&mut self, path: &PathBuf) -> Result<DataFrame> {
        let start_time = std::time::Instant::now();
        
        let mut reader = LazyFrame::scan_parquet(path, ScanArgsParquet::default())?;
        
        // Apply data sampling if configured
        reader = self.apply_data_sampling(reader)?;
        
        let df = reader.collect()?;
        
        // Update statistics
        self.update_load_statistics(df.height(), start_time.elapsed().as_millis() as f64);
        
        Ok(df)
    }

    /// Loads a CSV file efficiently with comprehensive parsing options
    pub async fn load_csv(&mut self, path: &PathBuf) -> Result<DataFrame> {
        let start_time = std::time::Instant::now();
        
        let mut reader = LazyFrame::scan_csv(path, ScanArgsCSV::default())?;
        
        reader = self.apply_data_sampling(reader)?;
        let df = reader.collect()?;
        
        self.update_load_statistics(df.height(), start_time.elapsed().as_millis() as f64);
        
        Ok(df)
    }

    /// Loads multiple data sources and creates a unified database
    pub async fn load_database(&mut self, sources: &[DataSource]) -> Result<Database> {
        let mut tables = HashMap::new();
        
        for source in sources {
            let df = match &source.format {
                DataFormat::Parquet => self.load_parquet(&source.path).await?,
                DataFormat::CSV => self.load_csv(&source.path).await?,
                DataFormat::JSON => self.load_json(&source.path).await?,
            };
            
            let table = self.create_table_from_dataframe(df, &source.config)?;
            tables.insert(source.name.clone(), table);
        }
        
        Ok(Database::new(tables))
    }

    /// Creates graph batches from database for model training
    pub async fn create_graph_batches(
        &mut self, 
        database: &Database,
        batch_size: usize,
    ) -> Result<Vec<GraphBatch>> {
        let mut batches = Vec::new();
        let mut current_batch_samples = Vec::new();
        
        // Extract relational data and create subgraph samples
        let subgraph_samples = self.extract_subgraph_samples(database).await?;
        
        for sample in subgraph_samples {
            current_batch_samples.push(sample);
            
            if current_batch_samples.len() >= batch_size {
                let batch = GraphBatch::new(current_batch_samples, &self.device)?;
                batches.push(batch);
                current_batch_samples = Vec::new();
                self.statistics.total_graphs_created += 1;
            }
        }
        
        // Handle remaining samples
        if !current_batch_samples.is_empty() {
            let batch = GraphBatch::new(current_batch_samples, &self.device)?;
            batches.push(batch);
            self.statistics.total_graphs_created += 1;
        }
        
        Ok(batches)
    }

    /// Creates model inputs directly from database tables
    pub async fn create_model_inputs(
        &mut self,
        database: &Database,
        target_table: &str,
        feature_columns: &[String],
    ) -> Result<Vec<ModelInput>> {
        let mut inputs = Vec::new();
        
        let table = database.get_table(target_table)
            .ok_or_else(|| crate::error::Error::data(&format!("Table {} not found", target_table)))?;
        
        // Extract features and create tensors
        let node_features = self.extract_node_features(table, feature_columns)?;
        let edge_indices = self.extract_edge_indices(database, target_table)?;
        
        // Create model input
        let input = ModelInput::homogeneous(node_features, edge_indices);
        inputs.push(input);
        
        Ok(inputs)
    }

    /// Loads a JSON file efficiently
    pub async fn load_json(&mut self, path: &PathBuf) -> Result<DataFrame> {
        let start_time = std::time::Instant::now();
        
        let mut reader = LazyFrame::scan_ndjson(path, ScanArgsNdJson::default())?;
        
        if let Some(chunk_size) = self.config.chunk_size {
            reader = reader.with_streaming(true);
        }
        
        reader = self.apply_data_sampling(reader)?;
        let df = reader.collect()?;
        
        self.update_load_statistics(df.height(), start_time.elapsed().as_millis() as f64);
        
        Ok(df)
    }

    /// Converts a polars DataFrame to a RelBench Table with enhanced configuration
    pub fn create_table_from_dataframe(&self, df: DataFrame, config: &TableConfig) -> Result<Table> {
        Table::from_polars_enhanced(df, config.clone())
    }

    /// Get comprehensive loader statistics
    pub fn get_statistics(&self) -> &LoaderStatistics {
        &self.statistics
    }

    /// Optimize loader configuration based on performance metrics
    pub fn optimize_configuration(&mut self) -> Result<()> {
        // Adjust batch size based on memory usage
        if self.statistics.memory_usage_mb > self.config.max_memory_usage_mb.unwrap_or(4096) as f64 {
            self.config.batch_size = (self.config.batch_size as f64 * 0.8) as usize;
        }
        
        // Adjust chunk size based on load times
        if self.statistics.avg_load_time_ms > 5000.0 { // 5 seconds
            if let Some(chunk_size) = self.config.chunk_size {
                self.config.chunk_size = Some((chunk_size as f64 * 0.7) as usize);
            }
        }
        
        Ok(())
    }

    /// Apply data sampling based on configuration
    fn apply_data_sampling(&self, reader: LazyFrame) -> Result<LazyFrame> {
        let mut reader = reader;
        
        // Apply row sampling if configured
        match &self.config.graph_config.node_sampling_strategy {
            NodeSamplingStrategy::Random { fraction } => {
                reader = reader.sample_frac(*fraction, Some(42), true, true)?;
            }
            NodeSamplingStrategy::TopK { k, .. } => {
                reader = reader.limit(*k as u32);
            }
            _ => {}
        }
        
        Ok(reader)
    }

    /// Extract subgraph samples from database
    async fn extract_subgraph_samples(&self, database: &Database) -> Result<Vec<SubgraphSample>> {
        let mut samples = Vec::new();
        
        // For each table, create subgraph samples based on the configuration
        for (table_name, table) in database.table_names().iter().enumerate() {
            let sample = SubgraphSample::new(
                format!("sample_{}_{}", table_name, samples.len()),
                vec![], // Would extract actual seed nodes
                SamplingStrategy {
                    strategy_type: SamplingType::NeighborSampling {
                        fanouts: vec![10, 5],
                        replace: false,
                    },
                    parameters: HashMap::new(),
                    max_nodes: self.config.graph_config.max_nodes_per_graph,
                    max_edges: self.config.graph_config.max_edges_per_graph,
                    max_hops: 3,
                    temporal_window: None,
                },
            );
            samples.push(sample);
        }
        
        Ok(samples)
    }

    /// Extract node features from table
    fn extract_node_features(&self, table: &Table, feature_columns: &[String]) -> Result<Tensor> {
        let num_nodes = table.len();
        let feature_dim = feature_columns.len().max(1);
        
        Tensor::zeros((num_nodes, feature_dim), candle_core::DType::F32, &self.device)
    }

    /// Extract edge indices from database relationships
    fn extract_edge_indices(&self, database: &Database, target_table: &str) -> Result<Tensor> {
        let num_edges = 100; // placeholder
        
        Tensor::zeros((2, num_edges), candle_core::DType::I64, &self.device)
    }

    /// Update loading statistics
    fn update_load_statistics(&mut self, rows_processed: usize, load_time_ms: f64) {
        self.statistics.total_files_loaded += 1;
        self.statistics.total_rows_processed += rows_processed as u64;
        self.statistics.avg_load_time_ms = 
            (self.statistics.avg_load_time_ms * (self.statistics.total_files_loaded - 1) as f64 + 
             load_time_ms) / self.statistics.total_files_loaded as f64;
    }
}

/// Enhanced configuration for table conversion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableConfig {
    pub pkey_col: String,
    pub time_col: Option<String>,
    pub fkey_cols: Vec<(String, String)>, // (column, referenced table)
    pub node_features: Vec<String>,
    pub edge_features: Vec<String>,
    pub entity_type_col: Option<String>,
    pub relation_type_col: Option<String>,
}

/// Data source specification for loading
#[derive(Debug, Clone)]
pub struct DataSource {
    pub name: String,
    pub path: PathBuf,
    pub format: DataFormat,
    pub config: TableConfig,
}

/// Supported data formats
#[derive(Debug, Clone)]
pub enum DataFormat {
    Parquet,
    CSV,
    JSON,
}

/// Streaming data loader for large datasets
pub struct StreamingDataLoader {
    config: DataLoaderConfig,
    buffer_size: usize,
    current_buffer: Vec<DataFrame>,
}

impl StreamingDataLoader {
    pub fn new(config: DataLoaderConfig, buffer_size: usize) -> Self {
        Self {
            config,
            buffer_size,
            current_buffer: Vec::new(),
        }
    }

    /// Stream data in chunks for memory-efficient processing
    pub async fn stream_chunks<F>(&mut self, path: &PathBuf, processor: F) -> Result<()> 
    where
        F: Fn(DataFrame) -> Result<()>,
    {
        let reader = LazyFrame::scan_parquet(path, ScanArgsParquet::default())?
            .with_streaming(true);
        
        // Process in chunks
        let chunk_size = self.config.chunk_size.unwrap_or(10000);
        let mut offset = 0;
        
        loop {
            let chunk = reader
                .clone()
                .slice(offset, chunk_size as u32)
                .collect()?;
            
            if chunk.is_empty() {
                break;
            }
            
            processor(chunk)?;
            offset += chunk_size as i64;
        }
        
        Ok(())
    }
}

/// Legacy data loader for compatibility
pub struct DataLoader {
    config: DataLoaderConfig,
}

impl DataLoader {
    /// Creates a new data loader
    pub fn new(config: DataLoaderConfig) -> Self {
        Self { config }
    }

    /// Loads a parquet file efficiently (legacy method)
    pub async fn load_parquet(&self, path: &PathBuf) -> Result<DataFrame> {
        let reader = LazyFrame::scan_parquet(path, ScanArgsParquet::default())?;
        Ok(reader.collect()?)
    }

    /// Loads a CSV file efficiently (legacy method)
    pub async fn load_csv(&self, path: &PathBuf) -> Result<DataFrame> {
        let reader = LazyFrame::scan_csv(path, ScanArgsCSV::default())?;
        Ok(reader.collect()?)
    }

    /// Loads a JSON file efficiently (legacy method)
    pub async fn load_json(&self, path: &PathBuf) -> Result<DataFrame> {
        let reader = LazyFrame::scan_ndjson(path, ScanArgsNdJson::default())?;
        Ok(reader.collect()?)
    }

    /// Converts a polars DataFrame to a RelBench Table (legacy method)
    pub fn to_table(&self, df: DataFrame, config: TableConfig) -> Result<Table> {
        Table::from_polars(df, config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use candle_core::Device;

    #[tokio::test]
    async fn test_enhanced_parquet_loading() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.parquet");
        
        // Create test data
        let df = df! {
            "id" => [1, 2, 3],
            "value" => [10, 20, 30],
        }.unwrap();
        
        // Save test data
        let mut file = std::fs::File::create(&path).unwrap();
        ParquetWriter::new(&mut file).finish(&df).unwrap();
        
        // Test loading
        let config = DataLoaderConfig::default();
        let device = Device::Cpu;
        let mut loader = GraphDataLoader::new(config, device);
        let loaded_df = loader.load_parquet(&path).await.unwrap();
        
        assert_eq!(loaded_df.shape(), (3, 2));
        assert!(loader.get_statistics().total_files_loaded > 0);
    }

    #[tokio::test]
    async fn test_graph_batch_creation() {
        let config = DataLoaderConfig::default();
        let device = Device::Cpu;
        let mut loader = GraphDataLoader::new(config, device);
        
        // Create a simple database
        let tables = HashMap::new();
        let database = Database::new(tables);
        
        // Test graph batch creation
        let batches = loader.create_graph_batches(&database, 2).await.unwrap();
        // Would have more meaningful assertions with actual data
    }

    #[test]
    fn test_configuration_optimization() {
        let config = DataLoaderConfig::default();
        let device = Device::Cpu;
        let mut loader = GraphDataLoader::new(config, device);
        
        // Simulate high memory usage
        loader.statistics.memory_usage_mb = 5000.0;
        
        loader.optimize_configuration().unwrap();
        
        // Batch size should be reduced
        assert!(loader.config.batch_size < 32);
    }

    #[tokio::test]
    async fn test_legacy_parquet_loading() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.parquet");
        
        // Create test data
        let df = df! {
            "id" => [1, 2, 3],
            "value" => [10, 20, 30],
        }.unwrap();
        
        // Save test data
        let mut file = std::fs::File::create(&path).unwrap();
        ParquetWriter::new(&mut file).finish(&df).unwrap();
        
        // Test loading
        let loader = DataLoader::new(DataLoaderConfig::default());
        let loaded_df = loader.load_parquet(&path).await.unwrap();
        
        assert_eq!(loaded_df.shape(), (3, 2));
    }
} 