use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use gaussrdl_core::{Dataset, CacheConfig, DownloadConfig};
use gaussrdl_core::{Error, Result};

/// Dataset registry to manage available datasets
#[derive(Debug, Default)]
pub struct DatasetRegistry {
    datasets: HashMap<String, Arc<dyn Dataset>>,
}

impl DatasetRegistry {
    /// Creates a new dataset registry
    pub fn new() -> Self {
        let mut registry = Self::default();
        registry.register_defaults();
        registry
    }

    /// Registers default datasets
    fn register_defaults(&mut self) {
        self.register("rel-amazon", Arc::new(RelAmazonDataset::new()));
        self.register("rel-f1", Arc::new(RelF1Dataset::new()));
        self.register("rel-hm", Arc::new(RelHMDataset::new()));
    }

    /// Registers a new dataset
    pub fn register(&mut self, name: &str, dataset: Arc<dyn Dataset>) {
        self.datasets.insert(name.to_string(), dataset);
    }

    /// Gets a dataset by name
    pub fn get(&self, name: &str) -> Option<Arc<dyn Dataset>> {
        self.datasets.get(name).cloned()
    }

    /// Lists all available datasets
    pub fn list(&self) -> Vec<String> {
        self.datasets.keys().cloned().collect()
    }
}

/// Dataset provider trait
pub trait DatasetProvider: std::fmt::Debug + Send + Sync {
    /// Gets the dataset name
    fn name(&self) -> &str;
    
    /// Gets the dataset description
    fn description(&self) -> &str;
    
    /// Gets the dataset version
    fn version(&self) -> &str;
    
    /// Gets the dataset homepage
    fn homepage(&self) -> Option<&str>;
    
    /// Gets the dataset citation
    fn citation(&self) -> Option<&str>;
    
    /// Downloads and loads the dataset
    fn load(&self, config: &DownloadConfig, cache_config: &CacheConfig) -> Result<Box<dyn Dataset>>;
    
    /// Gets the dataset cache path
    fn cache_path(&self, cache_dir: &PathBuf) -> PathBuf {
        cache_dir.join(self.name())
    }
    
    /// Validates the dataset
    fn validate(&self, dataset: &dyn Dataset) -> Result<()>;
}

/// Base implementation for dataset providers
#[derive(Debug, Clone)]
pub struct BaseDatasetProvider {
    _name: String,
    _description: String,
    _version: String,
    _homepage: Option<String>,
    _citation: Option<String>,
}

impl BaseDatasetProvider {
    /// Creates a new base dataset provider
    pub fn new(
        name: &str,
        description: &str,
        version: &str,
        homepage: Option<&str>,
        citation: Option<&str>,
    ) -> Self {
        Self {
            _name: name.to_string(),
            _description: description.to_string(),
            _version: version.to_string(),
            _homepage: homepage.map(String::from),
            _citation: citation.map(String::from),
        }
    }
}

/// Gets a dataset by name
pub fn get_dataset(name: &str, download: bool) -> Result<Box<dyn Dataset>> {
    let mut dataset: Box<dyn Dataset> = match name {
        "rel-amazon" => Box::new(RelAmazonDataset::new()),
        "rel-f1" => Box::new(RelF1Dataset::new()),
        "rel-hm" => Box::new(RelHMDataset::new()),
        "rel-trial" => Box::new(RelTrialDataset::new("rel-trial")),
        "rel-avito" => Box::new(RelAvitoDataset::new()),
        _ => return Err(Error::dataset(format!("Dataset {} not found", name))),
    };
    
    if download {
        let config = gaussrdl_core::DatasetConfig::default();
        dataset.load(&config)?;
    }
    
    Ok(dataset)
}

// Individual dataset implementations
mod rel_amazon;
mod rel_f1;
mod rel_hm;
mod rel_trial;
mod rel_avito;

pub use rel_amazon::RelAmazonDataset;
pub use rel_f1::RelF1Dataset;
pub use rel_hm::RelHMDataset;
pub use rel_trial::RelTrialDataset;
pub use rel_avito::RelAvitoDataset;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_creation() {
        let registry = DatasetRegistry::new();
        assert!(!registry.list().is_empty());
    }

    #[test]
    fn test_dataset_retrieval() {
        let registry = DatasetRegistry::new();
        assert!(registry.get("rel-amazon").is_some());
        assert!(registry.get("rel-f1").is_some());
        assert!(registry.get("rel-hm").is_some());
        assert!(registry.get("non-existent").is_none());
    }
} 