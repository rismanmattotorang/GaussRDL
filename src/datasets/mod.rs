use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use crate::base::{Dataset, Database, CacheConfig, DownloadConfig};
use crate::error::{Error, Result};

/// Dataset registry to manage available datasets
#[derive(Debug, Default)]
pub struct DatasetRegistry {
    datasets: HashMap<String, Arc<dyn DatasetProvider>>,
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
        // Register rel-amazon dataset
        self.register("rel-amazon", Arc::new(RelAmazonDataset::default()));
        
        // Register rel-f1 dataset
        self.register("rel-f1", Arc::new(RelF1Dataset::default()));
        
        // Register rel-hm dataset
        self.register("rel-hm", Arc::new(RelHMDataset::default()));
        
        // Register rel-trial dataset
        self.register("rel-trial", Arc::new(RelTrialDataset::default()));
        
        // Register rel-event dataset
        self.register("rel-event", Arc::new(RelEventDataset::default()));
        
        // Register rel-avito dataset
        self.register("rel-avito", Arc::new(RelAvitoDataset::default()));
    }

    /// Registers a new dataset provider
    pub fn register(&mut self, name: &str, provider: Arc<dyn DatasetProvider>) {
        self.datasets.insert(name.to_string(), provider);
    }

    /// Gets a dataset by name
    pub fn get(&self, name: &str) -> Option<Arc<dyn DatasetProvider>> {
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
    fn load(&self, config: &DownloadConfig, cache_config: &CacheConfig) -> Result<Dataset>;
    
    /// Gets the dataset cache path
    fn cache_path(&self, cache_dir: &PathBuf) -> PathBuf {
        cache_dir.join(self.name())
    }
    
    /// Validates the dataset
    fn validate(&self, dataset: &Dataset) -> Result<()>;
}

/// Base implementation for dataset providers
#[derive(Debug)]
pub struct BaseDatasetProvider {
    name: String,
    description: String,
    version: String,
    homepage: Option<String>,
    citation: Option<String>,
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
            name: name.to_string(),
            description: description.to_string(),
            version: version.to_string(),
            homepage: homepage.map(String::from),
            citation: citation.map(String::from),
        }
    }
}

/// Gets a dataset by name
pub fn get_dataset(name: &str, download: bool) -> Result<Dataset> {
    let registry = DatasetRegistry::new();
    let provider = registry
        .get(name)
        .ok_or_else(|| Error::dataset(format!("Dataset {} not found", name)))?;
        
    let config = DownloadConfig {
        download,
        ..Default::default()
    };
    
    let cache_config = CacheConfig::default();
    
    provider.load(&config, &cache_config)
}

// Individual dataset implementations
mod rel_amazon;
mod rel_f1;
mod rel_hm;
mod rel_trial;
mod rel_event;
mod rel_avito;

pub use rel_amazon::RelAmazonDataset;
pub use rel_f1::RelF1Dataset;
pub use rel_hm::RelHMDataset;
pub use rel_trial::RelTrialDataset;
pub use rel_event::RelEventDataset;
pub use rel_avito::RelAvitoDataset;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_registry() {
        let registry = DatasetRegistry::new();
        assert!(registry.get("rel-amazon").is_some());
        assert!(registry.get("rel-f1").is_some());
        assert!(registry.get("rel-hm").is_some());
        assert!(registry.get("non-existent").is_none());
    }

    #[test]
    fn test_get_dataset() {
        let result = get_dataset("rel-amazon", false);
        assert!(result.is_ok());

        let result = get_dataset("non-existent", false);
        assert!(result.is_err());
    }
} 