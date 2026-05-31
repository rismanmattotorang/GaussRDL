// src/model/tokenizer.rs
use candle_core::{Device, Tensor};
use candle_nn::{Module, VarBuilder};
use crate::{Result, ModelInput, UnifiedModelConfig};
use crate::ModelError;
use gaussrdl_graph::graph::*;
use gaussrdl_graph::graph::sampler::SubgraphSample;
use std::collections::{HashMap, HashSet};
use serde::{Serialize, Deserialize};
use serde::de::{MapAccess};
use serde::de::Visitor;
// use gaussrdl_graph::graph::SubgraphSample;

/// Enhanced tokenizer for relational graph data
pub struct RelgtTokenizer {
    config: TokenizerConfig,
    device: Device,
    
    // Vocabulary mappings
    node_type_vocab: HashMap<String, usize>,
    relation_type_vocab: HashMap<String, usize>,
    feature_vocab: HashMap<String, usize>,
    
    // Special tokens
    pad_token_id: usize,
    unk_token_id: usize,
    cls_token_id: usize,
    sep_token_id: usize,
    
    // Statistics
    statistics: TokenizerStatistics,
}

/// Configuration for the tokenizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerConfig {
    /// Maximum sequence length
    pub max_length: usize,
    /// Maximum number of nodes per subgraph
    pub max_nodes: usize,
    /// Maximum number of edges per subgraph
    pub max_edges: usize,
    /// Vocabulary size for node types
    pub node_type_vocab_size: usize,
    /// Vocabulary size for relation types
    pub relation_type_vocab_size: usize,
    /// Whether to use positional encoding
    pub use_positional_encoding: bool,
    /// Positional encoding dimension
    pub positional_encoding_dim: usize,
    /// Whether to use temporal encoding
    pub use_temporal_encoding: bool,
    /// Padding strategy
    pub padding_strategy: PaddingStrategy,
    /// Truncation strategy
    pub truncation_strategy: TruncationStrategy,
    /// Whether to normalize features
    pub normalize_features: bool,
    /// Feature scaling method
    pub feature_scaling: FeatureScaling,
}

/// Padding strategy enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PaddingStrategy {
    /// Pad to maximum length
    MaxLength,
    /// Pad to longest in batch
    LongestInBatch,
    /// No padding
    NoPadding,
}

/// Truncation strategy enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TruncationStrategy {
    /// Truncate from the end
    TruncateEnd,
    /// Truncate from the beginning
    TruncateBeginning,
    /// Truncate randomly
    TruncateRandom,
    /// No truncation
    NoTruncation,
}

/// Feature scaling methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeatureScaling {
    /// No scaling
    None,
    /// Min-max normalization
    MinMax,
    /// Z-score normalization
    ZScore,
    /// Robust scaling
    Robust,
}

/// Tokenizer statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerStatistics {
    pub total_tokens_processed: usize,
    pub avg_sequence_length: f64,
    pub vocab_coverage: f64,
    pub oov_rate: f64,
    pub padding_rate: f64,
    pub truncation_rate: f64,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            max_length: 512,
            max_nodes: 1000,
            max_edges: 5000,
            node_type_vocab_size: 1000,
            relation_type_vocab_size: 100,
            use_positional_encoding: true,
            positional_encoding_dim: 128,
            use_temporal_encoding: true,
            padding_strategy: PaddingStrategy::MaxLength,
            truncation_strategy: TruncationStrategy::TruncateEnd,
            normalize_features: true,
            feature_scaling: FeatureScaling::ZScore,
        }
    }
}

impl RelgtTokenizer {
    /// Create a new tokenizer
    pub fn new(config: TokenizerConfig, device: Device) -> Result<Self> {
        let mut node_type_vocab = HashMap::new();
        let mut relation_type_vocab = HashMap::new();
        let feature_vocab = HashMap::new();
        
        // Initialize special tokens
        let pad_token_id = 0;
        let unk_token_id = 1;
        let cls_token_id = 2;
        let sep_token_id = 3;
        
        // Add special tokens to vocabularies
        node_type_vocab.insert("[PAD]".to_string(), pad_token_id);
        node_type_vocab.insert("[UNK]".to_string(), unk_token_id);
        node_type_vocab.insert("[CLS]".to_string(), cls_token_id);
        node_type_vocab.insert("[SEP]".to_string(), sep_token_id);
        
        relation_type_vocab.insert("[PAD]".to_string(), pad_token_id);
        relation_type_vocab.insert("[UNK]".to_string(), unk_token_id);
        
        let statistics = TokenizerStatistics {
            total_tokens_processed: 0,
            avg_sequence_length: 0.0,
            vocab_coverage: 0.0,
            oov_rate: 0.0,
            padding_rate: 0.0,
            truncation_rate: 0.0,
        };
        
        Ok(Self {
            config,
            device,
            node_type_vocab,
            relation_type_vocab,
            feature_vocab,
            pad_token_id,
            unk_token_id,
            cls_token_id,
            sep_token_id,
            statistics,
        })
    }
    
    /// Build vocabulary from a collection of subgraphs
    pub fn build_vocabulary(&mut self, subgraphs: &[SubgraphSample]) -> Result<()> {
        let mut node_types = HashSet::new();
        let mut relation_types = HashSet::new();
        let mut features = HashSet::new();
        
        // Collect all unique types and features
        for subgraph in subgraphs {
            for node in &subgraph.nodes {
                node_types.insert(node.entity_type.name.clone());
                
                // Collect feature names from attributes
                for key in node.attributes.keys() {
                    features.insert(key.clone());
                }
            }
            
            for edge in &subgraph.edges {
                relation_types.insert(edge.relation_type.name.clone());
                
                // Collect edge feature names
                for key in edge.attributes.keys() {
                    features.insert(key.clone());
                }
            }
        }
        
        // Build vocabularies
        let mut node_id = 4; // Start after special tokens
        for node_type in node_types {
            if !self.node_type_vocab.contains_key(&node_type) {
                self.node_type_vocab.insert(node_type, node_id);
                node_id += 1;
            }
        }
        
        let mut rel_id = 2; // Start after special tokens (no CLS/SEP for relations)
        for relation_type in relation_types {
            if !self.relation_type_vocab.contains_key(&relation_type) {
                self.relation_type_vocab.insert(relation_type, rel_id);
                rel_id += 1;
            }
        }
        
        let mut feat_id = 0;
        for feature in features {
            if !self.feature_vocab.contains_key(&feature) {
                self.feature_vocab.insert(feature, feat_id);
                feat_id += 1;
            }
        }
        
        Ok(())
    }
    
    /// Tokenize a single subgraph into model input
    pub fn tokenize_subgraph(&self, subgraph: &SubgraphSample) -> Result<ModelInput> {
        let num_nodes = subgraph.nodes.len();
        let num_edges = subgraph.edges.len();
        
        if num_nodes == 0 {
            return Err(ModelError::Model("Cannot tokenize empty subgraph".to_string()));
        }
        
        // Extract node features
        let node_features = self.extract_node_features(subgraph)?;
        
        // Extract edge indices
        let edge_index = self.extract_edge_indices(subgraph)?;
        
        // Extract edge features if available
        let edge_features = self.extract_edge_features(subgraph)?;
        
        // Extract node types
        let node_types = self.extract_node_types(subgraph)?;
        
        // Extract edge types
        let edge_types = self.extract_edge_types(subgraph)?;
        
        // Create model input
        let mut input = ModelInput::heterogeneous(
            node_features,
            edge_index,
            edge_types,
            Some(node_types),
        );
        
        if let Some(edge_feat) = edge_features {
            input = input.with_edge_features(edge_feat);
        }
        
        // Add positional encoding if enabled
        if self.config.use_positional_encoding {
            let pos_encoding = self.create_positional_encoding(num_nodes)?;
            input = input.with_metadata("positional_encoding".to_string(), pos_encoding);
        }
        
        // Add temporal encoding if enabled
        if self.config.use_temporal_encoding {
            let temp_encoding = self.create_temporal_encoding(subgraph)?;
            input = input.with_metadata("temporal_encoding".to_string(), temp_encoding);
        }
        
        Ok(input)
    }
    
    /// Tokenize a batch of subgraphs
    pub fn tokenize_batch(&self, subgraphs: &[SubgraphSample]) -> Result<ModelInput> {
        if subgraphs.is_empty() {
            return Err(ModelError::Model("Cannot tokenize empty batch".to_string()));
        }
        
        // Tokenize individual subgraphs
        let mut tokenized_inputs = Vec::new();
        for subgraph in subgraphs {
            let input = self.tokenize_subgraph(subgraph)?;
            tokenized_inputs.push(input);
        }
        
        // Batch the inputs
        self.batch_inputs(&tokenized_inputs)
    }
    
    /// Extract node features from subgraph
    fn extract_node_features(&self, subgraph: &SubgraphSample) -> Result<Tensor> {
        let num_nodes = subgraph.nodes.len();
        
        if num_nodes == 0 {
            return Err(ModelError::Model("No nodes to extract features from".to_string()));
        }
        
        // Get feature dimension from first node
        let feature_dim = subgraph.nodes[0].feature_dim()?;
        let mut features_data = Vec::with_capacity(num_nodes * feature_dim);
        
        for node in &subgraph.nodes {
            let node_features = node.features.to_vec1::<f32>()?;
            features_data.extend(node_features);
        }
        
        let features = Tensor::from_vec(
            features_data,
            (num_nodes, feature_dim),
            &self.device,
        )?;
        
        // Apply feature scaling if enabled
        if self.config.normalize_features {
            self.scale_features(features)
        } else {
            Ok(features)
        }
    }
    
    /// Extract edge indices from subgraph
    fn extract_edge_indices(&self, subgraph: &SubgraphSample) -> Result<Tensor> {
        let num_edges = subgraph.edges.len();
        let mut edge_indices = Vec::with_capacity(num_edges * 2);
        
        for edge in &subgraph.edges {
            let source_idx = subgraph.node_mapping.get(&edge.source).unwrap_or(&0);
            let target_idx = subgraph.node_mapping.get(&edge.target).unwrap_or(&0);
            edge_indices.push(*source_idx as i64);
            edge_indices.push(*target_idx as i64);
        }
        
        Ok(Tensor::from_vec(edge_indices, (2, num_edges), &self.device)
            .map_err(|e| ModelError::Model(format!("Failed to create edge indices tensor: {}", e)))?)
    }
    
    /// Extract edge features from subgraph
    fn extract_edge_features(&self, subgraph: &SubgraphSample) -> Result<Option<Tensor>> {
        let edges_with_features: Vec<_> = subgraph.edges.iter()
            .filter(|e| e.features.is_some())
            .collect();
        
        if edges_with_features.is_empty() {
            return Ok(None);
        }
        
        let num_edges = subgraph.edges.len();
        let feature_dim = edges_with_features[0].features.as_ref().unwrap().dims()[0];
        let mut features_data = Vec::with_capacity(num_edges * feature_dim);
        
        for edge in &subgraph.edges {
            if let Some(ref features) = edge.features {
                let features_ref: &candle_core::Tensor = features;
                let features = if features_ref.dtype() != candle_core::DType::F32 {
                    features_ref.to_dtype(candle_core::DType::F32).unwrap_or_else(|_| features_ref.clone())
                } else {
                    features_ref.clone()
                };
                let edge_features: Vec<f32> = features.to_vec1::<f32>().unwrap_or_else(|_| vec![0.0_f32; features.dims()[0]]);
                features_data.extend(edge_features);
            } else {
                features_data.extend(vec![0.0; feature_dim]);
            }
        }
        
        let features = Tensor::from_vec(
            features_data,
            (num_edges, feature_dim),
            &self.device,
        )?;
        
        Ok(Some(features))
    }
    
    /// Extract node types from subgraph
    fn extract_node_types(&self, subgraph: &SubgraphSample) -> Result<Tensor> {
        let mut node_types = Vec::with_capacity(subgraph.nodes.len());
        
        for node in &subgraph.nodes {
            let type_id = self.node_type_vocab.get(&node.entity_type.name).unwrap_or(&0);
            node_types.push(*type_id as i64);
        }
        
        Ok(Tensor::from_vec(node_types, (subgraph.nodes.len(),), &self.device)
            .map_err(|e| ModelError::Model(format!("Failed to create node types tensor: {}", e)))?)
    }
    
    /// Extract edge types from subgraph
    fn extract_edge_types(&self, subgraph: &SubgraphSample) -> Result<Tensor> {
        let mut edge_types = Vec::with_capacity(subgraph.edges.len());
        
        for edge in &subgraph.edges {
            let type_id = self.relation_type_vocab.get(&edge.relation_type.name).unwrap_or(&0);
            edge_types.push(*type_id as i64);
        }
        
        Ok(Tensor::from_vec(edge_types, (subgraph.edges.len(),), &self.device)
            .map_err(|e| ModelError::Model(format!("Failed to create edge types tensor: {}", e)))?)
    }
    
    /// Create positional encoding for nodes
    fn create_positional_encoding(&self, num_nodes: usize) -> Result<Tensor> {
        let encoding_dim = self.config.positional_encoding_dim;
        let mut encoding_data = Vec::with_capacity(num_nodes * encoding_dim);
        
        for pos in 0..num_nodes {
            for i in 0..encoding_dim {
                let angle = pos as f32 / 10000.0_f32.powf(2.0 * i as f32 / encoding_dim as f32);
                if i % 2 == 0 {
                    encoding_data.push(angle.sin());
                } else {
                    encoding_data.push(angle.cos());
                }
            }
        }
        
        Ok(Tensor::from_vec(encoding_data, (num_nodes, encoding_dim), &self.device)
            .map_err(|e| ModelError::Model(format!("Failed to create positional encoding: {}", e)))?)
    }
    
    /// Create temporal encoding for subgraph
    fn create_temporal_encoding(&self, subgraph: &SubgraphSample) -> Result<Tensor> {
        let num_nodes = subgraph.nodes.len();
        let mut temporal_data = Vec::with_capacity(num_nodes);
        
        for node in &subgraph.nodes {
            let timestamp = node.timestamp.map(|t| t.timestamp() as f32).unwrap_or(0.0);
            temporal_data.push(timestamp);
        }
        
        Ok(Tensor::from_vec(temporal_data, (num_nodes, 1), &self.device)
            .map_err(|e| ModelError::Model(format!("Failed to create temporal encoding: {}", e)))?)
    }
    
    /// Scale features according to configuration
    fn scale_features(&self, features: Tensor) -> Result<Tensor> {
        match self.config.feature_scaling {
            FeatureScaling::None => Ok(features),
            FeatureScaling::MinMax => {
                // Min-max normalization: (x - min) / (max - min)
                let min_vals = features.min(0)
                    .map_err(|e| ModelError::Model(format!("Failed to compute min: {}", e)))?;
                let max_vals = features.max(0)
                    .map_err(|e| ModelError::Model(format!("Failed to compute max: {}", e)))?;
                let range = max_vals.sub(&min_vals)
                    .map_err(|e| ModelError::Model(format!("Failed to compute range: {}", e)))?;
                let normalized = features.sub(&min_vals)
                    .map_err(|e| ModelError::Model(format!("Failed to subtract min: {}", e)))?
                    .div(&range)
                    .map_err(|e| ModelError::Model(format!("Failed to divide by range: {}", e)))?;
                Ok(normalized)
            }
            FeatureScaling::ZScore => {
                // Z-score normalization: (x - mean) / std
                let mean = features.mean(0)
                    .map_err(|e| ModelError::Model(format!("Failed to compute mean: {}", e)))?;
                let std = features.var(0)
                    .map_err(|e| ModelError::Model(format!("Failed to compute variance: {}", e)))?
                    .sqrt()
                    .map_err(|e| ModelError::Model(format!("Failed to compute sqrt: {}", e)))?;
                let normalized = features.sub(&mean)
                    .map_err(|e| ModelError::Model(format!("Failed to subtract mean: {}", e)))?
                    .div(&std)
                    .map_err(|e| ModelError::Model(format!("Failed to divide by std: {}", e)))?;
                Ok(normalized)
            }
            FeatureScaling::Robust => {
                // Robust scaling: (x - median) / IQR
                // Simplified implementation using mean and std for now
                let mean = features.mean(0)
                    .map_err(|e| ModelError::Model(format!("Failed to compute mean: {}", e)))?;
                let std = features.var(0)
                    .map_err(|e| ModelError::Model(format!("Failed to compute variance: {}", e)))?
                    .sqrt()
                    .map_err(|e| ModelError::Model(format!("Failed to compute sqrt: {}", e)))?;
                let normalized = features.sub(&mean)
                    .map_err(|e| ModelError::Model(format!("Failed to subtract mean: {}", e)))?
                    .div(&std)
                    .map_err(|e| ModelError::Model(format!("Failed to divide by std: {}", e)))?;
                Ok(normalized)
            }
        }
    }
    
    /// Batch multiple model inputs
    fn batch_inputs(&self, inputs: &[ModelInput]) -> Result<ModelInput> {
        if inputs.is_empty() {
            return Err(ModelError::Model("Cannot batch empty inputs".to_string()));
        }
        
        // Concatenate node features
        let node_features_list: Vec<&Tensor> = inputs.iter()
            .map(|input| &input.node_features)
            .collect();
        let batched_node_features = Tensor::cat(&node_features_list, 0)?;
        
        // Concatenate and offset edge indices
        let mut all_edge_indices = Vec::new();
        let mut node_offset = 0;
        
        for input in inputs {
            let edge_indices = input.edge_index.to_vec2::<i64>()?;
            for edge_pair in edge_indices {
                all_edge_indices.push(edge_pair[0] + node_offset);
                all_edge_indices.push(edge_pair[1] + node_offset);
            }
            node_offset += input.num_nodes()? as i64;
        }
        
        let total_edges = all_edge_indices.len() / 2;
        let batched_edge_index = Tensor::from_vec(
            all_edge_indices,
            (2, total_edges),
            &self.device,
        )?;
        
        // Create batch indices
        let mut batch_indices = Vec::new();
        for (batch_idx, input) in inputs.iter().enumerate() {
            let num_nodes = input.num_nodes()?;
            batch_indices.extend(vec![batch_idx as i64; num_nodes]);
        }
        let batch_tensor = Tensor::from_vec(batch_indices, (batched_node_features.dim(0)?,), &self.device)?;
        
        // Create batched input
        let mut batched_input = ModelInput::homogeneous(batched_node_features, batched_edge_index);
        batched_input = batched_input.with_batch(batch_tensor);
        
        // Batch other optional tensors if present
        if inputs.iter().all(|input| input.node_types.is_some()) {
            let node_types_list: Vec<&Tensor> = inputs.iter()
                .map(|input| input.node_types.as_ref().unwrap())
                .collect();
            let batched_node_types = Tensor::cat(&node_types_list, 0)?;
            batched_input.node_types = Some(batched_node_types);
        }
        
        if inputs.iter().all(|input| input.edge_types.is_some()) {
            let edge_types_list: Vec<&Tensor> = inputs.iter()
                .map(|input| input.edge_types.as_ref().unwrap())
                .collect();
            let batched_edge_types = Tensor::cat(&edge_types_list, 0)?;
            batched_input.edge_types = Some(batched_edge_types);
        }
        
        Ok(batched_input)
    }
    
    /// Get tokenizer configuration
    pub fn get_config(&self) -> &TokenizerConfig {
        &self.config
    }
    
    /// Get tokenizer statistics
    pub fn get_statistics(&self) -> &TokenizerStatistics {
        &self.statistics
    }
    
    /// Get vocabulary sizes
    pub fn vocab_sizes(&self) -> (usize, usize, usize) {
        (
            self.node_type_vocab.len(),
            self.relation_type_vocab.len(),
            self.feature_vocab.len(),
        )
    }
    
    /// Save tokenizer to file
    pub fn save(&self, path: &str) -> Result<()> {
        let serialized = serde_json::to_string_pretty(self)?;
        std::fs::write(path, serialized)?;
        Ok(())
    }
    
    /// Load tokenizer from file
    pub fn load(path: &str, device: Device) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let mut tokenizer: Self = serde_json::from_str(&content)?;
        tokenizer.device = device;
        Ok(tokenizer)
    }
}

impl Serialize for RelgtTokenizer {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("RelgtTokenizer", 7)?;
        state.serialize_field("config", &self.config)?;
        state.serialize_field("node_type_vocab", &self.node_type_vocab)?;
        state.serialize_field("relation_type_vocab", &self.relation_type_vocab)?;
        state.serialize_field("feature_vocab", &self.feature_vocab)?;
        state.serialize_field("pad_token_id", &self.pad_token_id)?;
        state.serialize_field("unk_token_id", &self.unk_token_id)?;
        state.serialize_field("statistics", &self.statistics)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for RelgtTokenizer {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::{self, Deserialize, Deserializer, MapAccess};
        use std::fmt;
        
        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "snake_case")]
        enum Field {
            Config,
            NodeTypeVocab,
            RelationTypeVocab,
            FeatureVocab,
            PadTokenId,
            UnkTokenId,
            Statistics,
        }
        
        struct TokenizerVisitor;
        
        impl<'de> Visitor<'de> for TokenizerVisitor {
            type Value = RelgtTokenizer;
            
            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct RelgtTokenizer")
            }
            
            fn visit_map<V>(self, mut map: V) -> std::result::Result<RelgtTokenizer, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut config = None;
                let mut node_type_vocab = None;
                let mut relation_type_vocab = None;
                let mut feature_vocab = None;
                let mut pad_token_id = None;
                let mut unk_token_id = None;
                let mut statistics = None;
                
                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Config => {
                            if config.is_some() {
                                return Err(de::Error::duplicate_field("config"));
                            }
                            config = Some(map.next_value()?);
                        }
                        Field::NodeTypeVocab => {
                            if node_type_vocab.is_some() {
                                return Err(de::Error::duplicate_field("node_type_vocab"));
                            }
                            node_type_vocab = Some(map.next_value()?);
                        }
                        Field::RelationTypeVocab => {
                            if relation_type_vocab.is_some() {
                                return Err(de::Error::duplicate_field("relation_type_vocab"));
                            }
                            relation_type_vocab = Some(map.next_value()?);
                        }
                        Field::FeatureVocab => {
                            if feature_vocab.is_some() {
                                return Err(de::Error::duplicate_field("feature_vocab"));
                            }
                            feature_vocab = Some(map.next_value()?);
                        }
                        Field::PadTokenId => {
                            if pad_token_id.is_some() {
                                return Err(de::Error::duplicate_field("pad_token_id"));
                            }
                            pad_token_id = Some(map.next_value()?);
                        }
                        Field::UnkTokenId => {
                            if unk_token_id.is_some() {
                                return Err(de::Error::duplicate_field("unk_token_id"));
                            }
                            unk_token_id = Some(map.next_value()?);
                        }
                        Field::Statistics => {
                            if statistics.is_some() {
                                return Err(de::Error::duplicate_field("statistics"));
                            }
                            statistics = Some(map.next_value()?);
                        }
                    }
                }
                
                let config = config.ok_or_else(|| de::Error::missing_field("config"))?;
                let node_type_vocab = node_type_vocab.ok_or_else(|| de::Error::missing_field("node_type_vocab"))?;
                let relation_type_vocab = relation_type_vocab.ok_or_else(|| de::Error::missing_field("relation_type_vocab"))?;
                let feature_vocab = feature_vocab.ok_or_else(|| de::Error::missing_field("feature_vocab"))?;
                let pad_token_id = pad_token_id.ok_or_else(|| de::Error::missing_field("pad_token_id"))?;
                let unk_token_id = unk_token_id.ok_or_else(|| de::Error::missing_field("unk_token_id"))?;
                let statistics = statistics.ok_or_else(|| de::Error::missing_field("statistics"))?;
                
                Ok(RelgtTokenizer {
                    config,
                    device: Device::Cpu, // Will be set when loading
                    node_type_vocab,
                    relation_type_vocab,
                    feature_vocab,
                    pad_token_id,
                    unk_token_id,
                    cls_token_id: 2, // Fixed
                    sep_token_id: 3, // Fixed
                    statistics,
                })
            }
        }
        
        const FIELDS: &'static [&'static str] = &[
            "config", "node_type_vocab", "relation_type_vocab", "feature_vocab",
            "pad_token_id", "unk_token_id", "statistics"
        ];
        deserializer.deserialize_struct("RelgtTokenizer", FIELDS, TokenizerVisitor)
    }
}