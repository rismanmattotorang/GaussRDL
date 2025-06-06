// src/data/loader.rs
use std::path::Path;
use candle_core::Device;
use crate::{Result, graph::RelationalEntityGraph};

pub struct DataLoader {
    data_dir: std::path::PathBuf,
    device: Device,
}

impl DataLoader {
    pub fn from_directory(data_dir: &Path) -> Result<Self> {
        let device = crate::utils::get_device()?;
        
        Ok(Self {
            data_dir: data_dir.to_path_buf(),
            device,
        })
    }
    
    pub fn load_splits(&self) -> Result<(Vec<super::TrainingBatch>, Vec<super::TrainingBatch>, Vec<super::TestBatch>)> {
        let train_data = self.load_training_data("train")?;
        let val_data = self.load_training_data("val")?;
        let test_data = self.load_test_data("test")?;
        
        Ok((train_data, val_data, test_data))
    }
    
    fn load_training_data(&self, split: &str) -> Result<Vec<super::TrainingBatch>> {
        // Create dummy training batch
        let dummy_graph = self.create_dummy_graph()?;
        let dummy_batch = super::TrainingBatch {
            graph: dummy_graph,
            seed_nodes: vec![(0, 1), (0, 2), (1, 1)],
            subgraphs: vec![
                vec![(0, 1), (0, 2), (1, 1)],
                vec![(0, 2), (0, 3), (1, 2)],
                vec![(1, 1), (0, 1), (1, 3)],
            ],
            timestamps: vec![1000.0, 1001.0, 1002.0],
            targets: candle_core::Tensor::new(&[1.0f32, 0.0, 1.0], &self.device)?,
            batch_size: 3,
        };
        
        Ok(vec![dummy_batch])
    }
    
    fn load_test_data(&self, split: &str) -> Result<Vec<super::TestBatch>> {
        // Create dummy test batch
        let dummy_graph = self.create_dummy_graph()?;
        let dummy_batch = super::TestBatch {
            graph: dummy_graph,
            seed_nodes: vec![(0, 4), (0, 5), (1, 4)],
            subgraphs: vec![
                vec![(0, 4), (0, 5), (1, 4)],
                vec![(0, 5), (0, 6), (1, 5)],
                vec![(1, 4), (0, 4), (1, 6)],
            ],
            timestamps: vec![1003.0, 1004.0, 1005.0],
            targets: candle_core::Tensor::new(&[0.0f32, 1.0, 0.0], &self.device)?,
            batch_size: 3,
        };
        
        Ok(vec![dummy_batch])
    }
    
    fn create_dummy_graph(&self) -> Result<RelationalEntityGraph> {
        use crate::database::{DatabaseSchema, DatabaseInfo};
        use crate::graph::{RelationalEntityGraph, EntityNode};
        use std::collections::HashMap;
        
        let schema = DatabaseSchema {
            database_info: DatabaseInfo {
                name: "dummy".to_string(),
                version: "1.0".to_string(),
                table_count: 2,
                total_rows: 100,
                supports_foreign_keys: true,
                supports_temporal: true,
            },
            tables: vec![],
            relationships: vec![],
            extracted_at: chrono::Utc::now(),
        };
        
        let mut graph = RelationalEntityGraph::new(schema, self.device.clone());
        
        // Add dummy nodes
        for i in 1..=10 {
            let features = candle_core::Tensor::new(&[i as f32, (i * 2) as f32], &self.device)?;
            let node = EntityNode {
                table_type: i % 2,
                node_id: i,
                features,
                timestamp: 1000.0 + i as f64,
                attributes: HashMap::new(),
            };
            graph.add_node(node)?;
        }
        
        // Add dummy edges
        for i in 1..=5 {
            graph.add_edge((0, i), (1, i))?;
        }
        
        Ok(graph)
    }
}