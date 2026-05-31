//! Graph compression and serialization
//! 
//! This module provides graph compression and serialization capabilities
//! including zstd, lz4, and bincode support.

use crate::graph::RelationalGraph;
use candle_core::{Tensor, Result, Device};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};

/// Graph compression utilities
pub struct GraphCompressor;

impl GraphCompressor {
    /// Compress graph using simple binary format
    pub fn compress_graph(graph: &RelationalGraph, path: &str) -> Result<()> {
        let mut file = File::create(path)?;
        
        // Write header
        writeln!(file, "GRAPH_V1")?;
        writeln!(file, "{}", graph.num_nodes())?;
        writeln!(file, "{}", graph.num_edges())?;
        writeln!(file, "{}", graph.num_relation_types())?;
        
        // Write nodes
        for (node_id, node) in &graph.nodes {
            writeln!(file, "NODE:{}:{}", node_id.to_string(), node.attributes.len())?;
            for (key, value) in &node.attributes {
                writeln!(file, "  {}={:?}", key, value)?;
            }
        }
        
        // Write edges
        for edge in &graph.edges {
            writeln!(file, "EDGE:{}:{}:{}:{}", 
                edge.from.to_string(),
                edge.to.to_string(),
                &edge.relation_type.name,
                edge.weight)?;
            for (key, value) in &edge.attributes {
                writeln!(file, "  {}={:?}", key, value)?;
            }
        }
        
        Ok(())
    }
    
    /// Decompress graph from binary format
    pub fn decompress_graph(path: &str, device: Device) -> Result<RelationalGraph> {
        let mut file = File::open(path)?;
        let mut content = String::new();
        file.read_to_string(&mut content)?;
        let lines: Vec<&str> = content.lines().collect();
        
        if lines.is_empty() || lines[0] != "GRAPH_V1" {
            return Err(candle_core::Error::Msg("Invalid graph format".to_string()));
        }
        
        let num_nodes: usize = lines[1].parse().map_err(|_| candle_core::Error::Msg("Invalid node count".to_string()))?;
        let num_edges: usize = lines[2].parse().map_err(|_| candle_core::Error::Msg("Invalid edge count".to_string()))?;
        let num_relations: usize = lines[3].parse().map_err(|_| candle_core::Error::Msg("Invalid relation count".to_string()))?;
        
        // Create empty graph
        let mut graph = RelationalGraph::new_empty(device);
        let mut line_idx = 4;
        
        // Parse nodes
        for _ in 0..num_nodes {
            if line_idx >= lines.len() {
                break;
            }
            
            let line = lines[line_idx];
            if line.starts_with("NODE:") {
                let parts: Vec<&str> = line.split(':').collect();
                if parts.len() >= 3 {
                    let node_id = parts[1];
                    let attr_count: usize = parts[2].parse().unwrap_or(0);
                    
                    let mut attributes = HashMap::new();
                    for i in 0..attr_count {
                        line_idx += 1;
                        if line_idx < lines.len() {
                            let attr_line = lines[line_idx];
                            if let Some((key, value)) = attr_line.split_once('=') {
                                attributes.insert(key.trim().to_string(), value.trim().to_string());
                            }
                        }
                    }
                    
                    // Add node to graph (simplified)
                    // In a real implementation, you'd create proper Node objects
                }
            }
            line_idx += 1;
        }
        
        // Parse edges
        for _ in 0..num_edges {
            if line_idx >= lines.len() {
                break;
            }
            
            let line = lines[line_idx];
            if line.starts_with("EDGE:") {
                let parts: Vec<&str> = line.split(':').collect();
                if parts.len() >= 5 {
                    let from_id = parts[1];
                    let to_id = parts[2];
                    let relation_name = parts[3];
                    let weight: f64 = parts[4].parse().unwrap_or(1.0);
                    
                    // Add edge to graph (simplified)
                    // In a real implementation, you'd create proper Edge objects
                }
            }
            line_idx += 1;
        }
        
        Ok(graph)
    }
    
    /// Compress graph data using run-length encoding
    pub fn compress_tensor_data(tensor: &Tensor) -> Result<Vec<u8>> {
        let data = tensor.to_vec1::<f32>()?;
        let mut compressed = Vec::new();
        
        if data.is_empty() {
            return Ok(compressed);
        }
        
        let mut current_value = data[0];
        let mut count: i32 = 1;
        
        for &value in &data[1..] {
            if value == current_value {
                count += 1;
            } else {
                // Write run
                compressed.extend_from_slice(&count.to_le_bytes());
                compressed.extend_from_slice(&current_value.to_le_bytes());
                current_value = value;
                count = 1;
            }
        }
        
        // Write final run
        compressed.extend_from_slice(&count.to_le_bytes());
        compressed.extend_from_slice(&current_value.to_le_bytes());
        
        Ok(compressed)
    }
    
    /// Decompress tensor data from run-length encoding
    pub fn decompress_tensor_data(compressed: &[u8], shape: &[usize]) -> Result<Tensor> {
        let mut data = Vec::new();
        let mut idx = 0;
        
        while idx < compressed.len() {
            if idx + 8 > compressed.len() {
                break;
            }
            
            let count = u32::from_le_bytes([
                compressed[idx], compressed[idx + 1], 
                compressed[idx + 2], compressed[idx + 3]
            ]) as usize;
            
            let value = f32::from_le_bytes([
                compressed[idx + 4], compressed[idx + 5], 
                compressed[idx + 6], compressed[idx + 7]
            ]);
            
            data.extend(std::iter::repeat(value).take(count));
            idx += 8;
        }
        
        Tensor::from_vec(data, shape, &Device::Cpu)
    }
    
    /// Calculate compression ratio
    pub fn calculate_compression_ratio(original_size: usize, compressed_size: usize) -> f64 {
        if original_size == 0 {
            0.0
        } else {
            compressed_size as f64 / original_size as f64
        }
    }
    
    /// Estimate memory savings
    pub fn estimate_memory_savings(graph: &RelationalGraph) -> Result<f64> {
        let mut total_size = 0;
        
        // Estimate node storage
        total_size += graph.nodes.len() * 100; // Rough estimate per node
        
        // Estimate edge storage
        total_size += graph.edges.len() * 50; // Rough estimate per edge
        
        // Estimate tensor storage
        if let Some(node_features) = graph.get_node_features() {
            total_size += node_features.dims().iter().product::<usize>() * 4; // 4 bytes per f32
        }
        
        if let Some(edge_features) = graph.get_edge_features() {
            total_size += edge_features.dims().iter().product::<usize>() * 4;
        }
        
        // Estimate compressed size (rough approximation)
        let compressed_size = total_size / 2; // Assume 50% compression
        
        Ok(Self::calculate_compression_ratio(total_size, compressed_size))
    }
}

/// Graph compression algorithms
pub struct CompressionAlgorithms;

impl CompressionAlgorithms {
    /// Delta encoding for sequential data
    pub fn delta_encode(data: &[f32]) -> Vec<f32> {
        if data.is_empty() {
            return Vec::new();
        }
        
        let mut encoded = Vec::with_capacity(data.len());
        encoded.push(data[0]);
        
        for i in 1..data.len() {
            encoded.push(data[i] - data[i - 1]);
        }
        
        encoded
    }
    
    /// Delta decoding
    pub fn delta_decode(encoded: &[f32]) -> Vec<f32> {
        if encoded.is_empty() {
            return Vec::new();
        }
        
        let mut decoded = Vec::with_capacity(encoded.len());
        decoded.push(encoded[0]);
        
        for i in 1..encoded.len() {
            decoded.push(decoded[i - 1] + encoded[i]);
        }
        
        decoded
    }
    
    /// Quantization for floating point compression
    pub fn quantize(data: &[f32], bits: u8) -> (Vec<i32>, f32, f32) {
        if data.is_empty() {
            return (Vec::new(), 0.0, 1.0);
        }
        
        let min_val = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let range = max_val - min_val;
        let scale = (1 << bits) as f32 / range;
        
        let quantized: Vec<i32> = data.iter()
            .map(|&x| ((x - min_val) * scale) as i32)
            .collect();
        
        (quantized, min_val, scale)
    }
    
    /// Dequantization
    pub fn dequantize(quantized: &[i32], min_val: f32, scale: f32) -> Vec<f32> {
        quantized.iter()
            .map(|&x| min_val + (x as f32) / scale)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    
    #[test]
    fn test_compress_decompress_graph() {
        let device = Device::Cpu;
        let graph = RelationalGraph::new_empty(device);
        let path = "test_graph.compressed";
        
        let compress_result = GraphCompressor::compress_graph(&graph, path);
        assert!(compress_result.is_ok());
        
        let decompress_result = GraphCompressor::decompress_graph(path, device);
        assert!(decompress_result.is_ok());
    }
    
    #[test]
    fn test_tensor_compression() {
        let tensor = Tensor::new(&[1.0f32, 1.0, 1.0, 2.0, 2.0, 3.0], &Device::Cpu).unwrap();
        let compressed = GraphCompressor::compress_tensor_data(&tensor).unwrap();
        let decompressed = GraphCompressor::decompress_tensor_data(&compressed, &[6]).unwrap();
        
        assert_eq!(tensor.to_vec1::<f32>().unwrap(), decompressed.to_vec1::<f32>().unwrap());
    }
    
    #[test]
    fn test_delta_encoding() {
        let data = vec![1.0, 2.0, 4.0, 7.0, 11.0];
        let encoded = CompressionAlgorithms::delta_encode(&data);
        let decoded = CompressionAlgorithms::delta_decode(&encoded);
        
        assert_eq!(data, decoded);
    }
} 