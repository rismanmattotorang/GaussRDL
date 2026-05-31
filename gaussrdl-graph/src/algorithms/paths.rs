//! Path finding algorithms
//! 
//! This module provides implementations of path finding algorithms
//! including shortest paths and all-pairs shortest paths.

use crate::graph::{RelationalGraph, NodeId};
use candle_core::{Device, Tensor, Result as CandleResult};
use std::collections::{HashMap, HashSet, VecDeque, BinaryHeap};
use std::cmp::{Ordering, Reverse};

/// Dijkstra's shortest path algorithm
pub struct DijkstraShortestPath;

impl DijkstraShortestPath {
    pub fn compute(&self, graph: &RelationalGraph, source: &NodeId) -> CandleResult<HashMap<NodeId, (f64, Vec<NodeId>)>> {
        let mut distances = HashMap::new();
        let mut previous = HashMap::new();
        let mut visited = HashSet::new();
        
        // Initialize distances
        for node_id in graph.nodes.keys() {
            distances.insert(node_id.clone(), f64::INFINITY);
        }
        distances.insert(source.clone(), 0.0);
        
        // Priority queue for Dijkstra's algorithm
        let mut queue = BinaryHeap::new();
        queue.push(QueueItem {
            node_id: source.clone(),
            distance: 0.0,
        });
        
        while let Some(QueueItem { node_id, distance }) = queue.pop() {
            if visited.contains(&node_id) {
                continue;
            }
            visited.insert(node_id.clone());
            
            // Find all neighbors of current node
            for edge in &graph.edges {
                let neighbor = if edge.from == node_id {
                    &edge.to
                } else if edge.to == node_id {
                    &edge.from
                } else {
                    continue;
                };
                
                let new_distance = distance + edge.weight as f64;
                if new_distance < distances[neighbor] {
                    distances.insert(neighbor.clone(), new_distance);
                    previous.insert(neighbor.clone(), node_id.clone());
                    
                    queue.push(QueueItem {
                        node_id: neighbor.clone(),
                        distance: new_distance,
                    });
                }
            }
        }
        
        // Reconstruct paths
        let mut paths = HashMap::new();
        for node_id in graph.nodes.keys() {
            if distances[node_id] != f64::INFINITY {
                let path = self.reconstruct_path(&previous, source, node_id);
                paths.insert(node_id.clone(), (distances[node_id], path));
            }
        }
        
        Ok(paths)
    }
    
    fn reconstruct_path(&self, previous: &HashMap<NodeId, NodeId>, source: &NodeId, target: &NodeId) -> Vec<NodeId> {
        let mut path = Vec::new();
        let mut current = target.clone();
        
        while current != *source {
            path.push(current.clone());
            current = previous[&current].clone();
        }
        path.push(source.clone());
        path.reverse();
        path
    }
}

/// Queue item for Dijkstra's algorithm
#[derive(Debug, Clone)]
struct QueueItem {
    node_id: NodeId,
    distance: f64,
}

impl PartialEq for QueueItem {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for QueueItem {}

impl PartialOrd for QueueItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other.distance.partial_cmp(&self.distance)
    }
}

impl Ord for QueueItem {
    fn cmp(&self, other: &Self) -> Ordering {
        other.distance.partial_cmp(&self.distance).unwrap_or(Ordering::Equal)
    }
}

/// All-pairs shortest paths using Floyd-Warshall
pub struct FloydWarshall;

impl FloydWarshall {
    pub fn compute(&self, graph: &RelationalGraph) -> CandleResult<HashMap<(NodeId, NodeId), f64>> {
        let num_nodes = graph.num_nodes();
        let node_mapping: HashMap<NodeId, usize> = graph.nodes.keys()
            .enumerate()
            .map(|(idx, node_id)| (node_id.clone(), idx))
            .collect();
        
        // Initialize distance matrix
        let mut distances = vec![vec![f64::INFINITY; num_nodes]; num_nodes];
        
        // Set diagonal to 0
        for i in 0..num_nodes {
            distances[i][i] = 0.0;
        }
        
        // Set direct edge weights
        for edge in &graph.edges {
            if let (Some(&from_idx), Some(&to_idx)) = (
                node_mapping.get(&edge.from),
                node_mapping.get(&edge.to)
            ) {
                distances[from_idx][to_idx] = edge.weight as f64;
                distances[to_idx][from_idx] = edge.weight as f64; // Undirected graph
            }
        }
        
        // Floyd-Warshall algorithm
        for k in 0..num_nodes {
            for i in 0..num_nodes {
                for j in 0..num_nodes {
                    if distances[i][k] != f64::INFINITY && distances[k][j] != f64::INFINITY {
                        let new_distance = distances[i][k] + distances[k][j];
                        if new_distance < distances[i][j] {
                            distances[i][j] = new_distance;
                        }
                    }
                }
            }
        }
        
        // Convert back to node IDs
        let mut result = HashMap::new();
        let reverse_mapping: HashMap<usize, NodeId> = node_mapping.iter()
            .map(|(node_id, &idx)| (idx, node_id.clone()))
            .collect();
        
        for i in 0..num_nodes {
            for j in 0..num_nodes {
                if distances[i][j] != f64::INFINITY {
                    let from_id = reverse_mapping[&i].clone();
                    let to_id = reverse_mapping[&j].clone();
                    result.insert((from_id, to_id), distances[i][j]);
                }
            }
        }
        
        Ok(result)
    }
}

/// Breadth-first search for unweighted shortest paths
pub struct BFSShortestPath;

impl BFSShortestPath {
    pub fn compute(&self, graph: &RelationalGraph, source: &NodeId) -> CandleResult<HashMap<NodeId, (usize, Vec<NodeId>)>> {
        let mut distances = HashMap::new();
        let mut previous = HashMap::new();
        let mut queue = VecDeque::new();
        
        // Initialize
        for node_id in graph.nodes.keys() {
            distances.insert(node_id.clone(), usize::MAX);
        }
        distances.insert(source.clone(), 0);
        queue.push_back(source.clone());
        
        while let Some(current) = queue.pop_front() {
            let current_dist = distances[&current];
            
            // Find all neighbors of current node
            for edge in &graph.edges {
                let neighbor = if edge.from == current {
                    &edge.to
                } else if edge.to == current {
                    &edge.from
                } else {
                    continue;
                };
                
                if distances[neighbor] == usize::MAX {
                    distances.insert(neighbor.clone(), current_dist + 1);
                    previous.insert(neighbor.clone(), current.clone());
                    queue.push_back(neighbor.clone());
                }
            }
        }
        
        // Reconstruct paths
        let mut paths = HashMap::new();
        for node_id in graph.nodes.keys() {
            if distances[node_id] != usize::MAX {
                let path = self.reconstruct_path(&previous, source, node_id);
                paths.insert(node_id.clone(), (distances[node_id], path));
            }
        }
        
        Ok(paths)
    }
    
    fn reconstruct_path(&self, previous: &HashMap<NodeId, NodeId>, source: &NodeId, target: &NodeId) -> Vec<NodeId> {
        let mut path = Vec::new();
        let mut current = target.clone();
        
        while current != *source {
            path.push(current.clone());
            current = previous[&current].clone();
        }
        path.push(source.clone());
        path.reverse();
        path
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{RelationalGraph, SparseTensor};
    use candle_core::Device;
    
    #[test]
    fn test_dijkstra_shortest_path() {
        let device = Device::Cpu;
        let graph = RelationalGraph::new_empty(device);
        let dijkstra = DijkstraShortestPath;
        let source = NodeId::new(0, 1);
        let result = dijkstra.compute(&graph, &source);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_floyd_warshall() {
        let device = Device::Cpu;
        let graph = RelationalGraph::new_empty(device);
        let floyd = FloydWarshall;
        let result = floyd.compute(&graph);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_bfs_shortest_path() {
        let device = Device::Cpu;
        let graph = RelationalGraph::new_empty(device);
        let bfs = BFSShortestPath;
        let source = NodeId::new(0, 1);
        let result = bfs.compute(&graph, &source);
        assert!(result.is_ok());
    }
} 