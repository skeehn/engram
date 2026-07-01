//! HNSW (Hierarchical Navigable Small World) vector index via usearch.
//!
//! Replaces the flat O(n) scan with O(log n) approximate nearest neighbor search.
//! Scales to millions of vectors while maintaining sub-10ms query times.

use engram_core::{
    error::{EngramError, Result},
    id::NodeId,
};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};
use tracing::{debug, info};

/// HNSW index configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswConfig {
    /// Number of bi-directional links per node (higher = more memory, better recall)
    pub connectivity: usize,
    /// Expansion factor during construction (higher = slower build, better quality)
    pub expansion_add: usize,
    /// Expansion factor during search (higher = slower search, better recall)
    pub expansion_search: usize,
    /// Quantization type: F32 (precise), I8 (4x smaller, <1% recall loss), B1 (32x smaller)
    pub quantization: QuantizationType,
}

/// Quantization options for storage/speed tradeoffs.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub enum QuantizationType {
    /// Full precision (1536 bytes for 384d). Best recall, most memory.
    #[default]
    F32,
    /// 8-bit scalar quantization (384 bytes for 384d). <1% recall loss, 4x smaller.
    I8,
    /// Binary quantization (48 bytes for 384d). 5-15% recall loss, 32x smaller.
    /// Best for coarse filtering + rerank.
    B1,
}

impl QuantizationType {
    fn to_scalar_kind(self) -> ScalarKind {
        match self {
            QuantizationType::F32 => ScalarKind::F32,
            QuantizationType::I8 => ScalarKind::I8,
            QuantizationType::B1 => ScalarKind::B1,
        }
    }
    
    /// Bytes per vector for given dimensions.
    pub fn bytes_per_vector(self, dims: usize) -> usize {
        match self {
            QuantizationType::F32 => dims * 4,
            QuantizationType::I8 => dims,
            QuantizationType::B1 => (dims + 7) / 8,
        }
    }
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            connectivity: 16,        // M parameter - good balance
            expansion_add: 128,      // ef_construction - quality during build
            expansion_search: 64,    // ef_search - accuracy vs speed
            quantization: QuantizationType::I8, // Default to i8 for 4x compression
        }
    }
}

/// Mapping between NodeId and internal usearch u64 keys.
#[derive(Debug, Default, Serialize, Deserialize)]
struct IdMapping {
    /// NodeId -> internal key
    node_to_key: HashMap<String, u64>,
    /// Internal key -> NodeId
    key_to_node: HashMap<u64, String>,
    /// Next available key
    next_key: u64,
}

impl IdMapping {
    fn get_or_insert(&mut self, node_id: &NodeId) -> u64 {
        let key_str = node_id.as_ref().to_string();
        if let Some(&key) = self.node_to_key.get(&key_str) {
            key
        } else {
            let key = self.next_key;
            self.next_key += 1;
            self.node_to_key.insert(key_str.clone(), key);
            self.key_to_node.insert(key, key_str);
            key
        }
    }

    fn get_key(&self, node_id: &NodeId) -> Option<u64> {
        self.node_to_key.get(node_id.as_ref()).copied()
    }

    fn get_node(&self, key: u64) -> Option<&str> {
        self.key_to_node.get(&key).map(|s| s.as_str())
    }

    fn remove(&mut self, node_id: &NodeId) -> Option<u64> {
        let key_str = node_id.as_ref().to_string();
        if let Some(key) = self.node_to_key.remove(&key_str) {
            self.key_to_node.remove(&key);
            Some(key)
        } else {
            None
        }
    }
}

/// HNSW vector index with persistent storage.
pub struct HnswIndex {
    index: Index,
    mapping: RwLock<IdMapping>,
    dimensions: usize,
    index_path: PathBuf,
    mapping_path: PathBuf,
    config: HnswConfig,
}

impl HnswIndex {
    /// Create or reopen an HNSW index.
    pub fn new(dimensions: usize, path: impl AsRef<Path>, config: HnswConfig) -> Result<Self> {
        let index_path = path.as_ref().to_path_buf();
        let mapping_path = index_path.with_extension("mapping.json");

        // Configure index options
        let options = IndexOptions {
            dimensions,
            metric: MetricKind::Cos,  // Cosine similarity
            quantization: config.quantization.to_scalar_kind(),
            connectivity: config.connectivity,
            expansion_add: config.expansion_add,
            expansion_search: config.expansion_search,
            multi: false,  // Single vector per key
        };

        let index = Index::new(&options)
            .map_err(|e| EngramError::Index(format!("failed to create index: {e}")))?;

        // Load existing index if present
        let mut mapping = IdMapping::default();
        if index_path.exists() {
            info!(path = ?index_path, "Loading existing HNSW index");
            index
                .load(index_path.to_str().ok_or_else(|| EngramError::Index("invalid path".into()))?)
                .map_err(|e| EngramError::Index(format!("failed to load index: {e}")))?;
            
            if mapping_path.exists() {
                let data = std::fs::read(&mapping_path)?;
                mapping = serde_json::from_slice(&data)
                    .map_err(|e| EngramError::Storage(format!("failed to load mapping: {e}")))?;
            }
            info!(vectors = index.size(), "Index loaded");
        } else {
            // Reserve capacity only for new index
            index.reserve(10_000)
                .map_err(|e| EngramError::Index(format!("failed to reserve capacity: {e}")))?;
        }

        Ok(Self {
            index,
            mapping: RwLock::new(mapping),
            dimensions,
            index_path,
            mapping_path,
            config,
        })
    }

    /// Create with default configuration.
    pub fn with_defaults(dimensions: usize, path: impl AsRef<Path>) -> Result<Self> {
        Self::new(dimensions, path, HnswConfig::default())
    }

    /// Get index dimensions.
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Get number of indexed vectors.
    pub fn len(&self) -> usize {
        self.index.size()
    }

    /// Check if index is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Insert or update a vector.
    pub fn upsert(&self, node_id: &NodeId, embedding: &[f32]) -> Result<()> {
        if embedding.len() != self.dimensions {
            return Err(EngramError::Index(format!(
                "dimension mismatch: expected {}, got {}",
                self.dimensions,
                embedding.len()
            )));
        }

        let mut mapping = self.mapping.write();
        let key = mapping.get_or_insert(node_id);

        // Ensure capacity (usearch needs reserve before add)
        let current_size = self.index.size();
        let current_capacity = self.index.capacity();
        if current_size >= current_capacity {
            let new_capacity = std::cmp::max(current_capacity * 2, 1000);
            self.index.reserve(new_capacity)
                .map_err(|e| EngramError::Index(format!("failed to expand capacity: {e}")))?;
            debug!(old = current_capacity, new = new_capacity, "Index capacity expanded");
        }

        // Remove old entry if exists (usearch doesn't have update)
        let _ = self.index.remove(key);

        self.index.add(key, embedding)
            .map_err(|e| EngramError::Index(format!("failed to add vector: {e}")))?;

        debug!(node_id = %node_id.as_ref(), key = key, "Vector upserted");
        Ok(())
    }

    /// Remove a vector by NodeId.
    pub fn remove(&self, node_id: &NodeId) -> Result<bool> {
        let mut mapping = self.mapping.write();
        if let Some(key) = mapping.remove(node_id) {
            self.index.remove(key)
                .map_err(|e| EngramError::Index(format!("failed to remove vector: {e}")))?;
            debug!(node_id = %node_id.as_ref(), key = key, "Vector removed");
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Search for the k most similar vectors.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(NodeId, f32)>> {
        if query.len() != self.dimensions {
            return Err(EngramError::Index(format!(
                "query dimension mismatch: expected {}, got {}",
                self.dimensions,
                query.len()
            )));
        }

        if self.is_empty() {
            return Ok(vec![]);
        }

        let results = self.index.search(query, k)
            .map_err(|e| EngramError::Index(format!("search failed: {e}")))?;

        let mapping = self.mapping.read();
        let mut output = Vec::with_capacity(results.keys.len());

        for (key, distance) in results.keys.iter().zip(results.distances.iter()) {
            if let Some(node_str) = mapping.get_node(*key) {
                // usearch returns distance, convert to similarity for cosine
                let similarity = 1.0 - distance;
                output.push((NodeId::from(node_str.to_string()), similarity));
            }
        }

        Ok(output)
    }

    /// Persist the index to disk.
    pub fn save(&self) -> Result<()> {
        if let Some(parent) = self.index_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        self.index
            .save(self.index_path.to_str().ok_or_else(|| EngramError::Index("invalid path".into()))?)
            .map_err(|e| EngramError::Index(format!("failed to save index: {e}")))?;

        let mapping = self.mapping.read();
        let data = serde_json::to_vec(&*mapping)?;
        std::fs::write(&self.mapping_path, data)?;

        info!(
            path = ?self.index_path,
            vectors = self.len(),
            "Index saved"
        );
        Ok(())
    }

    /// Get the configuration used for this index.
    pub fn config(&self) -> &HnswConfig {
        &self.config
    }

    /// Check if a node exists in the index.
    pub fn contains(&self, node_id: &NodeId) -> bool {
        self.mapping.read().get_key(node_id).is_some()
    }
}

impl Drop for HnswIndex {
    fn drop(&mut self) {
        if let Err(e) = self.save() {
            tracing::error!(error = %e, "Failed to save index on drop");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_hnsw_basic() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.hnsw");
        
        let index = HnswIndex::with_defaults(3, &path).unwrap();
        
        // Insert some vectors
        let id1 = NodeId::from("node1".to_string());
        let id2 = NodeId::from("node2".to_string());
        
        index.upsert(&id1, &[1.0, 0.0, 0.0]).unwrap();
        index.upsert(&id2, &[0.0, 1.0, 0.0]).unwrap();
        
        assert_eq!(index.len(), 2);
        
        // Search
        let results = index.search(&[1.0, 0.0, 0.0], 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0.as_ref(), "node1");
        assert!(results[0].1 > 0.99); // Should be ~1.0 (identical)
    }

    #[test]
    fn test_hnsw_persistence() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.hnsw");
        
        {
            let index = HnswIndex::with_defaults(3, &path).unwrap();
            let id = NodeId::from("persistent".to_string());
            index.upsert(&id, &[1.0, 2.0, 3.0]).unwrap();
            index.save().unwrap();
        }
        
        // Reopen
        {
            let index = HnswIndex::with_defaults(3, &path).unwrap();
            assert_eq!(index.len(), 1);
            assert!(index.contains(&NodeId::from("persistent".to_string())));
        }
    }
}
