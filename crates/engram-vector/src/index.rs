use std::path::{Path, PathBuf};
use std::collections::HashMap;
use parking_lot::RwLock;
use serde::{Serialize, Deserialize};
use engram_core::{
    id::NodeId,
    error::{EngramError, Result},
};

/// Entry stored in the flat index.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Entry {
    node_id: NodeId,
    embedding: Vec<f32>,
}

/// Simple in-memory vector index with cosine similarity search and JSON persistence.
/// Uses a flat linear scan -- correct and fast enough for dev up to ~100k vectors.
pub struct VectorIndex {
    entries: RwLock<Vec<Entry>>,
    /// Map from NodeId -> index in `entries` for O(1) upsert lookup.
    id_map: RwLock<HashMap<String, usize>>,
    dimensions: usize,
    path: PathBuf,
}

impl VectorIndex {
    /// Create or reopen a vector index. If a persisted file exists at `path`, it is loaded.
    pub fn new(dimensions: usize, path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut entries: Vec<Entry> = Vec::new();

        if path.exists() {
            let data = std::fs::read(&path)?;
            entries = serde_json::from_slice(&data)
                .map_err(|e| EngramError::Storage(format!("failed to load vector index: {e}")))?;
        }

        let id_map: HashMap<String, usize> = entries
            .iter()
            .enumerate()
            .map(|(i, e)| (e.node_id.as_ref().to_string(), i))
            .collect();

        Ok(Self {
            entries: RwLock::new(entries),
            id_map: RwLock::new(id_map),
            dimensions,
            path,
        })
    }

    /// Insert or replace the embedding for a node.
    pub fn upsert(&self, node_id: &NodeId, embedding: &[f32]) -> Result<()> {
        if embedding.len() != self.dimensions {
            return Err(EngramError::Index(format!(
                "embedding dimension mismatch: expected {}, got {}",
                self.dimensions,
                embedding.len()
            )));
        }

        let key = node_id.as_ref().to_string();
        let entry = Entry {
            node_id: node_id.clone(),
            embedding: embedding.to_vec(),
        };

        let mut entries = self.entries.write();
        let mut id_map = self.id_map.write();

        if let Some(&idx) = id_map.get(&key) {
            entries[idx] = entry;
        } else {
            let idx = entries.len();
            entries.push(entry);
            id_map.insert(key, idx);
        }

        Ok(())
    }

    /// Remove a node's embedding. The slot is tombstoned (zeroed) and the id_map entry removed.
    /// A compaction pass runs if >25% of entries are tombstoned.
    pub fn remove(&self, node_id: &NodeId) -> Result<()> {
        let key = node_id.as_ref().to_string();
        let mut entries = self.entries.write();
        let mut id_map = self.id_map.write();

        if let Some(idx) = id_map.remove(&key) {
            // Mark as tombstone by setting an empty embedding.
            entries[idx].embedding.clear();
        }

        // Compact if needed.
        let tombstones = entries.iter().filter(|e| e.embedding.is_empty()).count();
        if tombstones > entries.len() / 4 {
            let live: Vec<Entry> = entries.drain(..).filter(|e| !e.embedding.is_empty()).collect();
            *entries = live;
            id_map.clear();
            for (i, e) in entries.iter().enumerate() {
                id_map.insert(e.node_id.as_ref().to_string(), i);
            }
        }

        Ok(())
    }

    /// Return the top-k most similar nodes by cosine similarity.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(NodeId, f32)>> {
        if query.len() != self.dimensions {
            return Err(EngramError::Index(format!(
                "query dimension mismatch: expected {}, got {}",
                self.dimensions,
                query.len()
            )));
        }

        let entries = self.entries.read();
        let query_norm = l2_norm(query);

        let mut scored: Vec<(usize, f32)> = entries
            .iter()
            .enumerate()
            .filter(|(_, e)| !e.embedding.is_empty())
            .map(|(i, e)| {
                let sim = cosine_similarity(query, &e.embedding, query_norm);
                (i, sim)
            })
            .collect();

        // Sort descending by similarity.
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);

        let results = scored
            .into_iter()
            .map(|(i, sim)| (entries[i].node_id.clone(), sim))
            .collect();

        Ok(results)
    }

    /// Persist the index to disk as JSON.
    pub fn save(&self) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let entries = self.entries.read();
        let live: Vec<&Entry> = entries.iter().filter(|e| !e.embedding.is_empty()).collect();
        let data = serde_json::to_vec(&live)?;
        std::fs::write(&self.path, data)?;
        Ok(())
    }

    /// Number of live (non-tombstoned) entries.
    pub fn len(&self) -> usize {
        self.entries.read().iter().filter(|e| !e.embedding.is_empty()).count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn cosine_similarity(a: &[f32], b: &[f32], a_norm: f32) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let b_norm = l2_norm(b);
    if a_norm == 0.0 || b_norm == 0.0 {
        return 0.0;
    }
    dot / (a_norm * b_norm)
}
