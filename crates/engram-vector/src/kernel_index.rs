//! KernelIndex: Production flat-scan index using kernel primitives.
//!
//! Uses AlignedF32Store (cache-line aligned) + BatchScanner (tiled prefetch)
//! for maximum throughput brute-force search.
//!
//! When to use vs HybridIndex:
//! - < 100K vectors: KernelIndex is faster (no quantization overhead)
//! - > 100K vectors: HybridIndex wins (binary coarse → rescore)
//!
//! Performance targets:
//! - 10K vectors @ 384d: < 2ms (5M vectors/sec)
//! - 100K vectors @ 384d: < 15ms
//! - Apple M1/M2: > 1M vectors/sec sustained

use crate::kernel::{AlignedF32Store, MmapVectors};
use engram_core::error::{EngramError, Result};
use engram_core::id::NodeId;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tracing::info;

/// Configuration for KernelIndex.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelConfig {
    /// Vector dimensions.
    pub dims: usize,
    /// Use memory-mapped file storage (persistent, zero-copy).
    /// If false, uses in-memory AlignedF32Store.
    pub mmap: bool,
    /// BatchScanner tile size (vectors per prefetch tile).
    pub tile_size: usize,
    /// Pre-compute and cache norms for fast cosine.
    pub cache_norms: bool,
}

impl KernelConfig {
    pub fn new(dims: usize) -> Self {
        Self {
            dims,
            mmap: true,
            tile_size: 256,
            cache_norms: true,
        }
    }

    /// In-memory only (for tests or ephemeral workspaces).
    pub fn in_memory(dims: usize) -> Self {
        Self {
            dims,
            mmap: false,
            tile_size: 256,
            cache_norms: true,
        }
    }
}

/// Storage backend: either mmap file or heap-allocated aligned store.
enum Storage {
    Mmap(MmapVectors),
    InMemory(AlignedF32Store),
}

impl Storage {
    fn push(&mut self, vector: &[f32]) -> std::io::Result<usize> {
        match self {
            Storage::Mmap(m) => m.push(vector),
            Storage::InMemory(s) => Ok(s.push(vector)),
        }
    }

    fn get(&self, index: usize) -> &[f32] {
        match self {
            Storage::Mmap(m) => m.get(index),
            Storage::InMemory(s) => s.get(index),
        }
    }

    fn len(&self) -> usize {
        match self {
            Storage::Mmap(m) => m.len(),
            Storage::InMemory(s) => s.len(),
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        match self {
            Storage::Mmap(m) => m.flush(),
            Storage::InMemory(_) => Ok(()),
        }
    }

    fn prefetch(&self, index: usize) {
        match self {
            Storage::Mmap(m) => m.prefetch(index),
            Storage::InMemory(s) => s.prefetch(index),
        }
    }
}

/// Production flat-scan index with kernel optimizations.
///
/// Uses cache-line aligned storage + tiled prefetch scanning
/// for throughput that saturates memory bandwidth.
///
/// NOTE: Tiled scanning is inlined rather than delegated to BatchScanner
/// because Storage enum abstracts over mmap vs in-memory, and BatchScanner
/// only accepts AlignedF32Store directly. The tile_size from config is used.
pub struct KernelIndex {
    config: KernelConfig,
    storage: Storage,
    /// Cached L2 norms for fast cosine (avoids recomputing per search).
    norms: RwLock<Vec<f32>>,
    /// NodeId mapping.
    id_mapping: RwLock<Vec<NodeId>>,
    /// Reverse mapping for O(1) lookup.
    reverse_mapping: RwLock<HashMap<String, usize>>,
    /// Base path (for persistence).
    base_path: Option<PathBuf>,
}

impl KernelIndex {
    /// Create a new persistent KernelIndex.
    pub fn new(config: KernelConfig, base_path: impl AsRef<Path>) -> Result<Self> {
        let base_path = base_path.as_ref().to_path_buf();
        std::fs::create_dir_all(&base_path)?;

        let vectors_path = base_path.join("vectors.engram");
        let storage = Storage::Mmap(
            MmapVectors::open(&vectors_path, config.dims)
                .map_err(|e| EngramError::Storage(format!("mmap open failed: {e}")))?,
        );

        // Load existing mappings if present
        let mapping_path = base_path.join("mapping.json");
        let (id_mapping, reverse_mapping) = if mapping_path.exists() {
            let data = std::fs::read(&mapping_path)?;
            let ids: Vec<NodeId> = serde_json::from_slice(&data)
                .map_err(|e| EngramError::Storage(format!("mapping load failed: {e}")))?;
            let reverse: HashMap<String, usize> = ids
                .iter()
                .enumerate()
                .map(|(i, id)| (id.as_ref().to_string(), i))
                .collect();
            (ids, reverse)
        } else {
            (Vec::new(), HashMap::new())
        };

        // Load or recompute norms
        let norms_path = base_path.join("norms.bin");
        let norms = if config.cache_norms && norms_path.exists() {
            let data = std::fs::read(&norms_path)?;
            let n = data.len() / 4;
            let mut v = Vec::with_capacity(n);
            for i in 0..n {
                let bytes: [u8; 4] = data[i * 4..(i + 1) * 4].try_into().unwrap();
                v.push(f32::from_le_bytes(bytes));
            }
            v
        } else {
            // Recompute from storage
            let count = match &storage {
                Storage::Mmap(m) => m.len(),
                Storage::InMemory(s) => s.len(),
            };
            let mut v = Vec::with_capacity(count);
            for i in 0..count {
                let vec = match &storage {
                    Storage::Mmap(m) => m.get(i),
                    Storage::InMemory(s) => s.get(i),
                };
                v.push(l2_norm(vec));
            }
            v
        };

        info!(
            path = ?base_path,
            vectors = id_mapping.len(),
            dims = config.dims,
            "KernelIndex opened"
        );

        Ok(Self {
            config,
            storage,
            norms: RwLock::new(norms),
            id_mapping: RwLock::new(id_mapping),
            reverse_mapping: RwLock::new(reverse_mapping),
            base_path: Some(base_path),
        })
    }

    /// Create an in-memory (non-persistent) index.
    pub fn in_memory(dims: usize) -> Self {
        let config = KernelConfig::in_memory(dims);
        let storage = Storage::InMemory(AlignedF32Store::new(dims, 1024));

        Self {
            config,
            storage,
            norms: RwLock::new(Vec::new()),
            id_mapping: RwLock::new(Vec::new()),
            reverse_mapping: RwLock::new(HashMap::new()),
            base_path: None,
        }
    }

    /// Add a vector. Returns internal index.
    pub fn add(&mut self, node_id: NodeId, embedding: &[f32]) -> Result<usize> {
        if embedding.len() != self.config.dims {
            return Err(EngramError::Index(format!(
                "dimension mismatch: expected {}, got {}",
                self.config.dims,
                embedding.len()
            )));
        }

        // Check for existing
        let key = node_id.as_ref().to_string();
        {
            let reverse = self.reverse_mapping.read();
            if reverse.contains_key(&key) {
                // Update existing
                drop(reverse);
                return self.update(&node_id, embedding);
            }
        }

        let idx = self
            .storage
            .push(embedding)
            .map_err(|e| EngramError::Storage(format!("push failed: {e}")))?;

        // Cache norm
        if self.config.cache_norms {
            self.norms.write().push(l2_norm(embedding));
        }

        self.id_mapping.write().push(node_id);
        self.reverse_mapping.write().insert(key, idx);

        Ok(idx)
    }

    /// Update an existing vector.
    pub fn update(&mut self, node_id: &NodeId, embedding: &[f32]) -> Result<usize> {
        let key = node_id.as_ref().to_string();
        let idx = {
            let reverse = self.reverse_mapping.read();
            *reverse
                .get(&key)
                .ok_or_else(|| EngramError::NodeNotFound(format!("Node {} not found", key)))?
        };

        // For mmap, we need to overwrite in place
        match &mut self.storage {
            Storage::Mmap(m) => {
                m.overwrite(idx, embedding)
                    .map_err(|e| EngramError::Storage(format!("overwrite failed: {e}")))?;
            }
            Storage::InMemory(s) => {
                s.overwrite(idx, embedding);
            }
        }

        // Update cached norm
        if self.config.cache_norms {
            let mut norms = self.norms.write();
            if idx < norms.len() {
                norms[idx] = l2_norm(embedding);
            }
        }

        Ok(idx)
    }

    /// Remove a vector (marks as deleted, doesn't compact).
    pub fn remove(&mut self, node_id: &NodeId) -> Result<bool> {
        let key = node_id.as_ref().to_string();
        let mut reverse = self.reverse_mapping.write();

        if let Some(idx) = reverse.remove(&key) {
            // Zero out the vector (acts as tombstone for search)
            let zeros = vec![0.0f32; self.config.dims];
            match &mut self.storage {
                Storage::Mmap(m) => {
                    let _ = m.overwrite(idx, &zeros);
                }
                Storage::InMemory(s) => {
                    s.overwrite(idx, &zeros);
                }
            }
            // Zero the norm (will be skipped in search)
            if self.config.cache_norms {
                let mut norms = self.norms.write();
                if idx < norms.len() {
                    norms[idx] = 0.0;
                }
            }
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Search for top-k most similar vectors.
    ///
    /// Uses kernel BatchScanner with tiled prefetch for maximum throughput.
    /// If norms are cached, uses the fast cosine path (pre-computed norms).
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(NodeId, f32)>> {
        if query.len() != self.config.dims {
            return Err(EngramError::Index(format!(
                "query dimension mismatch: expected {}, got {}",
                self.config.dims,
                query.len()
            )));
        }

        let n = self.storage.len();
        if n == 0 {
            return Ok(vec![]);
        }

        // Use cached norms for fast cosine if available
        let results = if self.config.cache_norms {
            let norms = self.norms.read();
            self.search_with_norms(query, k, &norms)
        } else {
            self.search_brute(query, k)
        };

        // Map internal IDs to NodeIds, skip tombstones
        let id_mapping = self.id_mapping.read();
        let reverse = self.reverse_mapping.read();

        let mapped: Vec<(NodeId, f32)> = results
            .into_iter()
            .filter_map(|(idx, score)| {
                let idx = idx as usize;
                if idx < id_mapping.len() {
                    let node_id = &id_mapping[idx];
                    // Skip if removed (not in reverse mapping)
                    if reverse.contains_key(node_id.as_ref()) {
                        return Some((node_id.clone(), score));
                    }
                }
                None
            })
            .collect();

        Ok(mapped)
    }

    /// Fast search with pre-computed norms.
    fn search_with_norms(&self, query: &[f32], k: usize, norms: &[f32]) -> Vec<(u32, f32)> {
        use crate::kernel::SearchArena;

        let n = self.storage.len();
        let query_norm = l2_norm(query);
        let mut arena = SearchArena::new(k);
        let lookahead = 4;
        let tile_size = self.config.tile_size;

        for tile_start in (0..n).step_by(tile_size) {
            let tile_end = (tile_start + tile_size).min(n);

            for i in tile_start..tile_end {
                // Skip tombstones (zero norm)
                if i < norms.len() && norms[i] == 0.0 {
                    continue;
                }

                // Prefetch
                if i + lookahead < tile_end {
                    self.storage.prefetch(i + lookahead);
                }

                let vec = self.storage.get(i);
                let score = fast_cosine(query, vec, query_norm, norms.get(i).copied().unwrap_or(0.0));
                arena.try_insert(i as u32, score);
            }
        }

        arena.drain_sorted().to_vec()
    }

    /// Brute-force search without cached norms.
    fn search_brute(&self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        use crate::kernel::SearchArena;

        let n = self.storage.len();
        let mut arena = SearchArena::new(k);
        let lookahead = 4;
        let tile_size = self.config.tile_size;

        for tile_start in (0..n).step_by(tile_size) {
            let tile_end = (tile_start + tile_size).min(n);

            for i in tile_start..tile_end {
                if i + lookahead < tile_end {
                    self.storage.prefetch(i + lookahead);
                }

                let vec = self.storage.get(i);
                let score = full_cosine(query, vec);
                arena.try_insert(i as u32, score);
            }
        }

        arena.drain_sorted().to_vec()
    }

    /// Batch search: multiple queries at once.
    pub fn search_batch(
        &self,
        queries: &[Vec<f32>],
        k: usize,
    ) -> Result<Vec<Vec<(NodeId, f32)>>> {
        queries.iter().map(|q| self.search(q, k)).collect()
    }

    /// Save index to disk.
    pub fn save(&mut self) -> Result<()> {
        self.storage
            .flush()
            .map_err(|e| EngramError::Storage(format!("flush failed: {e}")))?;

        if let Some(ref base_path) = self.base_path {
            // Save mappings
            let mapping_path = base_path.join("mapping.json");
            let id_mapping = self.id_mapping.read();
            let data = serde_json::to_vec(&*id_mapping)?;
            std::fs::write(&mapping_path, data)?;

            // Save norms
            if self.config.cache_norms {
                let norms_path = base_path.join("norms.bin");
                let norms = self.norms.read();
                let mut bytes = Vec::with_capacity(norms.len() * 4);
                for &n in norms.iter() {
                    bytes.extend_from_slice(&n.to_le_bytes());
                }
                std::fs::write(&norms_path, bytes)?;
            }

            // Save config
            let config_path = base_path.join("kernel_config.json");
            std::fs::write(&config_path, serde_json::to_vec_pretty(&self.config)?)?;

            info!(
                path = ?base_path,
                vectors = id_mapping.len(),
                "KernelIndex saved"
            );
        }

        Ok(())
    }

    /// Number of live vectors.
    pub fn len(&self) -> usize {
        self.reverse_mapping.read().len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Total vectors including tombstones.
    pub fn capacity(&self) -> usize {
        self.storage.len()
    }

    /// Get dimensions.
    pub fn dims(&self) -> usize {
        self.config.dims
    }

    /// Stats.
    pub fn stats(&self) -> KernelIndexStats {
        let n = self.storage.len();
        let live = self.len();
        let bytes_per_vec = self.config.dims * 4;
        let vector_bytes = n * bytes_per_vec;
        let norm_bytes = if self.config.cache_norms { n * 4 } else { 0 };

        KernelIndexStats {
            live_vectors: live,
            total_slots: n,
            dims: self.config.dims,
            vector_bytes,
            norm_bytes,
            total_bytes: vector_bytes + norm_bytes,
            mmap: self.config.mmap,
        }
    }
}

/// Stats for KernelIndex.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelIndexStats {
    pub live_vectors: usize,
    pub total_slots: usize,
    pub dims: usize,
    pub vector_bytes: usize,
    pub norm_bytes: usize,
    pub total_bytes: usize,
    pub mmap: bool,
}

impl std::fmt::Display for KernelIndexStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "KernelIndex: {}/{} vectors ({}d), {:.2}MB vectors + {:.2}MB norms = {:.2}MB total{}",
            self.live_vectors,
            self.total_slots,
            self.dims,
            self.vector_bytes as f64 / 1024.0 / 1024.0,
            self.norm_bytes as f64 / 1024.0 / 1024.0,
            self.total_bytes as f64 / 1024.0 / 1024.0,
            if self.mmap { " [mmap]" } else { " [heap]" }
        )
    }
}

/// Fast cosine with pre-computed norms.
#[inline]
fn fast_cosine(a: &[f32], b: &[f32], a_norm: f32, b_norm: f32) -> f32 {
    let denom = a_norm * b_norm;
    if denom < 1e-10 {
        return 0.0;
    }

    let mut dot = 0.0f32;
    let chunks = a.len() / 8;
    let remainder = a.len() % 8;

    for i in 0..chunks {
        let base = i * 8;
        for j in 0..8 {
            dot += unsafe { *a.get_unchecked(base + j) * *b.get_unchecked(base + j) };
        }
    }

    let base = chunks * 8;
    for j in 0..remainder {
        dot += unsafe { *a.get_unchecked(base + j) * *b.get_unchecked(base + j) };
    }

    dot / denom
}

/// Full cosine similarity (computes both norms).
#[inline]
fn full_cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    let denom = (norm_a * norm_b).sqrt();
    if denom < 1e-10 {
        0.0
    } else {
        dot / denom
    }
}

/// L2 norm.
#[inline]
fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_kernel_index_basic() {
        let mut index = KernelIndex::in_memory(4);

        let id1 = NodeId::from("a".to_string());
        let id2 = NodeId::from("b".to_string());
        let id3 = NodeId::from("c".to_string());

        index.add(id1.clone(), &[1.0, 0.0, 0.0, 0.0]).unwrap();
        index.add(id2.clone(), &[0.0, 1.0, 0.0, 0.0]).unwrap();
        index.add(id3.clone(), &[0.9, 0.1, 0.0, 0.0]).unwrap();

        assert_eq!(index.len(), 3);

        let results = index.search(&[1.0, 0.0, 0.0, 0.0], 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, id1);
        assert!(results[0].1 > 0.99);
    }

    #[test]
    fn test_kernel_index_persistence() {
        let dir = tempdir().unwrap();
        let config = KernelConfig::new(4);

        // Create and populate
        {
            let mut index = KernelIndex::new(config.clone(), dir.path()).unwrap();
            let id = NodeId::from("persistent".to_string());
            index.add(id, &[1.0, 2.0, 3.0, 4.0]).unwrap();
            index.save().unwrap();
        }

        // Reload
        {
            let index = KernelIndex::new(config, dir.path()).unwrap();
            assert_eq!(index.len(), 1);

            let results = index.search(&[1.0, 2.0, 3.0, 4.0], 1).unwrap();
            assert_eq!(results.len(), 1);
            assert!(results[0].1 > 0.99);
        }
    }

    #[test]
    fn test_kernel_index_update() {
        let mut index = KernelIndex::in_memory(4);

        let id = NodeId::from("x".to_string());
        index.add(id.clone(), &[1.0, 0.0, 0.0, 0.0]).unwrap();

        // Update to orthogonal direction
        index.update(&id, &[0.0, 1.0, 0.0, 0.0]).unwrap();

        let results = index.search(&[0.0, 1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(results[0].0, id);
        assert!(results[0].1 > 0.99);

        // Old direction should score low
        let results = index.search(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
        assert!(results[0].1 < 0.01);
    }

    #[test]
    fn test_kernel_index_remove() {
        let mut index = KernelIndex::in_memory(4);

        let id1 = NodeId::from("keep".to_string());
        let id2 = NodeId::from("remove".to_string());

        index.add(id1.clone(), &[1.0, 0.0, 0.0, 0.0]).unwrap();
        index.add(id2.clone(), &[0.9, 0.1, 0.0, 0.0]).unwrap();

        assert_eq!(index.len(), 2);
        index.remove(&id2).unwrap();
        assert_eq!(index.len(), 1);

        let results = index.search(&[1.0, 0.0, 0.0, 0.0], 5).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, id1);
    }

    #[test]
    fn test_kernel_index_throughput() {
        let dims = 384;
        let n = 10_000;
        let mut index = KernelIndex::in_memory(dims);

        // Fill
        for i in 0..n {
            let id = NodeId::from(format!("node{}", i));
            let vec: Vec<f32> = (0..dims)
                .map(|d| ((i * 7 + d * 13) % 1000) as f32 / 1000.0)
                .collect();
            index.add(id, &vec).unwrap();
        }

        let query: Vec<f32> = (0..dims).map(|d| (d % 100) as f32 / 100.0).collect();

        let start = std::time::Instant::now();
        let iterations = 100;
        for _ in 0..iterations {
            let _ = index.search(&query, 10).unwrap();
        }
        let elapsed = start.elapsed();

        let total_vectors = n * iterations;
        let vecs_per_sec = total_vectors as f64 / elapsed.as_secs_f64();

        eprintln!(
            "KernelIndex throughput: {:.0} vectors/sec ({} dims, {} vecs, {:.1}ms total)",
            vecs_per_sec,
            dims,
            n,
            elapsed.as_millis()
        );

        // Must achieve at least 500K vectors/sec
        assert!(
            vecs_per_sec > 500_000.0,
            "Too slow: {:.0} vectors/sec",
            vecs_per_sec
        );
    }

    #[test]
    fn test_kernel_index_stats() {
        let mut index = KernelIndex::in_memory(384);

        for i in 0..10 {
            let id = NodeId::from(format!("n{}", i));
            let vec = vec![i as f32 / 10.0; 384];
            index.add(id, &vec).unwrap();
        }

        let stats = index.stats();
        assert_eq!(stats.live_vectors, 10);
        assert_eq!(stats.dims, 384);
        assert_eq!(stats.vector_bytes, 10 * 384 * 4); // 15,360
        assert_eq!(stats.norm_bytes, 10 * 4); // 40
    }
}
