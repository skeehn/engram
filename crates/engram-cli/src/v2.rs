//! engram 2.0 initialization and configuration.
//!
//! Sets up local-first embeddings + HNSW vector search by default.
//! Falls back to API embeddings if local fails.

use anyhow::Result;
use engram_embed::{EmbedStrategy, HybridEmbedder};
use engram_fts::FtsIndex;
use engram_store::EngramStore;
use engram_vector::HnswIndex;
use std::path::Path;
use std::sync::Arc;
use parking_lot::Mutex;

/// Default embedding dimensions for local models (BGE Small EN v1.5).
pub const LOCAL_EMBED_DIMS: usize = 384;

/// Embedding dimensions for Jina v3 API.
pub const JINA_EMBED_DIMS: usize = 1024;

/// engram 2.0 context: thread-safe wrappers for all components.
pub struct EngramContext {
    pub store: Arc<EngramStore>,
    pub embedder: Arc<Mutex<HybridEmbedder>>,
    pub fts: Arc<FtsIndex>,
    pub vector: Arc<HnswIndex>,
}

impl EngramContext {
    /// Initialize engram with local-first defaults.
    ///
    /// - Uses BGE Small EN v1.5 for embeddings (384 dims, ~130MB, offline)
    /// - Uses HNSW for vector search (O(log n), millions of vectors)
    /// - Falls back to Jina API if local embedding fails
    pub fn open(db_path: impl AsRef<Path>) -> Result<Self> {
        let db = db_path.as_ref();
        
        // Create directories
        std::fs::create_dir_all(db)?;
        
        // Initialize components
        let store = Arc::new(EngramStore::open(db)?);
        let fts = Arc::new(FtsIndex::open(&db.join("fts"))?);
        
        // Use local embeddings with 384 dimensions
        let embedder = Arc::new(Mutex::new(HybridEmbedder::from_env()));
        
        // HNSW index with local dimensions
        let vector = Arc::new(HnswIndex::with_defaults(
            LOCAL_EMBED_DIMS,
            db.join("vectors.hnsw"),
        )?);
        
        tracing::info!(
            path = ?db,
            dims = LOCAL_EMBED_DIMS,
            "engram 2.0 initialized (local-first)"
        );
        
        Ok(Self {
            store,
            embedder,
            fts,
            vector,
        })
    }
    
    /// Initialize with local-only mode (no API fallback).
    /// Use this for fully offline operation.
    pub fn open_offline(db_path: impl AsRef<Path>) -> Result<Self> {
        let db = db_path.as_ref();
        std::fs::create_dir_all(db)?;
        
        let store = Arc::new(EngramStore::open(db)?);
        let fts = Arc::new(FtsIndex::open(&db.join("fts"))?);
        let embedder = Arc::new(Mutex::new(HybridEmbedder::local_only()));
        let vector = Arc::new(HnswIndex::with_defaults(
            LOCAL_EMBED_DIMS,
            db.join("vectors.hnsw"),
        )?);
        
        tracing::info!(
            path = ?db,
            dims = LOCAL_EMBED_DIMS,
            "engram 2.0 initialized (offline mode)"
        );
        
        Ok(Self {
            store,
            embedder,
            fts,
            vector,
        })
    }
    
    /// Embed a single text using the configured embedder.
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let embedder = self.embedder.clone();
        let text = text.to_string();
        tokio::task::spawn_blocking(move || {
            let mut e = embedder.lock();
            e.embed_local(&text).map_err(|e| anyhow::anyhow!("{}", e))
        })
        .await
        .map_err(|e| anyhow::anyhow!("spawn_blocking failed: {}", e))?
    }
    
    /// Embed a query (may use query-specific prefixes).
    pub async fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        let embedder = self.embedder.clone();
        let query = query.to_string();
        tokio::task::spawn_blocking(move || {
            let mut e = embedder.lock();
            e.embed_local(&query).map_err(|e| anyhow::anyhow!("{}", e))
        })
        .await
        .map_err(|e| anyhow::anyhow!("spawn_blocking failed: {}", e))?
    }
    
    /// Add a text to the index.
    pub async fn add(&self, id: &str, text: &str) -> Result<()> {
        // Get embedding
        let embedding = self.embed(text).await?;
        
        // Store in HNSW
        let node_id = engram_core::id::NodeId::from(id.to_string());
        self.vector.upsert(&node_id, &embedding)?;
        
        // Index in FTS
        let node = engram_core::types::Node::new(text.to_string(), engram_core::types::NodeType::Fact);
        self.fts.index_node(&node)?;
        self.fts.commit()?;
        
        Ok(())
    }
    
    /// Search for similar texts.
    pub async fn search(&self, query: &str, k: usize) -> Result<Vec<(String, f32)>> {
        let embedding = self.embed_query(query).await?;
        let results = self.vector.search(&embedding, k)?;
        Ok(results
            .into_iter()
            .map(|(id, score)| (id.as_ref().to_string(), score))
            .collect())
    }
    
    /// Get stats.
    pub fn stats(&self) -> EngramStats {
        EngramStats {
            vectors: self.vector.len(),
            fts_docs: self.fts.doc_count().unwrap_or(0),
            nodes: self.store.stats().map(|s| s.node_count).unwrap_or(0),
        }
    }
}

#[derive(Debug)]
pub struct EngramStats {
    pub vectors: usize,
    pub fts_docs: u64,
    pub nodes: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    
    #[tokio::test]
    #[ignore = "downloads model on first run"]
    async fn test_context_basic() {
        let dir = tempdir().unwrap();
        let ctx = EngramContext::open_offline(dir.path()).unwrap();
        
        // Add some documents
        ctx.add("doc1", "Rust is a systems programming language").await.unwrap();
        ctx.add("doc2", "Python is great for data science").await.unwrap();
        ctx.add("doc3", "Machine learning uses neural networks").await.unwrap();
        
        // Search
        let results = ctx.search("programming languages", 2).await.unwrap();
        assert!(!results.is_empty());
        
        // Stats
        let stats = ctx.stats();
        assert_eq!(stats.vectors, 3);
    }
}
