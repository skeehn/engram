//! Hybrid embedder: tries local first, falls back to API.
//!
//! This is the default embedder for engram 2.0:
//! - Local: Fast, free, offline, no rate limits (~3ms/embed)
//! - API: Higher quality (Jina v3), but requires API key + network

use crate::{
    jina::JinaClient,
    local::{LocalEmbedder, LocalModel},
    types::{EmbedConfig, ReaderResponse},
};
use engram_core::{error::Result, types::Node};
use std::sync::Arc;

/// Strategy for embedding: local-first, API-first, or local-only
#[derive(Debug, Clone, Copy, Default)]
pub enum EmbedStrategy {
    /// Try local first, fall back to API on error
    #[default]
    LocalFirst,
    /// Try API first, fall back to local on error  
    ApiFirst,
    /// Local only, no API fallback
    LocalOnly,
    /// API only, no local fallback
    ApiOnly,
}

pub struct HybridEmbedder {
    local: Option<Arc<LocalEmbedder>>,
    jina: Option<Arc<JinaClient>>,
    strategy: EmbedStrategy,
}

impl HybridEmbedder {
    /// Create a new hybrid embedder with specified strategy.
    /// Local embedder is lazily initialized on first use if strategy allows local.
    pub fn new(config: EmbedConfig, strategy: EmbedStrategy) -> Self {
        let jina = match strategy {
            EmbedStrategy::LocalOnly => None,
            _ => Some(Arc::new(JinaClient::new(config))),
        };
        
        // Local is None initially; we'll init lazily to avoid slow startup
        Self {
            local: None,
            jina,
            strategy,
        }
    }
    
    /// Create with local-first strategy (default for engram 2.0)
    pub fn local_first(config: EmbedConfig) -> Self {
        Self::new(config, EmbedStrategy::LocalFirst)
    }
    
    /// Create with local-only (no API calls, fully offline)
    pub fn local_only() -> Self {
        Self {
            local: None,
            jina: None,
            strategy: EmbedStrategy::LocalOnly,
        }
    }
    
    /// Create from environment, defaulting to local-first
    pub fn from_env() -> Self {
        Self::local_first(EmbedConfig::from_env())
    }
    
    /// Initialize local embedder if not already done.
    /// Called lazily on first local embed request.
    fn ensure_local(&mut self) -> Result<&Arc<LocalEmbedder>> {
        if self.local.is_none() {
            tracing::info!("Initializing local embedder (BGE Small EN v1.5)...");
            let embedder = LocalEmbedder::new(LocalModel::BgeSmallEnV15)?;
            self.local = Some(Arc::new(embedder));
            tracing::info!("Local embedder ready");
        }
        Ok(self.local.as_ref().unwrap())
    }
    
    /// Embed a single text using the configured strategy.
    pub async fn embed_one(&mut self, text: &str) -> Result<Vec<f32>> {
        match self.strategy {
            EmbedStrategy::LocalFirst | EmbedStrategy::LocalOnly => {
                // Try local first
                match self.embed_local(text) {
                    Ok(v) => Ok(v),
                    Err(e) if matches!(self.strategy, EmbedStrategy::LocalFirst) => {
                        tracing::debug!("Local embed failed ({}), trying API", e);
                        self.embed_api(text).await
                    }
                    Err(e) => Err(e),
                }
            }
            EmbedStrategy::ApiFirst => {
                match self.embed_api(text).await {
                    Ok(v) => Ok(v),
                    Err(e) => {
                        tracing::debug!("API embed failed ({}), trying local", e);
                        self.embed_local(text)
                    }
                }
            }
            EmbedStrategy::ApiOnly => self.embed_api(text).await,
        }
    }
    
    /// Embed multiple texts.
    pub async fn embed_many(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        match self.strategy {
            EmbedStrategy::LocalFirst | EmbedStrategy::LocalOnly => {
                match self.embed_local_batch(texts) {
                    Ok(v) => Ok(v),
                    Err(e) if matches!(self.strategy, EmbedStrategy::LocalFirst) => {
                        tracing::debug!("Local batch embed failed ({}), trying API", e);
                        self.embed_api_batch(texts).await
                    }
                    Err(e) => Err(e),
                }
            }
            EmbedStrategy::ApiFirst => {
                match self.embed_api_batch(texts).await {
                    Ok(v) => Ok(v),
                    Err(e) => {
                        tracing::debug!("API batch embed failed ({}), trying local", e);
                        self.embed_local_batch(texts)
                    }
                }
            }
            EmbedStrategy::ApiOnly => self.embed_api_batch(texts).await,
        }
    }
    
    /// Embed a query (may use different model/prefix for asymmetric retrieval).
    pub async fn embed_query(&mut self, query: &str) -> Result<Vec<f32>> {
        // For local models like BGE, query prefix is already handled
        // For Jina, embed_query uses "retrieval.query" task
        self.embed_one(query).await
    }
    
    /// Embed a node's body, storing result in node.embedding.
    pub async fn embed_node(&mut self, node: &mut Node) -> Result<()> {
        const MAX_EMBED_CHARS: usize = 6000;
        let text = if node.body.len() > MAX_EMBED_CHARS {
            let mut end = MAX_EMBED_CHARS;
            while end > 0 && !node.body.is_char_boundary(end) {
                end -= 1;
            }
            &node.body[..end]
        } else {
            &node.body
        };
        let embedding = self.embed_one(text).await?;
        node.embedding = Some(embedding);
        Ok(())
    }
    
    /// Rerank candidates by relevance to query.
    /// Falls back to local cosine similarity if API unavailable.
    pub async fn rerank_nodes(
        &mut self,
        query: &str,
        candidates: &[(String, String)],
    ) -> Result<Vec<(String, f32)>> {
        // Try API reranker first (Jina reranker is high quality)
        if let Some(ref jina) = self.jina {
            let texts: Vec<String> = candidates.iter().map(|(_, t)| t.clone()).collect();
            match jina.rerank(query, &texts, None).await {
                Ok(ranked) => {
                    return Ok(ranked
                        .into_iter()
                        .map(|(idx, score)| (candidates[idx].0.clone(), score))
                        .collect());
                }
                Err(e) => {
                    tracing::debug!("API rerank failed ({}), using local cosine", e);
                }
            }
        }
        
        // Fallback: local cosine similarity reranking
        self.rerank_local(query, candidates)
    }
    
    /// Read URL content via Jina Reader API.
    pub async fn read_url(&self, url: &str) -> Result<ReaderResponse> {
        match &self.jina {
            Some(jina) => jina.read_url(url).await,
            None => Err(engram_core::error::EngramError::Embedding(
                "URL reading requires API (local-only mode)".into(),
            )),
        }
    }
    
    /// Get embedding dimensions (depends on current model).
    pub fn dimensions(&self) -> usize {
        // BGE Small EN v1.5 = 384 dims
        // Jina v3 = 1024 dims
        match self.strategy {
            EmbedStrategy::LocalFirst | EmbedStrategy::LocalOnly => 384,
            EmbedStrategy::ApiFirst | EmbedStrategy::ApiOnly => 1024,
        }
    }
    
    // --- Private helpers ---
    
    fn embed_local(&mut self, text: &str) -> Result<Vec<f32>> {
        let local = self.ensure_local()?;
        local.embed_one(text)
    }
    
    fn embed_local_batch(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let local = self.ensure_local()?;
        local.embed_many(texts)
    }
    
    async fn embed_api(&self, text: &str) -> Result<Vec<f32>> {
        match &self.jina {
            Some(jina) => {
                let results = jina.embed_passages(&[text.to_string()]).await?;
                results.into_iter().next().ok_or_else(|| {
                    engram_core::error::EngramError::Embedding("empty API result".into())
                })
            }
            None => Err(engram_core::error::EngramError::Embedding(
                "API not available (local-only mode)".into(),
            )),
        }
    }
    
    async fn embed_api_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        match &self.jina {
            Some(jina) => jina.embed_passages(texts).await,
            None => Err(engram_core::error::EngramError::Embedding(
                "API not available (local-only mode)".into(),
            )),
        }
    }
    
    fn rerank_local(
        &mut self,
        query: &str,
        candidates: &[(String, String)],
    ) -> Result<Vec<(String, f32)>> {
        let local = self.ensure_local()?;
        let query_emb = local.embed_one(query)?;
        
        let texts: Vec<String> = candidates.iter().map(|(_, t)| t.clone()).collect();
        let doc_embs = local.embed_many(&texts)?;
        
        let mut scored: Vec<(String, f32)> = candidates
            .iter()
            .zip(doc_embs.iter())
            .map(|((id, _), emb)| {
                let score = cosine_similarity(&query_emb, emb);
                (id.clone(), score)
            })
            .collect();
        
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scored)
    }
}

/// Cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);
        
        let c = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &c).abs() < 0.001);
    }
}
