//! Local ONNX embedding via fastembed.
//!
//! Default model: bge-small-en-v1.5 (384 dimensions, ~130MB, MIT license)
//! Alternative: all-MiniLM-L6-v2 (384 dimensions, ~80MB)
//!
//! This removes the Jina API dependency and enables fully offline operation.

use engram_core::error::{EngramError, Result};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use parking_lot::Mutex;
use std::sync::Arc;
use tracing::{debug, info};

/// Supported embedding models for local inference.
#[derive(Debug, Clone, Copy, Default)]
pub enum LocalModel {
    /// BGE Small EN v1.5 - 384 dimensions, good quality/speed balance (default)
    #[default]
    BgeSmallEnV15,
    /// All MiniLM L6 v2 - 384 dimensions, fastest, most tested
    AllMiniLmL6V2,
    /// BGE Base EN v1.5 - 768 dimensions, higher quality
    BgeBaseEnV15,
    /// All MiniLM L12 v2 - 384 dimensions, better than L6
    AllMiniLmL12V2,
}

impl LocalModel {
    fn to_fastembed(&self) -> EmbeddingModel {
        match self {
            LocalModel::BgeSmallEnV15 => EmbeddingModel::BGESmallENV15,
            LocalModel::AllMiniLmL6V2 => EmbeddingModel::AllMiniLML6V2,
            LocalModel::BgeBaseEnV15 => EmbeddingModel::BGEBaseENV15,
            LocalModel::AllMiniLmL12V2 => EmbeddingModel::AllMiniLML12V2,
        }
    }

    /// Returns the embedding dimensions for this model.
    pub fn dimensions(&self) -> usize {
        match self {
            LocalModel::BgeSmallEnV15 => 384,
            LocalModel::AllMiniLmL6V2 => 384,
            LocalModel::BgeBaseEnV15 => 768,
            LocalModel::AllMiniLmL12V2 => 384,
        }
    }

    /// Returns a human-readable name for the model.
    pub fn name(&self) -> &'static str {
        match self {
            LocalModel::BgeSmallEnV15 => "bge-small-en-v1.5",
            LocalModel::AllMiniLmL6V2 => "all-MiniLM-L6-v2",
            LocalModel::BgeBaseEnV15 => "bge-base-en-v1.5",
            LocalModel::AllMiniLmL12V2 => "all-MiniLM-L12-v2",
        }
    }
}

/// Local embedding client using ONNX models via fastembed.
/// Uses interior mutability (Mutex) because fastembed's embed() requires &mut self.
pub struct LocalEmbedder {
    model: Mutex<TextEmbedding>,
    model_info: LocalModel,
}

impl LocalEmbedder {
    /// Create a new local embedder with the specified model.
    ///
    /// The model will be downloaded automatically on first use (~80-130MB).
    pub fn new(model: LocalModel) -> Result<Self> {
        info!(model = model.name(), "Initializing local embedder");

        let embedding_model = TextEmbedding::try_new(
            InitOptions::new(model.to_fastembed())
                .with_show_download_progress(true)
                .with_execution_providers(vec![]), // Use default (CPU)
        )
        .map_err(|e| EngramError::Embedding(format!("failed to load model: {e}")))?;

        info!(
            model = model.name(),
            dimensions = model.dimensions(),
            "Local embedder initialized"
        );

        Ok(Self {
            model: Mutex::new(embedding_model),
            model_info: model,
        })
    }

    /// Create with the default model (BGE Small EN v1.5).
    pub fn default_model() -> Result<Self> {
        Self::new(LocalModel::default())
    }

    /// Get the embedding dimensions for this model.
    pub fn dimensions(&self) -> usize {
        self.model_info.dimensions()
    }

    /// Get the model name.
    pub fn model_name(&self) -> &'static str {
        self.model_info.name()
    }

    /// Embed a single text.
    pub fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        let results = self.embed_many(&[text.to_string()])?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| EngramError::Embedding("empty result".into()))
    }

    /// Embed multiple texts in a batch.
    pub fn embed_many(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        debug!(count = texts.len(), "Embedding batch");

        let mut model = self.model.lock();
        model
            .embed(texts.to_vec(), None)
            .map_err(|e| EngramError::Embedding(format!("embedding failed: {e}")))
    }

    /// Embed a query (for retrieval - some models have different query vs passage encoding).
    pub fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        // BGE models work better with "query: " prefix for retrieval
        let prefixed = if matches!(
            self.model_info,
            LocalModel::BgeSmallEnV15 | LocalModel::BgeBaseEnV15
        ) {
            format!("query: {}", query)
        } else {
            query.to_string()
        };

        self.embed_one(&prefixed)
    }

    /// Embed a passage (for storage - some models have different query vs passage encoding).
    pub fn embed_passage(&self, passage: &str) -> Result<Vec<f32>> {
        // BGE models work better with "passage: " prefix for passages
        let prefixed = if matches!(
            self.model_info,
            LocalModel::BgeSmallEnV15 | LocalModel::BgeBaseEnV15
        ) {
            format!("passage: {}", passage)
        } else {
            passage.to_string()
        };

        self.embed_one(&prefixed)
    }
}

/// Thread-safe wrapper for LocalEmbedder.
pub struct SharedLocalEmbedder(Arc<LocalEmbedder>);

impl SharedLocalEmbedder {
    pub fn new(model: LocalModel) -> Result<Self> {
        Ok(Self(Arc::new(LocalEmbedder::new(model)?)))
    }

    pub fn default_model() -> Result<Self> {
        Ok(Self(Arc::new(LocalEmbedder::default_model()?)))
    }

    pub fn inner(&self) -> &LocalEmbedder {
        &self.0
    }
}

impl Clone for SharedLocalEmbedder {
    fn clone(&self) -> Self {
        Self(Arc::clone(&self.0))
    }
}

impl std::ops::Deref for SharedLocalEmbedder {
    type Target = LocalEmbedder;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_local_model_dimensions() {
        assert_eq!(LocalModel::BgeSmallEnV15.dimensions(), 384);
        assert_eq!(LocalModel::AllMiniLmL6V2.dimensions(), 384);
        assert_eq!(LocalModel::BgeBaseEnV15.dimensions(), 768);
    }

    // Integration test - requires model download
    #[test]
    #[ignore = "downloads model on first run"]
    fn test_embed_one() {
        let embedder = LocalEmbedder::default_model().unwrap();
        let embedding = embedder.embed_one("Hello, world!").unwrap();
        assert_eq!(embedding.len(), 384);
    }
}
