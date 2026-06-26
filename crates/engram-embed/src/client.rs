use crate::{
    jina::JinaClient,
    types::{EmbedConfig, ReaderResponse},
};
use engram_core::{error::Result, types::Node};
use std::sync::Arc;

pub struct EmbedClient {
    jina: Arc<JinaClient>,
}

impl EmbedClient {
    pub fn new(config: EmbedConfig) -> Self {
        Self {
            jina: Arc::new(JinaClient::new(config)),
        }
    }

    pub fn from_env() -> Self {
        Self::new(EmbedConfig::from_env())
    }

    pub async fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        let results = self.jina.embed_passages(&[text.to_string()]).await?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| engram_core::error::EngramError::Embedding("empty result".into()))
    }

    pub async fn embed_many(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.jina.embed_passages(texts).await
    }

    pub async fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        self.jina.embed_query(query).await
    }

    pub async fn rerank_nodes(
        &self,
        query: &str,
        candidates: &[(String, String)],
    ) -> Result<Vec<(String, f32)>> {
        let texts: Vec<String> = candidates.iter().map(|(_, t)| t.clone()).collect();
        let ranked = self.jina.rerank(query, &texts, None).await?;
        Ok(ranked
            .into_iter()
            .map(|(idx, score)| (candidates[idx].0.clone(), score))
            .collect())
    }

    pub async fn read_url(&self, url: &str) -> Result<ReaderResponse> {
        self.jina.read_url(url).await
    }

    pub async fn embed_node(&self, node: &mut Node) -> Result<()> {
        // Jina v3 has an 8192 token limit -- truncate at ~6000 chars to stay safe.
        // We embed a summary/head of the text; the full body stays in storage.
        const MAX_EMBED_CHARS: usize = 6000;
        let text = if node.body.len() > MAX_EMBED_CHARS {
            // Snap down to a UTF-8 char boundary so multi-byte content
            // (emoji, accents) never panics on slicing.
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
}
