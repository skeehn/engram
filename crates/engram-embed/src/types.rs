use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct EmbedConfig {
    pub jina_api_key: Option<String>,
    pub base_url: String,
    pub reader_url: String,
    pub embed_model: String,
    pub rerank_model: String,
    pub embed_dimensions: u32,
    pub timeout_secs: u64,
    pub max_batch_size: usize,
}

impl Default for EmbedConfig {
    fn default() -> Self {
        Self {
            jina_api_key: std::env::var("JINA_API_KEY").ok(),
            base_url: "https://api.jina.ai/v1".into(),
            reader_url: "https://r.jina.ai".into(),
            embed_model: "jina-embeddings-v3".into(),
            rerank_model: "jina-reranker-v2-base-multilingual".into(),
            embed_dimensions: 1024,
            timeout_secs: 30,
            max_batch_size: 64,
        }
    }
}

impl EmbedConfig {
    pub fn from_env() -> Self {
        Self::default()
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EmbedRequest {
    pub model: String,
    pub input: Vec<String>,
    pub task: String,
    pub dimensions: u32,
    pub embedding_type: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EmbedResponse {
    pub model: String,
    pub data: Vec<EmbedData>,
    pub usage: Usage,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EmbedData {
    pub index: usize,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RerankRequest {
    pub model: String,
    pub query: String,
    pub documents: Vec<serde_json::Value>,
    pub top_n: Option<usize>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RerankResponse {
    pub model: String,
    pub results: Vec<RerankResult>,
    pub usage: Usage,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RerankResult {
    pub index: usize,
    pub document: RerankDoc,
    pub relevance_score: f32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RerankDoc {
    pub text: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Usage {
    pub total_tokens: u64,
    pub prompt_tokens: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReaderResponse {
    pub url: String,
    pub title: String,
    pub content: String,
    pub description: Option<String>,
}
