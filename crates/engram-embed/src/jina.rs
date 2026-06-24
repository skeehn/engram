use std::sync::Arc;
use reqwest::Client;
use engram_core::error::{EngramError, Result};
use crate::types::*;

pub struct JinaClient {
    client: Client,
    pub config: Arc<EmbedConfig>,
}

impl JinaClient {
    pub fn new(config: EmbedConfig) -> Self {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_secs))
            .build()
            .expect("failed to build reqwest client");
        Self {
            client,
            config: Arc::new(config),
        }
    }

    pub async fn embed_passages(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let key = self.api_key()?;
        let mut all_embeddings = Vec::new();
        for chunk in texts.chunks(self.config.max_batch_size) {
            let req = EmbedRequest {
                model: self.config.embed_model.clone(),
                input: chunk.to_vec(),
                task: "retrieval.passage".into(),
                dimensions: self.config.embed_dimensions,
                embedding_type: "float".into(),
            };
            let resp: EmbedResponse = self.post("/embeddings", &req, &key).await?;
            let mut embeddings: Vec<Vec<f32>> = vec![Vec::new(); chunk.len()];
            for d in resp.data {
                if d.index < embeddings.len() {
                    embeddings[d.index] = d.embedding;
                }
            }
            all_embeddings.extend(embeddings);
        }
        Ok(all_embeddings)
    }

    pub async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let key = self.api_key()?;
        let req = EmbedRequest {
            model: self.config.embed_model.clone(),
            input: vec![text.to_string()],
            task: "retrieval.query".into(),
            dimensions: self.config.embed_dimensions,
            embedding_type: "float".into(),
        };
        let resp: EmbedResponse = self.post("/embeddings", &req, &key).await?;
        resp.data
            .into_iter()
            .next()
            .map(|d| d.embedding)
            .ok_or_else(|| EngramError::Embedding("empty embedding response".into()))
    }

    pub async fn rerank(
        &self,
        query: &str,
        documents: &[String],
        top_n: Option<usize>,
    ) -> Result<Vec<(usize, f32)>> {
        let key = self.api_key()?;
        let req = RerankRequest {
            model: self.config.rerank_model.clone(),
            query: query.to_string(),
            documents: documents
                .iter()
                .map(|d| serde_json::Value::String(d.clone()))
                .collect(),
            top_n,
        };
        let resp: RerankResponse = self.post("/rerank", &req, &key).await?;
        let mut results: Vec<(usize, f32)> = resp
            .results
            .into_iter()
            .map(|r| (r.index, r.relevance_score))
            .collect();
        results.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(results)
    }

    pub async fn read_url(&self, url: &str) -> Result<ReaderResponse> {
        let reader_url = format!("{}/{}", self.config.reader_url, url);
        let mut builder = self
            .client
            .get(&reader_url)
            .header("Accept", "application/json");
        // Auth is optional for Jina Reader -- include key if available, but don't require it
        if let Some(ref key) = self.config.jina_api_key {
            builder = builder.header("Authorization", format!("Bearer {}", key));
        }
        let resp = builder
            .send()
            .await
            .map_err(|e| EngramError::Http(e.to_string()))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(EngramError::Http(format!(
                "jina reader {} : {}",
                status, body
            )));
        }
        let json: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| EngramError::Http(e.to_string()))?;
        Ok(ReaderResponse {
            url: url.to_string(),
            title: json["data"]["title"]
                .as_str()
                .unwrap_or("")
                .to_string(),
            content: json["data"]["content"]
                .as_str()
                .unwrap_or("")
                .to_string(),
            description: json["data"]["description"]
                .as_str()
                .map(|s| s.to_string()),
        })
    }

    fn api_key(&self) -> Result<String> {
        self.config
            .jina_api_key
            .clone()
            .ok_or_else(|| {
                EngramError::Embedding(
                    "JINA_API_KEY not set -- set env var or pass key in EmbedConfig".into(),
                )
            })
    }

    async fn post<Req: serde::Serialize, Resp: serde::de::DeserializeOwned>(
        &self,
        path: &str,
        body: &Req,
        api_key: &str,
    ) -> Result<Resp> {
        let url = format!("{}{}", self.config.base_url, path);
        let resp = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {}", api_key))
            .json(body)
            .send()
            .await
            .map_err(|e| EngramError::Http(e.to_string()))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let err_body = resp.text().await.unwrap_or_default();
            return Err(EngramError::Http(format!(
                "jina api {} {}: {}",
                path, status, err_body
            )));
        }
        resp.json::<Resp>()
            .await
            .map_err(|e| EngramError::Http(format!("json decode: {}", e)))
    }
}
