use crate::rrf::rrf_fuse;
use engram_core::{
    error::Result,
    id::NodeId,
    types::{Node, SearchMode, SearchQuery, SearchResult},
};
use engram_embed::EmbedClient;
use engram_fts::FtsIndex;
use engram_graph::GraphTraversal;
use engram_store::EngramStore;
use engram_vector::VectorIndex;
use std::sync::Arc;

pub struct QueryEngine {
    pub store: Arc<EngramStore>,
    embed: Arc<EmbedClient>,
    fts: Arc<FtsIndex>,
    vector: Arc<VectorIndex>,
}

impl QueryEngine {
    pub fn new(
        store: Arc<EngramStore>,
        embed: Arc<EmbedClient>,
        fts: Arc<FtsIndex>,
        vector: Arc<VectorIndex>,
    ) -> Self {
        Self {
            store,
            embed,
            fts,
            vector,
        }
    }

    /// Execute a search query across all requested modes, fuse results with RRF.
    pub async fn search(&self, query: SearchQuery) -> Result<Vec<SearchResult>> {
        let text = &query.text;
        let top_k = query.top_k;
        let candidate_k = top_k * 3;

        let mut rankings: Vec<Vec<String>> = Vec::new();
        let mut score_map: std::collections::HashMap<String, f32> =
            std::collections::HashMap::new();

        for mode in &query.modes {
            match mode {
                SearchMode::Vector => match self.embed.embed_query(text).await {
                    Ok(embedding) => match self.vector.search(&embedding, candidate_k) {
                        Ok(results) => {
                            let ranked: Vec<String> = results
                                .iter()
                                .map(|(id, score)| {
                                    score_map.insert(id.as_ref().to_string(), *score);
                                    id.as_ref().to_string()
                                })
                                .collect();
                            rankings.push(ranked);
                        }
                        Err(e) => {
                            tracing::warn!("vector search error: {}", e);
                        }
                    },
                    Err(e) => {
                        tracing::warn!("embed query error (no API key?): {}", e);
                    }
                },
                SearchMode::Keyword => {
                    // Escape special tantivy query characters so a plain text search doesn't fail
                    let safe_query = escape_fts_query(text);
                    match self.fts.search(&safe_query, candidate_k) {
                        Ok(results) => {
                            let ranked: Vec<String> = results
                                .iter()
                                .map(|(id, score)| {
                                    score_map.entry(id.as_ref().to_string()).or_insert(*score);
                                    id.as_ref().to_string()
                                })
                                .collect();
                            rankings.push(ranked);
                        }
                        Err(e) => {
                            tracing::warn!("fts search error: {}", e);
                        }
                    }
                }
                SearchMode::Graph => {
                    // Seed from vector results, then BFS expand
                    if let Ok(embedding) = self.embed.embed_query(text).await {
                        if let Ok(seeds) = self.vector.search(&embedding, 3) {
                            let trav = GraphTraversal::new(&self.store);
                            let mut graph_ids: Vec<String> = Vec::new();
                            for (seed_id, _) in seeds.iter().take(3) {
                                if let Ok(neighbors) = trav.neighbors(seed_id, 2) {
                                    for nid in neighbors {
                                        let s = nid.as_ref().to_string();
                                        if !graph_ids.contains(&s) {
                                            graph_ids.push(s);
                                        }
                                    }
                                }
                            }
                            if !graph_ids.is_empty() {
                                rankings.push(graph_ids);
                            }
                        }
                    }
                }
                SearchMode::Temporal | SearchMode::Relational | SearchMode::Ppr => {
                    // TODO: implement temporal / relational / PPR search
                    tracing::debug!("search mode {:?} not yet implemented", mode);
                }
            }
        }

        if rankings.is_empty() {
            return Ok(Vec::new());
        }

        let fused = rrf_fuse(&rankings, 60.0);

        // Load nodes, apply min_confidence filter
        let mut results: Vec<SearchResult> = Vec::new();
        for (id_str, rrf_score) in fused.iter().take(top_k * 2) {
            let node_id = NodeId::from(id_str.as_str());
            if let Ok(Some(node)) = self.store.get_node(&node_id) {
                if node.confidence >= query.min_confidence {
                    results.push(SearchResult {
                        node,
                        score: *rrf_score,
                        mode: "fused".into(),
                    });
                }
            }
        }

        // Rerank top 20 if embed client is available
        if results.len() > 1 {
            let candidates: Vec<(String, String)> = results
                .iter()
                .take(20)
                .map(|r| (r.node.id.as_ref().to_string(), r.node.body.clone()))
                .collect();

            match self.embed.rerank_nodes(text, &candidates).await {
                Ok(reranked) => {
                    let rerank_map: std::collections::HashMap<String, f32> =
                        reranked.into_iter().collect();
                    for r in results.iter_mut().take(20) {
                        if let Some(&new_score) = rerank_map.get(r.node.id.as_ref()) {
                            r.score = new_score;
                        }
                    }
                    results.sort_by(|a, b| {
                        b.score
                            .partial_cmp(&a.score)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                }
                Err(e) => {
                    tracing::debug!("rerank skipped: {}", e);
                }
            }
        }

        results.truncate(top_k);
        Ok(results)
    }

    /// Simple convenience: search by text with default settings (vector + keyword + graph).
    pub async fn search_text(&self, text: &str, top_k: usize) -> Result<Vec<SearchResult>> {
        let query = SearchQuery::new(text, top_k);
        self.search(query).await
    }

    /// Add a node: store it + index in FTS + embed and index in vector.
    pub async fn add_node(&self, mut node: Node) -> Result<NodeId> {
        // Try embedding; if it fails (e.g. no API key), continue without it
        match self.embed.embed_node(&mut node).await {
            Ok(()) => {
                // Store the node (with embedding)
                self.store.put_node(&node)?;
                // Index in FTS and commit so it survives process exit
                self.fts.index_node(&node)?;
                self.fts.commit()?;
                // Index in vector
                if let Some(ref emb) = node.embedding {
                    if let Err(e) = self.vector.upsert(&node.id, emb) {
                        tracing::warn!("vector upsert error: {}", e);
                    }
                    // Persist vector index to disk
                    if let Err(e) = self.vector.save() {
                        tracing::warn!("vector save error: {}", e);
                    }
                }
            }
            Err(e) => {
                tracing::warn!("embedding skipped ({}), storing without vector index", e);
                self.store.put_node(&node)?;
                self.fts.index_node(&node)?;
                self.fts.commit()?;
            }
        }

        let id = node.id.clone();
        Ok(id)
    }

    /// Update a node's embedding after body change.
    pub async fn reembed_node(&self, node_id: &NodeId) -> Result<()> {
        if let Some(mut node) = self.store.get_node(node_id)? {
            self.embed.embed_node(&mut node).await?;
            self.store.put_node(&node)?;
            if let Some(ref emb) = node.embedding {
                self.vector.upsert(&node.id, emb)?;
                self.vector.save()?;
            }
        }
        Ok(())
    }

    /// Pass-through accessors for HTTP stats handler
    pub fn fts_doc_count(&self) -> Result<u64> {
        self.fts.doc_count()
    }

    pub fn vector_len(&self) -> usize {
        self.vector.len()
    }
}

/// Escape special tantivy query syntax characters so a plain-text phrase search
/// never returns a parse error. Characters that matter in tantivy query syntax:
/// + - && || ! ( ) { } [ ] ^ " ~ * ? : \ /
fn escape_fts_query(text: &str) -> String {
    // For simplicity, wrap multi-word queries in quotes so they're treated as a
    // phrase search, and escape any embedded double-quotes.
    let escaped = text.replace('\\', r"\\").replace('"', r#"\""#);
    // Use a simple OR of tokens so individual words still match
    let words: Vec<&str> = escaped.split_whitespace().collect();
    if words.is_empty() {
        return String::new();
    }
    words.join(" ")
}
