//! Document embedding with late-interaction retrieval.
//!
//! Implements a ColBERT-style approach for documents:
//! - Chunk documents by page/section/paragraph
//! - Embed each chunk independently
//! - MaxSim retrieval: max similarity across all chunk pairs
//!
//! This gives late-interaction benefits without needing ColPali's VLM.

use std::path::Path;
use tracing::{debug, info};

use crate::multimodal::{ContentType, MultimodalEmbedder};
use engram_core::error::{EngramError, Result};

/// Document chunk with metadata.
#[derive(Debug, Clone)]
pub struct DocumentChunk {
    /// Chunk content
    pub content: String,
    /// Page number (1-indexed, 0 if unknown)
    pub page: usize,
    /// Section heading (if detected)
    pub section: Option<String>,
    /// Chunk index within document
    pub chunk_index: usize,
    /// Character offset in original document
    pub char_offset: usize,
}

/// Embedded document chunk.
#[derive(Debug, Clone)]
pub struct EmbeddedChunk {
    /// Original chunk metadata
    pub chunk: DocumentChunk,
    /// Embedding vector
    pub embedding: Vec<f32>,
    /// Content type used for embedding
    pub content_type: ContentType,
}

/// Document with all its embedded chunks.
#[derive(Debug)]
pub struct EmbeddedDocument {
    /// Source path or identifier
    pub source: String,
    /// All embedded chunks
    pub chunks: Vec<EmbeddedChunk>,
    /// Total character count
    pub total_chars: usize,
}

impl EmbeddedDocument {
    /// Get number of chunks.
    pub fn num_chunks(&self) -> usize {
        self.chunks.len()
    }
    
    /// Get all embeddings as a matrix (for batch operations).
    pub fn embeddings_matrix(&self) -> Vec<&[f32]> {
        self.chunks.iter().map(|c| c.embedding.as_slice()).collect()
    }
}

/// Chunking strategy for documents.
#[derive(Debug, Clone, Copy, Default)]
pub enum ChunkStrategy {
    /// Fixed size chunks with overlap
    #[default]
    FixedSize,
    /// Chunk by paragraph (double newline)
    Paragraph,
    /// Chunk by sentence
    Sentence,
    /// Chunk by markdown sections (headers)
    MarkdownSections,
}

/// Configuration for document chunking.
#[derive(Debug, Clone)]
pub struct ChunkConfig {
    /// Target chunk size in characters
    pub chunk_size: usize,
    /// Overlap between chunks in characters
    pub overlap: usize,
    /// Chunking strategy
    pub strategy: ChunkStrategy,
    /// Minimum chunk size (skip smaller chunks)
    pub min_chunk_size: usize,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            chunk_size: 512,
            overlap: 64,
            strategy: ChunkStrategy::FixedSize,
            min_chunk_size: 32,
        }
    }
}

/// Document embedder with late-interaction retrieval.
pub struct DocumentEmbedder {
    embedder: MultimodalEmbedder,
    config: ChunkConfig,
}

impl DocumentEmbedder {
    /// Create a new document embedder.
    pub fn new(embedder: MultimodalEmbedder, config: ChunkConfig) -> Self {
        Self { embedder, config }
    }
    
    /// Create with default config.
    pub fn with_defaults(embedder: MultimodalEmbedder) -> Self {
        Self::new(embedder, ChunkConfig::default())
    }
    
    /// Chunk a document into pieces.
    pub fn chunk_document(&self, content: &str, source: &str) -> Vec<DocumentChunk> {
        match self.config.strategy {
            ChunkStrategy::FixedSize => self.chunk_fixed_size(content),
            ChunkStrategy::Paragraph => self.chunk_by_paragraph(content),
            ChunkStrategy::Sentence => self.chunk_by_sentence(content),
            ChunkStrategy::MarkdownSections => self.chunk_by_markdown(content),
        }
    }
    
    /// Fixed-size chunking with overlap.
    fn chunk_fixed_size(&self, content: &str) -> Vec<DocumentChunk> {
        let mut chunks = Vec::new();
        let chars: Vec<char> = content.chars().collect();
        let total = chars.len();
        
        if total == 0 {
            return chunks;
        }
        
        let mut offset = 0;
        let mut chunk_index = 0;
        
        while offset < total {
            let end = (offset + self.config.chunk_size).min(total);
            let chunk_chars: String = chars[offset..end].iter().collect();
            
            if chunk_chars.trim().len() >= self.config.min_chunk_size {
                chunks.push(DocumentChunk {
                    content: chunk_chars.trim().to_string(),
                    page: 0,
                    section: None,
                    chunk_index,
                    char_offset: offset,
                });
                chunk_index += 1;
            }
            
            // Move forward, accounting for overlap
            let step = self.config.chunk_size.saturating_sub(self.config.overlap);
            offset += step.max(1);
        }
        
        chunks
    }
    
    /// Chunk by paragraph (double newline).
    fn chunk_by_paragraph(&self, content: &str) -> Vec<DocumentChunk> {
        let mut chunks = Vec::new();
        let mut char_offset = 0;
        
        for (chunk_index, para) in content.split("\n\n").enumerate() {
            let trimmed = para.trim();
            if trimmed.len() >= self.config.min_chunk_size {
                // If paragraph is too long, sub-chunk it
                if trimmed.len() > self.config.chunk_size * 2 {
                    let sub_chunks = self.chunk_fixed_size(trimmed);
                    for mut sub in sub_chunks {
                        sub.char_offset += char_offset;
                        sub.chunk_index = chunks.len();
                        chunks.push(sub);
                    }
                } else {
                    chunks.push(DocumentChunk {
                        content: trimmed.to_string(),
                        page: 0,
                        section: None,
                        chunk_index,
                        char_offset,
                    });
                }
            }
            char_offset += para.len() + 2; // +2 for \n\n
        }
        
        chunks
    }
    
    /// Chunk by sentence (simple heuristic).
    fn chunk_by_sentence(&self, content: &str) -> Vec<DocumentChunk> {
        let mut chunks = Vec::new();
        let mut current_chunk = String::new();
        let mut chunk_index = 0;
        let mut char_offset = 0;
        let mut chunk_start_offset = 0;
        
        for (i, c) in content.chars().enumerate() {
            current_chunk.push(c);
            
            // Sentence boundary: . ! ? followed by space/newline
            let is_sentence_end = (c == '.' || c == '!' || c == '?') && 
                content.chars().nth(i + 1).map_or(true, |next| next.is_whitespace());
            
            if is_sentence_end && current_chunk.len() >= self.config.min_chunk_size {
                // Check if we should flush
                if current_chunk.len() >= self.config.chunk_size {
                    let trimmed = current_chunk.trim().to_string();
                    if trimmed.len() >= self.config.min_chunk_size {
                        chunks.push(DocumentChunk {
                            content: trimmed,
                            page: 0,
                            section: None,
                            chunk_index,
                            char_offset: chunk_start_offset,
                        });
                        chunk_index += 1;
                    }
                    current_chunk.clear();
                    chunk_start_offset = i + 1;
                }
            }
            char_offset = i + 1;
        }
        
        // Flush remaining
        let trimmed = current_chunk.trim().to_string();
        if trimmed.len() >= self.config.min_chunk_size {
            chunks.push(DocumentChunk {
                content: trimmed,
                page: 0,
                section: None,
                chunk_index,
                char_offset: chunk_start_offset,
            });
        }
        
        chunks
    }
    
    /// Chunk by markdown sections (# headers).
    fn chunk_by_markdown(&self, content: &str) -> Vec<DocumentChunk> {
        let mut chunks = Vec::new();
        let mut current_section: Option<String> = None;
        let mut current_content = String::new();
        let mut chunk_index = 0;
        let mut section_start = 0;
        let mut char_offset = 0;
        
        for line in content.lines() {
            let trimmed = line.trim();
            
            // Detect markdown header
            if trimmed.starts_with('#') {
                // Flush previous section
                if current_content.trim().len() >= self.config.min_chunk_size {
                    // Sub-chunk if too long
                    if current_content.len() > self.config.chunk_size * 2 {
                        let sub_chunks = self.chunk_fixed_size(&current_content);
                        for mut sub in sub_chunks {
                            sub.section = current_section.clone();
                            sub.char_offset += section_start;
                            sub.chunk_index = chunks.len();
                            chunks.push(sub);
                        }
                    } else {
                        chunks.push(DocumentChunk {
                            content: current_content.trim().to_string(),
                            page: 0,
                            section: current_section.clone(),
                            chunk_index,
                            char_offset: section_start,
                        });
                        chunk_index += 1;
                    }
                }
                
                // Start new section
                current_section = Some(trimmed.trim_start_matches('#').trim().to_string());
                current_content.clear();
                section_start = char_offset;
            } else {
                current_content.push_str(line);
                current_content.push('\n');
            }
            
            char_offset += line.len() + 1; // +1 for newline
        }
        
        // Flush last section
        if current_content.trim().len() >= self.config.min_chunk_size {
            if current_content.len() > self.config.chunk_size * 2 {
                let sub_chunks = self.chunk_fixed_size(&current_content);
                for mut sub in sub_chunks {
                    sub.section = current_section.clone();
                    sub.char_offset += section_start;
                    sub.chunk_index = chunks.len();
                    chunks.push(sub);
                }
            } else {
                chunks.push(DocumentChunk {
                    content: current_content.trim().to_string(),
                    page: 0,
                    section: current_section,
                    chunk_index,
                    char_offset: section_start,
                });
            }
        }
        
        chunks
    }
    
    /// Embed a document, returning all chunk embeddings.
    pub fn embed_document(&self, content: &str, source: &str) -> Result<EmbeddedDocument> {
        let chunks = self.chunk_document(content, source);
        
        if chunks.is_empty() {
            return Ok(EmbeddedDocument {
                source: source.to_string(),
                chunks: vec![],
                total_chars: content.len(),
            });
        }
        
        info!(chunks = chunks.len(), source = source, "Embedding document chunks");
        
        // Batch embed all chunks
        let texts: Vec<&str> = chunks.iter().map(|c| c.content.as_str()).collect();
        
        // Detect content type from first chunk
        let content_type = ContentType::from_content(&chunks[0].content);
        
        let embeddings = match content_type {
            ContentType::Text => self.embedder.embed_text(&texts)?,
            ContentType::Code => self.embedder.embed_code(&texts)?,
            ContentType::Image => {
                return Err(EngramError::Embedding(
                    "Cannot embed image content as text chunks".to_string()
                ));
            }
        };
        
        // Combine chunks with embeddings
        let embedded_chunks: Vec<EmbeddedChunk> = chunks
            .into_iter()
            .zip(embeddings)
            .map(|(chunk, embedding)| EmbeddedChunk {
                chunk,
                embedding,
                content_type,
            })
            .collect();
        
        debug!(
            num_chunks = embedded_chunks.len(),
            dims = embedded_chunks.first().map(|c| c.embedding.len()).unwrap_or(0),
            "Document embedded"
        );
        
        Ok(EmbeddedDocument {
            source: source.to_string(),
            chunks: embedded_chunks,
            total_chars: content.len(),
        })
    }
    
    /// MaxSim retrieval: find max similarity across all query-document chunk pairs.
    /// Returns (max_similarity, best_chunk_index).
    pub fn max_sim_score(
        query_embedding: &[f32],
        document: &EmbeddedDocument,
    ) -> (f32, usize) {
        let mut max_sim = f32::NEG_INFINITY;
        let mut best_idx = 0;
        
        for (i, chunk) in document.chunks.iter().enumerate() {
            let sim = cosine_similarity(query_embedding, &chunk.embedding);
            if sim > max_sim {
                max_sim = sim;
                best_idx = i;
            }
        }
        
        (max_sim, best_idx)
    }
    
    /// Late interaction score: sum of max similarities for multi-vector query.
    /// Used when query itself is chunked (long query).
    pub fn late_interaction_score(
        query_embeddings: &[Vec<f32>],
        document: &EmbeddedDocument,
    ) -> f32 {
        let mut total = 0.0;
        
        for q_emb in query_embeddings {
            let (max_sim, _) = Self::max_sim_score(q_emb, document);
            total += max_sim;
        }
        
        // Normalize by number of query vectors
        total / query_embeddings.len() as f32
    }
    
    /// Rank documents by MaxSim score.
    pub fn rank_documents<'a>(
        query_embedding: &[f32],
        documents: &'a [EmbeddedDocument],
        top_k: usize,
    ) -> Vec<(f32, &'a EmbeddedDocument, usize)> {
        let mut scored: Vec<_> = documents
            .iter()
            .map(|doc| {
                let (score, chunk_idx) = Self::max_sim_score(query_embedding, doc);
                (score, doc, chunk_idx)
            })
            .collect();
        
        // Sort by score descending
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        
        scored.into_iter().take(top_k).collect()
    }
}

/// Cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    
    dot / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_fixed_size_chunking() {
        let embedder = MultimodalEmbedder::new(None::<&str>);
        let config = ChunkConfig {
            chunk_size: 50,
            overlap: 10,
            min_chunk_size: 10,
            ..Default::default()
        };
        let doc_embedder = DocumentEmbedder::new(embedder, config);
        
        let content = "This is a test document with multiple sentences. It should be chunked into smaller pieces. Each piece will be embedded separately.";
        let chunks = doc_embedder.chunk_document(content, "test.txt");
        
        assert!(chunks.len() > 1, "Should produce multiple chunks");
        for chunk in &chunks {
            assert!(chunk.content.len() >= 10, "Chunks should meet min size");
        }
    }
    
    #[test]
    fn test_paragraph_chunking() {
        let embedder = MultimodalEmbedder::new(None::<&str>);
        let config = ChunkConfig {
            strategy: ChunkStrategy::Paragraph,
            min_chunk_size: 10,
            ..Default::default()
        };
        let doc_embedder = DocumentEmbedder::new(embedder, config);
        
        let content = "First paragraph here.\n\nSecond paragraph is longer and has more content.\n\nThird paragraph.";
        let chunks = doc_embedder.chunk_document(content, "test.txt");
        
        assert_eq!(chunks.len(), 3, "Should produce 3 paragraphs");
    }
    
    #[test]
    fn test_markdown_chunking() {
        let embedder = MultimodalEmbedder::new(None::<&str>);
        let config = ChunkConfig {
            strategy: ChunkStrategy::MarkdownSections,
            min_chunk_size: 10,
            ..Default::default()
        };
        let doc_embedder = DocumentEmbedder::new(embedder, config);
        
        let content = "# Introduction\n\nThis is the intro section.\n\n# Methods\n\nThis describes the methods used.\n\n# Results\n\nHere are the results.";
        let chunks = doc_embedder.chunk_document(content, "test.md");
        
        assert_eq!(chunks.len(), 3, "Should produce 3 sections");
        assert_eq!(chunks[0].section, Some("Introduction".to_string()));
        assert_eq!(chunks[1].section, Some("Methods".to_string()));
        assert_eq!(chunks[2].section, Some("Results".to_string()));
    }
    
    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);
        
        let c = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &c) - 0.0).abs() < 0.001);
        
        let d = vec![-1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &d) - (-1.0)).abs() < 0.001);
    }
}
