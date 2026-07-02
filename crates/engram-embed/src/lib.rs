pub mod client;
pub mod document;
pub mod hybrid;
pub mod jina;
pub mod local;
pub mod multimodal;
pub mod types;

pub use client::EmbedClient;
pub use document::{ChunkConfig, ChunkStrategy, DocumentChunk, DocumentEmbedder, EmbeddedChunk, EmbeddedDocument};
pub use hybrid::{EmbedStrategy, HybridEmbedder};
pub use local::{LocalEmbedder, LocalModel, SharedLocalEmbedder};
pub use multimodal::{ContentType, MultimodalEmbedder, MultimodalStats};
