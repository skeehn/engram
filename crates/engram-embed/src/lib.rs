pub mod client;
pub mod hybrid;
pub mod jina;
pub mod local;
pub mod types;

pub use client::EmbedClient;
pub use hybrid::{EmbedStrategy, HybridEmbedder};
pub use local::{LocalEmbedder, LocalModel, SharedLocalEmbedder};
