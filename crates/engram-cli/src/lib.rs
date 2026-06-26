/// engram public library API — re-exports the core building blocks
/// so downstream crates and grain can use engram as a library
/// without depending on each internal crate individually.

pub use engram_core::{
    error::{EngramError, Result},
    id::{ClusterId, EdgeId, NodeId, ObjectId},
    types::{
        Edge, EdgeType, Node, NodeType, SearchMode, SearchQuery, SearchResult,
    },
};

pub use engram_store::EngramStore;
pub use engram_embed::EmbedClient;
pub use engram_fts::FtsIndex;
pub use engram_vector::VectorIndex;
pub use engram_query::QueryEngine;
pub use engram_graph::GraphTraversal;
pub use engram_temporal::TemporalQuery;

use std::path::{Path, PathBuf};
use std::sync::Arc;

/// All-in-one handle: open every sub-component from a single db path.
/// Useful for embedding engram in other Rust programs without wiring
/// each crate manually.
pub struct Engram {
    pub store: Arc<EngramStore>,
    pub embed: Arc<EmbedClient>,
    pub fts: Arc<FtsIndex>,
    pub vector: Arc<VectorIndex>,
    pub engine: Arc<QueryEngine>,
    db_path: PathBuf,
}

impl Engram {
    /// Open (or create) an engram database at `path`.
    /// Pass `embed_dimensions` as 1024 for jina-embeddings-v3 (default).
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_with_dimensions(path, 1024)
    }

    pub fn open_with_dimensions(path: impl AsRef<Path>, embed_dimensions: usize) -> Result<Self> {
        let db_path = path.as_ref().to_path_buf();
        let store = Arc::new(EngramStore::open(&db_path)?);
        let fts_path = db_path.join("fts");
        let fts = Arc::new(FtsIndex::open(&fts_path)?);
        let vec_path = db_path.join("vectors.json");
        let vector = Arc::new(VectorIndex::new(embed_dimensions, &vec_path)?);
        let embed = Arc::new(EmbedClient::from_env());
        let engine = Arc::new(QueryEngine::new(
            store.clone(),
            embed.clone(),
            fts.clone(),
            vector.clone(),
        ));
        Ok(Self { store, embed, fts, vector, engine, db_path })
    }

    pub fn db_path(&self) -> &Path {
        &self.db_path
    }
}
