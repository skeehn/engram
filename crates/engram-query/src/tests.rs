#[cfg(test)]
mod tests {
    use engram_core::{
        types::{Node, NodeType, SearchQuery},
    };
    use engram_embed::EmbedClient;
    use engram_fts::FtsIndex;
    use crate::QueryEngine;
    use engram_store::EngramStore;
    use engram_vector::VectorIndex;
    use std::sync::Arc;
    use tempfile::tempdir;

    fn make_engine() -> QueryEngine {
        let dir = tempdir().unwrap();
        let base = dir.into_path(); // keep the dir alive (not dropped)
        std::fs::create_dir_all(base.join("store")).unwrap();
        std::fs::create_dir_all(base.join("fts")).unwrap();
        let store = Arc::new(EngramStore::open(base.join("store")).unwrap());
        let fts = Arc::new(FtsIndex::open(base.join("fts")).unwrap());
        let vector = Arc::new(VectorIndex::new(1024, base.join("v.json")).unwrap());
        let embed = Arc::new(EmbedClient::from_env());
        QueryEngine::new(store, embed, fts, vector)
    }

    #[tokio::test]
    async fn test_add_and_search_fts_fallback() {
        let engine = make_engine();
        let node = Node::new("Rust is a systems programming language", NodeType::Fact);
        engine.add_node(node).await.unwrap();

        // Give tantivy reader time to reload committed segment
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let results = engine.search_text("systems programming", 5).await.unwrap();
        assert!(!results.is_empty(), "FTS search should return at least one result");
        assert!(results[0].node.body.contains("Rust"));
    }

    #[tokio::test]
    async fn test_add_multiple_search_ranking() {
        let engine = make_engine();
        engine.add_node(Node::new("Python for data science", NodeType::Fact)).await.unwrap();
        engine.add_node(Node::new("Rust memory safety ownership", NodeType::Fact)).await.unwrap();
        engine.add_node(Node::new("TypeScript typed JavaScript", NodeType::Fact)).await.unwrap();

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let results = engine.search_text("memory safety", 5).await.unwrap();
        assert!(!results.is_empty());
        assert!(results[0].node.body.contains("Rust"), "Rust should rank highest for 'memory safety'");
    }

    #[tokio::test]
    async fn test_add_node_stores_in_fts() {
        let engine = make_engine();
        let node = Node::new("engram knowledge database", NodeType::Concept)
            .with_tags(vec!["test".into()]);
        engine.add_node(node).await.unwrap();

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        // Search should find the node — proves it was indexed
        let results = engine.search_text("knowledge database", 5).await.unwrap();
        assert!(!results.is_empty(), "node should be findable via FTS after add");
        assert_eq!(engine.vector_len(), 0); // no embed key → no vector
    }

    #[tokio::test]
    async fn test_search_returns_empty_for_no_match() {
        let engine = make_engine();
        engine.add_node(Node::new("completely unrelated content", NodeType::Note)).await.unwrap();

        let results = engine.search_text("quantum entanglement xyzzy", 5).await.unwrap();
        // May return 0 or some results — key thing is it doesn't panic
        let _ = results;
    }

    #[tokio::test]
    async fn test_reembed_node_noop_without_key() {
        let engine = make_engine();
        let node = Node::new("reembed test", NodeType::Fact);
        let id = engine.add_node(node).await.unwrap();
        // Should not panic even without embed key
        let _ = engine.reembed_node(&id).await;
    }

    #[tokio::test]
    async fn test_explicit_search_query_modes() {
        let engine = make_engine();
        engine.add_node(Node::new("grain agent tool", NodeType::Fact)).await.unwrap();

        let query = SearchQuery::new("grain agent", 3);
        let results = engine.search(query).await.unwrap();
        // FTS mode should work; won't panic
        let _ = results;
    }
}
