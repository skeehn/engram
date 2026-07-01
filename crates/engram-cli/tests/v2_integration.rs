//! Phase 2 integration test: Local embeddings + HNSW + EngramContext
//!
//! Tests the full v2 pipeline with local-first semantics.

use engram_cli::v2::EngramContext;
use std::time::Instant;
use tempfile::tempdir;

/// Test the full v2 context with local embeddings and HNSW.
#[tokio::test]
#[ignore = "downloads model on first run (~130MB)"]
async fn test_v2_context_full_pipeline() {
    let dir = tempdir().unwrap();
    
    println!("\n=== engram 2.0 Full Pipeline Test ===\n");
    
    // Initialize context
    let start = Instant::now();
    let ctx = EngramContext::open_offline(dir.path()).unwrap();
    println!("Context initialized in {:?}", start.elapsed());
    
    // Test documents
    let docs = vec![
        ("doc1", "Rust is a systems programming language focused on safety and performance. It has no garbage collector."),
        ("doc2", "Python is a high-level interpreted language popular for data science and machine learning applications."),
        ("doc3", "JavaScript runs in browsers and enables interactive web pages. Node.js allows server-side JS."),
        ("doc4", "Machine learning uses neural networks to learn patterns from data. Deep learning uses many layers."),
        ("doc5", "Memory safety in Rust prevents common bugs like null pointer dereference and buffer overflows."),
        ("doc6", "PostgreSQL is a powerful open-source relational database with ACID compliance."),
        ("doc7", "Docker containers package applications with their dependencies for consistent deployment."),
        ("doc8", "Kubernetes orchestrates container deployments at scale across clusters of machines."),
    ];
    
    // Add all documents
    println!("\nIndexing {} documents...", docs.len());
    let start = Instant::now();
    for (id, text) in &docs {
        ctx.add(id, text).await.unwrap();
    }
    println!("Indexed in {:?}", start.elapsed());
    
    // Verify stats
    let stats = ctx.stats();
    assert_eq!(stats.vectors, docs.len());
    println!("Vectors indexed: {}", stats.vectors);
    
    // Test queries
    let queries = vec![
        ("memory safety programming", vec!["doc5", "doc1"]),  // Should find Rust safety docs
        ("machine learning data science", vec!["doc4", "doc2"]),  // ML + Python
        ("container deployment", vec!["doc7", "doc8"]),  // Docker + K8s
        ("database storage", vec!["doc6"]),  // PostgreSQL
        ("web browser javascript", vec!["doc3"]),  // JS
    ];
    
    println!("\n--- Search Results ---\n");
    for (query, expected_top) in queries {
        let start = Instant::now();
        let results = ctx.search(query, 3).await.unwrap();
        let elapsed = start.elapsed();
        
        println!("Query: '{}'  ({:?})", query, elapsed);
        for (i, (id, score)) in results.iter().enumerate() {
            let is_expected = expected_top.contains(&id.as_str());
            let marker = if is_expected { "✓" } else { " " };
            println!("  {} {}. [score={:.4}] {}", marker, i + 1, score, id);
        }
        
        // Verify at least one expected result in top 3
        let found = results.iter().any(|(id, _)| expected_top.contains(&id.as_str()));
        assert!(found, "Expected one of {:?} in results for '{}'", expected_top, query);
        println!();
    }
    
    println!("=== Test Complete ===\n");
}

/// Benchmark embedding speed.
#[tokio::test]
#[ignore = "downloads model on first run"]
async fn bench_v2_embedding() {
    use engram_embed::{HybridEmbedder, EmbedStrategy};
    use engram_embed::types::EmbedConfig;
    
    println!("\n=== Embedding Benchmark ===\n");
    
    let mut embedder = HybridEmbedder::new(EmbedConfig::from_env(), EmbedStrategy::LocalOnly);
    
    // First call loads model
    let start = Instant::now();
    let _ = embedder.embed_one("warmup").await.unwrap();
    println!("Model load + first embed: {:?}", start.elapsed());
    
    // Single embed
    let start = Instant::now();
    let _ = embedder.embed_one("This is a test sentence.").await.unwrap();
    println!("Single embed: {:?}", start.elapsed());
    
    // Batch of 10
    let texts: Vec<String> = (0..10).map(|i| format!("Test sentence number {}", i)).collect();
    let start = Instant::now();
    let embeddings = embedder.embed_many(&texts).await.unwrap();
    let elapsed = start.elapsed();
    println!("Batch of 10: {:?} ({:.2?}/text)", elapsed, elapsed / 10);
    assert_eq!(embeddings.len(), 10);
    
    // Batch of 100
    let texts: Vec<String> = (0..100).map(|i| format!("Test sentence number {}", i)).collect();
    let start = Instant::now();
    let embeddings = embedder.embed_many(&texts).await.unwrap();
    let elapsed = start.elapsed();
    println!("Batch of 100: {:?} ({:.2?}/text)", elapsed, elapsed / 100);
    assert_eq!(embeddings.len(), 100);
    
    println!("\n=== Benchmark Complete ===\n");
}
