//! Integration test: Local embeddings + HNSW index
//! 
//! This tests the full engram 2.0 pipeline:
//! 1. Local ONNX embeddings (no API calls)
//! 2. HNSW vector search (O(log n) instead of O(n))

use engram_embed::{LocalEmbedder, LocalModel};
use engram_vector::{HnswIndex, HnswConfig};
use engram_core::id::NodeId;
use std::time::Instant;
use tempfile::tempdir;

#[test]
#[ignore = "downloads model on first run (~130MB)"]
fn test_local_embed_and_hnsw_search() {
    println!("\n=== engram 2.0 Local Embedding + HNSW Test ===\n");
    
    // Create local embedder (downloads model on first run)
    let start = Instant::now();
    let embedder = LocalEmbedder::new(LocalModel::BgeSmallEnV15).unwrap();
    println!("Model loaded in {:?}", start.elapsed());
    println!("Model: {} ({} dimensions)", embedder.model_name(), embedder.dimensions());
    
    // Create HNSW index
    let dir = tempdir().unwrap();
    let index_path = dir.path().join("test.hnsw");
    let index = HnswIndex::with_defaults(embedder.dimensions(), &index_path).unwrap();
    
    // Test data
    let documents = vec![
        ("doc1", "Rust is a systems programming language focused on safety and performance."),
        ("doc2", "Python is great for data science and machine learning."),
        ("doc3", "The cargo build system makes Rust dependency management easy."),
        ("doc4", "PyTorch and TensorFlow are popular deep learning frameworks."),
        ("doc5", "Memory safety in Rust prevents common bugs like null pointer dereferences."),
    ];
    
    // Embed and index documents
    println!("\nIndexing {} documents...", documents.len());
    let start = Instant::now();
    for (id, text) in &documents {
        let embedding = embedder.embed_passage(text).unwrap();
        index.upsert(&NodeId::from(id.to_string()), &embedding).unwrap();
    }
    println!("Indexed in {:?}", start.elapsed());
    
    // Search for Rust-related content
    let queries = vec![
        "memory safety programming",
        "machine learning frameworks",
        "package management tools",
    ];
    
    println!("\n--- Search Results ---");
    for query in &queries {
        let start = Instant::now();
        let query_embedding = embedder.embed_query(query).unwrap();
        let results = index.search(&query_embedding, 3).unwrap();
        let search_time = start.elapsed();
        
        println!("\nQuery: '{}'  (search took {:?})", query, search_time);
        for (i, (node_id, score)) in results.iter().enumerate() {
            let doc_text = documents.iter()
                .find(|(id, _)| *id == node_id.as_ref())
                .map(|(_, text)| *text)
                .unwrap_or("(not found)");
            println!("  {}. [score={:.4}] {}: {}...", 
                i + 1, 
                score, 
                node_id.as_ref(),
                &doc_text[..doc_text.len().min(60)]
            );
        }
    }
    
    // Verify Rust queries return Rust docs
    let query_embedding = embedder.embed_query("Rust programming language").unwrap();
    let results = index.search(&query_embedding, 3).unwrap();
    let top_result = &results[0].0;
    assert!(
        top_result.as_ref() == "doc1" || 
        top_result.as_ref() == "doc3" || 
        top_result.as_ref() == "doc5",
        "Expected Rust-related doc, got: {}",
        top_result.as_ref()
    );
    
    // Save and reload index
    index.save().unwrap();
    drop(index);
    
    let reloaded = HnswIndex::with_defaults(embedder.dimensions(), &index_path).unwrap();
    assert_eq!(reloaded.len(), documents.len());
    println!("\n✓ Index persisted and reloaded successfully ({} vectors)", reloaded.len());
    
    println!("\n=== Test Complete ===\n");
}

#[test]
#[ignore = "downloads model, runs benchmark"]
fn bench_local_embeddings() {
    let embedder = LocalEmbedder::new(LocalModel::AllMiniLmL6V2).unwrap();
    
    // Single text
    let start = Instant::now();
    for _ in 0..100 {
        let _ = embedder.embed_one("Hello world").unwrap();
    }
    let single_time = start.elapsed() / 100;
    println!("Single embed: {:?}", single_time);
    
    // Batch
    let texts: Vec<String> = (0..100).map(|i| format!("Document number {}", i)).collect();
    let start = Instant::now();
    let _ = embedder.embed_many(&texts).unwrap();
    let batch_time = start.elapsed();
    println!("Batch embed (100 texts): {:?} ({:?}/text)", batch_time, batch_time / 100);
}
