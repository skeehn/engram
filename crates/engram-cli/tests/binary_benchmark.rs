//! Benchmark binary quantization vs other methods.
//!
//! Run: cargo test --release --features binary -p engram-vector binary_benchmark -- --nocapture

use std::time::Instant;

#[cfg(feature = "binary")]
use engram_vector::binary::{BinaryIndex, BinaryIndexConfig};

/// Generate random vectors for testing.
fn random_vectors(count: usize, dims: usize) -> Vec<Vec<f32>> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    (0..count)
        .map(|i| {
            (0..dims)
                .map(|j| {
                    let mut hasher = DefaultHasher::new();
                    (i, j).hash(&mut hasher);
                    let h = hasher.finish();
                    // Convert to f32 in range [-1, 1]
                    ((h as f64 / u64::MAX as f64) * 2.0 - 1.0) as f32
                })
                .collect()
        })
        .collect()
}

#[test]
#[cfg(feature = "binary")]
fn binary_benchmark() {
    let dims = 384; // BGE Small EN
    let num_vectors = 100_000;
    
    println!("\n=== Binary Quantization Benchmark ===");
    println!("Vectors: {}", num_vectors);
    println!("Dimensions: {}", dims);
    
    // Generate test data
    println!("\nGenerating {} random vectors...", num_vectors);
    let start = Instant::now();
    let vectors = random_vectors(num_vectors, dims);
    println!("Generated in {:.2}s", start.elapsed().as_secs_f64());
    
    // Calculate theoretical storage sizes
    let f32_bytes = num_vectors * dims * 4;
    let i8_bytes = num_vectors * dims;
    let binary_bytes = num_vectors * ((dims + 7) / 8);
    
    println!("\n--- Theoretical Storage ---");
    println!("f32:    {:.2} MB", f32_bytes as f64 / 1024.0 / 1024.0);
    println!("i8:     {:.2} MB ({:.1}x compression)", 
             i8_bytes as f64 / 1024.0 / 1024.0,
             f32_bytes as f64 / i8_bytes as f64);
    println!("binary: {:.2} MB ({:.1}x compression)", 
             binary_bytes as f64 / 1024.0 / 1024.0,
             f32_bytes as f64 / binary_bytes as f64);
    
    // Build binary index
    println!("\n--- Binary Index Build ---");
    let config = BinaryIndexConfig::new(dims);
    let mut index = BinaryIndex::new(config);
    
    let start = Instant::now();
    for vec in &vectors {
        index.add(vec, None).unwrap();
    }
    let build_time = start.elapsed();
    
    println!("Build time: {:.2}s ({:.0} vec/sec)", 
             build_time.as_secs_f64(),
             num_vectors as f64 / build_time.as_secs_f64());
    
    let stats = index.stats();
    println!("Index stats: {}", stats);
    
    // Search benchmark
    println!("\n--- Search Benchmark ---");
    let query = &vectors[0];
    let k = 10;
    let num_queries = 1000;
    
    let start = Instant::now();
    for i in 0..num_queries {
        let q = &vectors[i % vectors.len()];
        let _ = index.search_binary(q, k);
    }
    let search_time = start.elapsed();
    
    println!("Binary search: {} queries in {:.2}s ({:.0} QPS, {:.2}ms avg)",
             num_queries,
             search_time.as_secs_f64(),
             num_queries as f64 / search_time.as_secs_f64(),
             search_time.as_millis() as f64 / num_queries as f64);
    
    // Test recall (binary search should find similar vectors)
    println!("\n--- Recall Test ---");
    let results = index.search_binary(query, k);
    
    // The first result should be the query itself (id 0)
    println!("Query: vector[0]");
    println!("Top {} results:", k);
    for (id, score) in &results {
        println!("  id={}, similarity={:.4}", id, score);
    }
    
    // Verify the first result is the query itself
    assert_eq!(results[0].0, 0, "First result should be the query vector");
    assert!(results[0].1 > 0.99, "Query should have near-perfect self-similarity");
    
    // Summary
    println!("\n=== Summary ===");
    println!("Storage: {:.2} MB binary ({:.1}x smaller than f32)",
             stats.binary_bytes as f64 / 1024.0 / 1024.0,
             f32_bytes as f64 / stats.binary_bytes as f64);
    println!("Build: {:.0} vec/sec", num_vectors as f64 / build_time.as_secs_f64());
    println!("Search: {:.0} QPS", num_queries as f64 / search_time.as_secs_f64());
    
    // Project to 1M vectors
    let million_binary_mb = 1_000_000.0 * stats.bytes_per_vec_binary as f64 / 1024.0 / 1024.0;
    println!("\n--- Projected for 1M vectors ---");
    println!("Binary only: {:.1} MB", million_binary_mb);
    println!("Binary + f32 originals: {:.1} MB", 
             million_binary_mb + 1_000_000.0 * dims as f64 * 4.0 / 1024.0 / 1024.0);
}

#[test]
#[cfg(feature = "binary")]
fn recall_accuracy_test() {
    println!("\n=== Recall Accuracy Test ===");
    
    let dims = 384;
    let num_vectors = 10_000;
    let k = 10;
    
    let vectors = random_vectors(num_vectors, dims);
    let config = BinaryIndexConfig::new(dims);
    let mut index = BinaryIndex::new(config);
    
    for vec in &vectors {
        index.add(vec, None).unwrap();
    }
    
    // Compute ground truth with exact cosine similarity
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let mut dot = 0.0f32;
        let mut norm_a = 0.0f32;
        let mut norm_b = 0.0f32;
        
        for (x, y) in a.iter().zip(b.iter()) {
            dot += x * y;
            norm_a += x * x;
            norm_b += y * y;
        }
        
        dot / (norm_a.sqrt() * norm_b.sqrt() + 1e-10)
    }
    
    // Test recall across 100 queries
    let num_queries = 100;
    let mut total_recall = 0.0f64;
    
    for qi in 0..num_queries {
        let query = &vectors[qi];
        
        // Get binary results
        let binary_results: Vec<usize> = index.search_binary(query, k)
            .iter()
            .map(|(id, _)| *id)
            .collect();
        
        // Compute exact top-k
        let mut exact_scores: Vec<(usize, f32)> = vectors.iter()
            .enumerate()
            .map(|(i, v)| (i, cosine_similarity(query, v)))
            .collect();
        exact_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let exact_top_k: Vec<usize> = exact_scores.iter().take(k).map(|(i, _)| *i).collect();
        
        // Compute recall@k
        let hits = binary_results.iter()
            .filter(|id| exact_top_k.contains(id))
            .count();
        
        total_recall += hits as f64 / k as f64;
    }
    
    let avg_recall = total_recall / num_queries as f64;
    println!("Recall@{}: {:.1}%", k, avg_recall * 100.0);
    println!("(Binary search alone, no rescoring)");
    
    // Binary quantization typically achieves 85-95% recall WITH rescoring
    // Without rescoring on random data, recall is lower
    assert!(avg_recall > 0.20, "Recall should be >20% even without rescoring");
    
    println!("\nNote: With rescoring (20x oversampling + cosine rescore),");
    println!("recall typically improves to 97-99%.");
}
