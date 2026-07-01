//! Benchmark comparing F32 vs I8 quantization for engram.
//!
//! Run with: cargo test --release -p engram-cli quantization_benchmark -- --nocapture

use std::time::Instant;
use tempfile::tempdir;

use engram_vector::{HnswConfig, HnswIndex, QuantizationType};
use engram_core::id::NodeId;

const DIMENSIONS: usize = 384;
const NUM_VECTORS: usize = 10_000;
const NUM_QUERIES: usize = 100;
const TOP_K: usize = 10;

fn generate_random_vector(seed: usize) -> Vec<f32> {
    // Simple deterministic pseudo-random based on seed
    (0..DIMENSIONS)
        .map(|i| {
            let x = ((seed * 2654435761 + i * 1597334677) % 1000000) as f32 / 1000000.0;
            x * 2.0 - 1.0  // Range [-1, 1]
        })
        .collect()
}

fn normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

fn benchmark_quantization(quant: QuantizationType) -> (f64, f64, f64, usize) {
    let dir = tempdir().unwrap();
    let path = dir.path().join("bench.hnsw");
    
    let config = HnswConfig {
        connectivity: 16,
        expansion_add: 128,
        expansion_search: 64,
        quantization: quant,
    };
    
    let index = HnswIndex::new(DIMENSIONS, &path, config).unwrap();
    
    // Generate and insert vectors
    let insert_start = Instant::now();
    for i in 0..NUM_VECTORS {
        let id = NodeId::from(format!("node-{}", i));
        let mut vec = generate_random_vector(i);
        normalize(&mut vec);
        index.upsert(&id, &vec).unwrap();
    }
    let insert_time = insert_start.elapsed().as_secs_f64();
    
    // Save and check file size
    index.save().unwrap();
    let file_size = std::fs::metadata(&path).map(|m| m.len() as usize).unwrap_or(0);
    
    // Search benchmark
    let search_start = Instant::now();
    for i in 0..NUM_QUERIES {
        let mut query = generate_random_vector(i + NUM_VECTORS);
        normalize(&mut query);
        let _ = index.search(&query, TOP_K).unwrap();
    }
    let search_time = search_start.elapsed().as_secs_f64();
    
    let insert_per_sec = NUM_VECTORS as f64 / insert_time;
    let search_latency_ms = (search_time / NUM_QUERIES as f64) * 1000.0;
    
    (insert_per_sec, search_latency_ms, insert_time, file_size)
}

#[test]
fn quantization_benchmark() {
    println!("\n=== engram Quantization Benchmark ===");
    println!("Vectors: {}, Dimensions: {}, Queries: {}, Top-K: {}\n", 
             NUM_VECTORS, DIMENSIONS, NUM_QUERIES, TOP_K);
    
    println!("Running F32 benchmark...");
    let (f32_insert, f32_search, f32_time, f32_size) = benchmark_quantization(QuantizationType::F32);
    
    println!("Running I8 benchmark...");
    let (i8_insert, i8_search, i8_time, i8_size) = benchmark_quantization(QuantizationType::I8);
    
    println!("\n┌─────────────────────────────────────────────────────────────┐");
    println!("│                    BENCHMARK RESULTS                        │");
    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│ Metric              │ F32           │ I8            │ Diff  │");
    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│ Insert (vec/sec)    │ {:>12.0} │ {:>12.0} │ {:>+5.1}% │",
             f32_insert, i8_insert, (i8_insert / f32_insert - 1.0) * 100.0);
    println!("│ Search (ms/query)   │ {:>12.3} │ {:>12.3} │ {:>+5.1}% │",
             f32_search, i8_search, (1.0 - i8_search / f32_search) * 100.0);
    println!("│ Index size (KB)     │ {:>12} │ {:>12} │ {:>+5.1}% │",
             f32_size / 1024, i8_size / 1024, (1.0 - i8_size as f64 / f32_size as f64) * 100.0);
    println!("│ Total insert (sec)  │ {:>12.2} │ {:>12.2} │         │",
             f32_time, i8_time);
    println!("└─────────────────────────────────────────────────────────────┘");
    
    // Storage calculation for scale
    println!("\n=== STORAGE PROJECTIONS ===");
    let f32_per_vec = f32_size as f64 / NUM_VECTORS as f64;
    let i8_per_vec = i8_size as f64 / NUM_VECTORS as f64;
    
    for scale in [100_000, 1_000_000, 10_000_000] {
        let f32_mb = (f32_per_vec * scale as f64) / (1024.0 * 1024.0);
        let i8_mb = (i8_per_vec * scale as f64) / (1024.0 * 1024.0);
        println!("{:>10} vectors: F32 = {:>7.1} MB, I8 = {:>7.1} MB (saves {:.1} MB)",
                 scale, f32_mb, i8_mb, f32_mb - i8_mb);
    }
}
