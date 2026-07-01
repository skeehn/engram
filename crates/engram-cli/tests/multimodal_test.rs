//! Test multimodal embeddings (text vs code).
//!
//! Run with: cargo test --release -p engram-cli multimodal_test -- --nocapture

use engram_embed::{ContentType, MultimodalEmbedder};
use std::time::Instant;

#[test]
#[ignore] // Requires model download
fn test_multimodal_embeddings() {
    let embedder = MultimodalEmbedder::new(None::<&str>);
    
    // Test text embedding
    println!("\n=== Text Embedding Test ===");
    let text_samples = &[
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming industries worldwide.",
        "Today we discuss the future of artificial intelligence.",
    ];
    
    let start = Instant::now();
    let text_embs = embedder.embed_text(text_samples).expect("text embed failed");
    let text_time = start.elapsed();
    
    println!("Text embeddings: {} samples in {:?}", text_embs.len(), text_time);
    println!("Dimensions: {}", text_embs[0].len());
    assert_eq!(text_embs[0].len(), 384); // BGE Small EN
    
    // Test code embedding
    println!("\n=== Code Embedding Test ===");
    let code_samples = &[
        "fn main() {\n    println!(\"Hello, world!\");\n}",
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "async function fetchData(url) {\n    const response = await fetch(url);\n    return response.json();\n}",
    ];
    
    let start = Instant::now();
    let code_embs = embedder.embed_code(code_samples).expect("code embed failed");
    let code_time = start.elapsed();
    
    println!("Code embeddings: {} samples in {:?}", code_embs.len(), code_time);
    println!("Dimensions: {}", code_embs[0].len());
    assert_eq!(code_embs[0].len(), 768); // Jina Code v2
    
    // Test content type detection
    println!("\n=== Content Type Detection ===");
    
    let rust_code = "pub struct Config {\n    pub name: String,\n    pub value: i32,\n}";
    let prose = "This is a simple document about programming concepts.";
    
    assert_eq!(ContentType::from_content(rust_code), ContentType::Code);
    assert_eq!(ContentType::from_content(prose), ContentType::Text);
    println!("Content detection: OK");
    
    // Test auto-embed
    println!("\n=== Auto-Embed Test ===");
    let (detected_type, emb) = embedder.embed_auto(rust_code, None).expect("auto embed failed");
    assert_eq!(detected_type, ContentType::Code);
    assert_eq!(emb.len(), 768);
    println!("Auto-detected Rust code -> {:?}, dims={}", detected_type, emb.len());
    
    let (detected_type, emb) = embedder.embed_auto(prose, None).expect("auto embed failed");
    assert_eq!(detected_type, ContentType::Text);
    assert_eq!(emb.len(), 384);
    println!("Auto-detected prose -> {:?}, dims={}", detected_type, emb.len());
    
    // Stats
    let stats = embedder.stats();
    println!("\n=== Model Stats ===");
    println!("Text model loaded: {}", stats.text_loaded);
    println!("Code model loaded: {}", stats.code_loaded);
    println!("Image model loaded: {}", stats.image_loaded);
    
    println!("\n✓ All multimodal tests passed!");
}
