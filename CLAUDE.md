# engram - Multi-Modal Knowledge Database

## Architecture

11 Rust crates forming a knowledge database optimized for AI agents:
- engram-core: Core types and traits
- engram-store: Sled key-value storage (pure Rust, no rocksdb)
- engram-graph: Graph operations and PPR
- engram-fts: Tantivy full-text search
- engram-vector: Flat cosine vector index (JSON storage)
- engram-embed: Jina v3 1024-dim embedding client
- engram-rerank: Jina reranker client (not yet wired)
- engram-query: RRF fusion for hybrid search
- engram-extract: Jina Reader for URL ingestion
- engram-temporal: Temporal operations (not yet wired)
- engram-cli: CLI binary

## Binary Location

~/engram/target/release/engram (also symlinked at ~/bin/engram)

## Key Commands

```bash
# Build (release mode only, debug has issues)
cargo build --release

# Add knowledge node
engram add "text content" --tags "tag1,tag2"

# Search (hybrid FTS + vector with RRF fusion)
engram search "query text"

# Get specific node
engram get <node_id>

# List all nodes
engram list

# Show stats
engram stats

# Ingest URL via Jina Reader
engram ingest <url>

# Show graph neighborhood
engram graph <node_id>
```

## Storage Structure

Default: .engram/ (in current directory)
Override: engram -d /path/to/db <command>

Internal structure:
- Sled database for key-value storage
- Tantivy index for full-text search
- JSON file for flat vector index
- Column families: nodes, edges, clusters, objects

## Jina API Integration

Required: JINA_API_KEY environment variable (NOT YET WIRED)
Models:
- Embed: jina-embeddings-v3 (1024-dim)
- Rerank: jina-reranker-v2-base-multilingual

## Code Standards

- Pure Rust, no unsafe blocks
- sled for storage (NOT rocksdb - rocksdb fills disk)
- Error handling via anyhow + thiserror
- Tests use workspace-level integration
- All 11 crates compile clean with warnings only (unused mut)

## Known Issues

1. JINA_API_KEY not yet wired - embeddings/rerank will fail without it
2. Reranker client exists but not integrated into query engine
3. Temporal crate exists but not wired into pipeline
4. Vector index is flat (no HNSW/IVF) - will be slow at scale
5. engram-unified crate (DualEdgeVamana, CompositeEmbedding) NOT in workspace yet

## molt Integration

molt (TypeScript agent at ~/molt) uses engram for knowledge:
- Config: ~/.molt/config.json sets engram_db path
- Before each LLM call, molt searches engram with last user message
- molt can call engram tool explicitly
- molt auto-learns via finish tool with learnings array

## Testing Pattern

Each crate has tests/ directory with integration tests. Run with:
```bash
cargo test --release
```

DO NOT run debug builds - sled has debug mode issues.
