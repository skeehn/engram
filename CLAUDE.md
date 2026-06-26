# engram - Multi-Modal Knowledge Database

## Architecture

10 Rust crates forming a knowledge database optimized for AI agents:
- engram-core: Core types and traits
- engram-store: Sled key-value storage (pure Rust, no rocksdb)
- engram-graph: Graph operations and PPR
- engram-fts: Tantivy full-text search
- engram-vector: Flat cosine vector index (JSON storage)
- engram-embed: Jina v3 1024-dim embedding client
- engram-query: RRF fusion for hybrid search
- engram-extract: Jina Reader for URL ingestion
- engram-temporal: Temporal operations (not yet wired)
- engram-cli: CLI binary + HTTP server (`engram serve`)

## Binary Location

~/engram/target/release/engram (also symlinked at ~/bin/engram)

## Key Commands

```bash
# Build (release recommended; debug builds and tests also work)
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

Required: JINA_API_KEY environment variable (wired via `EmbedClient::from_env`)
Models:
- Embed: jina-embeddings-v3 (1024-dim)

## Code Standards

- Pure Rust, no unsafe blocks
- sled for storage (NOT rocksdb - rocksdb fills disk)
- Error handling via anyhow + thiserror
- Tests use workspace-level integration
- All 10 crates compile clean

## Known Issues

1. Vector index is flat cosine (no HNSW/IVF) - fine to ~100k vectors, slower beyond.
2. Graph/temporal modes are wired structurally but inert: no CLI path creates
   edges yet, so `graph` shows 0 edges and Temporal/PPR search modes are TODO.
3. engram-unified crate (DualEdgeVamana, CompositeEmbedding) is not in the workspace yet.

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

Release is recommended for perf/benchmarking, but debug builds and `cargo test` work fine.
