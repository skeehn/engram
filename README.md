# engram

Multi-modal knowledge database for AI agents. Sled + Tantivy + flat vector search with RRF fusion.

## Features

- Sled key-value storage
- Tantivy full-text search
- Flat cosine vector index (JSON)
- Jina v3 embeddings (1024-dim)
- Jina reranker integration
- RRF fusion for hybrid search
- URL ingestion via Jina Reader
- Graph operations

## Architecture

11 Rust crates:
- engram-core: Core types and traits
- engram-store: Sled storage
- engram-graph: Graph operations
- engram-fts: Tantivy FTS
- engram-vector: Flat vector index
- engram-embed: Jina embed client
- engram-rerank: Jina reranker client
- engram-query: RRF fusion
- engram-extract: Jina Reader
- engram-ingest: Ingestion pipeline
- engram-cli: CLI binary

## Quick Start

```bash
# Build
cargo build --release

# Add knowledge
engram add "Knowledge fact" --tags "tag1,tag2"

# Search
engram search "query"

# Stats
engram stats
```

## Usage in molt

molt automatically searches engram for context before each LLM call. Configure in ~/.molt/config.json:

```json
{
  "engram_db": "~/engram/.engram"
}
```

## License

MIT
