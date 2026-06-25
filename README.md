# engram

Multi-modal knowledge database for AI agents. Sled + Tantivy + flat vector search with RRF fusion.

[![CI](https://github.com/skeehn/engram/actions/workflows/ci.yml/badge.svg)](https://github.com/skeehn/engram/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Features

- Sled key-value storage (pure Rust, no RocksDB)
- Tantivy full-text search
- Flat cosine vector index over 1024-dim Jina v3 embeddings
- RRF fusion for hybrid (FTS + vector) retrieval
- URL ingestion via Jina Reader
- Graph operations (neighborhood traversal)
- Bitemporal timestamps on every node (tx-time + valid-time)
- HTTP API (`engram serve`)

## Architecture

10 Rust crates:
- engram-core: Core types, IDs, and traits
- engram-store: Sled storage (nodes, edges, clusters, objects)
- engram-graph: Graph operations and traversal
- engram-fts: Tantivy full-text search
- engram-vector: Flat cosine vector index
- engram-embed: Jina v3 embedding client
- engram-query: RRF hybrid fusion (the read path)
- engram-extract: Jina Reader URL ingestion
- engram-temporal: Bitemporal validity helpers
- engram-cli: CLI binary + HTTP server

## Requirements

`engram` builds and runs with no external services. The embedding-backed
commands (`add`, `search`, `ingest`) call the Jina AI API, so set a key first:

```bash
export JINA_API_KEY=your_key   # free key at https://jina.ai
```

## Quick Start

```bash
# Build (release recommended — dev profile is also optimized)
cargo build --release

# Add knowledge (embeds via Jina)
engram add "Knowledge fact" --tags "tag1,tag2"

# Hybrid search (FTS + vector, RRF-fused)
engram search "query"

# Inspect: stats, list, single node, graph neighborhood
engram stats
engram list
engram get <node_id>
engram graph <node_id>

# Ingest a URL via Jina Reader
engram ingest https://example.com

# Serve the HTTP API: GET /health, GET /search?q=…&top_k=…, POST /add
engram serve --port 7474
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
