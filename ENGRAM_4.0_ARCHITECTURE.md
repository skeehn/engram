# engram 4.0: World-Class Local AI Context System

## Vision
The fastest, most storage-efficient, AI-native local context/memory/search system.
Beat everything: Chroma, Qdrant, Milvus, Pinecone, Weaviate, LanceDB, SuperMemory.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         MCP SERVER                               │
│  JSON-RPC over stdio/HTTP - AI agent interface                  │
├─────────────────────────────────────────────────────────────────┤
│                         HTTP DAEMON                              │
│  REST API + WebSocket for real-time updates                     │
├─────────────────────────────────────────────────────────────────┤
│                      QUERY ENGINE                                │
│  Hybrid: FTS + Semantic + Graph + Temporal                      │
├──────────────┬──────────────┬──────────────┬───────────────────┤
│   BINARY     │    HNSW      │     FTS      │      GRAPH        │
│   INDEX      │   (usearch)  │  (tantivy)   │   (petgraph)      │
│  Ultra-fast  │   Refined    │   Keyword    │   Relations       │
│  coarse      │   search     │   search     │   & entities      │
├──────────────┴──────────────┴──────────────┴───────────────────┤
│                    SIMD KERNEL LAYER                            │
│  AVX2/NEON Hamming, cache-aligned, zero-copy, io_uring         │
├─────────────────────────────────────────────────────────────────┤
│                    STORAGE LAYER                                 │
│  mmap'd binary vectors | sled KV | optional f32 rescore file   │
├─────────────────────────────────────────────────────────────────┤
│                    FILE WATCHER                                  │
│  notify-rs, content-hash dedup, auto-index on change           │
└─────────────────────────────────────────────────────────────────┘
```

## Storage Tiers (1M vectors @ 384d)

| Tier | Size | Use Case |
|------|------|----------|
| Binary (1-bit) | 46 MB | Coarse search, always in RAM |
| I8 quant | 366 MB | HNSW refined search |
| F32 original | 1,465 MB | Rescoring (mmap'd, lazy) |
| **Hybrid** | **~100 MB** | Binary index + I8 HNSW |

## Optimizations

### 1. SIMD Kernel (binary_simd.rs)
- AVX2 `vpopcount` for Hamming distance (x86_64)
- NEON `vcnt` for ARM (Apple Silicon)
- Process 256 bits per instruction = 8x speedup
- Explicit cache prefetch for sequential scans

### 2. Memory Layout
- Cache-line aligned (64 bytes) for all hot data
- Struct of Arrays (SoA) over Array of Structs (AoS)
- Contiguous binary vectors for SIMD streaming
- mmap with `MAP_POPULATE` for pre-fault

### 3. Two-Stage Search
```
Query → Binary Index (Hamming, top-1000) → HNSW (cosine, top-10)
        ~0.5ms @ 1M                        ~0.1ms @ 1000
        Total: <1ms for 1M vectors
```

### 4. I/O Optimizations
- `io_uring` on Linux for async reads
- `kqueue` on macOS
- Direct I/O for large sequential scans
- Write coalescing for batch inserts

### 5. AI-Native Features
- Matryoshka embeddings: store 384d, query at 64/128/256
- Temporal decay: recent > old (configurable half-life)
- Context budgeting: pack results into token limits
- Agent memory: working/episodic/semantic separation

## File Structure
```
~/.engram/
├── config.toml           # Configuration
├── data/
│   ├── binary.idx        # Binary vectors (mmap'd)
│   ├── hnsw.idx          # HNSW graph (usearch)
│   ├── originals.f32     # Original vectors (lazy mmap)
│   ├── metadata.sled/    # KV store for metadata
│   └── fts.tantivy/      # Full-text index
├── workspaces/
│   ├── default/          # Per-workspace isolation
│   └── project-x/
└── daemon.sock           # Unix socket for IPC
```

## API (MCP + HTTP)

### MCP Tools
```json
{
  "tools": [
    {"name": "engram_add", "description": "Add content to memory"},
    {"name": "engram_search", "description": "Semantic search"},
    {"name": "engram_recall", "description": "Get context for prompt"},
    {"name": "engram_forget", "description": "Remove from memory"},
    {"name": "engram_link", "description": "Create entity relation"}
  ]
}
```

### HTTP Endpoints
```
POST /v1/add          # Add content
POST /v1/search       # Semantic search
POST /v1/recall       # Context retrieval
GET  /v1/stats        # Index statistics
WS   /v1/stream       # Real-time updates
```

## Benchmarks Target

| Metric | Target | Current Best |
|--------|--------|--------------|
| Insert | 500K vec/sec | Qdrant: 100K |
| Search@1M | <1ms p99 | Milvus: 2ms |
| Storage | 100MB/1M | Chroma: 500MB |
| Recall@10 | >95% | Standard: 95% |

## Implementation Order

1. **SIMD Kernel** - binary_simd.rs with AVX2/NEON
2. **Hybrid Index** - Binary coarse + HNSW fine
3. **Daemon** - HTTP server + file watcher integration
4. **MCP Server** - AI agent protocol
5. **Temporal + Agent Memory** - Advanced features
