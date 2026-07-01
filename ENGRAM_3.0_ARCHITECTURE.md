# engram 3.0: High-Performance Multimodal Context Filesystem

## Vision
A local-first knowledge filesystem that makes AI agents dramatically smarter by providing:
- **Speed**: <5ms semantic search across millions of items
- **Multimodal**: Text, code, images, audio, documents - all unified
- **Filesystem**: Mount your knowledge as a FUSE filesystem
- **Storage-efficient**: 10x compression via quantization
- **Agent-native**: MCP server + direct integration with Claude Code, Codex, grain

## Architecture Layers

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER INTERFACES                              │
├─────────────────────────────────────────────────────────────────────┤
│  FUSE Mount    │  CLI (engram)  │  MCP Server   │  HTTP API        │
│  /engram/      │  add/search    │  tools/       │  :7474           │
└────────┬───────┴───────┬────────┴───────┬───────┴────────┬─────────┘
         │               │                │                │
┌────────▼───────────────▼────────────────▼────────────────▼─────────┐
│                      QUERY ENGINE (v3)                              │
├─────────────────────────────────────────────────────────────────────┤
│  Hybrid Search: Vector + FTS + Graph                                │
│  Reranking: Cross-encoder or cosine                                 │
│  Query Planning: Route to optimal index                             │
└────────┬────────────────────────────────────────────────────────────┘
         │
┌────────▼────────────────────────────────────────────────────────────┐
│                    MULTIMODAL EMBEDDINGS                             │
├──────────────┬──────────────┬──────────────┬────────────────────────┤
│  Text        │  Code        │  Images      │  Documents             │
│  BGE-M3      │  CodeBERT    │  SigLIP      │  ColPali               │
│  384d        │  768d        │  512d        │  128 patches           │
│  ONNX/local  │  ONNX/local  │  ONNX/local  │  ONNX/local            │
└──────────────┴──────────────┴──────────────┴────────────────────────┘
         │
┌────────▼────────────────────────────────────────────────────────────┐
│                    VECTOR INDEX (usearch + i8)                       │
├─────────────────────────────────────────────────────────────────────┤
│  Primary: HNSW with i8 scalar quantization                          │
│  - 4x smaller than f32                                              │
│  - 60% faster search                                                │
│  - <1% recall loss                                                  │
│  Binary: Optional SimHash for coarse filtering                      │
└────────┬────────────────────────────────────────────────────────────┘
         │
┌────────▼────────────────────────────────────────────────────────────┐
│                    STORAGE LAYER                                     │
├──────────────┬──────────────┬──────────────┬────────────────────────┤
│  Nodes       │  FTS Index   │  Graph       │  Temporal              │
│  redb/SQLite │  tantivy     │  Adjacency   │  COW snapshots         │
│  CBOR encoded│  Incremental │  In-memory   │  blake3 hashes         │
└──────────────┴──────────────┴──────────────┴────────────────────────┘
```

## Data Types Support

### Text (Current)
- Embedder: BGE Small EN v1.5 (384d) or BGE-M3 (multilingual)
- Chunking: Sentence-level with overlap
- Special: Markdown header-aware splitting

### Code (New)
- Embedder: CodeBERT or StarEncoder (768d)
- Chunking: AST-aware (function/class boundaries)
- Metadata: Language, imports, exports, symbols
- Special: Dependency graph extraction

### Images (New)
- Embedder: SigLIP (512d, better than CLIP for retrieval)
- Processing: Resize to 224x224, normalize
- Metadata: EXIF, dimensions, detected objects
- Special: OCR for text in images (via tesseract or doctr)

### Documents (New)
- Embedder: ColPali (128 patch embeddings) or BGE-M3
- Processing: PDF → images → ColPali patches
- Metadata: Page count, table of contents, structure
- Special: Table/figure extraction

### Audio (Future)
- Embedder: CLAP (512d) or wav2vec
- Processing: Resample to 16kHz, segment
- Metadata: Duration, speech vs music
- Special: Transcript via Whisper, then text embedding

## FUSE Filesystem Design

### Mount Structure
```
/engram/
├── nodes/                    # Canonical storage
│   └── {ulid}/
│       ├── content.md        # Raw content
│       ├── meta.json         # Type, tags, timestamps
│       └── links/            # Symlinks to related nodes
│
├── by-type/                  # Virtual: type views
│   ├── fact/
│   ├── concept/
│   ├── code/
│   └── image/
│
├── by-tag/                   # Virtual: tag views
│   ├── rust/
│   ├── ml/
│   └── {custom}/
│
├── clusters/                 # Virtual: semantic clusters
│   └── {cluster-id}/         # Auto-generated groupings
│
├── temporal/                 # Virtual: point-in-time
│   └── at/{ISO-timestamp}/   # See graph as it was then
│
├── queries/                  # Virtual: live query results
│   └── {encoded-query}/      # e.g., /queries/type:code%20lang:rust/
│
└── .engram/                  # Hidden: raw database access
    ├── store.db
    ├── vectors.hnsw
    └── fts/
```

### Path Semantics
- `ls /engram/by-tag/rust/` → Lists all nodes tagged "rust"
- `cat /engram/nodes/{id}/content.md` → Read node content
- `echo "..." > /engram/nodes/new/content.md` → Create new node
- `grep -r "pattern" /engram/queries/type:code/` → Search code nodes

## Performance Targets

| Operation | Target | Current | Improvement |
|-----------|--------|---------|-------------|
| Single embed (text) | <5ms | 8ms | 1.6x |
| Batch embed (100) | <100ms | 200ms | 2x |
| Vector search (1M) | <5ms | 6ms | 1.2x |
| FTS search | <10ms | 15ms | 1.5x |
| Hybrid search | <15ms | N/A | New |
| Path resolution | <1ms | N/A | New |
| Directory listing | <10ms | N/A | New |

## Storage Efficiency

| Scale | f32 (current) | i8 (target) | Savings |
|-------|---------------|-------------|---------|
| 100K nodes | 150 MB | 40 MB | 73% |
| 1M nodes | 1.5 GB | 400 MB | 73% |
| 10M nodes | 15 GB | 4 GB | 73% |

## Implementation Phases

### Phase 1: i8 Quantization (1 week)
- [x] Add `ScalarKind::I8` to HnswIndex
- [ ] Quantization at insert time (f32 → i8)
- [ ] Dequantization for reranking
- [ ] Benchmark: verify <1% recall loss

### Phase 2: Multimodal Embeddings (2 weeks)
- [ ] Abstract `Embedder` trait for any modality
- [ ] Add SigLIP for images (ONNX)
- [ ] Add CodeBERT for code
- [ ] Content-type detection + routing

### Phase 3: FUSE Filesystem (2 weeks)
- [ ] Basic fuser integration
- [ ] `/nodes/` read/write
- [ ] `/by-type/` and `/by-tag/` virtual dirs
- [ ] File watcher for auto-indexing

### Phase 4: Query Engine v3 (1 week)
- [ ] Hybrid vector + FTS + graph
- [ ] Query planning and routing
- [ ] Cross-encoder reranking option

### Phase 5: MCP Server (1 week)
- [ ] Expose as MCP tools
- [ ] Integration with Claude Code
- [ ] Integration with Codex

## Crate Dependencies (New)

```toml
[workspace.dependencies]
# FUSE
fuser = "0.17"

# File watching
notify = "9.0"

# Multimodal (via fastembed or direct ONNX)
fastembed = { version = "5.17", features = ["image"] }
ort = "2.0"  # ONNX Runtime

# Storage
redb = "2.0"  # Faster than SQLite for embedded use
tantivy = "0.26"  # FTS

# Hashing
blake3 = "1.5"  # Content-addressable storage

# Async
tokio = { version = "1", features = ["full"] }
```

## Agent Integration

### MCP Tools
```json
{
  "tools": [
    {"name": "engram_search", "description": "Semantic search across all knowledge"},
    {"name": "engram_add", "description": "Add new knowledge (text, code, image)"},
    {"name": "engram_relate", "description": "Create relationship between nodes"},
    {"name": "engram_context", "description": "Get relevant context for a task"},
    {"name": "engram_remember", "description": "Store agent memory/learning"}
  ]
}
```

### Claude Code Integration
```bash
# In .claude/settings.json
{
  "mcpServers": {
    "engram": {
      "command": "engram",
      "args": ["mcp", "--db", "~/.engram"]
    }
  }
}
```

### Codex Integration
```bash
# In .codex/config.yaml
mcp_servers:
  - name: engram
    command: engram mcp --db ~/.engram
```

## Success Metrics

1. **Performance**: 5ms search @ 1M nodes
2. **Storage**: 4x reduction via i8 quantization
3. **Multimodal**: Support text + code + images + documents
4. **Integration**: Works with Claude Code, Codex, grain
5. **Filesystem**: Usable via standard UNIX tools

## Open Questions

1. Should images store raw bytes in engram or just path references?
2. How to handle large documents (>1MB)? Chunk vs summarize?
3. Graph storage: in-memory vs persistent? Recompute from edges?
4. Temporal: Full COW or diff-based? Storage vs performance tradeoff.
