# ENGRAM 2.0: The World's Best Local AI Context Database

**Goal:** Beat Chroma, SuperMemory, smfs.ai, OmniGraph, Mem0, MemGPT/Letta, and every other context system as of June 2026.

**Philosophy:** Git for agent memory - local-first, universal protocol, typed schemas, provenance tracking, cross-agent sharing.

---

## CURRENT STATE (engram 1.0)

### Architecture (343 nodes, 1028 edges from graphify)
- 10 Rust crates, ~3,900 LOC
- **Storage:** Sled KV (trees: nodes, edges, edge_rev, objects, clusters, delta_log)
- **FTS:** Tantivy BM25 full-text search
- **Vector:** Flat O(n) cosine scan, JSON persistence (~100K limit)
- **Graph:** petgraph BFS/DFS/PPR
- **Embeddings:** Jina v3 1024-dim (requires API)
- **Fusion:** RRF (keyword + vector + graph) + Jina reranker
- **API:** 9 HTTP endpoints on port 7474

### Critical Limitations
1. **O(n) vector search** - flat scan, doesn't scale
2. **Jina API required** - no offline/local operation
3. **No MCP server** - can't integrate with Claude Code/Codex
4. **No memory types** - all nodes are unstructured
5. **No provenance tracking** - can't trace why a memory exists
6. **No cross-agent sharing** - isolated memory per instance
7. **No temporal queries** - code exists but not wired to CLI
8. **JSON vector persistence** - inefficient for large indices

---

## COMPETITIVE ANALYSIS

### What They Do Well

| System | Strength | We Must Beat |
|--------|----------|--------------|
| **Chroma** | HNSW (hnswlib), SIMD (simsimd), RaBitQ quantization, battle-tested | Vector search performance |
| **SuperMemory** | Knowledge graph (not entity-relation), dynamic dreaming, auto-forget | Memory intelligence |
| **smfs.ai** | Filesystem interface, grep=semantic search, Unix philosophy | Agent integration |
| **Letta/MemGPT** | Self-improving memory, dreaming, memory doctor, skills | Memory lifecycle |
| **Mem0** | Universal layer, multiple backends, OpenAI-compatible API | Ecosystem support |

### What They ALL Lack (Our Opportunity)

1. **No universal agent protocol** - each requires custom integration
2. **No temporal intelligence** - can't query "context 3 days ago"
3. **No cross-agent sharing** - isolated memories per agent
4. **No memory provenance** - can't trace source of facts
5. **No active/push-based memory** - all pull-based
6. **No typed schemas** - all unstructured text
7. **No local-first with optional sync** - cloud-dependent or fully offline
8. **No memory views/compositions** - can't combine memory subsets

---

## ENGRAM 2.0 ARCHITECTURE

### Phase 1: Foundation (Make It Work)

#### 1.1 Local ONNX Embeddings
Replace Jina API with local inference using `fastembed-rs`:

```toml
# Cargo.toml additions
fastembed = "0.4"  # or latest
ort = "2.0"        # ONNX Runtime
```

**Default Model:** `bge-small-en-v1.5` (384d, ~130MB, MIT license)
- Alternative: `all-MiniLM-L6-v2` (384d, ~80MB, most tested)
- Advanced: `nomic-embed-text-v1.5` (768d, Matryoshka, variable dims)

**Implementation:**
```rust
// crates/engram-embed/src/local.rs
use fastembed::{TextEmbedding, TextInitOptions, EmbeddingModel};

pub struct LocalEmbedder {
    model: TextEmbedding,
    dimensions: usize,
}

impl LocalEmbedder {
    pub fn new(model: EmbeddingModel) -> Result<Self> {
        let model = TextEmbedding::try_new(
            TextInitOptions::new(model)
                .with_show_download_progress(true)
                .with_intra_threads(num_cpus::get())
        )?;
        Ok(Self { model, dimensions: 384 })
    }
    
    pub fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.model.embed(texts.to_vec(), None)
    }
}
```

#### 1.2 HNSW Vector Index
Replace flat scan with HNSW using `usearch` (smaller, faster than hnswlib):

```toml
# Cargo.toml
usearch = "3.0"  # or hnswlib = "0.3"
```

**Implementation:**
```rust
// crates/engram-vector/src/hnsw.rs
use usearch::Index;

pub struct HnswIndex {
    index: Index,
    dimensions: usize,
    path: PathBuf,
}

impl HnswIndex {
    pub fn new(dimensions: usize, path: impl AsRef<Path>) -> Result<Self> {
        let index = Index::new(
            usearch::Metric::Cosine,
            dimensions,
            usearch::IndexConfig {
                m: 16,                    // neighbors per node
                ef_construction: 128,      // build quality
                expansion_search: 64,      // query accuracy
            }
        )?;
        
        if path.exists() {
            index.load(&path)?;
        }
        
        Ok(Self { index, dimensions, path: path.as_ref().to_path_buf() })
    }
    
    pub fn upsert(&mut self, id: u64, embedding: &[f32]) -> Result<()> {
        self.index.add(id, embedding)?;
        Ok(())
    }
    
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u64, f32)>> {
        let results = self.index.search(query, k)?;
        Ok(results.keys.into_iter().zip(results.distances).collect())
    }
    
    pub fn save(&self) -> Result<()> {
        self.index.save(&self.path)?;
        Ok(())
    }
}
```

**HNSW Parameters (research-backed):**
- `m=16`: Good balance of memory/quality
- `ef_construction=128-200`: Higher = better index, slower build
- `ef_search=64-100`: Higher = better recall, slower query

#### 1.3 MCP Server
Native MCP support for Claude Code, Codex, grain integration:

```rust
// crates/engram-mcp/src/server.rs
use mcp_sdk::{Server, Tool, Resource};

pub struct EngramMcpServer {
    engram: Arc<Engram>,
}

impl Server for EngramMcpServer {
    fn tools(&self) -> Vec<Tool> {
        vec![
            Tool::new("engram_search", "Search memories by text query")
                .with_input("query", "string"),
            Tool::new("engram_add", "Add a new memory")
                .with_input("content", "string")
                .with_input("type", "string"),
            Tool::new("engram_get", "Get memory by ID")
                .with_input("id", "string"),
            Tool::new("engram_list", "List recent memories")
                .with_input("limit", "number"),
            Tool::new("engram_related", "Find related memories")
                .with_input("id", "string"),
        ]
    }
    
    fn resources(&self) -> Vec<Resource> {
        vec![
            Resource::new("engram://memory/{id}", "Memory by ID"),
            Resource::new("engram://recent", "Recent memories"),
            Resource::new("engram://profile", "Synthesized user profile"),
        ]
    }
}
```

**Config for agents:**
```json
// claude_desktop_config.json
{
  "mcpServers": {
    "engram": {
      "command": "engram",
      "args": ["mcp"],
      "env": { "ENGRAM_PATH": "~/.engram" }
    }
  }
}
```

---

### Phase 2: Intelligence (Make It Smart)

#### 2.1 Typed Memory Schemas

```rust
// crates/engram-core/src/types.rs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryType {
    Fact { source: String, confidence: f32 },
    Pattern { language: String, usage_count: u32 },
    Preference { user: String, last_updated: DateTime<Utc> },
    Decision { context: String, alternatives: Vec<String> },
    Error { error_type: String, resolution: Option<String> },
    Relationship { from: NodeId, to: NodeId, kind: String },
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypedNode {
    pub id: NodeId,
    pub content: String,
    pub memory_type: MemoryType,
    pub embedding: Option<Vec<f32>>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub provenance: Provenance,
    pub tags: Vec<String>,
}
```

#### 2.2 Provenance Tracking

```rust
// crates/engram-core/src/provenance.rs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Provenance {
    /// Original source (file, URL, conversation)
    pub source: SourceRef,
    /// How this memory was created
    pub method: CreationMethod,
    /// Parent memories this was derived from
    pub derived_from: Vec<NodeId>,
    /// Confidence in this memory (0.0-1.0)
    pub confidence: f32,
    /// Who/what created this memory
    pub creator: Creator,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CreationMethod {
    DirectInput,           // User explicitly added
    Extracted { model: String },  // LLM extracted
    Inferred { rule: String },    // Rule-based inference
    Consolidated { from: Vec<NodeId> },  // Merged from multiple
    Imported { format: String },  // Imported from other system
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Creator {
    User(String),
    Agent(String),
    System,
}
```

#### 2.3 Memory Lifecycle (SuperMemory-inspired)

```rust
// crates/engram-lifecycle/src/lib.rs

pub struct MemoryLifecycle {
    engram: Arc<Engram>,
}

impl MemoryLifecycle {
    /// Mark memory as updated (new version supersedes old)
    pub async fn update(&self, old_id: &NodeId, new_content: &str) -> Result<NodeId> {
        let new_id = self.engram.add_node(new_content)?;
        self.engram.add_edge(new_id, old_id, EdgeKind::Updates)?;
        self.engram.mark_obsolete(old_id)?;
        Ok(new_id)
    }
    
    /// Mark memory as extending another
    pub async fn extend(&self, base_id: &NodeId, extension: &str) -> Result<NodeId> {
        let ext_id = self.engram.add_node(extension)?;
        self.engram.add_edge(ext_id, base_id, EdgeKind::Extends)?;
        Ok(ext_id)
    }
    
    /// Derive new memory from existing ones
    pub async fn derive(&self, sources: &[NodeId], derived: &str) -> Result<NodeId> {
        let der_id = self.engram.add_node(derived)?;
        for src in sources {
            self.engram.add_edge(der_id, src, EdgeKind::DerivedFrom)?;
        }
        Ok(der_id)
    }
    
    /// Background consolidation (dreaming)
    pub async fn consolidate(&self) -> Result<ConsolidationReport> {
        // Find duplicate/similar memories
        // Merge redundant facts
        // Strengthen frequently accessed memories
        // Decay unused memories
        todo!()
    }
}
```

---

### Phase 3: Integration (Make It Universal)

#### 3.1 Filesystem Interface (smfs-inspired)

```rust
// crates/engram-fuse/src/lib.rs
use fuser::{Filesystem, MountOption};

pub struct EngramFs {
    engram: Arc<Engram>,
}

impl Filesystem for EngramFs {
    // Virtual filesystem structure:
    // /engram/
    //   memories/
    //     <id>.md          # Individual memories
    //   types/
    //     facts/
    //     patterns/
    //     preferences/
    //   profile.md         # Synthesized user profile
    //   recent.md          # Recent memories
    //   search/            # Write query, read results
}
```

**Usage:**
```bash
# Mount engram as filesystem
engram mount ~/engram-fs

# Search via grep (transparently semantic)
grep "authentication pattern" ~/engram-fs/memories/

# Read synthesized profile
cat ~/engram-fs/profile.md

# Add memory by writing file
echo "User prefers dark mode" > ~/engram-fs/memories/new.md
```

#### 3.2 Cross-Agent Memory Bus

```rust
// crates/engram-bus/src/lib.rs
pub struct MemoryBus {
    pubsub: PubSub,
    namespaces: HashMap<String, Vec<AgentId>>,
}

impl MemoryBus {
    /// Subscribe agent to namespace changes
    pub fn subscribe(&mut self, agent: AgentId, namespace: &str) -> Receiver<MemoryEvent> {
        self.pubsub.subscribe(namespace)
    }
    
    /// Publish memory to namespace
    pub fn publish(&self, namespace: &str, memory: &TypedNode) {
        self.pubsub.publish(namespace, MemoryEvent::Added(memory.clone()));
    }
    
    /// Share namespace between agents
    pub fn share(&mut self, namespace: &str, agents: &[AgentId]) {
        self.namespaces.insert(namespace.to_string(), agents.to_vec());
    }
}

// Usage: coding agent shares patterns with review agent
bus.share("code-patterns", &[codex_agent, review_agent]);
```

#### 3.3 Active Memory Triggers

```rust
// crates/engram-triggers/src/lib.rs
pub struct MemoryTrigger {
    condition: TriggerCondition,
    action: TriggerAction,
}

pub enum TriggerCondition {
    PathMatch(Glob),           // Working in auth/ directory
    ContentMatch(String),      // Editing authentication code
    TimeRange(DateTime, DateTime),  // During work hours
    MemoryAge(Duration),       // Memory older than X
    Custom(Box<dyn Fn(&Context) -> bool>),
}

pub enum TriggerAction {
    InjectContext(Vec<NodeId>),  // Push memories to agent
    Notify(String),              // Alert user
    Consolidate,                 // Run memory cleanup
    Archive,                     // Move to archive
}

// Example: auto-inject security memories when in auth directory
triggers.add(MemoryTrigger {
    condition: PathMatch("**/auth/**"),
    action: InjectContext(security_memories),
});
```

---

### Phase 4: Performance (Make It Fast)

#### 4.1 SIMD Distance Functions

```toml
# Cargo.toml
simsimd = "5.0"  # Same library Chroma uses
```

```rust
// crates/engram-vector/src/simd.rs
use simsimd::SpatialSimilarity;

pub fn cosine_simd(a: &[f32], b: &[f32]) -> f32 {
    f32::cosine(a, b).unwrap_or(0.0)
}

pub fn dot_simd(a: &[f32], b: &[f32]) -> f32 {
    f32::dot(a, b).unwrap_or(0.0)
}

// Auto-detects: AVX-512, AVX2, NEON, SVE
```

#### 4.2 Quantization (for 1M+ memories)

```rust
// crates/engram-vector/src/quantize.rs
pub enum Quantization {
    None,           // f32 (default)
    Scalar8,        // i8 (4x compression)
    Binary,         // 1-bit (32x compression)
    ProductQuant,   // PQ (configurable)
}

impl VectorIndex {
    pub fn with_quantization(self, quant: Quantization) -> Self {
        // Apply quantization to reduce memory
    }
}
```

#### 4.3 Memory-Mapped Storage

```rust
// crates/engram-store/src/mmap.rs
use memmap2::MmapMut;

pub struct MmapVectorStore {
    mmap: MmapMut,
    header: StoreHeader,
}

// Zero-copy vector access for large indices
```

---

## IMPLEMENTATION PHASES

### Phase 1: Foundation (Week 1-2)
- [ ] Add `fastembed-rs` for local embeddings
- [ ] Replace flat index with HNSW (usearch)
- [ ] Add MCP server support
- [ ] Wire temporal queries to CLI
- [ ] Tests for all new components

### Phase 2: Intelligence (Week 3-4)
- [ ] Implement typed memory schemas
- [ ] Add provenance tracking
- [ ] Implement memory lifecycle (update/extend/derive)
- [ ] Add basic consolidation (duplicate detection)
- [ ] Memory doctor (stale/contradiction detection)

### Phase 3: Integration (Week 5-6)
- [ ] FUSE filesystem interface
- [ ] Cross-agent memory bus (pub/sub)
- [ ] Active memory triggers
- [ ] Memory views and compositions
- [ ] Agent-specific configurations

### Phase 4: Performance (Week 7-8)
- [ ] SIMD distance functions (simsimd)
- [ ] Scalar quantization support
- [ ] Memory-mapped vector storage
- [ ] Benchmark suite vs Chroma
- [ ] Documentation and examples

---

## SUCCESS METRICS

1. **Local embeddings:** <50ms for 384-dim embedding (CPU)
2. **Vector search:** <10ms for k=10 on 1M vectors (HNSW)
3. **Memory footprint:** <500MB RAM for 1M memories (with quantization)
4. **MCP latency:** <100ms tool response time
5. **Zero API dependencies:** Works fully offline
6. **Agent compatibility:** Works with Claude Code, Codex, grain out-of-box

---

## DIFFERENTIATORS vs COMPETITION

| Feature | Chroma | SuperMemory | smfs.ai | Mem0 | engram 2.0 |
|---------|--------|-------------|---------|------|------------|
| Local-first | ✓ | ✗ | ✗ | ✗ | **✓** |
| HNSW | ✓ | ? | ? | ✓ | **✓** |
| Typed schemas | ✗ | ✓ | ✗ | ✗ | **✓** |
| Provenance | ✗ | ✗ | ✗ | ✗ | **✓** |
| MCP native | ✗ | ✗ | ✗ | ✓ | **✓** |
| Filesystem | ✗ | ✗ | ✓ | ✗ | **✓** |
| Cross-agent | ✗ | ✗ | ✗ | ✗ | **✓** |
| Triggers | ✗ | ✗ | ✗ | ✗ | **✓** |
| Git-native | ✗ | ✗ | ✗ | ✗ | **✓** |

---

## FILES TO CREATE/MODIFY

### New Crates
- `crates/engram-mcp/` - MCP server
- `crates/engram-fuse/` - FUSE filesystem
- `crates/engram-bus/` - Cross-agent pub/sub
- `crates/engram-triggers/` - Active memory
- `crates/engram-lifecycle/` - Memory lifecycle
- `crates/engram-quantize/` - Vector quantization

### Modified Crates
- `crates/engram-embed/` - Add local ONNX
- `crates/engram-vector/` - Replace with HNSW
- `crates/engram-core/` - Typed schemas, provenance
- `crates/engram-cli/` - New commands (mcp, mount, trigger)

---

## BASETEN $18 CREDIT

With $18 Baseten credit, we could:
1. **Test Mixedbread Embed** - Higher quality embeddings for benchmark
2. **Run comparison benchmarks** - engram vs Chroma vs hosted solutions
3. **Profile embedding quality** - Compare BGE vs MiniLM vs Mixedbread

But for engram 2.0, the goal is **zero cloud dependency**. Baseten could be optional for users who want premium embeddings.
