//! engram-mcp: MCP server exposing engram memory to AI agents.
//!
//! Protocol: JSON-RPC 2.0 over stdio
//!
//! Tools exposed:
//! - engram_add: Add a memory (text, code, fact, etc.)
//! - engram_search: Semantic search across memories
//! - engram_recall: Get specific memory by ID
//! - engram_forget: Remove a memory
//! - engram_list: List recent memories
//! - engram_stats: Get memory statistics

use std::io::{self, BufRead, Write};
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use engram_core::id::NodeId;
use engram_core::types::{Node, NodeType};
use engram_fts::FtsIndex;
use engram_store::EngramStore;
use engram_temporal::{LifecycleConfig, LifecycleManager};
use engram_vector::VectorIndex;

// ── MCP Protocol Types ──────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct JsonRpcRequest {
    #[allow(dead_code)]
    jsonrpc: String,
    id: Option<Value>,
    method: String,
    #[serde(default)]
    params: Value,
}

#[derive(Debug, Serialize)]
struct JsonRpcResponse {
    jsonrpc: String,
    id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
}

#[derive(Debug, Serialize)]
struct JsonRpcError {
    code: i32,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<Value>,
}

// ── Tool Definitions ────────────────────────────────────────────────────────

fn get_tools() -> Vec<Value> {
    vec![
        json!({
            "name": "engram_add",
            "description": "Add a memory to engram. Use for facts, code snippets, decisions, or any information worth remembering across sessions.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The content to remember"
                    },
                    "type": {
                        "type": "string",
                        "enum": ["note", "fact", "concept", "entity", "event", "document", "chunk"],
                        "description": "Type of memory (default: note)"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags for categorization"
                    }
                },
                "required": ["content"]
            }
        }),
        json!({
            "name": "engram_search",
            "description": "Search memories by semantic similarity or keywords. Returns most relevant memories.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (semantic or keyword)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default: 5)"
                    },
                    "type": {
                        "type": "string",
                        "enum": ["note", "fact", "concept", "entity", "event", "document", "chunk"],
                        "description": "Filter by type (optional)"
                    }
                },
                "required": ["query"]
            }
        }),
        json!({
            "name": "engram_recall",
            "description": "Get a specific memory by its ID.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "Memory ID"
                    }
                },
                "required": ["id"]
            }
        }),
        json!({
            "name": "engram_forget",
            "description": "Remove a memory by ID. Use sparingly - only for incorrect or outdated information.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "Memory ID to forget"
                    }
                },
                "required": ["id"]
            }
        }),
        json!({
            "name": "engram_list",
            "description": "List recent memories, optionally filtered by type.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default: 10)"
                    },
                    "type": {
                        "type": "string",
                        "enum": ["note", "fact", "concept", "entity", "event", "document", "chunk"],
                        "description": "Filter by type (optional)"
                    }
                }
            }
        }),
        json!({
            "name": "engram_stats",
            "description": "Get memory statistics: total count, types, storage size.",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        }),
    ]
}

// ── Engram Server ───────────────────────────────────────────────────────────

struct EngramServer {
    store: EngramStore,
    fts: FtsIndex,
    #[allow(dead_code)]
    vector: VectorIndex,
    lifecycle: LifecycleManager,
}

impl EngramServer {
    fn new(data_dir: PathBuf) -> Result<Self, String> {
        let store_path = data_dir.join("store");
        let fts_path = data_dir.join("fts");
        let vector_path = data_dir.join("vector");

        let store =
            EngramStore::open(&store_path).map_err(|e| format!("Failed to open store: {}", e))?;

        let fts = FtsIndex::open(&fts_path).map_err(|e| format!("Failed to open FTS: {}", e))?;

        let vector = VectorIndex::new(384, &vector_path) // BGE small dim
            .map_err(|e| format!("Failed to open vector index: {}", e))?;

        let lifecycle = LifecycleManager::new(LifecycleConfig::default());

        Ok(Self {
            store,
            fts,
            vector,
            lifecycle,
        })
    }

    fn handle_add(&mut self, params: Value) -> Result<Value, String> {
        let content = params
            .get("content")
            .and_then(|v| v.as_str())
            .ok_or("content is required")?;

        let node_type = match params.get("type").and_then(|v| v.as_str()) {
            Some("fact") => NodeType::Fact,
            Some("concept") => NodeType::Concept,
            Some("entity") => NodeType::Entity,
            Some("event") => NodeType::Event,
            Some("document") => NodeType::Document,
            Some("chunk") => NodeType::Chunk,
            _ => NodeType::Note,
        };

        let tags: Vec<String> = params
            .get("tags")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        let mut node = Node::new(content, node_type);
        node.tags = tags;

        // Store
        self.store
            .put_node(&node)
            .map_err(|e| format!("Failed to store: {}", e))?;

        // Index in FTS
        self.fts
            .index_node(&node)
            .map_err(|e| format!("Failed to index: {}", e))?;
        
        // Commit FTS so search works immediately
        self.fts
            .commit()
            .map_err(|e| format!("Failed to commit FTS: {}", e))?;

        // Record in lifecycle manager
        self.lifecycle.record_access(&node.id);

        Ok(json!({
            "id": node.id.to_string(),
            "message": "Memory added successfully"
        }))
    }

    fn handle_search(&self, params: Value) -> Result<Value, String> {
        let query = params
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or("query is required")?;

        let limit = params.get("limit").and_then(|v| v.as_u64()).unwrap_or(5) as usize;

        // FTS search - returns Vec<(NodeId, f32)>
        let results = self
            .fts
            .search(query, limit)
            .map_err(|e| format!("Search failed: {}", e))?;

        let memories: Vec<Value> = results
            .iter()
            .filter_map(|(node_id, score)| {
                self.store.get_node(node_id).ok().flatten().map(|node| {
                    json!({
                        "id": node.id.to_string(),
                        "content": node.body,
                        "type": node.node_type.to_string(),
                        "tags": node.tags,
                        "score": score,
                        "created": node.tx_time.to_rfc3339()
                    })
                })
            })
            .collect();

        Ok(json!({
            "results": memories,
            "count": memories.len()
        }))
    }

    fn handle_recall(&self, params: Value) -> Result<Value, String> {
        let id_str = params
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or("id is required")?;

        let id = NodeId::from_string(id_str);

        match self.store.get_node(&id).map_err(|e| e.to_string())? {
            Some(node) => Ok(json!({
                "id": node.id.to_string(),
                "content": node.body,
                "type": node.node_type.to_string(),
                "tags": node.tags,
                "confidence": node.confidence,
                "created": node.tx_time.to_rfc3339()
            })),
            None => Err("Memory not found".to_string()),
        }
    }

    fn handle_forget(&mut self, params: Value) -> Result<Value, String> {
        let id_str = params
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or("id is required")?;

        let id = NodeId::from_string(id_str);

        self.store
            .delete_node(&id)
            .map_err(|e| format!("Failed to delete: {}", e))?;

        Ok(json!({
            "message": "Memory forgotten"
        }))
    }

    fn handle_list(&self, params: Value) -> Result<Value, String> {
        let limit = params.get("limit").and_then(|v| v.as_u64()).unwrap_or(10) as usize;

        let type_filter = params
            .get("type")
            .and_then(|v| v.as_str())
            .map(|s| match s {
                "fact" => NodeType::Fact,
                "concept" => NodeType::Concept,
                "entity" => NodeType::Entity,
                "event" => NodeType::Event,
                "document" => NodeType::Document,
                "chunk" => NodeType::Chunk,
                _ => NodeType::Note,
            });

        let nodes = self
            .store
            .list_nodes(type_filter, limit)
            .map_err(|e| format!("Failed to list: {}", e))?;

        let memories: Vec<Value> = nodes
            .iter()
            .map(|node| {
                json!({
                    "id": node.id.to_string(),
                    "content": if node.body.len() > 100 {
                        format!("{}...", &node.body[..100])
                    } else {
                        node.body.clone()
                    },
                    "type": node.node_type.to_string(),
                    "tags": node.tags,
                    "created": node.tx_time.to_rfc3339()
                })
            })
            .collect();

        Ok(json!({
            "memories": memories,
            "count": memories.len()
        }))
    }

    fn handle_stats(&self) -> Result<Value, String> {
        let stats = self
            .store
            .stats()
            .map_err(|e| format!("Failed to get stats: {}", e))?;

        let fts_count = self.fts.doc_count().unwrap_or(0);
        let vec_count = self.vector.len();

        Ok(json!({
            "total_nodes": stats.node_count,
            "total_edges": stats.edge_count,
            "fts_indexed": fts_count,
            "vector_indexed": vec_count
        }))
    }
}

// ── MCP Protocol Handler ────────────────────────────────────────────────────

fn handle_request(server: &mut EngramServer, request: JsonRpcRequest) -> JsonRpcResponse {
    let id = request.id.clone().unwrap_or(Value::Null);

    let result = match request.method.as_str() {
        // MCP lifecycle
        "initialize" => Ok(json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {}
            },
            "serverInfo": {
                "name": "engram",
                "version": "0.1.0"
            }
        })),

        "notifications/initialized" => Ok(json!({})),

        // Tool listing
        "tools/list" => Ok(json!({
            "tools": get_tools()
        })),

        // Tool calls
        "tools/call" => {
            let tool_name = request
                .params
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("");

            let args = request
                .params
                .get("arguments")
                .cloned()
                .unwrap_or(json!({}));

            let tool_result = match tool_name {
                "engram_add" => server.handle_add(args),
                "engram_search" => server.handle_search(args),
                "engram_recall" => server.handle_recall(args),
                "engram_forget" => server.handle_forget(args),
                "engram_list" => server.handle_list(args),
                "engram_stats" => server.handle_stats(),
                _ => Err(format!("Unknown tool: {}", tool_name)),
            };

            match tool_result {
                Ok(content) => Ok(json!({
                    "content": [{
                        "type": "text",
                        "text": serde_json::to_string_pretty(&content).unwrap_or_default()
                    }]
                })),
                Err(e) => Ok(json!({
                    "content": [{
                        "type": "text",
                        "text": format!("Error: {}", e)
                    }],
                    "isError": true
                })),
            }
        }

        _ => Err(format!("Unknown method: {}", request.method)),
    };

    match result {
        Ok(r) => JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id,
            result: Some(r),
            error: None,
        },
        Err(e) => JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id,
            result: None,
            error: Some(JsonRpcError {
                code: -32603,
                message: e,
                data: None,
            }),
        },
    }
}

// ── Main ────────────────────────────────────────────────────────────────────

fn main() {
    // Setup logging to stderr (stdout is for MCP protocol)
    tracing_subscriber::fmt()
        .with_writer(io::stderr)
        .with_env_filter("engram=info")
        .init();

    // Data directory
    let data_dir = dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("engram");

    std::fs::create_dir_all(&data_dir).expect("Failed to create data dir");

    // Initialize server
    let mut server = EngramServer::new(data_dir).expect("Failed to initialize engram server");

    eprintln!("engram-mcp server started");

    // Read JSON-RPC from stdin, write to stdout
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };

        if line.trim().is_empty() {
            continue;
        }

        let request: JsonRpcRequest = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("Invalid JSON-RPC: {}", e);
                continue;
            }
        };

        // Skip notifications (no id)
        if request.id.is_none() && request.method.starts_with("notifications/") {
            continue;
        }

        let response = handle_request(&mut server, request);

        let json = serde_json::to_string(&response).unwrap();
        writeln!(stdout, "{}", json).unwrap();
        stdout.flush().unwrap();
    }
}
