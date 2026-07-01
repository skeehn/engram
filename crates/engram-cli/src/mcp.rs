//! MCP (Model Context Protocol) server for engram.
//!
//! Implements JSON-RPC over stdin/stdout for AI agent integration.
//! Compatible with Claude Code, Cursor, Windsurf, and any MCP client.
//!
//! Tools exposed:
//! - engram_search: Semantic + FTS hybrid search
//! - engram_add: Add knowledge to the database
//! - engram_stats: Get index statistics
//! - engram_index_file: Index a file from disk

use std::io::{self, BufRead, Write};
use std::sync::Arc;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::{error, info};

use crate::v2::EngramContext;

// ── MCP Protocol Types ───────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct JsonRpcRequest {
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
}

#[derive(Debug, Serialize)]
struct ToolDescription {
    name: String,
    description: String,
    #[serde(rename = "inputSchema")]
    input_schema: Value,
}

// ── MCP Server ───────────────────────────────────────────────────────────────

pub struct McpServer {
    ctx: Arc<EngramContext>,
}

impl McpServer {
    pub fn new(ctx: EngramContext) -> Self {
        Self { ctx: Arc::new(ctx) }
    }

    /// Run the MCP server (blocks on stdin/stdout).
    pub async fn run(&self) -> Result<()> {
        info!("engram MCP server starting (stdin/stdout)");

        let stdin = io::stdin();
        let stdout = io::stdout();

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
                    error!(error = %e, "invalid JSON-RPC request");
                    continue;
                }
            };

            let response = self.handle_request(request).await;

            if let Some(resp) = response {
                let mut out = stdout.lock();
                let _ = serde_json::to_writer(&mut out, &resp);
                let _ = out.write_all(b"\n");
                let _ = out.flush();
            }
        }

        Ok(())
    }

    async fn handle_request(&self, req: JsonRpcRequest) -> Option<JsonRpcResponse> {
        let id = req.id.clone().unwrap_or(Value::Null);

        match req.method.as_str() {
            "initialize" => Some(self.handle_initialize(id)),
            "tools/list" => Some(self.handle_tools_list(id)),
            "tools/call" => Some(self.handle_tools_call(id, req.params).await),
            "notifications/initialized" => None, // No response needed
            "ping" => Some(JsonRpcResponse {
                jsonrpc: "2.0".into(),
                id,
                result: Some(serde_json::json!({})),
                error: None,
            }),
            _ => Some(JsonRpcResponse {
                jsonrpc: "2.0".into(),
                id,
                result: None,
                error: Some(JsonRpcError {
                    code: -32601,
                    message: format!("Method not found: {}", req.method),
                }),
            }),
        }
    }

    fn handle_initialize(&self, id: Value) -> JsonRpcResponse {
        JsonRpcResponse {
            jsonrpc: "2.0".into(),
            id,
            result: Some(serde_json::json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": "engram",
                    "version": "4.0.0"
                }
            })),
            error: None,
        }
    }

    fn handle_tools_list(&self, id: Value) -> JsonRpcResponse {
        let tools = vec![
            ToolDescription {
                name: "engram_search".into(),
                description: "Search engram knowledge base using hybrid semantic + full-text search. Returns ranked results by relevance.".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query (natural language or keywords)"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results to return (default: 10)",
                            "default": 10
                        }
                    },
                    "required": ["query"]
                }),
            },
            ToolDescription {
                name: "engram_add".into(),
                description: "Add a knowledge node to engram. Automatically embeds and indexes for future retrieval.".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "body": {
                            "type": "string",
                            "description": "Content to store"
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tags for categorization"
                        },
                        "node_type": {
                            "type": "string",
                            "enum": ["fact", "concept", "entity", "event", "document", "note"],
                            "description": "Type of knowledge (default: fact)",
                            "default": "fact"
                        }
                    },
                    "required": ["body"]
                }),
            },
            ToolDescription {
                name: "engram_stats".into(),
                description: "Get engram database statistics: node count, vector count, FTS docs.".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {}
                }),
            },
            ToolDescription {
                name: "engram_index_file".into(),
                description: "Index a file from disk into engram. Reads content, embeds, and stores for retrieval.".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Absolute path to file to index"
                        },
                        "project": {
                            "type": "string",
                            "description": "Optional project namespace tag"
                        }
                    },
                    "required": ["path"]
                }),
            },
        ];

        JsonRpcResponse {
            jsonrpc: "2.0".into(),
            id,
            result: Some(serde_json::json!({"tools": tools})),
            error: None,
        }
    }

    async fn handle_tools_call(&self, id: Value, params: Value) -> JsonRpcResponse {
        let name = params
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let arguments = params
            .get("arguments")
            .cloned()
            .unwrap_or(Value::Object(Default::default()));

        let result = match name {
            "engram_search" => self.tool_search(arguments).await,
            "engram_add" => self.tool_add(arguments).await,
            "engram_stats" => self.tool_stats().await,
            "engram_index_file" => self.tool_index_file(arguments).await,
            _ => Err(format!("Unknown tool: {}", name)),
        };

        match result {
            Ok(content) => JsonRpcResponse {
                jsonrpc: "2.0".into(),
                id,
                result: Some(serde_json::json!({
                    "content": [{
                        "type": "text",
                        "text": content
                    }]
                })),
                error: None,
            },
            Err(e) => JsonRpcResponse {
                jsonrpc: "2.0".into(),
                id,
                result: Some(serde_json::json!({
                    "content": [{
                        "type": "text",
                        "text": format!("Error: {}", e)
                    }],
                    "isError": true
                })),
                error: None,
            },
        }
    }

    // ── Tool Implementations ─────────────────────────────────────────────────

    async fn tool_search(&self, args: Value) -> std::result::Result<String, String> {
        let query = args
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or("missing 'query' argument")?;
        let top_k = args
            .get("top_k")
            .and_then(|v| v.as_u64())
            .unwrap_or(10) as usize;

        let results = self
            .ctx
            .search(query, top_k)
            .await
            .map_err(|e| e.to_string())?;

        if results.is_empty() {
            return Ok("No results found.".into());
        }

        // Load full nodes for display
        let mut output = String::new();
        for (i, (id, score)) in results.iter().enumerate() {
            use engram_core::id::NodeId;
            let node_id = NodeId::from(id.as_str());
            if let Ok(Some(node)) = self.ctx.store.get_node(&node_id) {
                let body_preview = if node.body.len() > 200 {
                    &node.body[..200]
                } else {
                    &node.body
                };
                output.push_str(&format!(
                    "[{}] score={:.3} type={} tags={:?}\n{}\n\n",
                    i + 1,
                    score,
                    node.node_type,
                    node.tags,
                    body_preview
                ));
            } else {
                output.push_str(&format!("[{}] id={} score={:.3}\n\n", i + 1, id, score));
            }
        }

        Ok(output)
    }

    async fn tool_add(&self, args: Value) -> std::result::Result<String, String> {
        use engram_core::types::{Node, NodeType};

        let body = args
            .get("body")
            .and_then(|v| v.as_str())
            .ok_or("missing 'body' argument")?;

        let node_type = args
            .get("node_type")
            .and_then(|v| v.as_str())
            .unwrap_or("fact");

        let tags: Vec<String> = args
            .get("tags")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        let nt = match node_type {
            "fact" => NodeType::Fact,
            "concept" => NodeType::Concept,
            "entity" => NodeType::Entity,
            "document" => NodeType::Document,
            "note" => NodeType::Note,
            other => NodeType::Custom(other.to_string()),
        };

        let node = Node::new(body.to_string(), nt).with_tags(tags);
        let id = node.id.clone();

        self.ctx.store.put_node(&node).map_err(|e| e.to_string())?;

        if let Err(e) = self.ctx.fts.index_node(&node) {
            error!(error = %e, "FTS index failed");
        }
        let _ = self.ctx.fts.commit();

        match self.ctx.embed(body).await {
            Ok(embedding) => {
                let _ = self.ctx.vector.upsert(&id, &embedding);
                let _ = self.ctx.vector.save();
            }
            Err(e) => error!(error = %e, "embedding failed"),
        }

        Ok(format!("Added node: {}", id.as_ref()))
    }

    async fn tool_stats(&self) -> std::result::Result<String, String> {
        let s = self.ctx.stats();
        Ok(format!(
            "Nodes: {}\nFTS docs: {}\nVectors: {} (HNSW, 384d)",
            s.nodes, s.fts_docs, s.vectors
        ))
    }

    async fn tool_index_file(&self, args: Value) -> std::result::Result<String, String> {
        use engram_core::types::{Node, NodeType};
        use std::path::PathBuf;

        let path_str = args
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or("missing 'path' argument")?;

        let path = PathBuf::from(path_str);
        if !path.exists() {
            return Err(format!("file not found: {}", path_str));
        }

        let content = std::fs::read_to_string(&path).map_err(|e| e.to_string())?;

        if content.is_empty() {
            return Err("file is empty".into());
        }

        let file_name = path
            .file_name()
            .map(|f| f.to_string_lossy().to_string())
            .unwrap_or_default();

        let mut tags = vec![format!("file:{}", file_name)];
        if let Some(ext) = path.extension() {
            tags.push(format!("ext:{}", ext.to_string_lossy()));
        }
        if let Some(p) = args.get("project").and_then(|v| v.as_str()) {
            tags.push(format!("project:{}", p));
        }
        tags.push(format!("path:{}", path.display()));

        let body = if content.len() > 4096 {
            content[..4096].to_string()
        } else {
            content.clone()
        };

        let node = Node::new(body.clone(), NodeType::Document).with_tags(tags);
        let id = node.id.clone();

        self.ctx.store.put_node(&node).map_err(|e| e.to_string())?;
        if let Err(e) = self.ctx.fts.index_node(&node) {
            error!(error = %e, "FTS failed");
        }
        let _ = self.ctx.fts.commit();

        match self.ctx.embed(&body).await {
            Ok(embedding) => {
                let _ = self.ctx.vector.upsert(&id, &embedding);
                let _ = self.ctx.vector.save();
            }
            Err(e) => error!(error = %e, "embedding failed"),
        }

        Ok(format!(
            "Indexed: {} ({} bytes) -> {}",
            file_name,
            content.len(),
            id.as_ref()
        ))
    }
}
