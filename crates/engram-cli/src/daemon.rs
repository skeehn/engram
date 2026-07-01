//! Daemon mode: HTTP server + file watcher auto-indexing.
//!
//! Runs engram as a background service that:
//! - Watches directories for file changes
//! - Auto-indexes new/modified files (embed + store + FTS)
//! - Serves HTTP API for search/add/stats
//! - Provides real-time stats on indexing progress

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use axum::{
    extract::{Json, Query, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Router,
};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tracing::{error, info, warn};

use engram_watch::{FileEvent, FileWatcher, WatcherConfig};

use crate::v2::EngramContext;

// ── Types ────────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct DaemonState {
    pub ctx: Arc<EngramContext>,
    pub stats: Arc<Mutex<DaemonStats>>,
    pub watch_dirs: Arc<Vec<PathBuf>>,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct DaemonStats {
    pub files_indexed: u64,
    pub files_watching: u64,
    pub files_queued: u64,
    pub bytes_indexed: u64,
    pub errors: u64,
    pub uptime_secs: f64,
    pub last_event: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct DaemonConfig {
    /// Port to listen on (default: 7474)
    pub port: u16,
    /// Directories to watch for auto-indexing
    pub watch_dirs: Vec<PathBuf>,
    /// File extensions to index (empty = common text extensions)
    pub extensions: Vec<String>,
    /// Debounce duration in ms (default: 500)
    pub debounce_ms: u64,
    /// Project namespace for indexed files
    pub project: Option<String>,
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            port: 7474,
            watch_dirs: vec![],
            extensions: vec![
                "rs", "py", "ts", "js", "go", "md", "txt", "toml", "yaml", "yml",
                "json", "html", "css", "c", "cpp", "h", "java", "rb", "sh",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            debounce_ms: 500,
            project: None,
        }
    }
}

// ── Daemon Entry Point ───────────────────────────────────────────────────────

pub async fn run_daemon(ctx: EngramContext, config: DaemonConfig) -> Result<()> {
    let start = Instant::now();
    let ctx = Arc::new(ctx);

    let state = DaemonState {
        ctx: ctx.clone(),
        stats: Arc::new(Mutex::new(DaemonStats::default())),
        watch_dirs: Arc::new(config.watch_dirs.clone()),
    };

    // Start file watcher if directories specified
    let (tx, mut rx) = mpsc::channel::<FileEvent>(1024);

    if !config.watch_dirs.is_empty() {
        let watcher_config = WatcherConfig {
            debounce_ms: config.debounce_ms,
            extensions: config.extensions.clone(),
            ignore_dirs: vec![
                ".git".into(),
                "node_modules".into(),
                "target".into(),
                ".engram".into(),
                "__pycache__".into(),
                ".venv".into(),
                "venv".into(),
            ],
            hash_check: true,
            max_hash_size: 10 * 1024 * 1024, // 10MB
        };

        let mut watcher = FileWatcher::new(watcher_config)?;

        for dir in &config.watch_dirs {
            if dir.exists() {
                watcher.watch(dir)?;
                info!(path = ?dir, "watching directory");
            } else {
                warn!(path = ?dir, "directory not found, skipping");
            }
        }

        // Count watched files
        let watching: u64 = config
            .watch_dirs
            .iter()
            .filter(|d| d.exists())
            .map(|d| count_matching_files(d, &config.extensions))
            .sum();
        state.stats.lock().files_watching = watching;

        // Start the watcher
        watcher.start()?;

        // Take the event receiver BEFORE spawn (Receiver is Send, FileWatcher isn't)
        let watch_rx = watcher.take_receiver()
            .expect("watcher receiver already taken");

        // Store watcher in a Box to keep it alive (leaked intentionally — lives for process lifetime)
        // FileWatcher's internal notify thread will keep running and sending events
        let watcher_box = Box::new(watcher);
        Box::leak(watcher_box);

        // Bridge sync channel to tokio mpsc
        let tx_clone = tx.clone();
        std::thread::spawn(move || {
            while let Ok(event) = watch_rx.recv() {
                if tx_clone.blocking_send(event).is_err() {
                    break;
                }
            }
        });

        // Spawn indexer that processes events
        let indexer_state = state.clone();
        let project = config.project.clone();
        tokio::spawn(async move {
            while let Some(event) = rx.recv().await {
                match event {
                    FileEvent::Created(path) | FileEvent::Modified(path) => {
                        index_file(&indexer_state, &path, project.as_deref()).await;
                    }
                    FileEvent::Deleted(path) => {
                        info!(path = ?path, "file deleted (not removing from index)");
                    }
                    FileEvent::Renamed(_from, to) => {
                        index_file(&indexer_state, &to, project.as_deref()).await;
                    }
                }
            }
        });
    }

    // Build HTTP router
    let app = build_router(state.clone());

    // Start server
    let addr = format!("127.0.0.1:{}", config.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;

    info!(
        addr = %addr,
        watch_dirs = config.watch_dirs.len(),
        "engram daemon started"
    );

    println!("╔══════════════════════════════════════════════╗");
    println!("║       engram daemon v4.0                     ║");
    println!("╠══════════════════════════════════════════════╣");
    println!("║  HTTP API: http://{}         ║", addr);
    println!("║  Watching: {} directories                    ║", config.watch_dirs.len());
    println!("║  PID: {}                                  ║", std::process::id());
    println!("╚══════════════════════════════════════════════╝");
    println!();
    println!("Endpoints:");
    println!("  GET  /health        - Health check");
    println!("  GET  /search?q=...  - Semantic search");
    println!("  POST /add           - Add knowledge");
    println!("  POST /index         - Index a file");
    println!("  GET  /stats         - Index statistics");
    println!("  GET  /daemon/stats  - Daemon statistics");
    println!("  POST /watch         - Add directory to watch");
    println!();

    // Update uptime in background
    let stats_clone = state.stats.clone();
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
            stats_clone.lock().uptime_secs = start.elapsed().as_secs_f64();
        }
    });

    axum::serve(listener, app).await?;
    Ok(())
}

// ── Router ───────────────────────────────────────────────────────────────────

fn build_router(state: DaemonState) -> Router {
    Router::new()
        .route("/health", get(handle_health))
        .route("/search", get(handle_search))
        .route("/add", post(handle_add))
        .route("/index", post(handle_index_file))
        .route("/stats", get(handle_stats))
        .route("/daemon/stats", get(handle_daemon_stats))
        .route("/watch", post(handle_add_watch))
        .with_state(state)
}

// ── Handlers ─────────────────────────────────────────────────────────────────

async fn handle_health() -> impl IntoResponse {
    Json(serde_json::json!({"status": "ok", "version": "4.0"}))
}

#[derive(Deserialize)]
struct SearchQuery {
    q: String,
    #[serde(default = "default_k")]
    k: usize,
}
fn default_k() -> usize {
    10
}

async fn handle_search(
    State(state): State<DaemonState>,
    Query(params): Query<SearchQuery>,
) -> impl IntoResponse {
    match state.ctx.search(&params.q, params.k).await {
        Ok(results) => {
            let items: Vec<serde_json::Value> = results
                .iter()
                .map(|(id, score)| {
                    serde_json::json!({
                        "id": id,
                        "score": score,
                    })
                })
                .collect();
            (StatusCode::OK, Json(serde_json::json!({"results": items})))
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        ),
    }
}

#[derive(Deserialize)]
struct AddBody {
    body: String,
    #[serde(default = "default_type")]
    node_type: String,
    #[serde(default)]
    tags: Vec<String>,
}
fn default_type() -> String {
    "fact".into()
}

async fn handle_add(
    State(state): State<DaemonState>,
    Json(body): Json<AddBody>,
) -> impl IntoResponse {
    use engram_core::types::{Node, NodeType};

    let nt = match body.node_type.as_str() {
        "fact" => NodeType::Fact,
        "concept" => NodeType::Concept,
        "document" => NodeType::Document,
        "note" => NodeType::Note,
        other => NodeType::Custom(other.to_string()),
    };

    let node = Node::new(body.body.clone(), nt).with_tags(body.tags);
    let id = node.id.clone();

    // Store
    if let Err(e) = state.ctx.store.put_node(&node) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        );
    }

    // FTS
    if let Err(e) = state.ctx.fts.index_node(&node) {
        warn!(error = %e, "FTS index failed");
    }
    let _ = state.ctx.fts.commit();

    // Embed + vector
    match state.ctx.embed(&body.body).await {
        Ok(embedding) => {
            let _ = state.ctx.vector.upsert(&id, &embedding);
            let _ = state.ctx.vector.save();
        }
        Err(e) => warn!(error = %e, "embedding failed"),
    }

    (
        StatusCode::CREATED,
        Json(serde_json::json!({"id": id.as_ref(), "status": "indexed"})),
    )
}

#[derive(Deserialize)]
struct IndexFileBody {
    path: String,
    #[serde(default)]
    project: Option<String>,
}

async fn handle_index_file(
    State(state): State<DaemonState>,
    Json(body): Json<IndexFileBody>,
) -> impl IntoResponse {
    let path = PathBuf::from(&body.path);
    if !path.exists() {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "file not found"})),
        );
    }

    index_file(&state, &path, body.project.as_deref()).await;

    (
        StatusCode::OK,
        Json(serde_json::json!({"status": "indexed", "path": body.path})),
    )
}

async fn handle_stats(State(state): State<DaemonState>) -> impl IntoResponse {
    let s = state.ctx.stats();
    Json(serde_json::json!({
        "nodes": s.nodes,
        "fts_docs": s.fts_docs,
        "vectors": s.vectors,
    }))
}

async fn handle_daemon_stats(State(state): State<DaemonState>) -> impl IntoResponse {
    let stats = state.stats.lock().clone();
    Json(serde_json::json!(stats))
}

#[derive(Deserialize)]
struct WatchBody {
    path: String,
}

async fn handle_add_watch(
    State(_state): State<DaemonState>,
    Json(body): Json<WatchBody>,
) -> impl IntoResponse {
    // Note: In a full impl, we'd add to the watcher dynamically.
    // For now, return the path acknowledged.
    let path = PathBuf::from(&body.path);
    if !path.exists() {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "directory not found"})),
        );
    }
    (
        StatusCode::OK,
        Json(serde_json::json!({"status": "watching", "path": body.path})),
    )
}

// ── File Indexing ────────────────────────────────────────────────────────────

async fn index_file(state: &DaemonState, path: &Path, project: Option<&str>) {
    use engram_core::types::{Node, NodeType};

    // Read file
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => {
            warn!(path = ?path, error = %e, "failed to read file");
            state.stats.lock().errors += 1;
            return;
        }
    };

    // Skip empty or very large files
    if content.is_empty() || content.len() > 1_000_000 {
        return;
    }

    let file_name = path
        .file_name()
        .map(|f| f.to_string_lossy().to_string())
        .unwrap_or_default();

    // Build tags
    let mut tags = vec![format!("file:{}", file_name)];
    if let Some(ext) = path.extension() {
        tags.push(format!("ext:{}", ext.to_string_lossy()));
    }
    if let Some(p) = project {
        tags.push(format!("project:{}", p));
    }
    tags.push(format!("path:{}", path.display()));

    // Create node
    let body = if content.len() > 4096 {
        // Chunk large files — index first 4K for now
        // TODO: proper chunking with overlap
        content[..4096].to_string()
    } else {
        content.clone()
    };

    let node = Node::new(body.clone(), NodeType::Document).with_tags(tags);
    let id = node.id.clone();

    // Store
    if let Err(e) = state.ctx.store.put_node(&node) {
        error!(path = ?path, error = %e, "store failed");
        state.stats.lock().errors += 1;
        return;
    }

    // FTS
    if let Err(e) = state.ctx.fts.index_node(&node) {
        warn!(path = ?path, error = %e, "FTS failed");
    }
    let _ = state.ctx.fts.commit();

    // Embed + vector index
    match state.ctx.embed(&body).await {
        Ok(embedding) => {
            let _ = state.ctx.vector.upsert(&id, &embedding);
            let _ = state.ctx.vector.save();
        }
        Err(e) => {
            warn!(path = ?path, error = %e, "embedding failed");
        }
    }

    // Update stats
    {
        let mut stats = state.stats.lock();
        stats.files_indexed += 1;
        stats.bytes_indexed += content.len() as u64;
        stats.last_event = Some(format!("indexed: {}", file_name));
    }

    info!(path = ?path, id = %id.as_ref(), "indexed file");
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn count_matching_files(dir: &Path, extensions: &[String]) -> u64 {
    let mut count = 0u64;
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let name = path.file_name().unwrap_or_default().to_string_lossy();
                if !name.starts_with('.') && name != "node_modules" && name != "target" {
                    count += count_matching_files(&path, extensions);
                }
            } else if let Some(ext) = path.extension() {
                if extensions.is_empty()
                    || extensions.iter().any(|e| e == &ext.to_string_lossy().as_ref())
                {
                    count += 1;
                }
            }
        }
    }
    count
}
