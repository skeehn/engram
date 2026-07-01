use anyhow::Result;
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    routing::{delete, get, post},
    Json, Router,
};
use clap::{Parser, Subcommand};
use engram_core::id::NodeId;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Parser)]
#[command(
    name = "engram",
    about = "Multi-modal knowledge database for AI agents"
)]
struct Cli {
    /// Path to engram store directory (default: .engram)
    #[arg(short, long, default_value = ".engram")]
    db: PathBuf,

    /// Use local embeddings + HNSW (v2 mode: offline, no API needed)
    #[arg(long)]
    local: bool,

    #[command(subcommand)]
    cmd: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Add a knowledge node
    Add {
        body: String,
        #[arg(short, long, default_value = "fact")]
        node_type: String,
        #[arg(short, long)]
        tags: Option<String>,
        /// Project namespace (e.g. --project myapp)
        #[arg(short, long)]
        project: Option<String>,
    },
    /// Search the knowledge base
    Search {
        query: String,
        #[arg(short, long, default_value = "10")]
        top_k: usize,
        #[arg(long)]
        json: bool,
        /// Restrict to a project namespace
        #[arg(short, long)]
        project: Option<String>,
    },
    /// Get a specific node by ID
    Get {
        id: String,
        #[arg(long)]
        json: bool,
    },
    /// List nodes
    List {
        #[arg(short, long)]
        node_type: Option<String>,
        #[arg(short, long, default_value = "20")]
        limit: usize,
        #[arg(long)]
        json: bool,
        #[arg(short, long)]
        project: Option<String>,
    },
    /// Delete a node by ID
    Delete { id: String },
    /// Relate two nodes (create an edge)
    Relate {
        from: String,
        edge_type: String,
        to: String,
        #[arg(short, long, default_value = "1.0")]
        weight: f32,
    },
    /// Ingest a URL via Jina reader
    Ingest {
        url: String,
        #[arg(short, long)]
        project: Option<String>,
    },
    /// Show database stats
    Stats,
    /// Show graph neighborhood of a node
    Graph {
        id: String,
        #[arg(short, long, default_value = "2")]
        depth: usize,
    },
    /// Start HTTP server (default port 7474)
    Serve {
        #[arg(short, long, default_value = "7474")]
        port: u16,
    },
}

// ── HTTP server types ──────────────────────────────────────────────────────────

#[derive(Clone)]
struct AppState {
    engine: Arc<engram_query::QueryEngine>,
}

#[derive(Deserialize)]
struct SearchParams {
    q: String,
    #[serde(default = "default_top_k")]
    top_k: usize,
    /// Optional project namespace filter
    project: Option<String>,
}
fn default_top_k() -> usize {
    10
}

#[derive(Deserialize)]
struct ListParams {
    #[serde(default = "default_limit")]
    limit: usize,
    node_type: Option<String>,
    project: Option<String>,
}
fn default_limit() -> usize {
    20
}

#[derive(Deserialize)]
struct AddBody {
    body: String,
    #[serde(default = "default_node_type")]
    node_type: String,
    #[serde(default)]
    tags: Vec<String>,
    /// Optional project namespace
    project: Option<String>,
}
fn default_node_type() -> String {
    "fact".to_string()
}

#[derive(Deserialize)]
struct RelateBody {
    from: String,
    edge_type: String,
    to: String,
    #[serde(default = "default_weight")]
    weight: f32,
}
fn default_weight() -> f32 {
    1.0
}

#[derive(Serialize)]
struct NodeResponse {
    id: String,
    score: Option<f32>,
    body: String,
    node_type: String,
    tags: Vec<String>,
    confidence: f32,
    tx_time: String,
}

#[derive(Serialize)]
struct AddResult {
    id: String,
}

#[derive(Serialize)]
struct StatsResponse {
    nodes: u64,
    edges: u64,
    clusters: u64,
    object_bytes: u64,
    fts_docs: u64,
    vectors: usize,
}

#[derive(Serialize)]
struct GraphResponse {
    nodes: Vec<NodeResponse>,
    edges: Vec<EdgeResponse>,
}

#[derive(Serialize)]
struct EdgeResponse {
    id: String,
    source: String,
    target: String,
    edge_type: String,
    weight: f32,
}

fn node_to_response(node: engram_core::types::Node, score: Option<f32>) -> NodeResponse {
    NodeResponse {
        id: node.id.to_string(),
        score,
        body: node.body,
        node_type: node.node_type.to_string(),
        tags: node.tags,
        confidence: node.confidence,
        tx_time: node.tx_time.to_rfc3339(),
    }
}

/// Inject project tag if provided
fn inject_project(mut tags: Vec<String>, project: Option<&str>) -> Vec<String> {
    if let Some(p) = project {
        let tag = format!("project:{}", p);
        if !tags.contains(&tag) {
            tags.push(tag);
        }
    }
    tags
}

// ── HTTP handlers ──────────────────────────────────────────────────────────────

async fn handle_health() -> &'static str {
    "ok"
}

async fn handle_search(
    State(state): State<AppState>,
    Query(params): Query<SearchParams>,
) -> Json<Vec<NodeResponse>> {
    let mut results = state
        .engine
        .search_text(&params.q, params.top_k)
        .await
        .unwrap_or_default();

    // Filter by project if specified
    if let Some(ref proj) = params.project {
        let tag = format!("project:{}", proj);
        results.retain(|r| r.node.tags.contains(&tag));
    }

    Json(
        results
            .into_iter()
            .map(|r| node_to_response(r.node, Some(r.score)))
            .collect(),
    )
}

async fn handle_add(
    State(state): State<AppState>,
    Json(body): Json<AddBody>,
) -> (StatusCode, Json<AddResult>) {
    use engram_core::types::Node;
    let nt = parse_node_type(&body.node_type);
    let tags = inject_project(body.tags, body.project.as_deref());
    let node = Node::new(body.body, nt).with_tags(tags);
    match state.engine.add_node(node).await {
        Ok(id) => (StatusCode::CREATED, Json(AddResult { id: id.to_string() })),
        Err(e) => {
            tracing::error!("add_node error: {}", e);
            (StatusCode::INTERNAL_SERVER_ERROR, Json(AddResult { id: String::new() }))
        }
    }
}

async fn handle_get_node(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<NodeResponse>, StatusCode> {
    let node_id = NodeId::from(id.as_str());
    match state.engine.store.get_node(&node_id) {
        Ok(Some(node)) => Ok(Json(node_to_response(node, None))),
        Ok(None) => Err(StatusCode::NOT_FOUND),
        Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
    }
}

async fn handle_delete_node(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> StatusCode {
    let node_id = NodeId::from(id.as_str());
    match state.engine.store.delete_node(&node_id) {
        Ok(_) => StatusCode::NO_CONTENT,
        Err(_) => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

async fn handle_list(
    State(state): State<AppState>,
    Query(params): Query<ListParams>,
) -> Json<Vec<NodeResponse>> {
    let nt = params.node_type.as_deref().map(parse_node_type);
    let mut nodes = state
        .engine
        .store
        .list_nodes(nt, params.limit * 4) // over-fetch for project filter
        .unwrap_or_default();

    if let Some(ref proj) = params.project {
        let tag = format!("project:{}", proj);
        nodes.retain(|n| n.tags.contains(&tag));
    }
    nodes.truncate(params.limit);

    Json(nodes.into_iter().map(|n| node_to_response(n, None)).collect())
}

async fn handle_relate(
    State(state): State<AppState>,
    Json(body): Json<RelateBody>,
) -> StatusCode {
    use engram_core::types::{Edge, EdgeType};
    let et = EdgeType::Custom(body.edge_type);
    let mut edge = Edge::new(
        NodeId::from(body.from.as_str()),
        NodeId::from(body.to.as_str()),
        et,
    );
    edge.weight = body.weight;
    match state.engine.store.put_edge(&edge) {
        Ok(_) => StatusCode::CREATED,
        Err(_) => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

async fn handle_stats(State(state): State<AppState>) -> Result<Json<StatsResponse>, StatusCode> {
    let (fts_docs, vec_len) = (
        state.engine.fts_doc_count().unwrap_or(0),
        state.engine.vector_len(),
    );
    match state.engine.store.stats() {
        Ok(s) => Ok(Json(StatsResponse {
            nodes: s.node_count,
            edges: s.edge_count,
            clusters: s.cluster_count,
            object_bytes: s.object_bytes,
            fts_docs,
            vectors: vec_len,
        })),
        Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
    }
}

async fn handle_graph(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Query(params): Query<std::collections::HashMap<String, String>>,
) -> Result<Json<GraphResponse>, StatusCode> {
    use engram_graph::GraphTraversal;
    let depth: usize = params
        .get("depth")
        .and_then(|d| d.parse().ok())
        .unwrap_or(2);
    let node_id = NodeId::from(id.as_str());
    let trav = GraphTraversal::new(&state.engine.store);
    match trav.subgraph(&node_id, depth) {
        Ok((nodes, edges)) => Ok(Json(GraphResponse {
            nodes: nodes.into_iter().map(|n| node_to_response(n, None)).collect(),
            edges: edges
                .into_iter()
                .map(|e| EdgeResponse {
                    id: e.id.to_string(),
                    source: e.source.to_string(),
                    target: e.target.to_string(),
                    edge_type: e.edge_type.to_string(),
                    weight: e.weight,
                })
                .collect(),
        })),
        Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
    }
}

// ── Main ───────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    // Branch: v2 mode uses local embeddings + HNSW
    if cli.local {
        return run_local_mode(&cli).await;
    }

    // Original v1 mode: API embeddings + flat vector index
    let store = Arc::new(engram_store::EngramStore::open(&cli.db)?);
    let fts_path = cli.db.join("fts");
    let fts = Arc::new(engram_fts::FtsIndex::open(&fts_path)?);
    let vec_path = cli.db.join("vectors.json");
    let vector = Arc::new(engram_vector::VectorIndex::new(1024, &vec_path)?);
    let embed = Arc::new(engram_embed::EmbedClient::from_env());
    let engine = Arc::new(engram_query::QueryEngine::new(
        store.clone(),
        embed.clone(),
        fts.clone(),
        vector.clone(),
    ));

    match cli.cmd {
        Commands::Add { body, node_type, tags, project } => {
            use engram_core::types::Node;
            let nt = parse_node_type(&node_type);
            let mut tag_list: Vec<String> = tags
                .unwrap_or_default()
                .split(',')
                .filter(|s| !s.is_empty())
                .map(|s| s.trim().to_string())
                .collect();
            tag_list = inject_project(tag_list, project.as_deref());
            let node = Node::new(body, nt).with_tags(tag_list);
            let id = engine.add_node(node).await?;
            println!("Added: {}", id);
        }
        Commands::Search { query, top_k, json, project } => {
            let mut results = engine.search_text(&query, top_k).await?;
            if let Some(ref proj) = project {
                let tag = format!("project:{}", proj);
                results.retain(|r| r.node.tags.contains(&tag));
            }
            if results.is_empty() {
                println!("No results found.");
            } else if json {
                let out: Vec<serde_json::Value> = results
                    .iter()
                    .map(|r| serde_json::json!({
                        "id": r.node.id.as_ref(),
                        "type": r.node.node_type.to_string(),
                        "score": r.score,
                        "body": truncate_str(&r.node.body, 200),
                        "tags": r.node.tags,
                    }))
                    .collect();
                println!("{}", serde_json::to_string_pretty(&out)?);
            } else {
                for (i, r) in results.iter().enumerate() {
                    let preview = truncate_str(&r.node.body, 120);
                    println!("[{}] {:.3} | {} | {}...", i + 1, r.score, r.node.id, preview);
                }
            }
        }
        Commands::Get { id, json } => {
            let node_id = NodeId::from(id.as_str());
            match store.get_node(&node_id)? {
                Some(node) => {
                    if json {
                        println!("{}", serde_json::to_string_pretty(&node)?);
                    } else {
                        println!("ID:         {}", node.id);
                        println!("Type:       {}", node.node_type);
                        println!("Confidence: {:.2}", node.confidence);
                        println!("Tags:       {}", node.tags.join(", "));
                        println!("TX time:    {}", node.tx_time.format("%Y-%m-%d %H:%M:%S UTC"));
                        println!("Valid time: {}", node.valid_time
                            .map(|t| t.format("%Y-%m-%d %H:%M:%S UTC").to_string())
                            .unwrap_or_else(|| "current".into()));
                        println!();
                        println!("{}", node.body);
                    }
                }
                None => eprintln!("Node not found: {}", node_id),
            }
        }
        Commands::List { node_type, limit, json, project } => {
            let nt = node_type.as_deref().map(parse_node_type);
            let mut nodes = store.list_nodes(nt, limit * 4)?;
            if let Some(ref proj) = project {
                let tag = format!("project:{}", proj);
                nodes.retain(|n| n.tags.contains(&tag));
            }
            nodes.truncate(limit);
            if json {
                println!("{}", serde_json::to_string_pretty(&nodes)?);
            } else {
                println!("{} nodes:", nodes.len());
                for node in &nodes {
                    let preview = truncate_str(&node.body, 80);
                    println!("  {} | {} | {:.2} | {}...", node.id, node.node_type, node.confidence, preview);
                }
            }
        }
        Commands::Delete { id } => {
            let node_id = NodeId::from(id.as_str());
            store.delete_node(&node_id)?;
            println!("Deleted: {}", node_id);
        }
        Commands::Relate { from, edge_type, to, weight } => {
            use engram_core::types::{Edge, EdgeType};
            let et = EdgeType::Custom(edge_type);
            let mut edge = Edge::new(
                NodeId::from(from.as_str()),
                NodeId::from(to.as_str()),
                et,
            );
            edge.weight = weight;
            store.put_edge(&edge)?;
            println!("Related: {} --{}--> {}", edge.source, edge.edge_type, edge.target);
        }
        Commands::Ingest { url, project } => {
            use engram_core::types::Node;
            let reader = embed.read_url(&url).await?;
            let mut tags = vec!["ingested".into(), "url".into()];
            tags = inject_project(tags, project.as_deref());
            let node = Node::new(
                format!("# {}\n\n{}", reader.title, reader.content),
                engram_core::types::NodeType::Document,
            ).with_tags(tags);
            let id = engine.add_node(node).await?;
            println!("Ingested {} -> {}", url, id);
        }
        Commands::Stats => {
            let stats = store.stats()?;
            println!("nodes:    {}", stats.node_count);
            println!("edges:    {}", stats.edge_count);
            println!("clusters: {}", stats.cluster_count);
            println!("objects:  {} bytes", stats.object_bytes);
            println!("fts docs: {}", fts.doc_count()?);
            println!("vectors:  {}", vector.len());
        }
        Commands::Graph { id, depth } => {
            use engram_graph::GraphTraversal;
            let node_id = NodeId::from(id.as_str());
            let trav = GraphTraversal::new(&store);
            let (nodes, edges) = trav.subgraph(&node_id, depth)?;
            println!("{} nodes, {} edges in neighborhood:", nodes.len(), edges.len());
            for n in &nodes {
                println!("  {} [{}] {}", n.id, n.node_type, truncate_str(&n.body, 60));
            }
            for e in &edges {
                println!("  {} --{}--> {}", e.source, e.edge_type, e.target);
            }
        }
        Commands::Serve { port } => {
            let state = AppState { engine: engine.clone() };
            let app = Router::new()
                .route("/health",        get(handle_health))
                .route("/search",        get(handle_search))
                .route("/add",           post(handle_add))
                .route("/nodes",         get(handle_list))
                .route("/nodes/:id",     get(handle_get_node))
                .route("/nodes/:id",     delete(handle_delete_node))
                .route("/relate",        post(handle_relate))
                .route("/stats",         get(handle_stats))
                .route("/graph/:id",     get(handle_graph))
                .with_state(state);
            let addr = format!("127.0.0.1:{}", port);
            let listener = tokio::net::TcpListener::bind(&addr).await?;
            println!("engram serving on http://{}", addr);
            axum::serve(listener, app).await?;
        }
    }

    Ok(())
}

/// v2 mode: local embeddings + HNSW, no API needed
async fn run_local_mode(cli: &Cli) -> Result<()> {
    use engram_cli::v2::EngramContext;
    use engram_core::types::Node;
    
    let ctx = EngramContext::open_offline(&cli.db)?;
    println!("engram v2 (local mode) initialized");
    
    match &cli.cmd {
        Commands::Add { body, node_type, tags, project } => {
            let nt = parse_node_type(node_type);
            let mut tag_list: Vec<String> = tags
                .clone()
                .unwrap_or_default()
                .split(',')
                .filter(|s| !s.is_empty())
                .map(|s| s.trim().to_string())
                .collect();
            tag_list = inject_project(tag_list, project.as_deref());
            
            let node = Node::new(body.clone(), nt).with_tags(tag_list);
            let id = node.id.clone();
            
            // Store node
            ctx.store.put_node(&node)?;
            
            // Index in FTS
            ctx.fts.index_node(&node)?;
            ctx.fts.commit()?;
            
            // Embed and index in HNSW
            let embedding = ctx.embed(&node.body).await?;
            ctx.vector.upsert(&id, &embedding)?;
            ctx.vector.save()?;
            
            println!("Added: {}", id);
        }
        Commands::Search { query, top_k, json, project } => {
            let results = ctx.search(query, *top_k).await?;
            
            // Load full nodes and filter by project
            let mut nodes: Vec<(engram_core::types::Node, f32)> = Vec::new();
            for (id, score) in results {
                let node_id = NodeId::from(id.as_str());
                if let Ok(Some(node)) = ctx.store.get_node(&node_id) {
                    if let Some(ref proj) = project {
                        let tag = format!("project:{}", proj);
                        if !node.tags.contains(&tag) {
                            continue;
                        }
                    }
                    nodes.push((node, score));
                }
            }
            
            if nodes.is_empty() {
                println!("No results found.");
            } else if *json {
                let out: Vec<serde_json::Value> = nodes
                    .iter()
                    .map(|(n, score)| serde_json::json!({
                        "id": n.id.as_ref(),
                        "type": n.node_type.to_string(),
                        "score": score,
                        "body": truncate_str(&n.body, 200),
                        "tags": n.tags,
                    }))
                    .collect();
                println!("{}", serde_json::to_string_pretty(&out)?);
            } else {
                for (i, (n, score)) in nodes.iter().enumerate() {
                    let preview = truncate_str(&n.body, 120);
                    println!("[{}] {:.3} | {} | {}...", i + 1, score, n.id, preview);
                }
            }
        }
        Commands::Stats => {
            let stats = ctx.stats();
            println!("Nodes:   {}", stats.nodes);
            println!("FTS:     {} docs", stats.fts_docs);
            println!("Vectors: {} (HNSW, 384d)", stats.vectors);
        }
        _ => {
            // For commands not yet implemented in v2, fall back
            println!("Command not yet implemented in local mode. Remove --local flag.");
        }
    }
    
    Ok(())
}

fn parse_node_type(s: &str) -> engram_core::types::NodeType {
    use engram_core::types::NodeType;
    match s.to_lowercase().as_str() {
        "fact"            => NodeType::Fact,
        "concept"         => NodeType::Concept,
        "entity"          => NodeType::Entity,
        "event"           => NodeType::Event,
        "document" | "doc"=> NodeType::Document,
        "chunk"           => NodeType::Chunk,
        "note"            => NodeType::Note,
        other             => NodeType::Custom(other.to_string()),
    }
}

fn truncate_str(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes { return s; }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) { end -= 1; }
    &s[..end]
}
