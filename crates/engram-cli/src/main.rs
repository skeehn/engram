use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::sync::Arc;
use anyhow::Result;

#[derive(Parser)]
#[command(name = "engram", about = "Multi-modal knowledge database for AI agents")]
struct Cli {
    /// Path to engram store directory (default: .engram)
    #[arg(short, long, default_value = ".engram")]
    db: PathBuf,

    #[command(subcommand)]
    cmd: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Add a knowledge node
    Add {
        /// The content to store
        body: String,
        /// Node type: fact, concept, entity, event, document, chunk, note, or any custom string
        #[arg(short, long, default_value = "fact")]
        node_type: String,
        /// Comma-separated tags
        #[arg(short, long)]
        tags: Option<String>,
    },
    /// Search the knowledge base
    Search {
        /// Query text
        query: String,
        /// Number of results
        #[arg(short, long, default_value = "10")]
        top_k: usize,
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },
    /// Get a specific node by ID
    Get {
        id: String,
        #[arg(long)]
        json: bool,
    },
    /// List nodes
    List {
        /// Filter by type
        #[arg(short, long)]
        node_type: Option<String>,
        #[arg(short, long, default_value = "20")]
        limit: usize,
        #[arg(long)]
        json: bool,
    },
    /// Ingest a URL via Jina reader
    Ingest {
        url: String,
    },
    /// Show database stats
    Stats,
    /// Show graph neighborhood of a node
    Graph {
        id: String,
        #[arg(short, long, default_value = "2")]
        depth: usize,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    // Open store
    let store = Arc::new(engram_store::EngramStore::open(&cli.db)?);

    // Open FTS index
    let fts_path = cli.db.join("fts");
    let fts = Arc::new(engram_fts::FtsIndex::open(&fts_path)?);

    // Open vector index (1024 dimensions for jina-embeddings-v3)
    let vec_path = cli.db.join("vectors.json");
    let vector = Arc::new(engram_vector::VectorIndex::new(1024, &vec_path)?);

    // Embed client (reads JINA_API_KEY from env)
    let embed = Arc::new(engram_embed::EmbedClient::from_env());

    // Query engine
    let engine = Arc::new(engram_query::QueryEngine::new(
        store.clone(),
        embed.clone(),
        fts.clone(),
        vector.clone(),
    ));

    match cli.cmd {
        Commands::Add { body, node_type, tags } => {
            use engram_core::types::Node;
            let nt = parse_node_type(&node_type);
            let tag_list: Vec<String> = tags
                .unwrap_or_default()
                .split(',')
                .filter(|s| !s.is_empty())
                .map(|s| s.trim().to_string())
                .collect();
            let node = Node::new(body, nt).with_tags(tag_list);
            let id = engine.add_node(node).await?;
            println!("Added: {}", id);
        }
        Commands::Search { query, top_k, json } => {
            let results = engine.search_text(&query, top_k).await?;
            if results.is_empty() {
                println!("No results found.");
            } else if json {
                let out: Vec<serde_json::Value> = results
                    .iter()
                    .map(|r| {
                        let preview_len = r.node.body.len().min(200);
                        serde_json::json!({
                            "id": r.node.id.as_ref(),
                            "type": r.node.node_type.to_string(),
                            "score": r.score,
                            "body": &r.node.body[..preview_len],
                        })
                    })
                    .collect();
                println!("{}", serde_json::to_string_pretty(&out)?);
            } else {
                for (i, r) in results.iter().enumerate() {
                    let preview_len = r.node.body.len().min(120);
                    let preview = &r.node.body[..preview_len];
                    println!(
                        "[{}] {:.3} | {} | {}...",
                        i + 1,
                        r.score,
                        r.node.id,
                        preview
                    );
                }
            }
        }
        Commands::Get { id, json } => {
            use engram_core::id::NodeId;
            let node_id = NodeId::from(id);
            match store.get_node(&node_id)? {
                Some(node) => {
                    if json {
                        println!("{}", serde_json::to_string_pretty(&node)?);
                    } else {
                        println!("ID:         {}", node.id);
                        println!("Type:       {}", node.node_type);
                        println!("Confidence: {:.2}", node.confidence);
                        println!("Tags:       {}", node.tags.join(", "));
                        println!(
                            "TX time:    {}",
                            node.tx_time.format("%Y-%m-%d %H:%M:%S UTC")
                        );
                        println!(
                            "Valid time: {}",
                            node.valid_time
                                .map(|t| t.format("%Y-%m-%d %H:%M:%S UTC").to_string())
                                .unwrap_or_else(|| "current".into())
                        );
                        println!();
                        println!("{}", node.body);
                    }
                }
                None => eprintln!("Node not found: {}", node_id),
            }
        }
        Commands::List { node_type, limit, json } => {
            let nt = node_type.as_deref().map(parse_node_type);
            let nodes = store.list_nodes(nt, limit)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&nodes)?);
            } else {
                println!("{} nodes:", nodes.len());
                for node in &nodes {
                    let preview_len = node.body.len().min(80);
                    let preview = &node.body[..preview_len];
                    println!(
                        "  {} | {} | {:.2} | {}...",
                        node.id, node.node_type, node.confidence, preview
                    );
                }
            }
        }
        Commands::Ingest { url } => {
            use engram_core::types::Node;
            let reader = embed.read_url(&url).await?;
            let node = Node::new(
                format!("# {}\n\n{}", reader.title, reader.content),
                engram_core::types::NodeType::Document,
            )
            .with_tags(vec!["ingested".into(), "url".into()]);
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
            use engram_core::id::NodeId;
            use engram_graph::GraphTraversal;
            let node_id = NodeId::from(id);
            let trav = GraphTraversal::new(&store);
            let (nodes, edges) = trav.subgraph(&node_id, depth)?;
            println!("{} nodes, {} edges in neighborhood:", nodes.len(), edges.len());
            for n in &nodes {
                let preview_len = n.body.len().min(60);
                println!(
                    "  {} [{}] {}",
                    n.id,
                    n.node_type,
                    &n.body[..preview_len]
                );
            }
            for e in &edges {
                println!("  {} --{}--> {}", e.source, e.edge_type, e.target);
            }
        }
    }

    Ok(())
}

fn parse_node_type(s: &str) -> engram_core::types::NodeType {
    use engram_core::types::NodeType;
    match s.to_lowercase().as_str() {
        "fact" => NodeType::Fact,
        "concept" => NodeType::Concept,
        "entity" => NodeType::Entity,
        "event" => NodeType::Event,
        "document" | "doc" => NodeType::Document,
        "chunk" => NodeType::Chunk,
        "note" => NodeType::Note,
        other => NodeType::Custom(other.to_string()),
    }
}
