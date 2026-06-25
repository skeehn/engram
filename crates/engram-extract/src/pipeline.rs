use engram_core::{
    error::Result,
    id::NodeId,
    types::{Node, NodeType},
};
use engram_query::QueryEngine;
use std::sync::Arc;
use tokio::sync::mpsc;

/// A raw write request (Lane 1 -- fast, synchronous)
#[derive(Debug)]
pub struct WriteRequest {
    pub body: String,
    pub node_type: NodeType,
    pub tags: Vec<String>,
    pub source_id: Option<String>,
}

pub struct ExtractPipeline {
    engine: Arc<QueryEngine>,
    /// Channel for async background processing
    tx: mpsc::Sender<Node>,
}

impl ExtractPipeline {
    /// Create pipeline. Spawns background worker tokio task.
    pub fn new(engine: Arc<QueryEngine>) -> Self {
        let (tx, rx) = mpsc::channel::<Node>(1000);
        let engine_clone = engine.clone();
        tokio::spawn(Self::process_background(engine_clone, rx));
        Self { engine, tx }
    }

    /// Lane 1: < 5ms. Store raw node immediately, queue for background embedding.
    pub async fn write_raw(&self, req: WriteRequest) -> Result<NodeId> {
        let node = Node::new(req.body, req.node_type).with_tags(req.tags);
        // Store immediately (no embedding yet)
        self.engine.store.put_node(&node)?;
        let id = node.id.clone();
        // Non-blocking send to background worker; if channel is full, log and skip
        if let Err(e) = self.tx.try_send(node) {
            tracing::warn!(
                "background queue full, skipping embedding for {}: {}",
                id,
                e
            );
        }
        Ok(id)
    }

    /// Lane 2+3 background: embed + index the node (runs in spawned task)
    async fn process_background(engine: Arc<QueryEngine>, mut rx: mpsc::Receiver<Node>) {
        while let Some(node) = rx.recv().await {
            if let Err(e) = engine.add_node(node).await {
                tracing::warn!("background embedding error: {}", e);
            }
        }
    }
}
