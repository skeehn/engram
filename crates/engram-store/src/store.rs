use crate::cf::*;
use crate::delta::{temporal_key, temporal_key_prefix, DeltaOp, NodeDelta};
use chrono::{DateTime, Utc};
use engram_core::{
    error::{EngramError, Result},
    id::{ClusterId, EdgeId, NodeId, ObjectId},
    types::{Edge, Node, NodeType},
};
use sled::{Db, Tree};
use std::path::{Path, PathBuf};

/// Simple cluster info stored alongside nodes.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ClusterInfo {
    pub id: ClusterId,
    pub name: String,
    pub members: Vec<NodeId>,
}

pub struct EngramStore {
    db: Db,
    #[allow(dead_code)]
    path: PathBuf,
}

fn sled_err(e: sled::Error) -> EngramError {
    EngramError::Storage(e.to_string())
}

impl EngramStore {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        std::fs::create_dir_all(&path).map_err(|e| EngramError::Storage(e.to_string()))?;
        let db = sled::open(&path).map_err(sled_err)?;
        Ok(Self { db, path })
    }

    fn tree(&self, name: &str) -> Result<Tree> {
        self.db.open_tree(name).map_err(sled_err)
    }

    // -----------------------------------------------------------------------
    // Node operations
    // -----------------------------------------------------------------------

    pub fn put_node(&self, node: &Node) -> Result<()> {
        let tree = self.tree(TREE_NODES)?;
        let value = serde_json::to_vec(node)?;
        tree.insert(node.id.as_ref().as_bytes(), value)
            .map_err(sled_err)?;
        let delta = NodeDelta::insert(node.clone());
        self.append_delta(&delta)?;
        Ok(())
    }

    pub fn get_node(&self, id: &NodeId) -> Result<Option<Node>> {
        let tree = self.tree(TREE_NODES)?;
        match tree.get(id.as_ref().as_bytes()).map_err(sled_err)? {
            Some(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
            None => Ok(None),
        }
    }

    pub fn delete_node(&self, id: &NodeId) -> Result<()> {
        let tree = self.tree(TREE_NODES)?;
        tree.remove(id.as_ref().as_bytes()).map_err(sled_err)?;
        Ok(())
    }

    pub fn list_nodes(&self, node_type: Option<NodeType>, limit: usize) -> Result<Vec<Node>> {
        let tree = self.tree(TREE_NODES)?;
        let mut nodes = Vec::new();
        for item in tree.iter() {
            let (_, value) = item.map_err(sled_err)?;
            let node: Node = serde_json::from_slice(&value)?;
            if let Some(ref nt) = node_type {
                if &node.node_type != nt {
                    continue;
                }
            }
            nodes.push(node);
            if nodes.len() >= limit {
                break;
            }
        }
        Ok(nodes)
    }

    pub fn scan_nodes(&self) -> Result<Vec<Node>> {
        self.list_nodes(None, usize::MAX)
    }

    // -----------------------------------------------------------------------
    // Edge operations
    // -----------------------------------------------------------------------

    pub fn put_edge(&self, edge: &Edge) -> Result<()> {
        let fwd_tree = self.tree(TREE_EDGES)?;
        let rev_tree = self.tree(TREE_EDGE_REV)?;
        let fwd_key = format!(
            "{}\x00{}\x00{}",
            edge.source.as_ref(),
            edge.edge_type,
            edge.id.as_ref()
        );
        let rev_key = format!(
            "{}\x00{}\x00{}",
            edge.target.as_ref(),
            edge.edge_type,
            edge.id.as_ref()
        );
        let value = serde_json::to_vec(edge)?;
        fwd_tree
            .insert(fwd_key.as_bytes(), value.clone())
            .map_err(sled_err)?;
        rev_tree
            .insert(rev_key.as_bytes(), value)
            .map_err(sled_err)?;
        Ok(())
    }

    pub fn get_edge(&self, id: &EdgeId) -> Result<Option<Edge>> {
        let tree = self.tree(TREE_EDGES)?;
        for item in tree.iter() {
            let (_, value) = item.map_err(sled_err)?;
            let edge: Edge = serde_json::from_slice(&value)?;
            if &edge.id == id {
                return Ok(Some(edge));
            }
        }
        Ok(None)
    }

    pub fn delete_edge(&self, id: &EdgeId) -> Result<()> {
        if let Some(edge) = self.get_edge(id)? {
            let fwd_tree = self.tree(TREE_EDGES)?;
            let rev_tree = self.tree(TREE_EDGE_REV)?;
            let fwd_key = format!(
                "{}\x00{}\x00{}",
                edge.source.as_ref(),
                edge.edge_type,
                edge.id.as_ref()
            );
            let rev_key = format!(
                "{}\x00{}\x00{}",
                edge.target.as_ref(),
                edge.edge_type,
                edge.id.as_ref()
            );
            fwd_tree.remove(fwd_key.as_bytes()).map_err(sled_err)?;
            rev_tree.remove(rev_key.as_bytes()).map_err(sled_err)?;
        }
        Ok(())
    }

    pub fn edges_from(&self, node_id: &NodeId) -> Result<Vec<Edge>> {
        let tree = self.tree(TREE_EDGES)?;
        let prefix = format!("{}\x00", node_id.as_ref());
        let mut edges = Vec::new();
        for item in tree.scan_prefix(prefix.as_bytes()) {
            let (_, value) = item.map_err(sled_err)?;
            let edge: Edge = serde_json::from_slice(&value)?;
            edges.push(edge);
        }
        Ok(edges)
    }

    pub fn edges_to(&self, node_id: &NodeId) -> Result<Vec<Edge>> {
        let tree = self.tree(TREE_EDGE_REV)?;
        let prefix = format!("{}\x00", node_id.as_ref());
        let mut edges = Vec::new();
        for item in tree.scan_prefix(prefix.as_bytes()) {
            let (_, value) = item.map_err(sled_err)?;
            let edge: Edge = serde_json::from_slice(&value)?;
            edges.push(edge);
        }
        Ok(edges)
    }

    pub fn edges_from_typed(&self, node_id: &NodeId, edge_type: &str) -> Result<Vec<Edge>> {
        let tree = self.tree(TREE_EDGES)?;
        let prefix = format!("{}\x00{}\x00", node_id.as_ref(), edge_type);
        let mut edges = Vec::new();
        for item in tree.scan_prefix(prefix.as_bytes()) {
            let (_, value) = item.map_err(sled_err)?;
            let edge: Edge = serde_json::from_slice(&value)?;
            edges.push(edge);
        }
        Ok(edges)
    }

    // -----------------------------------------------------------------------
    // Temporal / delta log
    // -----------------------------------------------------------------------

    pub fn get_node_as_of(&self, id: &NodeId, as_of: DateTime<Utc>) -> Result<Option<Node>> {
        let tree = self.tree(TREE_TEMPORAL)?;
        let upper = temporal_key_prefix(as_of);
        let mut current: Option<Node> = None;
        for item in tree.iter() {
            let (key, value) = item.map_err(sled_err)?;
            if key.len() >= 8 && key[..8] > upper[..] {
                break;
            }
            let delta = NodeDelta::from_bytes(&value)?;
            if &delta.node_id != id {
                continue;
            }
            match delta.op {
                DeltaOp::Insert(node) => current = Some(node),
                DeltaOp::UpdateBody(body) => {
                    if let Some(ref mut n) = current {
                        n.body = body;
                    }
                }
                DeltaOp::UpdateConfidence(c) => {
                    if let Some(ref mut n) = current {
                        n.confidence = c;
                    }
                }
                DeltaOp::UpdateMetadata(meta) => {
                    if let Some(ref mut n) = current {
                        n.metadata = meta;
                    }
                }
                DeltaOp::Invalidate { valid_time } => {
                    if let Some(ref mut n) = current {
                        n.valid_time = Some(valid_time);
                    }
                }
                DeltaOp::Delete => current = None,
            }
        }
        Ok(current)
    }

    pub fn append_delta(&self, delta: &NodeDelta) -> Result<()> {
        let tree = self.tree(TREE_TEMPORAL)?;
        let key = temporal_key(delta.timestamp, &delta.node_id);
        let value = delta.to_bytes()?;
        tree.insert(key, value).map_err(sled_err)?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Key-value namespace
    // -----------------------------------------------------------------------

    pub fn kv_put(&self, key: &str, value: &[u8]) -> Result<()> {
        let tree = self.tree(TREE_KV)?;
        tree.insert(key.as_bytes(), value).map_err(sled_err)?;
        Ok(())
    }

    pub fn kv_get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let tree = self.tree(TREE_KV)?;
        Ok(tree
            .get(key.as_bytes())
            .map_err(sled_err)?
            .map(|v| v.to_vec()))
    }

    pub fn kv_delete(&self, key: &str) -> Result<()> {
        let tree = self.tree(TREE_KV)?;
        tree.remove(key.as_bytes()).map_err(sled_err)?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Content-addressed object store
    // -----------------------------------------------------------------------

    pub fn object_put(&self, data: &[u8]) -> Result<ObjectId> {
        let id = ObjectId::from_bytes(data);
        let tree = self.tree(TREE_OBJECTS)?;
        tree.insert(id.as_ref().as_bytes(), data)
            .map_err(sled_err)?;
        Ok(id)
    }

    pub fn object_get(&self, id: &ObjectId) -> Result<Option<Vec<u8>>> {
        let tree = self.tree(TREE_OBJECTS)?;
        Ok(tree
            .get(id.as_ref().as_bytes())
            .map_err(sled_err)?
            .map(|v| v.to_vec()))
    }

    // -----------------------------------------------------------------------
    // Provenance / JTMS justifications
    // -----------------------------------------------------------------------

    pub fn put_justification(&self, node_id: &NodeId, justification: &[NodeId]) -> Result<()> {
        let tree = self.tree(TREE_PROVENANCE)?;
        let value = serde_json::to_vec(justification)?;
        tree.insert(node_id.as_ref().as_bytes(), value)
            .map_err(sled_err)?;
        Ok(())
    }

    pub fn get_justification(&self, node_id: &NodeId) -> Result<Vec<NodeId>> {
        let tree = self.tree(TREE_PROVENANCE)?;
        match tree.get(node_id.as_ref().as_bytes()).map_err(sled_err)? {
            Some(bytes) => Ok(serde_json::from_slice(&bytes)?),
            None => Ok(Vec::new()),
        }
    }

    // -----------------------------------------------------------------------
    // Cluster operations
    // -----------------------------------------------------------------------

    pub fn put_cluster(&self, cluster: &ClusterInfo) -> Result<()> {
        let tree = self.tree(TREE_CLUSTERS)?;
        let value = serde_json::to_vec(cluster)?;
        tree.insert(cluster.id.as_ref().as_bytes(), value)
            .map_err(sled_err)?;
        Ok(())
    }

    pub fn get_cluster(&self, id: &ClusterId) -> Result<Option<ClusterInfo>> {
        let tree = self.tree(TREE_CLUSTERS)?;
        match tree.get(id.as_ref().as_bytes()).map_err(sled_err)? {
            Some(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
            None => Ok(None),
        }
    }

    pub fn list_clusters(&self) -> Result<Vec<ClusterInfo>> {
        let tree = self.tree(TREE_CLUSTERS)?;
        let mut clusters = Vec::new();
        for item in tree.iter() {
            let (_, value) = item.map_err(sled_err)?;
            clusters.push(serde_json::from_slice(&value)?);
        }
        Ok(clusters)
    }

    // -----------------------------------------------------------------------
    // Stats
    // -----------------------------------------------------------------------

    pub fn stats(&self) -> Result<StoreStats> {
        let node_count = self.db.open_tree(TREE_NODES).map_err(sled_err)?.len() as u64;
        let edge_count = self.db.open_tree(TREE_EDGES).map_err(sled_err)?.len() as u64;
        let cluster_count = self.db.open_tree(TREE_CLUSTERS).map_err(sled_err)?.len() as u64;
        let object_bytes: u64 = {
            let tree = self.db.open_tree(TREE_OBJECTS).map_err(sled_err)?;
            let mut total = 0u64;
            for item in tree.iter() {
                let (_, v) = item.map_err(sled_err)?;
                total += v.len() as u64;
            }
            total
        };
        Ok(StoreStats {
            node_count,
            edge_count,
            object_bytes,
            cluster_count,
        })
    }

    pub fn flush(&self) -> Result<()> {
        self.db.flush().map_err(sled_err)?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct StoreStats {
    pub node_count: u64,
    pub edge_count: u64,
    pub object_bytes: u64,
    pub cluster_count: u64,
}
