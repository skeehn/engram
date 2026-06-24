use std::collections::{HashSet, VecDeque};
use engram_core::{
    id::NodeId,
    types::{Edge, Node},
    error::Result,
};
use engram_store::EngramStore;

pub struct GraphTraversal<'a> {
    store: &'a EngramStore,
}

impl<'a> GraphTraversal<'a> {
    pub fn new(store: &'a EngramStore) -> Self {
        Self { store }
    }

    /// BFS from a seed node up to `depth` hops. Returns (nodes, edges) in the subgraph.
    pub fn subgraph(&self, seed: &NodeId, depth: usize) -> Result<(Vec<Node>, Vec<Edge>)> {
        let mut visited: HashSet<String> = HashSet::new();
        let mut queue: VecDeque<(NodeId, usize)> = VecDeque::new();
        let mut nodes: Vec<Node> = Vec::new();
        let mut edges: Vec<Edge> = Vec::new();
        let mut edge_ids: HashSet<String> = HashSet::new();

        queue.push_back((seed.clone(), 0));
        visited.insert(seed.as_ref().to_string());

        while let Some((node_id, d)) = queue.pop_front() {
            if let Some(node) = self.store.get_node(&node_id)? {
                nodes.push(node);
            }

            if d >= depth {
                continue;
            }

            // Outgoing edges
            let out_edges = self.store.edges_from(&node_id)?;
            for edge in out_edges {
                let eid = edge.id.as_ref().to_string();
                if !edge_ids.contains(&eid) {
                    edge_ids.insert(eid);
                    let tid = edge.target.clone();
                    let tkey = tid.as_ref().to_string();
                    if !visited.contains(&tkey) {
                        visited.insert(tkey);
                        queue.push_back((tid, d + 1));
                    }
                    edges.push(edge);
                }
            }

            // Incoming edges
            let in_edges = self.store.edges_to(&node_id)?;
            for edge in in_edges {
                let eid = edge.id.as_ref().to_string();
                if !edge_ids.contains(&eid) {
                    edge_ids.insert(eid);
                    let sid = edge.source.clone();
                    let skey = sid.as_ref().to_string();
                    if !visited.contains(&skey) {
                        visited.insert(skey);
                        queue.push_back((sid, d + 1));
                    }
                    edges.push(edge);
                }
            }
        }

        Ok((nodes, edges))
    }

    /// BFS from a seed node, returning just the neighboring node IDs within `depth` hops.
    pub fn neighbors(&self, seed: &NodeId, depth: usize) -> Result<Vec<NodeId>> {
        let (nodes, _) = self.subgraph(seed, depth)?;
        Ok(nodes.into_iter().map(|n| n.id).collect())
    }
}
