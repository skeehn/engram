use std::collections::{HashMap, HashSet, VecDeque};
use engram_core::{error::Result, id::NodeId, types::{Edge, Node}};
use engram_store::EngramStore;

pub struct GraphTraversal<'a> {
    store: &'a EngramStore,
}

impl<'a> GraphTraversal<'a> {
    pub fn new(store: &'a EngramStore) -> Self {
        Self { store }
    }

    /// BFS from seed nodes up to max_depth hops. Returns all visited nodes.
    pub fn bfs(&self, seeds: &[NodeId], max_depth: usize) -> Result<Vec<Node>> {
        let mut visited: HashSet<NodeId> = HashSet::new();
        // queue holds (node_id, current_depth)
        let mut queue: VecDeque<(NodeId, usize)> = VecDeque::new();
        let mut result: Vec<Node> = Vec::new();

        for seed in seeds {
            if visited.insert(seed.clone()) {
                queue.push_back((seed.clone(), 0));
            }
        }

        while let Some((id, depth)) = queue.pop_front() {
            if let Some(node) = self.store.get_node(&id)? {
                result.push(node);
            }
            if depth < max_depth {
                let outbound = self.store.edges_from(&id)?;
                let inbound = self.store.edges_to(&id)?;
                for edge in outbound.iter().chain(inbound.iter()) {
                    let neighbor = if &edge.source == &id { edge.target.clone() } else { edge.source.clone() };
                    if visited.insert(neighbor.clone()) {
                        queue.push_back((neighbor, depth + 1));
                    }
                }
            }
        }

        Ok(result)
    }

    /// DFS from a single seed node up to max_depth hops.
    pub fn dfs(&self, seed: &NodeId, max_depth: usize) -> Result<Vec<Node>> {
        let mut visited: HashSet<NodeId> = HashSet::new();
        let mut result: Vec<Node> = Vec::new();
        self.dfs_inner(seed, max_depth, 0, &mut visited, &mut result)?;
        Ok(result)
    }

    fn dfs_inner(
        &self,
        id: &NodeId,
        max_depth: usize,
        depth: usize,
        visited: &mut HashSet<NodeId>,
        result: &mut Vec<Node>,
    ) -> Result<()> {
        if !visited.insert(id.clone()) {
            return Ok(());
        }
        if let Some(node) = self.store.get_node(id)? {
            result.push(node);
        }
        if depth < max_depth {
            let outbound = self.store.edges_from(id)?;
            let inbound = self.store.edges_to(id)?;
            for edge in outbound.iter().chain(inbound.iter()) {
                let neighbor = if &edge.source == id { edge.target.clone() } else { edge.source.clone() };
                self.dfs_inner(&neighbor, max_depth, depth + 1, visited, result)?;
            }
        }
        Ok(())
    }

    /// Find shortest path between two nodes (BFS). Returns node sequence or None.
    pub fn shortest_path(
        &self,
        from: &NodeId,
        to: &NodeId,
        max_depth: usize,
    ) -> Result<Option<Vec<Node>>> {
        if from == to {
            if let Some(node) = self.store.get_node(from)? {
                return Ok(Some(vec![node]));
            }
            return Ok(None);
        }

        // BFS with parent tracking
        let mut visited: HashSet<NodeId> = HashSet::new();
        let mut parent: HashMap<NodeId, NodeId> = HashMap::new();
        let mut queue: VecDeque<(NodeId, usize)> = VecDeque::new();

        visited.insert(from.clone());
        queue.push_back((from.clone(), 0));
        let mut found = false;

        'outer: while let Some((id, depth)) = queue.pop_front() {
            if depth >= max_depth {
                continue;
            }
            let outbound = self.store.edges_from(&id)?;
            let inbound = self.store.edges_to(&id)?;
            for edge in outbound.iter().chain(inbound.iter()) {
                let neighbor = if &edge.source == &id { edge.target.clone() } else { edge.source.clone() };
                if visited.insert(neighbor.clone()) {
                    parent.insert(neighbor.clone(), id.clone());
                    if &neighbor == to {
                        found = true;
                        break 'outer;
                    }
                    queue.push_back((neighbor, depth + 1));
                }
            }
        }

        if !found {
            return Ok(None);
        }

        // Reconstruct path
        let mut path_ids: Vec<NodeId> = Vec::new();
        let mut cur = to.clone();
        loop {
            path_ids.push(cur.clone());
            if &cur == from {
                break;
            }
            match parent.get(&cur) {
                Some(p) => cur = p.clone(),
                None => return Ok(None),
            }
        }
        path_ids.reverse();

        let mut nodes = Vec::new();
        for id in &path_ids {
            if let Some(node) = self.store.get_node(id)? {
                nodes.push(node);
            }
        }
        Ok(Some(nodes))
    }

    /// Get neighborhood: node + all nodes reachable in 1 hop (outbound + inbound).
    pub fn neighborhood(&self, node_id: &NodeId) -> Result<Vec<Node>> {
        let mut seen: HashSet<NodeId> = HashSet::new();
        let mut result: Vec<Node> = Vec::new();

        seen.insert(node_id.clone());
        if let Some(node) = self.store.get_node(node_id)? {
            result.push(node);
        }

        let outbound = self.store.edges_from(node_id)?;
        let inbound = self.store.edges_to(node_id)?;
        for edge in outbound.iter().chain(inbound.iter()) {
            let neighbor = if &edge.source == node_id { edge.target.clone() } else { edge.source.clone() };
            if seen.insert(neighbor.clone()) {
                if let Some(node) = self.store.get_node(&neighbor)? {
                    result.push(node);
                }
            }
        }

        Ok(result)
    }

    /// Get subgraph: all nodes and edges within depth hops of seed.
    pub fn subgraph(&self, seed: &NodeId, depth: usize) -> Result<(Vec<Node>, Vec<Edge>)> {
        let mut visited_nodes: HashSet<NodeId> = HashSet::new();
        let mut visited_edges: HashSet<String> = HashSet::new();
        let mut queue: VecDeque<(NodeId, usize)> = VecDeque::new();
        let mut result_nodes: Vec<Node> = Vec::new();
        let mut result_edges: Vec<Edge> = Vec::new();

        visited_nodes.insert(seed.clone());
        queue.push_back((seed.clone(), 0));

        while let Some((id, d)) = queue.pop_front() {
            if let Some(node) = self.store.get_node(&id)? {
                result_nodes.push(node);
            }

            if d < depth {
                let outbound = self.store.edges_from(&id)?;
                let inbound = self.store.edges_to(&id)?;
                for edge in outbound.into_iter().chain(inbound.into_iter()) {
                    let edge_key = edge.id.as_ref().to_string();
                    let neighbor = if &edge.source == &id { edge.target.clone() } else { edge.source.clone() };

                    if visited_edges.insert(edge_key) {
                        result_edges.push(edge);
                    }
                    if visited_nodes.insert(neighbor.clone()) {
                        queue.push_back((neighbor, d + 1));
                    }
                }
            }
        }

        Ok((result_nodes, result_edges))
    }
}
