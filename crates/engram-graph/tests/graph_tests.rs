use engram_core::{
    id::NodeId,
    types::{Edge, EdgeType, Node, NodeType},
};
use engram_graph::GraphTraversal;
use engram_store::EngramStore;
use tempfile::TempDir;

fn open_store(dir: &TempDir) -> EngramStore {
    EngramStore::open(dir.path().join("store")).expect("open store")
}

fn make_node(body: &str) -> Node {
    Node::new(body, NodeType::Entity)
}

/// Create chain: A -> B -> C -> D -> E
/// Returns node IDs in order [A, B, C, D, E]
fn build_chain(store: &EngramStore) -> Vec<NodeId> {
    let nodes: Vec<Node> = (0..5).map(|i| make_node(&format!("node_{i}"))).collect();
    for n in &nodes {
        store.put_node(n).expect("put node");
    }
    // Create directed edges A->B->C->D->E
    for i in 0..4 {
        let edge = Edge::new(nodes[i].id.clone(), nodes[i + 1].id.clone(), EdgeType::RelatedTo);
        store.put_edge(&edge).expect("put edge");
    }
    nodes.into_iter().map(|n| n.id).collect()
}

#[test]
fn test_subgraph_depth_2_from_start() {
    let dir = TempDir::new().expect("tempdir");
    let store = open_store(&dir);
    let ids = build_chain(&store);

    let trav = GraphTraversal::new(&store);
    // From A (ids[0]) with depth=2: should include A, B, C
    let (nodes, _edges) = trav.subgraph(&ids[0], 2).expect("subgraph");

    let node_ids: Vec<&NodeId> = nodes.iter().map(|n| &n.id).collect();
    assert!(
        node_ids.contains(&&ids[0]),
        "A should be in subgraph"
    );
    assert!(
        node_ids.contains(&&ids[1]),
        "B should be in subgraph at depth 1"
    );
    assert!(
        node_ids.contains(&&ids[2]),
        "C should be in subgraph at depth 2"
    );
    // D and E should NOT be included at depth=2 from A
    assert!(
        !node_ids.contains(&&ids[3]),
        "D should NOT be in depth-2 subgraph from A"
    );
    assert!(
        !node_ids.contains(&&ids[4]),
        "E should NOT be in depth-2 subgraph from A"
    );
    assert_eq!(nodes.len(), 3, "exactly 3 nodes at depth=2 from A");
}

#[test]
fn test_subgraph_depth_4_gets_all() {
    let dir = TempDir::new().expect("tempdir");
    let store = open_store(&dir);
    let ids = build_chain(&store);

    let trav = GraphTraversal::new(&store);
    // From A with depth=4: should reach all 5 nodes
    let (nodes, _edges) = trav.subgraph(&ids[0], 4).expect("subgraph");

    assert_eq!(nodes.len(), 5, "all 5 nodes should be in depth-4 subgraph");
    for (i, id) in ids.iter().enumerate() {
        let node_ids: Vec<&NodeId> = nodes.iter().map(|n| &n.id).collect();
        assert!(
            node_ids.contains(&id),
            "node {i} should be present"
        );
    }
}

#[test]
fn test_neighbors_returns_node_ids() {
    let dir = TempDir::new().expect("tempdir");
    let store = open_store(&dir);
    let ids = build_chain(&store);

    let trav = GraphTraversal::new(&store);
    // neighbors from B (ids[1]) at depth=1: should include B, A (inbound), C (outbound)
    let neighbors = trav.neighbors(&ids[1], 1).expect("neighbors");

    assert!(
        neighbors.contains(&ids[0]),
        "A (predecessor of B) should be a neighbor"
    );
    assert!(
        neighbors.contains(&ids[1]),
        "B itself is included in subgraph"
    );
    assert!(
        neighbors.contains(&ids[2]),
        "C (successor of B) should be a neighbor"
    );
}

#[test]
fn test_subgraph_middle_node() {
    let dir = TempDir::new().expect("tempdir");
    let store = open_store(&dir);
    let ids = build_chain(&store);

    let trav = GraphTraversal::new(&store);
    // Subgraph from C (ids[2]) at depth=1
    // Should include: C, B (inbound edge), D (outbound edge)
    let (nodes, edges) = trav.subgraph(&ids[2], 1).expect("subgraph from C");

    let node_ids: Vec<&NodeId> = nodes.iter().map(|n| &n.id).collect();
    assert!(
        node_ids.contains(&&ids[1]),
        "B should be in C's depth-1 neighborhood"
    );
    assert!(
        node_ids.contains(&&ids[2]),
        "C itself should be in result"
    );
    assert!(
        node_ids.contains(&&ids[3]),
        "D should be in C's depth-1 neighborhood"
    );
    assert_eq!(nodes.len(), 3, "C + B + D = 3 nodes at depth=1");
    assert_eq!(edges.len(), 2, "B->C and C->D = 2 edges at depth=1");
}

#[test]
fn test_subgraph_includes_edges() {
    let dir = TempDir::new().expect("tempdir");
    let store = open_store(&dir);
    let ids = build_chain(&store);

    let trav = GraphTraversal::new(&store);
    let (nodes, edges) = trav.subgraph(&ids[0], 4).expect("full subgraph");

    assert_eq!(nodes.len(), 5);
    // The chain A->B->C->D->E has 4 directed edges
    assert_eq!(edges.len(), 4, "should have 4 edges in the chain");

    // Verify each edge connects consecutive nodes
    for i in 0..4 {
        let has_edge = edges.iter().any(|e| {
            e.source == ids[i] && e.target == ids[i + 1]
        });
        assert!(
            has_edge,
            "edge from node_{i} to node_{} should be present",
            i + 1
        );
    }
}

#[test]
fn test_empty_graph() {
    let dir = TempDir::new().expect("tempdir");
    let store = open_store(&dir);

    let isolated = make_node("isolated");
    let id = isolated.id.clone();
    store.put_node(&isolated).expect("put isolated");

    let trav = GraphTraversal::new(&store);
    let (nodes, edges) = trav.subgraph(&id, 10).expect("subgraph of isolated");
    assert_eq!(nodes.len(), 1, "only the isolated node");
    assert_eq!(edges.len(), 0, "no edges");

    let neighbors = trav.neighbors(&id, 1).expect("neighbors");
    assert_eq!(neighbors.len(), 1, "only self in neighbors");
}
