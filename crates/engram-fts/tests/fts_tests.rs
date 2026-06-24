use engram_core::{
    id::NodeId,
    types::{Node, NodeType},
};
use engram_fts::FtsIndex;
use tempfile::TempDir;

fn open_fts() -> (TempDir, FtsIndex) {
    let dir = TempDir::new().expect("tempdir");
    let idx = FtsIndex::open(dir.path().join("fts")).expect("open fts");
    (dir, idx)
}

/// Commit and force reload so the reader sees new data immediately.
fn commit_and_reload(fts: &FtsIndex) {
    fts.commit().expect("commit");
    fts.reload().expect("reload");
}

#[test]
fn test_index_and_search_node() {
    let (_dir, fts) = open_fts();

    let node = Node::new("quantum computing is fascinating", NodeType::Fact);
    let node_id = node.id.clone();

    fts.index_node(&node).expect("index_node");
    commit_and_reload(&fts);

    // Search for a keyword in the body
    let results = fts.search("quantum", 10).expect("search");
    assert!(!results.is_empty(), "should find indexed node");
    let found = results.iter().any(|(id, _score)| id == &node_id);
    assert!(found, "should find the correct node_id");
}

#[test]
fn test_search_returns_correct_node_id() {
    let (_dir, fts) = open_fts();

    let node_a = Node::new("machine learning transforms industries", NodeType::Concept);
    let node_b = Node::new("gardening tips for spring flowers", NodeType::Note);
    let id_a = node_a.id.clone();
    let id_b = node_b.id.clone();

    fts.index_node(&node_a).expect("index a");
    fts.index_node(&node_b).expect("index b");
    commit_and_reload(&fts);

    // Search for "machine" -- should return node A only
    let results = fts.search("machine", 10).expect("search machine");
    assert!(!results.is_empty());
    let found_a = results.iter().any(|(id, _)| id == &id_a);
    let found_b = results.iter().any(|(id, _)| id == &id_b);
    assert!(found_a, "node A should appear in results");
    assert!(!found_b, "node B should not appear in results for 'machine'");

    // Search for "gardening" -- should return node B only
    let results2 = fts.search("gardening", 10).expect("search gardening");
    let found_a2 = results2.iter().any(|(id, _)| id == &id_a);
    let found_b2 = results2.iter().any(|(id, _)| id == &id_b);
    assert!(!found_a2, "node A should not appear for 'gardening'");
    assert!(found_b2, "node B should appear in results");
}

#[test]
fn test_remove_node_from_index() {
    let (_dir, fts) = open_fts();

    let node = Node::new("unique phrase about cryptography", NodeType::Document);
    let node_id = node.id.clone();

    fts.index_node(&node).expect("index_node");
    commit_and_reload(&fts);

    // Verify it appears
    let before = fts.search("cryptography", 10).expect("search before remove");
    assert!(
        before.iter().any(|(id, _)| id == &node_id),
        "should be present before removal"
    );

    // Remove from index
    fts.remove_node(&node_id).expect("remove_node");
    commit_and_reload(&fts);

    // Verify it's gone
    let after = fts.search("cryptography", 10).expect("search after remove");
    assert!(
        !after.iter().any(|(id, _)| id == &node_id),
        "should not appear after removal"
    );
}

#[test]
fn test_doc_count() {
    let (_dir, fts) = open_fts();

    let n1 = Node::new("first unique node content", NodeType::Fact);
    let n2 = Node::new("second unique node content", NodeType::Fact);
    let n3 = Node::new("third unique node content", NodeType::Fact);

    fts.index_node(&n1).expect("index 1");
    fts.index_node(&n2).expect("index 2");
    fts.index_node(&n3).expect("index 3");
    commit_and_reload(&fts);

    let count = fts.doc_count().expect("doc_count");
    assert_eq!(count, 3, "should have 3 indexed docs");
}

#[test]
fn test_search_by_tag() {
    let (_dir, fts) = open_fts();

    let mut node = Node::new("some content about science", NodeType::Fact);
    node.tags = vec!["science".to_string(), "physics".to_string()];
    let node_id = node.id.clone();

    fts.index_node(&node).expect("index_node");
    commit_and_reload(&fts);

    let results = fts.search_by_tag("science", 10).expect("search_by_tag");
    assert!(
        results.iter().any(|(id, _)| id == &node_id),
        "should find node by tag"
    );
}
