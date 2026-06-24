use engram_core::{id::NodeId, types::{Node, NodeType}};
use engram_vector::VectorIndex;
use tempfile::TempDir;

const DIM: usize = 4;

fn open_vector(dir: &TempDir) -> VectorIndex {
    VectorIndex::new(DIM, dir.path().join("vectors.json")).expect("open vector index")
}

fn make_id(s: &str) -> NodeId {
    NodeId::from_string(s)
}

#[test]
fn test_upsert_and_search() {
    let dir = TempDir::new().expect("tempdir");
    let idx = open_vector(&dir);

    // 5 nodes with 4-dimensional embeddings
    // node1 is at (1, 0, 0, 0)
    let id1 = make_id("node1");
    let id2 = make_id("node2");
    let id3 = make_id("node3");
    let id4 = make_id("node4");
    let id5 = make_id("node5");

    idx.upsert(&id1, &[1.0, 0.0, 0.0, 0.0]).expect("upsert 1");
    idx.upsert(&id2, &[0.0, 1.0, 0.0, 0.0]).expect("upsert 2");
    idx.upsert(&id3, &[0.0, 0.0, 1.0, 0.0]).expect("upsert 3");
    idx.upsert(&id4, &[0.0, 0.0, 0.0, 1.0]).expect("upsert 4");
    idx.upsert(&id5, &[-1.0, 0.0, 0.0, 0.0]).expect("upsert 5");

    assert_eq!(idx.len(), 5);

    // Query close to node1: (0.9, 0.1, 0.0, 0.0) normalized
    // cosine similarity to node1 should be highest
    let query = [0.9_f32, 0.1, 0.0, 0.0];
    let results = idx.search(&query, 5).expect("search");

    assert!(!results.is_empty(), "should return results");
    let top_id = &results[0].0;
    assert_eq!(top_id, &id1, "node1 should be top result (closest to query)");

    // node5 is opposite direction -- should be last (or near-last)
    let last_id = &results[results.len() - 1].0;
    assert_eq!(last_id, &id5, "node5 should be least similar");
}

#[test]
fn test_cosine_similarity_ordering() {
    let dir = TempDir::new().expect("tempdir");
    let idx = open_vector(&dir);

    let id_close = make_id("close");
    let id_far = make_id("far");
    let id_opposite = make_id("opposite");

    // close: same direction as query
    idx.upsert(&id_close, &[1.0, 1.0, 0.0, 0.0]).expect("upsert close");
    // far: orthogonal
    idx.upsert(&id_far, &[0.0, 0.0, 1.0, 0.0]).expect("upsert far");
    // opposite: opposite direction
    idx.upsert(&id_opposite, &[-1.0, -1.0, 0.0, 0.0]).expect("upsert opposite");

    // Query in direction (1, 1, 0, 0)
    let query = [1.0_f32, 1.0, 0.0, 0.0];
    let results = idx.search(&query, 3).expect("search");

    assert_eq!(results.len(), 3);

    // Verify ordering: close > far >= opposite (cosine similarity descending)
    let score_close = results.iter().find(|(id, _)| id == &id_close).map(|(_, s)| *s);
    let score_far = results.iter().find(|(id, _)| id == &id_far).map(|(_, s)| *s);
    let score_opposite = results.iter().find(|(id, _)| id == &id_opposite).map(|(_, s)| *s);

    let sc = score_close.expect("close in results");
    let sf = score_far.expect("far in results");
    let so = score_opposite.expect("opposite in results");

    assert!(sc > sf, "close should score higher than orthogonal (far)");
    assert!(sf > so, "orthogonal should score higher than opposite");
    assert!(so < 0.0, "opposite direction should have negative cosine similarity");
    assert!((sc - 1.0).abs() < 1e-5, "identical direction should have cosine ~1.0");
}

#[test]
fn test_upsert_replaces_existing() {
    let dir = TempDir::new().expect("tempdir");
    let idx = open_vector(&dir);

    let id = make_id("mynode");

    idx.upsert(&id, &[1.0, 0.0, 0.0, 0.0]).expect("first upsert");
    assert_eq!(idx.len(), 1);

    // Upsert again with different vector
    idx.upsert(&id, &[0.0, 1.0, 0.0, 0.0]).expect("second upsert");
    // Still only 1 entry
    assert_eq!(idx.len(), 1);

    // Search with (0, 1, 0, 0) query -- should return our node as top
    let results = idx.search(&[0.0, 1.0, 0.0, 0.0], 1).expect("search");
    assert_eq!(results[0].0, id);
    assert!((results[0].1 - 1.0).abs() < 1e-5, "cosine ~1.0 after upsert");
}

#[test]
fn test_remove_node() {
    let dir = TempDir::new().expect("tempdir");
    let idx = open_vector(&dir);

    let id1 = make_id("keep");
    let id2 = make_id("remove");

    idx.upsert(&id1, &[1.0, 0.0, 0.0, 0.0]).expect("upsert keep");
    idx.upsert(&id2, &[0.9, 0.1, 0.0, 0.0]).expect("upsert remove");
    assert_eq!(idx.len(), 2);

    idx.remove(&id2).expect("remove");
    assert_eq!(idx.len(), 1);

    // Search should only return the kept node
    let results = idx.search(&[1.0, 0.0, 0.0, 0.0], 10).expect("search after remove");
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, id1, "only id1 should remain");
}

#[test]
fn test_dimension_mismatch_error() {
    let dir = TempDir::new().expect("tempdir");
    let idx = open_vector(&dir);

    let id = make_id("bad");
    // Wrong dimension (5 instead of 4)
    let result = idx.upsert(&id, &[1.0, 0.0, 0.0, 0.0, 0.0]);
    assert!(result.is_err(), "should fail with dimension mismatch");

    let search_result = idx.search(&[1.0, 0.0, 0.0, 0.0, 0.0], 1);
    assert!(search_result.is_err(), "search with wrong dim should fail");
}

#[test]
fn test_save_and_reload() {
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().join("vectors.json");

    let id1 = make_id("persistent1");
    let id2 = make_id("persistent2");

    {
        let idx = VectorIndex::new(DIM, &path).expect("create");
        idx.upsert(&id1, &[1.0, 0.0, 0.0, 0.0]).expect("upsert 1");
        idx.upsert(&id2, &[0.0, 1.0, 0.0, 0.0]).expect("upsert 2");
        idx.save().expect("save");
    }

    // Reload and verify
    let idx2 = VectorIndex::new(DIM, &path).expect("reload");
    assert_eq!(idx2.len(), 2, "should reload 2 entries");

    let results = idx2.search(&[1.0, 0.0, 0.0, 0.0], 2).expect("search after reload");
    assert_eq!(results[0].0, id1, "id1 should be top after reload");
}
