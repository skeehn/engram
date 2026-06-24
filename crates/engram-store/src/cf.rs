// Tree names for sled
pub const TREE_NODES: &str = "nodes";
pub const TREE_EDGES: &str = "edges";
pub const TREE_EDGE_REV: &str = "edge_rev";
pub const TREE_TEMPORAL: &str = "temporal";
pub const TREE_KV: &str = "kv";
pub const TREE_OBJECTS: &str = "objects";
pub const TREE_PROVENANCE: &str = "provenance";
pub const TREE_CLUSTERS: &str = "clusters";

pub fn all_trees() -> Vec<&'static str> {
    vec![
        TREE_NODES,
        TREE_EDGES,
        TREE_EDGE_REV,
        TREE_TEMPORAL,
        TREE_KV,
        TREE_OBJECTS,
        TREE_PROVENANCE,
        TREE_CLUSTERS,
    ]
}
