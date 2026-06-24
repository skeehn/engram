use std::collections::HashMap;
use engram_core::{error::Result, id::NodeId, types::Node};
use engram_store::EngramStore;

pub struct PersonalizedPageRank<'a> {
    store: &'a EngramStore,
}

impl<'a> PersonalizedPageRank<'a> {
    pub fn new(store: &'a EngramStore) -> Self {
        Self { store }
    }

    /// Run PPR from seed nodes.
    /// alpha: teleport probability (0.15 typical)
    /// max_iter: iteration count (10 is enough)
    /// Returns HashMap<NodeId, f32> of PPR scores.
    pub fn run(
        &self,
        seeds: &[NodeId],
        alpha: f32,
        max_iter: usize,
    ) -> Result<HashMap<NodeId, f32>> {
        if seeds.is_empty() {
            return Ok(HashMap::new());
        }

        let n_seeds = seeds.len() as f32;
        let seed_weight = 1.0 / n_seeds;
        let seed_set: std::collections::HashSet<NodeId> =
            seeds.iter().cloned().collect();

        // Initialize scores
        let mut scores: HashMap<NodeId, f32> = HashMap::new();
        for seed in seeds {
            scores.insert(seed.clone(), seed_weight);
        }

        // Lazily-built adjacency: source -> Vec<(target, weight)>
        // We expand as we encounter new nodes.
        let mut adj: HashMap<NodeId, Vec<(NodeId, f32)>> = HashMap::new();

        // Pre-load adjacency for seed nodes
        for seed in seeds {
            self.load_adjacency(seed, &mut adj)?;
        }

        for _ in 0..max_iter {
            let mut new_scores: HashMap<NodeId, f32> = HashMap::new();

            for (v, &score_v) in &scores {
                if score_v == 0.0 {
                    continue;
                }

                // Lazily load adjacency for v if not present
                if !adj.contains_key(v) {
                    self.load_adjacency(v, &mut adj)?;
                }

                let neighbors = match adj.get(v) {
                    Some(n) => n,
                    None => continue,
                };

                if neighbors.is_empty() {
                    continue;
                }

                let total_weight: f32 = neighbors.iter().map(|(_, w)| w).sum();
                if total_weight == 0.0 {
                    continue;
                }

                for (u, w) in neighbors {
                    // Lazily load adjacency for discovered neighbor
                    if !adj.contains_key(u) {
                        self.load_adjacency(u, &mut adj)?;
                    }
                    let entry = new_scores.entry(u.clone()).or_insert(0.0);
                    *entry += alpha * score_v * w / total_weight;
                }
            }

            // Teleport back to seeds
            for seed in seeds {
                let entry = new_scores.entry(seed.clone()).or_insert(0.0);
                *entry += (1.0 - alpha) * seed_weight;
            }

            // Normalize
            let total: f32 = new_scores.values().sum();
            if total > 0.0 {
                for v in new_scores.values_mut() {
                    *v /= total;
                }
            }

            scores = new_scores;
        }

        Ok(scores)
    }

    /// Run PPR and return top-k nodes sorted by score descending.
    pub fn top_k(
        &self,
        seeds: &[NodeId],
        alpha: f32,
        max_iter: usize,
        k: usize,
    ) -> Result<Vec<(Node, f32)>> {
        let scores = self.run(seeds, alpha, max_iter)?;

        // Sort by score descending
        let mut sorted: Vec<(NodeId, f32)> = scores.into_iter().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted.truncate(k);

        let mut result = Vec::with_capacity(sorted.len());
        for (id, score) in sorted {
            if let Some(node) = self.store.get_node(&id)? {
                result.push((node, score));
            }
        }
        Ok(result)
    }

    fn load_adjacency(
        &self,
        id: &NodeId,
        adj: &mut HashMap<NodeId, Vec<(NodeId, f32)>>,
    ) -> Result<()> {
        if adj.contains_key(id) {
            return Ok(());
        }
        let edges = self.store.edges_from(id)?;
        let neighbors: Vec<(NodeId, f32)> = edges
            .into_iter()
            .map(|e| (e.target, e.weight))
            .collect();
        adj.insert(id.clone(), neighbors);
        Ok(())
    }
}
