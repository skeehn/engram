use std::collections::HashMap;

/// Classic RRF: score(d) = sum_over_rankings(1 / (k + rank))
/// k=60 is the standard constant
pub fn rrf_fuse(rankings: &[Vec<String>], k: f32) -> Vec<(String, f32)> {
    let mut scores: HashMap<String, f32> = HashMap::new();
    for ranking in rankings {
        for (rank, id) in ranking.iter().enumerate() {
            *scores.entry(id.clone()).or_insert(0.0) += 1.0 / (k + rank as f32 + 1.0);
        }
    }
    let mut result: Vec<(String, f32)> = scores.into_iter().collect();
    result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    result
}
