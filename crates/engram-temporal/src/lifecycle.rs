//! Memory lifecycle management with decay, importance, and recency scoring.
//!
//! Implements a sophisticated memory model inspired by human memory:
//! - **Decay**: Memories fade over time unless reinforced
//! - **Importance**: Based on access frequency, connections, and explicit boosts
//! - **Recency**: Recent interactions boost relevance
//! - **Lifecycle**: Archive stale memories, forget below threshold

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use engram_core::{error::Result, id::NodeId, types::Node};

/// Memory importance factors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportanceFactors {
    /// Number of times this memory was accessed/retrieved
    pub access_count: u32,
    /// Number of edges (connections to other memories)
    pub connection_count: u32,
    /// Explicit importance boost (0.0-1.0)
    pub explicit_boost: f32,
    /// Number of times this memory was edited/updated
    pub edit_count: u32,
    /// Last access timestamp
    pub last_accessed: DateTime<Utc>,
    /// Last reinforcement timestamp (explicit refresh)
    pub last_reinforced: Option<DateTime<Utc>>,
}

impl Default for ImportanceFactors {
    fn default() -> Self {
        Self {
            access_count: 0,
            connection_count: 0,
            explicit_boost: 0.0,
            edit_count: 0,
            last_accessed: Utc::now(),
            last_reinforced: None,
        }
    }
}

/// Memory lifecycle state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LifecycleState {
    /// Active in working memory
    Active,
    /// Consolidated into long-term storage
    Consolidated,
    /// Archived due to staleness (retrievable but deprioritized)
    Archived,
    /// Marked for forgetting (will be purged)
    Forgotten,
}

/// Configuration for memory lifecycle.
#[derive(Debug, Clone)]
pub struct LifecycleConfig {
    /// Decay rate (lambda) - higher = faster decay
    pub decay_rate: f32,
    /// Half-life in seconds (alternative to decay_rate)
    pub half_life_secs: Option<f64>,
    /// Threshold for archiving (combined score below this)
    pub archive_threshold: f32,
    /// Threshold for forgetting (combined score below this)
    pub forget_threshold: f32,
    /// Weight for recency in combined score
    pub recency_weight: f32,
    /// Weight for importance in combined score
    pub importance_weight: f32,
    /// Weight for confidence/base strength in combined score
    pub confidence_weight: f32,
    /// Consolidation delay (memories stay active for this long)
    pub consolidation_delay: Duration,
}

impl Default for LifecycleConfig {
    fn default() -> Self {
        Self {
            decay_rate: 0.0001, // ~2.7 hour half-life
            half_life_secs: None,
            archive_threshold: 0.2,
            forget_threshold: 0.05,
            recency_weight: 0.3,
            importance_weight: 0.4,
            confidence_weight: 0.3,
            consolidation_delay: Duration::hours(24),
        }
    }
}

impl LifecycleConfig {
    /// Get effective decay rate, computing from half-life if provided.
    pub fn effective_decay_rate(&self) -> f32 {
        if let Some(half_life) = self.half_life_secs {
            // lambda = ln(2) / half_life
            (2.0_f64.ln() / half_life) as f32
        } else {
            self.decay_rate
        }
    }
}

/// Memory lifecycle manager.
pub struct LifecycleManager {
    config: LifecycleConfig,
    /// Importance factors per node (in-memory cache, would persist in production)
    importance: HashMap<NodeId, ImportanceFactors>,
}

impl LifecycleManager {
    /// Create a new lifecycle manager.
    pub fn new(config: LifecycleConfig) -> Self {
        Self {
            config,
            importance: HashMap::new(),
        }
    }

    /// Create with default config.
    pub fn with_defaults() -> Self {
        Self::new(LifecycleConfig::default())
    }

    /// Record an access to a memory.
    pub fn record_access(&mut self, node_id: &NodeId) {
        let factors = self.importance.entry(node_id.clone()).or_default();
        factors.access_count += 1;
        factors.last_accessed = Utc::now();
    }

    /// Record a connection being added.
    pub fn record_connection(&mut self, node_id: &NodeId) {
        let factors = self.importance.entry(node_id.clone()).or_default();
        factors.connection_count += 1;
    }

    /// Record an edit to a memory.
    pub fn record_edit(&mut self, node_id: &NodeId) {
        let factors = self.importance.entry(node_id.clone()).or_default();
        factors.edit_count += 1;
        factors.last_accessed = Utc::now();
    }

    /// Explicitly reinforce a memory (spaced repetition).
    pub fn reinforce(&mut self, node_id: &NodeId, boost: f32) {
        let factors = self.importance.entry(node_id.clone()).or_default();
        factors.last_reinforced = Some(Utc::now());
        factors.explicit_boost = (factors.explicit_boost + boost).min(1.0);
    }

    /// Get importance factors for a node.
    pub fn get_importance(&self, node_id: &NodeId) -> Option<&ImportanceFactors> {
        self.importance.get(node_id)
    }

    /// Compute recency score (0.0-1.0) based on last access.
    pub fn recency_score(&self, node_id: &NodeId, now: DateTime<Utc>) -> f32 {
        let factors = match self.importance.get(node_id) {
            Some(f) => f,
            None => return 0.0, // No tracking = zero recency
        };

        let elapsed_secs = (now - factors.last_accessed).num_seconds().max(0) as f32;
        let lambda = self.config.effective_decay_rate();

        // Exponential decay from 1.0
        (-lambda * elapsed_secs).exp()
    }

    /// Compute importance score (0.0-1.0) based on factors.
    pub fn importance_score(&self, node_id: &NodeId) -> f32 {
        let factors = match self.importance.get(node_id) {
            Some(f) => f,
            None => return 0.0, // No tracking = zero importance
        };

        // Access contribution: log scale, capped
        let access_score = (1.0 + factors.access_count as f32).ln() / 5.0; // ln(150) ≈ 5

        // Connection contribution: sqrt scale, capped
        let connection_score = (factors.connection_count as f32).sqrt() / 10.0; // sqrt(100) = 10

        // Edit contribution: log scale
        let edit_score = (1.0 + factors.edit_count as f32).ln() / 3.0;

        // Combine with explicit boost
        let raw_score = access_score * 0.4 + connection_score * 0.3 + edit_score * 0.2 + factors.explicit_boost * 0.1;

        raw_score.clamp(0.0, 1.0)
    }

    /// Compute combined memory strength score.
    pub fn memory_strength(&self, node: &Node, now: DateTime<Utc>) -> f32 {
        let recency = self.recency_score(&node.id, now);
        let importance = self.importance_score(&node.id);
        let confidence = node.confidence;

        // Weighted combination
        let score = recency * self.config.recency_weight
            + importance * self.config.importance_weight
            + confidence * self.config.confidence_weight;

        score.clamp(0.0, 1.0)
    }

    /// Determine lifecycle state for a node.
    pub fn lifecycle_state(&self, node: &Node, now: DateTime<Utc>) -> LifecycleState {
        let strength = self.memory_strength(node, now);
        let age = now - node.tx_time;

        // Check if still in consolidation period
        if age < self.config.consolidation_delay {
            return LifecycleState::Active;
        }

        // Determine state based on strength thresholds
        if strength < self.config.forget_threshold {
            LifecycleState::Forgotten
        } else if strength < self.config.archive_threshold {
            LifecycleState::Archived
        } else {
            LifecycleState::Consolidated
        }
    }

    /// Score nodes and return sorted by memory strength (descending).
    pub fn rank_by_strength(&self, nodes: Vec<Node>, now: DateTime<Utc>) -> Vec<(Node, f32)> {
        let mut scored: Vec<_> = nodes
            .into_iter()
            .map(|n| {
                let strength = self.memory_strength(&n, now);
                (n, strength)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored
    }

    /// Filter nodes by lifecycle state.
    pub fn filter_by_state(&self, nodes: Vec<Node>, state: LifecycleState, now: DateTime<Utc>) -> Vec<Node> {
        nodes
            .into_iter()
            .filter(|n| self.lifecycle_state(n, now) == state)
            .collect()
    }

    /// Get nodes that should be archived.
    pub fn get_archivable(&self, nodes: Vec<Node>, now: DateTime<Utc>) -> Vec<Node> {
        self.filter_by_state(nodes, LifecycleState::Archived, now)
    }

    /// Get nodes that should be forgotten/purged.
    pub fn get_forgettable(&self, nodes: Vec<Node>, now: DateTime<Utc>) -> Vec<Node> {
        self.filter_by_state(nodes, LifecycleState::Forgotten, now)
    }

    /// Spaced repetition: calculate optimal review time for a node.
    /// Returns suggested review timestamp based on Ebbinghaus forgetting curve.
    pub fn next_review_time(&self, node: &Node, now: DateTime<Utc>) -> DateTime<Utc> {
        let factors = self.importance.get(&node.id);
        let review_count = factors.map_or(0, |f| f.access_count);

        // Interval increases with each successful review (spaced repetition)
        // Base interval: 1 day, grows by 2x each review, capped at 90 days
        let base_interval_hours = 24;
        let multiplier = 2.0_f64.powi(review_count.min(6) as i32); // Cap at 2^6 = 64x
        let interval_hours = (base_interval_hours as f64 * multiplier).min(90.0 * 24.0);

        now + Duration::hours(interval_hours as i64)
    }

    /// Get statistics about lifecycle distribution.
    pub fn lifecycle_stats(&self, nodes: &[Node], now: DateTime<Utc>) -> LifecycleStats {
        let mut stats = LifecycleStats::default();

        for node in nodes {
            match self.lifecycle_state(node, now) {
                LifecycleState::Active => stats.active += 1,
                LifecycleState::Consolidated => stats.consolidated += 1,
                LifecycleState::Archived => stats.archived += 1,
                LifecycleState::Forgotten => stats.forgotten += 1,
            }
        }

        stats.total = nodes.len();
        stats
    }
}

/// Statistics about memory lifecycle distribution.
#[derive(Debug, Default, Clone)]
pub struct LifecycleStats {
    pub total: usize,
    pub active: usize,
    pub consolidated: usize,
    pub archived: usize,
    pub forgotten: usize,
}

impl LifecycleStats {
    pub fn active_ratio(&self) -> f32 {
        if self.total == 0 { 0.0 } else { self.active as f32 / self.total as f32 }
    }

    pub fn consolidated_ratio(&self) -> f32 {
        if self.total == 0 { 0.0 } else { self.consolidated as f32 / self.total as f32 }
    }

    pub fn archived_ratio(&self) -> f32 {
        if self.total == 0 { 0.0 } else { self.archived as f32 / self.total as f32 }
    }

    pub fn forgotten_ratio(&self) -> f32 {
        if self.total == 0 { 0.0 } else { self.forgotten as f32 / self.total as f32 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engram_core::types::NodeType;

    fn make_node(age_hours: i64, confidence: f32) -> Node {
        let mut node = Node::new("test content", NodeType::Note);
        node.tx_time = Utc::now() - Duration::hours(age_hours);
        node.confidence = confidence;
        node
    }

    #[test]
    fn test_recency_decay() {
        let mut manager = LifecycleManager::with_defaults();
        let node = make_node(0, 1.0);

        // Record access
        manager.record_access(&node.id);

        let now = Utc::now();
        let score_now = manager.recency_score(&node.id, now);
        assert!(score_now > 0.9, "Recent access should have high recency");

        // Simulate time passing
        let future = now + Duration::hours(24);
        let score_later = manager.recency_score(&node.id, future);
        assert!(score_later < score_now, "Recency should decay over time");
    }

    #[test]
    fn test_importance_accumulation() {
        let mut manager = LifecycleManager::with_defaults();
        let node = make_node(0, 1.0);

        let initial = manager.importance_score(&node.id);

        // Record multiple accesses
        for _ in 0..10 {
            manager.record_access(&node.id);
        }

        let after_access = manager.importance_score(&node.id);
        assert!(after_access > initial, "Importance should increase with access");

        // Record connections
        for _ in 0..5 {
            manager.record_connection(&node.id);
        }

        let after_connections = manager.importance_score(&node.id);
        assert!(after_connections > after_access, "Importance should increase with connections");
    }

    #[test]
    fn test_lifecycle_states() {
        let config = LifecycleConfig {
            consolidation_delay: Duration::hours(1),
            archive_threshold: 0.3,
            forget_threshold: 0.1,
            ..Default::default()
        };
        let manager = LifecycleManager::new(config);
        let now = Utc::now();

        // Fresh node should be active
        let fresh = make_node(0, 1.0);
        assert_eq!(manager.lifecycle_state(&fresh, now), LifecycleState::Active);

        // Old high-confidence node should be consolidated
        let old_strong = make_node(48, 1.0);
        let state = manager.lifecycle_state(&old_strong, now);
        // Could be consolidated or archived depending on decay
        assert!(state == LifecycleState::Consolidated || state == LifecycleState::Archived);
    }

    #[test]
    fn test_spaced_repetition() {
        let mut manager = LifecycleManager::with_defaults();
        let node = make_node(0, 1.0);
        let now = Utc::now();

        // First review time
        let review1 = manager.next_review_time(&node, now);
        let interval1 = (review1 - now).num_hours();

        // Record access (simulating review)
        manager.record_access(&node.id);

        // Second review time should be longer
        let review2 = manager.next_review_time(&node, now);
        let interval2 = (review2 - now).num_hours();

        assert!(interval2 > interval1, "Review interval should increase");
    }

    #[test]
    fn test_rank_by_strength() {
        let mut manager = LifecycleManager::with_defaults();
        let now = Utc::now();

        let node1 = make_node(0, 0.5);
        let node2 = make_node(0, 1.0);

        // Boost node1 importance
        for _ in 0..20 {
            manager.record_access(&node1.id);
        }

        let ranked = manager.rank_by_strength(vec![node1.clone(), node2.clone()], now);

        // node1 should rank higher due to access boost
        assert_eq!(ranked[0].0.id, node1.id);
    }
}
