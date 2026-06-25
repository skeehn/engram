use chrono::{DateTime, Utc};
use engram_core::{error::Result, id::NodeId, types::Node};
use engram_store::EngramStore;

pub struct TemporalQuery<'a> {
    store: &'a EngramStore,
}

impl<'a> TemporalQuery<'a> {
    pub fn new(store: &'a EngramStore) -> Self {
        Self { store }
    }

    /// Get state of a node at a specific point in time (as-of query).
    /// Delegates to store.get_node_as_of which replays the temporal log.
    pub fn node_at(&self, id: &NodeId, as_of: DateTime<Utc>) -> Result<Option<Node>> {
        self.store.get_node_as_of(id, as_of)
    }

    /// Get all nodes that were valid at a given time.
    /// Filters by tx_time <= as_of (insertion happened before or at as_of),
    /// and valid_time is either None (still valid) or > as_of (not yet expired).
    pub fn nodes_valid_at(&self, as_of: DateTime<Utc>) -> Result<Vec<Node>> {
        let all = self.store.scan_nodes()?;
        let valid = all
            .into_iter()
            .filter(|n| {
                // The node must have been written by as_of
                n.tx_time <= as_of
                    // valid_time acts as valid_until: None means no expiry
                    && n.valid_time.map_or(true, |vt| vt > as_of)
            })
            .collect();
        Ok(valid)
    }

    /// Get nodes that changed in the given time range [from, to].
    /// Returns NodeIds whose tx_time falls within the range.
    pub fn nodes_changed_between(
        &self,
        from: DateTime<Utc>,
        to: DateTime<Utc>,
    ) -> Result<Vec<NodeId>> {
        let all = self.store.scan_nodes()?;
        let changed = all
            .into_iter()
            .filter(|n| n.tx_time >= from && n.tx_time <= to)
            .map(|n| n.id)
            .collect();
        Ok(changed)
    }

    /// Get the full history of a node as (timestamp, description) pairs.
    /// MVP: returns the current node's tx_time with description "current".
    pub fn node_history(&self, id: &NodeId) -> Result<Vec<(DateTime<Utc>, String)>> {
        match self.store.get_node(id)? {
            Some(node) => Ok(vec![(node.tx_time, "current".to_string())]),
            None => Ok(vec![]),
        }
    }

    /// Compute decayed confidence for a node.
    /// Formula: confidence = P_base + P_boost * exp(-lambda * elapsed_secs)
    /// Here we treat node.confidence as P_base and assume P_boost = 1 - P_base
    /// so the initial confidence at time zero is node.confidence + P_boost = 1.0,
    /// decaying toward node.confidence over time.
    ///
    /// Simplified: decayed = node.confidence * exp(-lambda * elapsed_secs)
    /// with a floor of 0.0.
    pub fn decayed_confidence(node: &Node, now: DateTime<Utc>, lambda: f32) -> f32 {
        let elapsed_secs = (now - node.tx_time).num_seconds().max(0) as f32;
        let p_base = node.confidence;
        let p_boost = (1.0_f32 - p_base).max(0.0);
        let decayed = p_base + p_boost * (-lambda * elapsed_secs).exp();
        decayed.clamp(0.0, 1.0)
    }

    /// Get nodes whose decayed confidence is below the given threshold.
    pub fn stale_nodes(&self, lambda: f32, threshold: f32) -> Result<Vec<Node>> {
        let now = Utc::now();
        let all = self.store.scan_nodes()?;
        let stale = all
            .into_iter()
            .filter(|n| Self::decayed_confidence(n, now, lambda) < threshold)
            .collect();
        Ok(stale)
    }
}
