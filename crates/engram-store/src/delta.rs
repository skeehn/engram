use chrono::{DateTime, Utc};
use engram_core::{
    error::{EngramError, Result},
    id::NodeId,
    types::Node,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeDelta {
    pub node_id: NodeId,
    pub timestamp: DateTime<Utc>,
    pub op: DeltaOp,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeltaOp {
    Insert(Node),
    UpdateBody(String),
    UpdateConfidence(f32),
    UpdateMetadata(serde_json::Value),
    Invalidate { valid_time: DateTime<Utc> },
    Delete,
}

impl NodeDelta {
    pub fn insert(node: Node) -> Self {
        Self {
            node_id: node.id.clone(),
            timestamp: Utc::now(),
            op: DeltaOp::Insert(node),
        }
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let json = serde_json::to_vec(self)?;
        let compressed = zstd::encode_all(json.as_slice(), 3)
            .map_err(|e| EngramError::Storage(e.to_string()))?;
        Ok(compressed)
    }

    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        let decompressed = zstd::decode_all(data)
            .map_err(|e| EngramError::Storage(e.to_string()))?;
        Ok(serde_json::from_slice(&decompressed)?)
    }
}

/// Build temporal key: big-endian timestamp_micros (8 bytes) + node_id bytes
pub fn temporal_key(ts: DateTime<Utc>, node_id: &NodeId) -> Vec<u8> {
    let micros = ts.timestamp_micros();
    let mut key = micros.to_be_bytes().to_vec();
    key.extend_from_slice(node_id.as_ref().as_bytes());
    key
}

/// Build 8-byte prefix for scanning all deltas up to a timestamp
pub fn temporal_key_prefix(ts: DateTime<Utc>) -> [u8; 8] {
    ts.timestamp_micros().to_be_bytes()
}
