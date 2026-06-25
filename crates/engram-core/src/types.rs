use crate::id::{EdgeId, NodeId};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fmt;

// ---------------------------------------------------------------------------
// Search types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SearchMode {
    Vector,
    Keyword,
    Graph,
    Temporal,
    Relational,
    Ppr,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchQuery {
    pub text: String,
    pub modes: Vec<SearchMode>,
    pub top_k: usize,
    pub min_confidence: f32,
}

impl SearchQuery {
    pub fn new(text: impl Into<String>, top_k: usize) -> Self {
        Self {
            text: text.into(),
            modes: vec![SearchMode::Vector, SearchMode::Keyword, SearchMode::Graph],
            top_k,
            min_confidence: 0.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub node: Node,
    pub score: f32,
    pub mode: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeType {
    Fact,
    Concept,
    Entity,
    Event,
    Document,
    Chunk,
    Note,
    Custom(String),
}

impl fmt::Display for NodeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NodeType::Fact => write!(f, "fact"),
            NodeType::Concept => write!(f, "concept"),
            NodeType::Entity => write!(f, "entity"),
            NodeType::Event => write!(f, "event"),
            NodeType::Document => write!(f, "document"),
            NodeType::Chunk => write!(f, "chunk"),
            NodeType::Note => write!(f, "note"),
            NodeType::Custom(s) => write!(f, "{}", s),
        }
    }
}

impl Default for NodeType {
    fn default() -> Self {
        NodeType::Note
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub id: NodeId,
    pub node_type: NodeType,
    pub body: String,
    pub tags: Vec<String>,
    pub confidence: f32,
    pub embedding: Option<Vec<f32>>,
    pub tx_time: DateTime<Utc>,
    pub valid_time: Option<DateTime<Utc>>,
    pub metadata: serde_json::Value,
}

impl Node {
    pub fn new(body: impl Into<String>, node_type: NodeType) -> Self {
        Self {
            id: NodeId::new(),
            node_type,
            body: body.into(),
            tags: Vec::new(),
            confidence: 1.0,
            embedding: None,
            tx_time: Utc::now(),
            valid_time: None,
            metadata: serde_json::Value::Null,
        }
    }

    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence;
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeType {
    RelatedTo,
    IsA,
    HasPart,
    Causes,
    Contradicts,
    Supports,
    References,
    DerivedFrom,
    Custom(String),
}

impl fmt::Display for EdgeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EdgeType::RelatedTo => write!(f, "related_to"),
            EdgeType::IsA => write!(f, "is_a"),
            EdgeType::HasPart => write!(f, "has_part"),
            EdgeType::Causes => write!(f, "causes"),
            EdgeType::Contradicts => write!(f, "contradicts"),
            EdgeType::Supports => write!(f, "supports"),
            EdgeType::References => write!(f, "references"),
            EdgeType::DerivedFrom => write!(f, "derived_from"),
            EdgeType::Custom(s) => write!(f, "{}", s),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub id: EdgeId,
    pub edge_type: EdgeType,
    pub source: NodeId,
    pub target: NodeId,
    pub weight: f32,
    pub tx_time: DateTime<Utc>,
    pub metadata: serde_json::Value,
}

impl Edge {
    pub fn new(source: NodeId, target: NodeId, edge_type: EdgeType) -> Self {
        Self {
            id: EdgeId::new(),
            edge_type,
            source,
            target,
            weight: 1.0,
            tx_time: Utc::now(),
            metadata: serde_json::Value::Null,
        }
    }
}
