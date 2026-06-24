#[derive(thiserror::Error, Debug)]
pub enum EngramError {
    #[error("node not found: {0}")]
    NodeNotFound(String),
    #[error("edge not found: {0}")]
    EdgeNotFound(String),
    #[error("storage error: {0}")]
    Storage(String),
    #[error("embedding error: {0}")]
    Embedding(String),
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid id: {0}")]
    InvalidId(String),
    #[error("query error: {0}")]
    Query(String),
    #[error("http error: {0}")]
    Http(String),
    #[error("index error: {0}")]
    Index(String),
}

pub type Result<T> = std::result::Result<T, EngramError>;
