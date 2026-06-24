use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;

macro_rules! typed_id {
    ($name:ident) => {
        #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
        pub struct $name(String);

        impl $name {
            pub fn new() -> Self {
                Self(ulid::Ulid::new().to_string())
            }

            pub fn from_string(s: impl Into<String>) -> Self {
                Self(s.into())
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }

        impl AsRef<str> for $name {
            fn as_ref(&self) -> &str {
                &self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}", self.0)
            }
        }

        impl From<String> for $name {
            fn from(s: String) -> Self {
                Self(s)
            }
        }

        impl From<&str> for $name {
            fn from(s: &str) -> Self {
                Self(s.to_string())
            }
        }
    };
}

typed_id!(NodeId);
typed_id!(EdgeId);
typed_id!(ClusterId);

/// Content-addressed object id (SHA-256 hex)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ObjectId(String);

impl ObjectId {
    pub fn from_bytes(data: &[u8]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(data);
        let hash = hasher.finalize();
        Self(hex::encode(hash))
    }

    pub fn from_string(s: impl Into<String>) -> Self {
        Self(s.into())
    }
}

impl AsRef<str> for ObjectId {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ObjectId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}
