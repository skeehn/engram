//! Vector index implementations for engram.
//!
//! By default, uses HNSW (usearch) for O(log n) approximate nearest neighbor search.
//! A flat O(n) index is available via the `flat` feature for small collections or testing.
//! Binary quantization provides 32x compression with rescoring for high recall.

pub mod index;

#[cfg(feature = "hnsw")]
pub mod hnsw;

#[cfg(feature = "binary")]
pub mod binary;

pub use index::VectorIndex;

#[cfg(feature = "hnsw")]
pub use hnsw::{HnswConfig, HnswIndex, QuantizationType};

#[cfg(feature = "binary")]
pub use binary::{BinaryIndex, BinaryIndexConfig, BinaryIndexStats, BinaryVector};
