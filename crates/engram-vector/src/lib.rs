//! Vector index implementations for engram.
//!
//! By default, uses HNSW (usearch) for O(log n) approximate nearest neighbor search.
//! A flat O(n) index is available via the `flat` feature for small collections or testing.

pub mod index;

#[cfg(feature = "hnsw")]
pub mod hnsw;

pub use index::VectorIndex;

#[cfg(feature = "hnsw")]
pub use hnsw::{HnswConfig, HnswIndex};
