//! Binary quantization for ultra-compact vector storage.
//!
//! Converts f32 vectors to binary (1-bit per dimension) for:
//! - 32x storage reduction (384 floats -> 48 bytes)
//! - Fast Hamming distance search
//! - Two-stage retrieval: binary for candidates, f32 for rescoring

use std::path::Path;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write, Seek, SeekFrom, BufWriter, BufReader};
use memmap2::{Mmap, MmapOptions};
use serde::{Serialize, Deserialize};

/// Binary vector - packed bits for ultra-compact storage.
#[derive(Clone)]
pub struct BinaryVector {
    /// Packed bits: ceil(dims/8) bytes
    pub bits: Vec<u8>,
    /// Original dimensions (needed for proper comparison)
    pub dims: usize,
}

impl BinaryVector {
    /// Quantize f32 vector to binary (1-bit per dimension).
    /// Positive values -> 1, non-positive -> 0
    pub fn from_f32(vec: &[f32]) -> Self {
        let dims = vec.len();
        let num_bytes = (dims + 7) / 8;
        let mut bits = vec![0u8; num_bytes];
        
        for (i, &val) in vec.iter().enumerate() {
            if val > 0.0 {
                bits[i / 8] |= 1 << (i % 8);
            }
        }
        
        Self { bits, dims }
    }
    
    /// Compute Hamming distance (number of differing bits).
    #[inline]
    pub fn hamming_distance(&self, other: &BinaryVector) -> u32 {
        debug_assert_eq!(self.bits.len(), other.bits.len());
        
        self.bits.iter()
            .zip(other.bits.iter())
            .map(|(a, b)| (a ^ b).count_ones())
            .sum()
    }
    
    /// Compute similarity score (1.0 = identical, 0.0 = opposite).
    #[inline]
    pub fn similarity(&self, other: &BinaryVector) -> f32 {
        let hamming = self.hamming_distance(other);
        1.0 - (hamming as f32 / self.dims as f32)
    }
    
    /// Get raw bytes for storage.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bits
    }
    
    /// Create from raw bytes.
    pub fn from_bytes(bytes: &[u8], dims: usize) -> Self {
        Self {
            bits: bytes.to_vec(),
            dims,
        }
    }
}

/// Configuration for binary index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinaryIndexConfig {
    /// Vector dimensions
    pub dims: usize,
    /// Number of bytes per binary vector (ceil(dims/8))
    pub bytes_per_vec: usize,
    /// Number of candidates to retrieve for rescoring (typically 10-50x final k)
    pub rescore_factor: usize,
}

impl BinaryIndexConfig {
    pub fn new(dims: usize) -> Self {
        Self {
            dims,
            bytes_per_vec: (dims + 7) / 8,
            rescore_factor: 20, // Retrieve 20x candidates, then rescore
        }
    }
}

/// Ultra-compact binary vector index with mmap support.
pub struct BinaryIndex {
    config: BinaryIndexConfig,
    /// Binary vectors stored contiguously
    binary_data: Vec<u8>,
    /// Number of vectors
    count: usize,
    /// Memory-mapped original vectors for rescoring (optional)
    original_mmap: Option<Mmap>,
    /// Path to original vectors file
    original_path: Option<std::path::PathBuf>,
}

impl BinaryIndex {
    /// Create a new empty binary index.
    pub fn new(config: BinaryIndexConfig) -> Self {
        Self {
            config,
            binary_data: Vec::new(),
            count: 0,
            original_mmap: None,
            original_path: None,
        }
    }
    
    /// Create index with rescoring support.
    pub fn with_rescoring(config: BinaryIndexConfig, original_path: impl AsRef<Path>) -> std::io::Result<Self> {
        let path = original_path.as_ref().to_path_buf();
        let file = File::open(&path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        
        Ok(Self {
            config,
            binary_data: Vec::new(),
            count: 0,
            original_mmap: Some(mmap),
            original_path: Some(path),
        })
    }
    
    /// Add a vector (stores both binary and optionally original).
    pub fn add(&mut self, vec: &[f32], original_writer: Option<&mut BufWriter<File>>) -> std::io::Result<usize> {
        let id = self.count;
        
        // Binary quantization
        let binary = BinaryVector::from_f32(vec);
        self.binary_data.extend_from_slice(&binary.bits);
        
        // Write original to file for rescoring
        if let Some(writer) = original_writer {
            for &val in vec {
                writer.write_all(&val.to_le_bytes())?;
            }
        }
        
        self.count += 1;
        Ok(id)
    }
    
    /// Add a batch of vectors.
    pub fn add_batch(&mut self, vecs: &[Vec<f32>], mut original_writer: Option<&mut BufWriter<File>>) -> std::io::Result<Vec<usize>> {
        let mut ids = Vec::with_capacity(vecs.len());
        for vec in vecs {
            ids.push(self.add(vec, original_writer.as_deref_mut())?);
        }
        Ok(ids)
    }
    
    /// Search using binary vectors only (fast, approximate).
    pub fn search_binary(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        let query_binary = BinaryVector::from_f32(query);
        let mut scores: Vec<(usize, f32)> = Vec::with_capacity(self.count);
        
        for i in 0..self.count {
            let offset = i * self.config.bytes_per_vec;
            let vec_binary = BinaryVector::from_bytes(
                &self.binary_data[offset..offset + self.config.bytes_per_vec],
                self.config.dims,
            );
            let sim = query_binary.similarity(&vec_binary);
            scores.push((i, sim));
        }
        
        // Sort by similarity descending
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scores.truncate(k);
        scores
    }
    
    /// Search with rescoring (accurate, uses original vectors).
    pub fn search_rescore(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        // First stage: binary search for candidates
        let candidates_k = k * self.config.rescore_factor;
        let candidates = self.search_binary(query, candidates_k.min(self.count));
        
        // Second stage: rescore with original vectors
        if let Some(ref mmap) = self.original_mmap {
            let bytes_per_vec = self.config.dims * 4; // f32 = 4 bytes
            
            let mut rescored: Vec<(usize, f32)> = candidates.iter().map(|(id, _)| {
                let offset = id * bytes_per_vec;
                let original = self.read_f32_vec(mmap, offset);
                let sim = cosine_similarity(query, &original);
                (*id, sim)
            }).collect();
            
            rescored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            rescored.truncate(k);
            rescored
        } else {
            // No rescoring data, return binary results
            candidates.into_iter().take(k).collect()
        }
    }
    
    /// Read f32 vector from mmap.
    fn read_f32_vec(&self, mmap: &Mmap, offset: usize) -> Vec<f32> {
        let mut vec = Vec::with_capacity(self.config.dims);
        for i in 0..self.config.dims {
            let byte_offset = offset + i * 4;
            let bytes: [u8; 4] = mmap[byte_offset..byte_offset + 4].try_into().unwrap();
            vec.push(f32::from_le_bytes(bytes));
        }
        vec
    }
    
    /// Get number of vectors.
    pub fn len(&self) -> usize {
        self.count
    }
    
    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
    
    /// Get storage statistics.
    pub fn stats(&self) -> BinaryIndexStats {
        let binary_bytes = self.binary_data.len();
        let original_bytes = self.original_mmap.as_ref().map(|m| m.len()).unwrap_or(0);
        
        BinaryIndexStats {
            count: self.count,
            binary_bytes,
            original_bytes,
            total_bytes: binary_bytes + original_bytes,
            bytes_per_vec_binary: self.config.bytes_per_vec,
            bytes_per_vec_original: self.config.dims * 4,
        }
    }
    
    /// Save binary index to file.
    pub fn save(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        
        // Write header
        let header = BinaryIndexHeader {
            magic: BINARY_INDEX_MAGIC,
            version: 1,
            dims: self.config.dims as u32,
            count: self.count as u64,
            bytes_per_vec: self.config.bytes_per_vec as u32,
            rescore_factor: self.config.rescore_factor as u32,
        };
        
        writer.write_all(&header.magic.to_le_bytes())?;
        writer.write_all(&header.version.to_le_bytes())?;
        writer.write_all(&header.dims.to_le_bytes())?;
        writer.write_all(&header.count.to_le_bytes())?;
        writer.write_all(&header.bytes_per_vec.to_le_bytes())?;
        writer.write_all(&header.rescore_factor.to_le_bytes())?;
        
        // Write binary data
        writer.write_all(&self.binary_data)?;
        writer.flush()?;
        
        Ok(())
    }
    
    /// Load binary index from file.
    pub fn load(path: impl AsRef<Path>, original_path: Option<impl AsRef<Path>>) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        
        // Read header
        let mut buf4 = [0u8; 4];
        let mut buf8 = [0u8; 8];
        
        reader.read_exact(&mut buf4)?;
        let magic = u32::from_le_bytes(buf4);
        if magic != BINARY_INDEX_MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid binary index magic number",
            ));
        }
        
        reader.read_exact(&mut buf4)?;
        let _version = u32::from_le_bytes(buf4);
        
        reader.read_exact(&mut buf4)?;
        let dims = u32::from_le_bytes(buf4) as usize;
        
        reader.read_exact(&mut buf8)?;
        let count = u64::from_le_bytes(buf8) as usize;
        
        reader.read_exact(&mut buf4)?;
        let bytes_per_vec = u32::from_le_bytes(buf4) as usize;
        
        reader.read_exact(&mut buf4)?;
        let rescore_factor = u32::from_le_bytes(buf4) as usize;
        
        // Read binary data
        let mut binary_data = vec![0u8; count * bytes_per_vec];
        reader.read_exact(&mut binary_data)?;
        
        let config = BinaryIndexConfig {
            dims,
            bytes_per_vec,
            rescore_factor,
        };
        
        // Load original vectors mmap if path provided
        let (original_mmap, original_path_buf) = if let Some(p) = original_path {
            let path_buf = p.as_ref().to_path_buf();
            let file = File::open(&path_buf)?;
            let mmap = unsafe { MmapOptions::new().map(&file)? };
            (Some(mmap), Some(path_buf))
        } else {
            (None, None)
        };
        
        Ok(Self {
            config,
            binary_data,
            count,
            original_mmap,
            original_path: original_path_buf,
        })
    }
}

/// Statistics about binary index.
#[derive(Debug, Clone)]
pub struct BinaryIndexStats {
    pub count: usize,
    pub binary_bytes: usize,
    pub original_bytes: usize,
    pub total_bytes: usize,
    pub bytes_per_vec_binary: usize,
    pub bytes_per_vec_original: usize,
}

impl std::fmt::Display for BinaryIndexStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, 
            "BinaryIndex: {} vectors, {:.2}MB binary, {:.2}MB original, {:.2}MB total",
            self.count,
            self.binary_bytes as f64 / 1024.0 / 1024.0,
            self.original_bytes as f64 / 1024.0 / 1024.0,
            self.total_bytes as f64 / 1024.0 / 1024.0,
        )
    }
}

const BINARY_INDEX_MAGIC: u32 = 0x454E4752; // "ENGR"

#[derive(Debug)]
struct BinaryIndexHeader {
    magic: u32,
    version: u32,
    dims: u32,
    count: u64,
    bytes_per_vec: u32,
    rescore_factor: u32,
}

/// Cosine similarity between two vectors.
#[inline]
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    
    dot / (norm_a.sqrt() * norm_b.sqrt() + 1e-10)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_binary_quantization() {
        let vec = vec![0.5, -0.3, 0.0, 0.1, -0.9, 0.8, -0.1, 0.2];
        let binary = BinaryVector::from_f32(&vec);
        
        // Expected: 0.5>0 (1), -0.3<0 (0), 0.0<=0 (0), 0.1>0 (1), ...
        // Bits: 1, 0, 0, 1, 0, 1, 0, 1 = 0b10101001 = 169
        assert_eq!(binary.bits[0], 0b10101001);
    }
    
    #[test]
    fn test_hamming_distance() {
        let a = BinaryVector::from_f32(&[1.0, 1.0, -1.0, -1.0]);
        let b = BinaryVector::from_f32(&[1.0, -1.0, 1.0, -1.0]);
        
        // a: 1100, b: 1010 -> XOR = 0110 -> 2 bits different
        assert_eq!(a.hamming_distance(&b), 2);
    }
    
    #[test]
    fn test_similarity() {
        let a = BinaryVector::from_f32(&[1.0, 1.0, 1.0, 1.0]);
        let b = BinaryVector::from_f32(&[1.0, 1.0, 1.0, 1.0]);
        
        assert!((a.similarity(&b) - 1.0).abs() < 0.001);
        
        let c = BinaryVector::from_f32(&[-1.0, -1.0, -1.0, -1.0]);
        assert!((a.similarity(&c) - 0.0).abs() < 0.001);
    }
    
    #[test]
    fn test_binary_index() {
        let config = BinaryIndexConfig::new(4);
        let mut index = BinaryIndex::new(config);
        
        let v1 = vec![1.0, 0.5, -0.3, 0.8];
        let v2 = vec![0.9, 0.4, -0.2, 0.7];
        let v3 = vec![-0.5, -0.3, 0.8, -0.2];
        
        index.add(&v1, None).unwrap();
        index.add(&v2, None).unwrap();
        index.add(&v3, None).unwrap();
        
        // Search should find v1 and v2 as most similar to query
        let query = vec![1.0, 0.5, -0.3, 0.8];
        let results = index.search_binary(&query, 3);
        
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].0, 0); // v1 should be most similar (identical)
        assert!(results[0].1 > 0.99);
    }
    
    #[test]
    fn test_storage_calculation() {
        // 384 dims -> 48 bytes binary
        let config = BinaryIndexConfig::new(384);
        assert_eq!(config.bytes_per_vec, 48);
        
        // 1M vectors @ 48 bytes = 48MB
        let expected_mb = 1_000_000.0 * 48.0 / 1024.0 / 1024.0;
        assert!((expected_mb - 45.78).abs() < 0.1);
    }
}
