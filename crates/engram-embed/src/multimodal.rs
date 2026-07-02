//! Multimodal embedding support for text, code, and images.
//!
//! Uses fastembed for all modalities with specialized models:
//! - Text: BGE Small EN v1.5 (384d) - general purpose
//! - Code: Jina Code v2 (768d) - programming languages
//! - Images: Nomic Embed Vision v1.5 (768d) - visual content
//!
//! Each modality has its own embedding space and index.

use std::path::Path;
use std::sync::Arc;
use parking_lot::RwLock;
use tracing::{debug, info, warn};

use engram_core::error::{EngramError, Result};
use fastembed::{EmbeddingModel, TextEmbedding, InitOptions, ImageEmbedding, ImageEmbeddingModel, ImageInitOptions};

/// Content types that engram can embed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ContentType {
    /// General text (markdown, notes, documents)
    Text,
    /// Source code (detected by extension or content)
    Code,
    /// Images (PNG, JPEG, etc.) - planned
    Image,
}

impl ContentType {
    /// Detect content type from file extension.
    pub fn from_extension(ext: &str) -> Self {
        match ext.to_lowercase().as_str() {
            // Code extensions
            "rs" | "py" | "js" | "ts" | "jsx" | "tsx" | "go" | "java" | "c" | "cpp" | "h" |
            "hpp" | "rb" | "php" | "swift" | "kt" | "scala" | "cs" | "fs" | "ex" | "exs" |
            "clj" | "cljs" | "hs" | "ml" | "mli" | "lua" | "r" | "jl" | "sh" | "bash" |
            "zsh" | "fish" | "ps1" | "vim" | "el" | "sql" | "graphql" | "proto" | "yaml" |
            "yml" | "toml" | "json" | "xml" | "html" | "css" | "scss" | "sass" | "less" |
            "dockerfile" | "makefile" | "cmake" | "gradle" | "zig" | "nim" | "v" | "d" |
            "sol" | "move" | "cairo" | "wasm" | "wat" => ContentType::Code,
            
            // Image extensions
            "png" | "jpg" | "jpeg" | "gif" | "webp" | "bmp" | "svg" | "ico" | "tiff" => ContentType::Image,
            
            // Default to text
            _ => ContentType::Text,
        }
    }
    
    /// Detect content type from content heuristics.
    pub fn from_content(content: &str) -> Self {
        // Quick heuristics for code detection
        let code_indicators = [
            "fn ", "def ", "func ", "function ", "class ", "struct ", "impl ",
            "pub fn", "async fn", "const ", "let ", "var ", "import ", "from ",
            "package ", "#include", "#define", "using namespace", "module ",
            "pub mod", "pub struct", "pub enum", "interface ", "type ",
            "export ", "require(", "console.log", "print(", "println!",
            "if __name__", "#!/", "// ", "/* ", "/// ", "#[", "@",
        ];
        
        let lines: Vec<&str> = content.lines().take(20).collect();
        let total_lines = lines.len();
        
        if total_lines == 0 {
            return ContentType::Text;
        }
        
        let code_line_count = lines.iter()
            .filter(|line| {
                code_indicators.iter().any(|ind| line.contains(ind)) ||
                line.trim().ends_with('{') ||
                line.trim().ends_with(';') ||
                line.trim().starts_with("}")
            })
            .count();
        
        if code_line_count as f64 / total_lines as f64 > 0.3 {
            ContentType::Code
        } else {
            ContentType::Text
        }
    }
    
    /// Get embedding dimensions for this content type.
    pub fn dimensions(self) -> usize {
        match self {
            ContentType::Text => 384,  // BGE Small EN
            ContentType::Code => 768,  // Jina Code v2
            ContentType::Image => 768, // Nomic Embed Vision v1.5
        }
    }
}

/// Multimodal embedder with specialized models per content type.
pub struct MultimodalEmbedder {
    text_model: Arc<RwLock<Option<TextEmbedding>>>,
    code_model: Arc<RwLock<Option<TextEmbedding>>>,
    image_model: Arc<RwLock<Option<ImageEmbedding>>>,
    cache_dir: Option<String>,
}

impl MultimodalEmbedder {
    /// Create a new multimodal embedder.
    /// Models are loaded lazily on first use.
    pub fn new(cache_dir: Option<impl AsRef<Path>>) -> Self {
        Self {
            text_model: Arc::new(RwLock::new(None)),
            code_model: Arc::new(RwLock::new(None)),
            image_model: Arc::new(RwLock::new(None)),
            cache_dir: cache_dir.map(|p| p.as_ref().to_string_lossy().to_string()),
        }
    }
    
    /// Embed text content.
    pub fn embed_text(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut guard = self.text_model.write();
        
        if guard.is_none() {
            info!("Loading text embedding model (BGE Small EN v1.5)...");
            let mut opts = InitOptions::new(EmbeddingModel::BGESmallENV15)
                .with_show_download_progress(true);
            
            if let Some(ref dir) = self.cache_dir {
                opts = opts.with_cache_dir(dir.clone().into());
            }
            
            let model = TextEmbedding::try_new(opts)
                .map_err(|e| EngramError::Embedding(format!("failed to load text model: {e}")))?;
            *guard = Some(model);
            info!("Text model loaded");
        }
        
        let model = guard.as_mut().unwrap();
        let embeddings = model.embed(texts.to_vec(), None)
            .map_err(|e| EngramError::Embedding(format!("text embedding failed: {e}")))?;
        
        debug!(count = texts.len(), "Embedded text chunks");
        Ok(embeddings)
    }
    
    /// Embed code content.
    /// Uses Jina Code v2 which understands programming languages.
    pub fn embed_code(&self, code_chunks: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut guard = self.code_model.write();
        
        if guard.is_none() {
            info!("Loading code embedding model (Jina Code v2)...");
            let mut opts = InitOptions::new(EmbeddingModel::JinaEmbeddingsV2BaseCode)
                .with_show_download_progress(true);
            
            if let Some(ref dir) = self.cache_dir {
                opts = opts.with_cache_dir(dir.clone().into());
            }
            
            let model = TextEmbedding::try_new(opts)
                .map_err(|e| EngramError::Embedding(format!("failed to load code model: {e}")))?;
            *guard = Some(model);
            info!("Code model loaded (768d, 8192 context)");
        }
        
        let model = guard.as_mut().unwrap();
        let embeddings = model.embed(code_chunks.to_vec(), None)
            .map_err(|e| EngramError::Embedding(format!("code embedding failed: {e}")))?;
        
        debug!(count = code_chunks.len(), "Embedded code chunks");
        Ok(embeddings)
    }
    
    /// Embed image files.
    /// Uses Nomic Embed Vision v1.5 (768d) for visual understanding.
    /// Input: list of file paths to images.
    pub fn embed_images(&self, image_paths: &[impl AsRef<Path>]) -> Result<Vec<Vec<f32>>> {
        let mut guard = self.image_model.write();
        
        if guard.is_none() {
            info!("Loading image embedding model (Nomic Embed Vision v1.5)...");
            let mut opts = ImageInitOptions::new(ImageEmbeddingModel::NomicEmbedVisionV15)
                .with_show_download_progress(true);
            
            if let Some(ref dir) = self.cache_dir {
                opts = opts.with_cache_dir(dir.clone().into());
            }
            
            let model = ImageEmbedding::try_new(opts)
                .map_err(|e| EngramError::Embedding(format!("failed to load image model: {e}")))?;
            *guard = Some(model);
            info!("Image model loaded (768d, Nomic Vision v1.5)");
        }
        
        let model = guard.as_mut().unwrap();
        
        // Convert paths to PathBuf for fastembed
        let paths: Vec<std::path::PathBuf> = image_paths
            .iter()
            .map(|p| p.as_ref().to_path_buf())
            .collect();
        
        let embeddings = model.embed(paths, None)
            .map_err(|e| EngramError::Embedding(format!("image embedding failed: {e}")))?;
        
        debug!(count = image_paths.len(), "Embedded images");
        Ok(embeddings)
    }
    
    /// Embed a single image file.
    pub fn embed_image(&self, image_path: impl AsRef<Path>) -> Result<Vec<f32>> {
        let embeddings = self.embed_images(&[image_path])?;
        Ok(embeddings.into_iter().next().unwrap())
    }
    
    /// Embed content with automatic type detection.
    pub fn embed_auto(&self, content: &str, hint: Option<ContentType>) -> Result<(ContentType, Vec<f32>)> {
        let content_type = hint.unwrap_or_else(|| ContentType::from_content(content));
        
        let embedding = match content_type {
            ContentType::Text => {
                let embs = self.embed_text(&[content])?;
                embs.into_iter().next().unwrap()
            }
            ContentType::Code => {
                let embs = self.embed_code(&[content])?;
                embs.into_iter().next().unwrap()
            }
            ContentType::Image => {
                warn!("Image embedding not yet implemented, falling back to text");
                let embs = self.embed_text(&[content])?;
                embs.into_iter().next().unwrap()
            }
        };
        
        Ok((content_type, embedding))
    }
    
    /// Batch embed with automatic type detection.
    pub fn embed_batch(&self, items: &[(String, Option<ContentType>)]) -> Result<Vec<(ContentType, Vec<f32>)>> {
        // Group by detected type for efficient batching
        let mut text_items: Vec<(usize, &str)> = vec![];
        let mut code_items: Vec<(usize, &str)> = vec![];
        
        for (i, (content, hint)) in items.iter().enumerate() {
            let content_type = hint.unwrap_or_else(|| ContentType::from_content(content));
            match content_type {
                ContentType::Text | ContentType::Image => text_items.push((i, content)),
                ContentType::Code => code_items.push((i, content)),
            }
        }
        
        let mut results: Vec<Option<(ContentType, Vec<f32>)>> = vec![None; items.len()];
        
        // Batch embed text
        if !text_items.is_empty() {
            let texts: Vec<&str> = text_items.iter().map(|(_, t)| *t).collect();
            let embeddings = self.embed_text(&texts)?;
            for ((idx, _), emb) in text_items.iter().zip(embeddings) {
                results[*idx] = Some((ContentType::Text, emb));
            }
        }
        
        // Batch embed code
        if !code_items.is_empty() {
            let codes: Vec<&str> = code_items.iter().map(|(_, c)| *c).collect();
            let embeddings = self.embed_code(&codes)?;
            for ((idx, _), emb) in code_items.iter().zip(embeddings) {
                results[*idx] = Some((ContentType::Code, emb));
            }
        }
        
        Ok(results.into_iter().map(|r| r.unwrap()).collect())
    }
    
    /// Get statistics about loaded models.
    pub fn stats(&self) -> MultimodalStats {
        MultimodalStats {
            text_loaded: self.text_model.read().is_some(),
            code_loaded: self.code_model.read().is_some(),
            image_loaded: self.image_model.read().is_some(),
        }
    }
}

/// Statistics about loaded multimodal models.
#[derive(Debug, Clone)]
pub struct MultimodalStats {
    pub text_loaded: bool,
    pub code_loaded: bool,
    pub image_loaded: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_content_type_detection() {
        assert_eq!(ContentType::from_extension("rs"), ContentType::Code);
        assert_eq!(ContentType::from_extension("py"), ContentType::Code);
        assert_eq!(ContentType::from_extension("md"), ContentType::Text);
        assert_eq!(ContentType::from_extension("png"), ContentType::Image);
        
        let code = "fn main() {\n    println!(\"Hello\");\n}";
        assert_eq!(ContentType::from_content(code), ContentType::Code);
        
        let text = "This is a simple note about something.";
        assert_eq!(ContentType::from_content(text), ContentType::Text);
    }
    
    #[test]
    fn test_dimensions() {
        assert_eq!(ContentType::Text.dimensions(), 384);
        assert_eq!(ContentType::Code.dimensions(), 768);
        assert_eq!(ContentType::Image.dimensions(), 768);
    }
}
