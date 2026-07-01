//! Cross-platform file watcher daemon using notify-rs.
//!
//! Features:
//! - Recursive directory watching
//! - Event debouncing (no duplicate/rapid fire events)
//! - Content-hash based change detection
//! - Cross-platform: macOS (FSEvents), Linux (inotify), Windows (ReadDirectoryChanges)

use notify::{
    Config, Event, EventKind, RecommendedWatcher, RecursiveMode, Result as NotifyResult, Watcher,
};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use thiserror::Error;
use tracing::{debug, error, info, warn};
use sha2::{Sha256, Digest};
use std::fs;

#[derive(Error, Debug)]
pub enum WatcherError {
    #[error("Notify error: {0}")]
    Notify(#[from] notify::Error),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Path not found: {0}")]
    PathNotFound(PathBuf),
    #[error("Watcher not running")]
    NotRunning,
}

/// Event types emitted by the file watcher.
#[derive(Debug, Clone)]
pub enum FileEvent {
    /// File was created
    Created(PathBuf),
    /// File was modified (content changed)
    Modified(PathBuf),
    /// File was deleted
    Deleted(PathBuf),
    /// File was renamed (from, to)
    Renamed(PathBuf, PathBuf),
}

/// Configuration for the file watcher.
#[derive(Debug, Clone)]
pub struct WatcherConfig {
    /// Debounce duration (default: 300ms)
    pub debounce_ms: u64,
    /// File extensions to watch (empty = all files)
    pub extensions: Vec<String>,
    /// Directories to ignore
    pub ignore_dirs: Vec<String>,
    /// Use content hash for change detection
    pub hash_check: bool,
    /// Maximum file size to hash (default: 10MB)
    pub max_hash_size: u64,
}

impl Default for WatcherConfig {
    fn default() -> Self {
        Self {
            debounce_ms: 300,
            extensions: vec![
                "md".into(), "txt".into(), "rs".into(), "py".into(),
                "js".into(), "ts".into(), "json".into(), "yaml".into(),
                "toml".into(), "html".into(), "css".into(),
            ],
            ignore_dirs: vec![
                ".git".into(), "node_modules".into(), "target".into(),
                "__pycache__".into(), ".venv".into(), "venv".into(),
            ],
            hash_check: true,
            max_hash_size: 10 * 1024 * 1024, // 10MB
        }
    }
}

/// File metadata cache for change detection.
#[derive(Debug, Clone)]
struct FileMetadata {
    hash: Option<[u8; 32]>,
    mtime: std::time::SystemTime,
    size: u64,
}

/// Cross-platform file watcher with debouncing and content-hash change detection.
pub struct FileWatcher {
    config: WatcherConfig,
    watcher: Option<RecommendedWatcher>,
    event_tx: Sender<FileEvent>,
    event_rx: Arc<RwLock<Option<Receiver<FileEvent>>>>,
    watched_paths: Arc<RwLock<Vec<PathBuf>>>,
    file_cache: Arc<RwLock<HashMap<PathBuf, FileMetadata>>>,
    pending_events: Arc<RwLock<HashMap<PathBuf, (EventKind, Instant)>>>,
}

impl FileWatcher {
    /// Create a new file watcher with the given configuration.
    pub fn new(config: WatcherConfig) -> Result<Self, WatcherError> {
        let (event_tx, event_rx) = channel();
        
        Ok(Self {
            config,
            watcher: None,
            event_tx,
            event_rx: Arc::new(RwLock::new(Some(event_rx))),
            watched_paths: Arc::new(RwLock::new(Vec::new())),
            file_cache: Arc::new(RwLock::new(HashMap::new())),
            pending_events: Arc::new(RwLock::new(HashMap::new())),
        })
    }
    
    /// Create with default configuration.
    pub fn with_defaults() -> Result<Self, WatcherError> {
        Self::new(WatcherConfig::default())
    }
    
    /// Start watching the filesystem.
    pub fn start(&mut self) -> Result<(), WatcherError> {
        let config = self.config.clone();
        let event_tx = self.event_tx.clone();
        let file_cache = Arc::clone(&self.file_cache);
        let pending_events = Arc::clone(&self.pending_events);
        
        let debounce_duration = Duration::from_millis(config.debounce_ms);
        
        // Create the internal notify watcher
        let watcher_event_tx = event_tx.clone();
        let watcher_config = config.clone();
        let watcher_file_cache = Arc::clone(&file_cache);
        let watcher_pending = Arc::clone(&pending_events);
        
        let mut watcher = RecommendedWatcher::new(
            move |res: NotifyResult<Event>| {
                match res {
                    Ok(event) => {
                        Self::handle_raw_event(
                            event,
                            &watcher_config,
                            &watcher_event_tx,
                            &watcher_file_cache,
                            &watcher_pending,
                            debounce_duration,
                        );
                    }
                    Err(e) => {
                        error!("Watch error: {:?}", e);
                    }
                }
            },
            Config::default(),
        )?;
        
        // Re-add all watched paths
        let paths = self.watched_paths.read().unwrap();
        for path in paths.iter() {
            watcher.watch(path, RecursiveMode::Recursive)?;
            info!(?path, "Watching directory");
        }
        
        self.watcher = Some(watcher);
        info!("File watcher started");
        Ok(())
    }
    
    /// Add a directory to watch.
    pub fn watch(&mut self, path: impl AsRef<Path>) -> Result<(), WatcherError> {
        let path = path.as_ref().canonicalize()?;
        
        if !path.exists() {
            return Err(WatcherError::PathNotFound(path));
        }
        
        // Add to watched paths
        {
            let mut watched = self.watched_paths.write().unwrap();
            if !watched.contains(&path) {
                watched.push(path.clone());
            }
        }
        
        // If watcher is running, add the watch
        if let Some(ref mut watcher) = self.watcher {
            watcher.watch(&path, RecursiveMode::Recursive)?;
            info!(?path, "Added watch");
            
            // Initial scan
            self.scan_directory(&path)?;
        }
        
        Ok(())
    }
    
    /// Remove a directory from watching.
    pub fn unwatch(&mut self, path: impl AsRef<Path>) -> Result<(), WatcherError> {
        let path = path.as_ref().canonicalize()?;
        
        // Remove from watched paths
        {
            let mut watched = self.watched_paths.write().unwrap();
            watched.retain(|p| p != &path);
        }
        
        // If watcher is running, remove the watch
        if let Some(ref mut watcher) = self.watcher {
            watcher.unwatch(&path)?;
            info!(?path, "Removed watch");
        }
        
        // Clean file cache
        {
            let mut cache = self.file_cache.write().unwrap();
            cache.retain(|p, _| !p.starts_with(&path));
        }
        
        Ok(())
    }
    
    /// Stop the watcher.
    pub fn stop(&mut self) {
        self.watcher = None;
        info!("File watcher stopped");
    }
    
    /// Take the event receiver (can only be called once).
    pub fn take_receiver(&self) -> Option<Receiver<FileEvent>> {
        self.event_rx.write().unwrap().take()
    }
    
    /// Scan a directory and populate the file cache.
    fn scan_directory(&self, path: &Path) -> Result<(), WatcherError> {
        let walker = walkdir::WalkDir::new(path)
            .follow_links(false)
            .into_iter()
            .filter_entry(|e| !self.should_ignore(e.path()));
        
        let mut cache = self.file_cache.write().unwrap();
        
        for entry in walker.filter_map(|e| e.ok()) {
            if entry.file_type().is_file() {
                let path = entry.path().to_path_buf();
                
                if !self.should_watch(&path) {
                    continue;
                }
                
                if let Ok(metadata) = self.compute_metadata(&path) {
                    debug!(?path, "Cached file metadata");
                    cache.insert(path, metadata);
                }
            }
        }
        
        info!(files = cache.len(), "Directory scan complete");
        Ok(())
    }
    
    /// Check if a path should be watched based on extension.
    fn should_watch(&self, path: &Path) -> bool {
        if self.config.extensions.is_empty() {
            return true;
        }
        
        path.extension()
            .and_then(|e| e.to_str())
            .map(|ext| self.config.extensions.iter().any(|e| e == ext))
            .unwrap_or(false)
    }
    
    /// Check if a path should be ignored.
    fn should_ignore(&self, path: &Path) -> bool {
        path.components()
            .any(|c| {
                c.as_os_str()
                    .to_str()
                    .map(|s| self.config.ignore_dirs.contains(&s.to_string()))
                    .unwrap_or(false)
            })
    }
    
    /// Compute file metadata including optional content hash.
    fn compute_metadata(&self, path: &Path) -> Result<FileMetadata, std::io::Error> {
        let meta = fs::metadata(path)?;
        let mtime = meta.modified()?;
        let size = meta.len();
        
        let hash = if self.config.hash_check && size <= self.config.max_hash_size {
            let content = fs::read(path)?;
            let mut hasher = Sha256::new();
            hasher.update(&content);
            Some(hasher.finalize().into())
        } else {
            None
        };
        
        Ok(FileMetadata { hash, mtime, size })
    }
    
    /// Handle a raw event from notify with debouncing.
    fn handle_raw_event(
        event: Event,
        config: &WatcherConfig,
        event_tx: &Sender<FileEvent>,
        file_cache: &Arc<RwLock<HashMap<PathBuf, FileMetadata>>>,
        pending_events: &Arc<RwLock<HashMap<PathBuf, (EventKind, Instant)>>>,
        debounce_duration: Duration,
    ) {
        let now = Instant::now();
        
        for path in event.paths {
            // Check extensions
            if !config.extensions.is_empty() {
                let ext_match = path.extension()
                    .and_then(|e| e.to_str())
                    .map(|ext| config.extensions.iter().any(|e| e == ext))
                    .unwrap_or(false);
                
                if !ext_match {
                    continue;
                }
            }
            
            // Check ignore patterns
            let should_ignore = path.components().any(|c| {
                c.as_os_str()
                    .to_str()
                    .map(|s| config.ignore_dirs.contains(&s.to_string()))
                    .unwrap_or(false)
            });
            
            if should_ignore {
                continue;
            }
            
            // Debounce check
            {
                let mut pending = pending_events.write().unwrap();
                if let Some((_, last_time)) = pending.get(&path) {
                    if now.duration_since(*last_time) < debounce_duration {
                        // Update timestamp but don't emit yet
                        pending.insert(path.clone(), (event.kind, now));
                        continue;
                    }
                }
                pending.insert(path.clone(), (event.kind, now));
            }
            
            // Process the event
            let file_event = match event.kind {
                EventKind::Create(_) => {
                    // Update cache
                    if path.exists() {
                        if let Ok(meta) = Self::compute_metadata_static(config, &path) {
                            let mut cache = file_cache.write().unwrap();
                            cache.insert(path.clone(), meta);
                        }
                    }
                    Some(FileEvent::Created(path))
                }
                EventKind::Modify(_) => {
                    // Check if content actually changed
                    if config.hash_check && path.exists() {
                        if let Ok(new_meta) = Self::compute_metadata_static(config, &path) {
                            let mut cache = file_cache.write().unwrap();
                            if let Some(old_meta) = cache.get(&path) {
                                if old_meta.hash == new_meta.hash {
                                    // Content unchanged (e.g., just mtime update)
                                    debug!(?path, "File touched but content unchanged");
                                    continue;
                                }
                            }
                            cache.insert(path.clone(), new_meta);
                        }
                    }
                    Some(FileEvent::Modified(path))
                }
                EventKind::Remove(_) => {
                    // Remove from cache
                    {
                        let mut cache = file_cache.write().unwrap();
                        cache.remove(&path);
                    }
                    Some(FileEvent::Deleted(path))
                }
                EventKind::Access(_) => None, // Ignore access events
                EventKind::Other => None,
                EventKind::Any => None,
            };
            
            if let Some(fe) = file_event {
                debug!(?fe, "Emitting file event");
                if event_tx.send(fe).is_err() {
                    warn!("Event receiver disconnected");
                }
            }
        }
    }
    
    /// Static version of compute_metadata for use in callback.
    fn compute_metadata_static(config: &WatcherConfig, path: &Path) -> Result<FileMetadata, std::io::Error> {
        let meta = fs::metadata(path)?;
        let mtime = meta.modified()?;
        let size = meta.len();
        
        let hash = if config.hash_check && size <= config.max_hash_size {
            let content = fs::read(path)?;
            let mut hasher = Sha256::new();
            hasher.update(&content);
            Some(hasher.finalize().into())
        } else {
            None
        };
        
        Ok(FileMetadata { hash, mtime, size })
    }
}

/// Statistics about the file watcher.
#[derive(Debug, Clone)]
pub struct WatcherStats {
    pub watched_dirs: usize,
    pub cached_files: usize,
}

impl FileWatcher {
    /// Get watcher statistics.
    pub fn stats(&self) -> WatcherStats {
        WatcherStats {
            watched_dirs: self.watched_paths.read().unwrap().len(),
            cached_files: self.file_cache.read().unwrap().len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use std::thread;
    use std::time::Duration;
    
    #[test]
    fn test_watcher_config_default() {
        let config = WatcherConfig::default();
        assert_eq!(config.debounce_ms, 300);
        assert!(config.extensions.contains(&"md".to_string()));
        assert!(config.ignore_dirs.contains(&".git".to_string()));
    }
    
    #[test]
    fn test_watcher_creation() {
        let watcher = FileWatcher::with_defaults();
        assert!(watcher.is_ok());
    }
    
    #[test]
    fn test_should_watch_extension() {
        let watcher = FileWatcher::with_defaults().unwrap();
        
        assert!(watcher.should_watch(Path::new("test.md")));
        assert!(watcher.should_watch(Path::new("test.rs")));
        assert!(!watcher.should_watch(Path::new("test.exe")));
        assert!(!watcher.should_watch(Path::new("test.dll")));
    }
    
    #[test]
    fn test_should_ignore() {
        let watcher = FileWatcher::with_defaults().unwrap();
        
        assert!(watcher.should_ignore(Path::new("/project/.git/config")));
        assert!(watcher.should_ignore(Path::new("/project/node_modules/pkg/index.js")));
        assert!(!watcher.should_ignore(Path::new("/project/src/main.rs")));
    }
    
    #[test]
    fn test_file_events() {
        let dir = tempdir().unwrap();
        let test_file = dir.path().join("test.md");
        
        let mut config = WatcherConfig::default();
        config.debounce_ms = 50; // Faster for tests
        
        let mut watcher = FileWatcher::new(config).unwrap();
        watcher.watch(dir.path()).unwrap();
        watcher.start().unwrap();
        
        let rx = watcher.take_receiver().unwrap();
        
        // Create a file
        fs::write(&test_file, "hello").unwrap();
        thread::sleep(Duration::from_millis(100));
        
        // Should receive create event
        let event = rx.recv_timeout(Duration::from_secs(1));
        assert!(event.is_ok());
        match event.unwrap() {
            FileEvent::Created(p) => assert_eq!(p, test_file),
            _ => panic!("Expected Created event"),
        }
    }
}
