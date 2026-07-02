//! Workspace isolation for multi-agent memory systems.
//!
//! Each workspace provides isolated memory space for an agent session
//! while allowing controlled access to a shared knowledge base.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use chrono::{DateTime, Duration, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};
use uuid::Uuid;

use engram_core::id::NodeId;

/// Unique identifier for a workspace.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WorkspaceId(String);

impl WorkspaceId {
    /// Create a new random workspace ID.
    pub fn new() -> Self {
        Self(Uuid::new_v4().to_string())
    }

    /// Create from a string.
    pub fn from_string(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    /// Get the inner string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for WorkspaceId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for WorkspaceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Configuration for workspace behavior.
#[derive(Debug, Clone)]
pub struct WorkspaceConfig {
    /// Maximum age before a workspace is considered stale
    pub max_idle_duration: Duration,
    /// Whether to allow reading from shared layer
    pub allow_shared_reads: bool,
    /// Whether to allow promoting local nodes to shared
    pub allow_promotion: bool,
    /// Maximum nodes per workspace (0 = unlimited)
    pub max_nodes: usize,
}

impl Default for WorkspaceConfig {
    fn default() -> Self {
        Self {
            max_idle_duration: Duration::hours(24),
            allow_shared_reads: true,
            allow_promotion: true,
            max_nodes: 10000,
        }
    }
}

/// Access level for nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AccessLevel {
    /// Only visible to the owning workspace
    Private,
    /// Visible to all workspaces (shared layer)
    Shared,
    /// Visible to specific workspaces
    Group(u32),
}

/// Metadata for a node's workspace association.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeWorkspaceInfo {
    /// Primary workspace that owns this node
    pub owner: WorkspaceId,
    /// Access level
    pub access: AccessLevel,
    /// When this node was added to the workspace
    pub added_at: DateTime<Utc>,
    /// Optional group ID for group-level sharing
    pub group_id: Option<u32>,
}

/// A workspace instance providing isolated memory.
#[derive(Debug)]
pub struct Workspace {
    /// Unique identifier
    pub id: WorkspaceId,
    /// Human-readable name (optional)
    pub name: Option<String>,
    /// Configuration
    pub config: WorkspaceConfig,
    /// Node IDs owned by this workspace
    owned_nodes: HashSet<NodeId>,
    /// Node workspace metadata
    node_info: HashMap<NodeId, NodeWorkspaceInfo>,
    /// When workspace was created
    pub created_at: DateTime<Utc>,
    /// When workspace was last accessed
    pub last_accessed: DateTime<Utc>,
    /// Parent workspace (for hierarchical isolation)
    pub parent: Option<WorkspaceId>,
    /// Tags for categorization
    pub tags: Vec<String>,
}

impl Workspace {
    /// Create a new workspace.
    pub fn new(config: WorkspaceConfig) -> Self {
        let now = Utc::now();
        Self {
            id: WorkspaceId::new(),
            name: None,
            config,
            owned_nodes: HashSet::new(),
            node_info: HashMap::new(),
            created_at: now,
            last_accessed: now,
            parent: None,
            tags: Vec::new(),
        }
    }

    /// Create with a specific ID.
    pub fn with_id(id: WorkspaceId, config: WorkspaceConfig) -> Self {
        let now = Utc::now();
        Self {
            id,
            name: None,
            config,
            owned_nodes: HashSet::new(),
            node_info: HashMap::new(),
            created_at: now,
            last_accessed: now,
            parent: None,
            tags: Vec::new(),
        }
    }

    /// Set workspace name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set parent workspace.
    pub fn with_parent(mut self, parent: WorkspaceId) -> Self {
        self.parent = Some(parent);
        self
    }

    /// Add a node to this workspace.
    pub fn add_node(&mut self, node_id: NodeId, access: AccessLevel) -> bool {
        // Check capacity
        if self.config.max_nodes > 0 && self.owned_nodes.len() >= self.config.max_nodes {
            warn!(
                workspace = %self.id,
                max_nodes = self.config.max_nodes,
                "Workspace at capacity"
            );
            return false;
        }

        self.owned_nodes.insert(node_id.clone());
        self.node_info.insert(
            node_id,
            NodeWorkspaceInfo {
                owner: self.id.clone(),
                access,
                added_at: Utc::now(),
                group_id: None,
            },
        );
        self.touch();
        true
    }

    /// Remove a node from this workspace.
    pub fn remove_node(&mut self, node_id: &NodeId) -> bool {
        self.touch();
        self.node_info.remove(node_id);
        self.owned_nodes.remove(node_id)
    }

    /// Check if this workspace owns a node.
    pub fn owns(&self, node_id: &NodeId) -> bool {
        self.owned_nodes.contains(node_id)
    }

    /// Check if this workspace can see a node (owns or has access).
    pub fn can_see(&self, node_id: &NodeId, info: Option<&NodeWorkspaceInfo>) -> bool {
        // Always see owned nodes
        if self.owns(node_id) {
            return true;
        }

        // Check access level
        if let Some(info) = info {
            match info.access {
                AccessLevel::Shared => self.config.allow_shared_reads,
                AccessLevel::Private => false,
                AccessLevel::Group(gid) => {
                    // Check if we're in the same group (simplified)
                    self.tags.iter().any(|t| t == &format!("group:{}", gid))
                }
            }
        } else {
            false
        }
    }

    /// Promote a node to shared access.
    pub fn promote_to_shared(&mut self, node_id: &NodeId) -> bool {
        if !self.config.allow_promotion {
            return false;
        }

        if let Some(info) = self.node_info.get_mut(node_id) {
            info.access = AccessLevel::Shared;
            self.touch();
            true
        } else {
            false
        }
    }

    /// Get node count.
    pub fn node_count(&self) -> usize {
        self.owned_nodes.len()
    }

    /// Get all owned node IDs.
    pub fn node_ids(&self) -> impl Iterator<Item = &NodeId> {
        self.owned_nodes.iter()
    }

    /// Update last accessed time.
    fn touch(&mut self) {
        self.last_accessed = Utc::now();
    }

    /// Check if workspace is stale.
    pub fn is_stale(&self, now: DateTime<Utc>) -> bool {
        now - self.last_accessed > self.config.max_idle_duration
    }

    /// Get workspace statistics.
    pub fn stats(&self) -> WorkspaceStats {
        let mut private_count = 0;
        let mut shared_count = 0;
        let mut group_count = 0;

        for info in self.node_info.values() {
            match info.access {
                AccessLevel::Private => private_count += 1,
                AccessLevel::Shared => shared_count += 1,
                AccessLevel::Group(_) => group_count += 1,
            }
        }

        WorkspaceStats {
            id: self.id.clone(),
            name: self.name.clone(),
            total_nodes: self.owned_nodes.len(),
            private_nodes: private_count,
            shared_nodes: shared_count,
            group_nodes: group_count,
            created_at: self.created_at,
            last_accessed: self.last_accessed,
            is_stale: self.is_stale(Utc::now()),
        }
    }
}

/// Statistics about a workspace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceStats {
    pub id: WorkspaceId,
    pub name: Option<String>,
    pub total_nodes: usize,
    pub private_nodes: usize,
    pub shared_nodes: usize,
    pub group_nodes: usize,
    pub created_at: DateTime<Utc>,
    pub last_accessed: DateTime<Utc>,
    pub is_stale: bool,
}

/// Manager for multiple workspaces.
pub struct WorkspaceManager {
    workspaces: RwLock<HashMap<WorkspaceId, Workspace>>,
    /// Global shared layer (nodes visible to all)
    shared_nodes: RwLock<HashSet<NodeId>>,
    /// Default config for new workspaces
    default_config: WorkspaceConfig,
}

impl WorkspaceManager {
    /// Create a new workspace manager.
    pub fn new(default_config: WorkspaceConfig) -> Self {
        Self {
            workspaces: RwLock::new(HashMap::new()),
            shared_nodes: RwLock::new(HashSet::new()),
            default_config,
        }
    }

    /// Create with default config.
    pub fn with_defaults() -> Self {
        Self::new(WorkspaceConfig::default())
    }

    /// Create a new workspace.
    pub fn create_workspace(&self, name: Option<String>) -> WorkspaceId {
        let mut ws = Workspace::new(self.default_config.clone());
        if let Some(n) = name {
            ws = ws.with_name(n);
        }
        let id = ws.id.clone();

        let mut workspaces = self.workspaces.write();
        workspaces.insert(id.clone(), ws);

        info!(workspace = %id, "Created workspace");
        id
    }

    /// Create a workspace with specific config.
    pub fn create_workspace_with_config(&self, name: Option<String>, config: WorkspaceConfig) -> WorkspaceId {
        let mut ws = Workspace::new(config);
        if let Some(n) = name {
            ws = ws.with_name(n);
        }
        let id = ws.id.clone();

        let mut workspaces = self.workspaces.write();
        workspaces.insert(id.clone(), ws);

        info!(workspace = %id, "Created workspace with custom config");
        id
    }

    /// Get a workspace by ID.
    pub fn get_workspace(&self, id: &WorkspaceId) -> Option<WorkspaceStats> {
        let workspaces = self.workspaces.read();
        workspaces.get(id).map(|ws| ws.stats())
    }

    /// Add a node to a workspace.
    pub fn add_node_to_workspace(
        &self,
        workspace_id: &WorkspaceId,
        node_id: NodeId,
        access: AccessLevel,
    ) -> bool {
        let mut workspaces = self.workspaces.write();
        if let Some(ws) = workspaces.get_mut(workspace_id) {
            let added = ws.add_node(node_id.clone(), access);

            // If shared, also add to global shared set
            if added && access == AccessLevel::Shared {
                let mut shared = self.shared_nodes.write();
                shared.insert(node_id);
            }

            added
        } else {
            false
        }
    }

    /// Remove a node from a workspace.
    pub fn remove_node_from_workspace(&self, workspace_id: &WorkspaceId, node_id: &NodeId) -> bool {
        let mut workspaces = self.workspaces.write();
        if let Some(ws) = workspaces.get_mut(workspace_id) {
            ws.remove_node(node_id)
        } else {
            false
        }
    }

    /// Check if a workspace can see a node.
    pub fn workspace_can_see(&self, workspace_id: &WorkspaceId, node_id: &NodeId) -> bool {
        let workspaces = self.workspaces.read();

        // Check if it's in shared layer
        {
            let shared = self.shared_nodes.read();
            if shared.contains(node_id) {
                if let Some(ws) = workspaces.get(workspace_id) {
                    if ws.config.allow_shared_reads {
                        return true;
                    }
                }
            }
        }

        // Check workspace ownership
        if let Some(ws) = workspaces.get(workspace_id) {
            if ws.owns(node_id) {
                return true;
            }

            // Check parent workspace if hierarchical
            if let Some(parent_id) = &ws.parent {
                if let Some(parent) = workspaces.get(parent_id) {
                    if parent.owns(node_id) {
                        return true;
                    }
                }
            }
        }

        false
    }

    /// Filter node IDs by workspace visibility.
    pub fn filter_visible(&self, workspace_id: &WorkspaceId, node_ids: Vec<NodeId>) -> Vec<NodeId> {
        node_ids
            .into_iter()
            .filter(|id| self.workspace_can_see(workspace_id, id))
            .collect()
    }

    /// Promote a node to shared access.
    pub fn promote_to_shared(&self, workspace_id: &WorkspaceId, node_id: &NodeId) -> bool {
        let mut workspaces = self.workspaces.write();
        if let Some(ws) = workspaces.get_mut(workspace_id) {
            if ws.promote_to_shared(node_id) {
                let mut shared = self.shared_nodes.write();
                shared.insert(node_id.clone());
                return true;
            }
        }
        false
    }

    /// Delete a workspace.
    pub fn delete_workspace(&self, workspace_id: &WorkspaceId) -> bool {
        let mut workspaces = self.workspaces.write();
        if workspaces.remove(workspace_id).is_some() {
            info!(workspace = %workspace_id, "Deleted workspace");
            true
        } else {
            false
        }
    }

    /// Garbage collect stale workspaces.
    pub fn gc_stale_workspaces(&self) -> Vec<WorkspaceId> {
        let now = Utc::now();
        let mut workspaces = self.workspaces.write();

        let stale_ids: Vec<_> = workspaces
            .iter()
            .filter(|(_, ws)| ws.is_stale(now))
            .map(|(id, _)| id.clone())
            .collect();

        for id in &stale_ids {
            workspaces.remove(id);
            debug!(workspace = %id, "Garbage collected stale workspace");
        }

        if !stale_ids.is_empty() {
            info!(count = stale_ids.len(), "Garbage collected stale workspaces");
        }

        stale_ids
    }

    /// List all workspace stats.
    pub fn list_workspaces(&self) -> Vec<WorkspaceStats> {
        let workspaces = self.workspaces.read();
        workspaces.values().map(|ws| ws.stats()).collect()
    }

    /// Get total node count across all workspaces.
    pub fn total_node_count(&self) -> usize {
        let workspaces = self.workspaces.read();
        workspaces.values().map(|ws| ws.node_count()).sum()
    }

    /// Get count of shared nodes.
    pub fn shared_node_count(&self) -> usize {
        let shared = self.shared_nodes.read();
        shared.len()
    }

    /// Merge two workspaces (source into target).
    pub fn merge_workspaces(&self, source_id: &WorkspaceId, target_id: &WorkspaceId) -> bool {
        let mut workspaces = self.workspaces.write();

        // Get source nodes
        let source_nodes: Vec<(NodeId, NodeWorkspaceInfo)> = {
            if let Some(source) = workspaces.get(source_id) {
                source
                    .node_info
                    .iter()
                    .map(|(id, info)| (id.clone(), info.clone()))
                    .collect()
            } else {
                return false;
            }
        };

        // Add to target
        if let Some(target) = workspaces.get_mut(target_id) {
            for (node_id, info) in source_nodes {
                target.owned_nodes.insert(node_id.clone());
                target.node_info.insert(node_id, NodeWorkspaceInfo {
                    owner: target.id.clone(),
                    access: info.access,
                    added_at: Utc::now(),
                    group_id: info.group_id,
                });
            }
            target.last_accessed = Utc::now();
        } else {
            return false;
        }

        // Remove source
        workspaces.remove(source_id);

        info!(
            source = %source_id,
            target = %target_id,
            "Merged workspaces"
        );
        true
    }
}

impl Default for WorkspaceManager {
    fn default() -> Self {
        Self::with_defaults()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workspace_creation() {
        let manager = WorkspaceManager::with_defaults();
        let ws_id = manager.create_workspace(Some("test".to_string()));

        let stats = manager.get_workspace(&ws_id).unwrap();
        assert_eq!(stats.name, Some("test".to_string()));
        assert_eq!(stats.total_nodes, 0);
    }

    #[test]
    fn test_node_isolation() {
        let manager = WorkspaceManager::with_defaults();

        let ws1 = manager.create_workspace(Some("agent1".to_string()));
        let ws2 = manager.create_workspace(Some("agent2".to_string()));

        let node1 = NodeId::new();
        let node2 = NodeId::new();

        // Add private node to ws1
        manager.add_node_to_workspace(&ws1, node1.clone(), AccessLevel::Private);

        // Add private node to ws2
        manager.add_node_to_workspace(&ws2, node2.clone(), AccessLevel::Private);

        // ws1 can see node1 but not node2
        assert!(manager.workspace_can_see(&ws1, &node1));
        assert!(!manager.workspace_can_see(&ws1, &node2));

        // ws2 can see node2 but not node1
        assert!(manager.workspace_can_see(&ws2, &node2));
        assert!(!manager.workspace_can_see(&ws2, &node1));
    }

    #[test]
    fn test_shared_access() {
        let manager = WorkspaceManager::with_defaults();

        let ws1 = manager.create_workspace(Some("agent1".to_string()));
        let ws2 = manager.create_workspace(Some("agent2".to_string()));

        let shared_node = NodeId::new();

        // Add shared node to ws1
        manager.add_node_to_workspace(&ws1, shared_node.clone(), AccessLevel::Shared);

        // Both can see it
        assert!(manager.workspace_can_see(&ws1, &shared_node));
        assert!(manager.workspace_can_see(&ws2, &shared_node));
    }

    #[test]
    fn test_promotion() {
        let manager = WorkspaceManager::with_defaults();

        let ws1 = manager.create_workspace(Some("agent1".to_string()));
        let ws2 = manager.create_workspace(Some("agent2".to_string()));

        let node = NodeId::new();

        // Add private node to ws1
        manager.add_node_to_workspace(&ws1, node.clone(), AccessLevel::Private);

        // ws2 can't see it
        assert!(!manager.workspace_can_see(&ws2, &node));

        // Promote to shared
        manager.promote_to_shared(&ws1, &node);

        // Now ws2 can see it
        assert!(manager.workspace_can_see(&ws2, &node));
    }

    #[test]
    fn test_workspace_gc() {
        let config = WorkspaceConfig {
            max_idle_duration: Duration::seconds(0),
            ..Default::default()
        };
        let manager = WorkspaceManager::new(config);

        let ws_id = manager.create_workspace(Some("stale".to_string()));

        // Should be immediately stale
        std::thread::sleep(std::time::Duration::from_millis(10));
        let stale = manager.gc_stale_workspaces();

        assert!(stale.contains(&ws_id));
        assert!(manager.get_workspace(&ws_id).is_none());
    }

    #[test]
    fn test_workspace_merge() {
        let manager = WorkspaceManager::with_defaults();

        let ws1 = manager.create_workspace(Some("source".to_string()));
        let ws2 = manager.create_workspace(Some("target".to_string()));

        let node1 = NodeId::new();
        let node2 = NodeId::new();

        manager.add_node_to_workspace(&ws1, node1.clone(), AccessLevel::Private);
        manager.add_node_to_workspace(&ws2, node2.clone(), AccessLevel::Private);

        // Merge ws1 into ws2
        assert!(manager.merge_workspaces(&ws1, &ws2));

        // ws1 should be deleted
        assert!(manager.get_workspace(&ws1).is_none());

        // ws2 should have both nodes
        assert!(manager.workspace_can_see(&ws2, &node1));
        assert!(manager.workspace_can_see(&ws2, &node2));

        let stats = manager.get_workspace(&ws2).unwrap();
        assert_eq!(stats.total_nodes, 2);
    }
}
