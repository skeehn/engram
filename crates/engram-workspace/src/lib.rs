//! Multi-agent workspace isolation for engram.
//!
//! Provides workspace-scoped memory isolation so multiple agents can
//! operate on the same engram instance without interfering with each other.
//!
//! Key features:
//! - **Workspace isolation**: Each agent/session has its own memory partition
//! - **Shared base layer**: Common knowledge accessible to all workspaces
//! - **Workspace merging**: Promote workspace-local memories to shared
//! - **Garbage collection**: Clean up stale workspaces

pub mod workspace;

pub use workspace::{Workspace, WorkspaceConfig, WorkspaceId, WorkspaceManager, WorkspaceStats};
