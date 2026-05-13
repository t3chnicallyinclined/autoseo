//! Workspaces DAO. v1 always operates against `ws_default`; this module
//! exists so call sites already thread a workspace_id through, and so that
//! v2 (SaaS) can add `create_workspace` / `list_workspaces` without
//! touching call sites.

use anyhow::Context;
use std::sync::Arc;

use crate::storage::Storage;

use super::WS_DEFAULT;

#[derive(Debug, Clone)]
pub struct Workspace {
    pub id: String,
    pub slug: String,
    pub name: String,
    pub created_at: i64,
    pub updated_at: i64,
}

/// Fetch a workspace by id. Returns `None` if it doesn't exist.
pub async fn get(storage: Arc<Storage>, id: &str) -> anyhow::Result<Option<Workspace>> {
    let id = id.to_string();
    let conn = storage.conn();
    tokio::task::spawn_blocking(move || -> anyhow::Result<Option<Workspace>> {
        let conn = conn.blocking_lock();
        let mut stmt = conn
            .prepare(
                "SELECT id, slug, name, created_at, updated_at \
                 FROM workspaces WHERE id = ?1",
            )
            .context("prepare workspace get")?;
        let row = stmt
            .query_row([id], |r| {
                Ok(Workspace {
                    id: r.get(0)?,
                    slug: r.get(1)?,
                    name: r.get(2)?,
                    created_at: r.get(3)?,
                    updated_at: r.get(4)?,
                })
            })
            .map(Some)
            .or_else(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => Ok(None),
                other => Err(other),
            })
            .context("query workspace")?;
        Ok(row)
    })
    .await
    .context("join workspace get")?
}

/// Fetch the default workspace. In v1 this row is seeded by the v2 migration
/// and should always exist; treat absence as a programmer bug.
pub async fn get_default(storage: Arc<Storage>) -> anyhow::Result<Workspace> {
    get(storage, WS_DEFAULT)
        .await?
        .ok_or_else(|| anyhow::anyhow!("default workspace missing — migration may have failed"))
}

/// List all workspaces. v1 returns exactly one row; v2 returns all rows the
/// caller has access to.
pub async fn list(storage: Arc<Storage>) -> anyhow::Result<Vec<Workspace>> {
    let conn = storage.conn();
    tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<Workspace>> {
        let conn = conn.blocking_lock();
        let mut stmt = conn
            .prepare(
                "SELECT id, slug, name, created_at, updated_at \
                 FROM workspaces ORDER BY created_at",
            )
            .context("prepare workspace list")?;
        let rows = stmt
            .query_map([], |r| {
                Ok(Workspace {
                    id: r.get(0)?,
                    slug: r.get(1)?,
                    name: r.get(2)?,
                    created_at: r.get(3)?,
                    updated_at: r.get(4)?,
                })
            })
            .context("query workspaces")?
            .collect::<Result<Vec<_>, _>>()
            .context("collect workspaces")?;
        Ok(rows)
    })
    .await
    .context("join workspace list")?
}
