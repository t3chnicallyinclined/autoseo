//! Audit log DAO. Every mutating route writes an entry here before
//! responding so that "who did what when" is recoverable forever.
//!
//! In v1 actor strings are: `'user'`, `'scheduler'`, `'analytics'`,
//! `'ingest'`, `'migration'`. Slice 13 surfaces these in the UI.

use anyhow::Context;
use std::sync::Arc;
use ulid::Ulid;

use crate::storage::Storage;

#[derive(Debug, Clone)]
pub struct AuditEntry {
    pub id: String,
    pub workspace_id: String,
    pub user_id: Option<String>,
    pub actor: String,
    pub action: String,
    pub target_kind: Option<String>,
    pub target_id: Option<String>,
    pub metadata_json: Option<String>,
    pub ip: Option<String>,
    pub created_at: i64,
}

/// Inputs for [`append`]. All fields except actor/action/workspace_id are
/// optional. Slice 13 will use these to render a filterable log view.
#[derive(Debug, Clone, Default)]
pub struct AppendArgs<'a> {
    pub workspace_id: &'a str,
    pub user_id: Option<&'a str>,
    pub actor: &'a str,
    pub action: &'a str,
    pub target_kind: Option<&'a str>,
    pub target_id: Option<&'a str>,
    pub metadata_json: Option<&'a str>,
    pub ip: Option<&'a str>,
}

/// Append a row to `audit_log`. Returns the new entry's ULID.
pub async fn append(storage: Arc<Storage>, args: AppendArgs<'_>) -> anyhow::Result<String> {
    let row = AuditEntry {
        id: Ulid::new().to_string(),
        workspace_id: args.workspace_id.to_string(),
        user_id: args.user_id.map(str::to_string),
        actor: args.actor.to_string(),
        action: args.action.to_string(),
        target_kind: args.target_kind.map(str::to_string),
        target_id: args.target_id.map(str::to_string),
        metadata_json: args.metadata_json.map(str::to_string),
        ip: args.ip.map(str::to_string),
        created_at: unix_now(),
    };
    let conn = storage.conn();
    let id = row.id.clone();
    tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
        let conn = conn.blocking_lock();
        conn.execute(
            "INSERT INTO audit_log (id, workspace_id, user_id, actor, action, \
             target_kind, target_id, metadata_json, ip, created_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            rusqlite::params![
                row.id,
                row.workspace_id,
                row.user_id,
                row.actor,
                row.action,
                row.target_kind,
                row.target_id,
                row.metadata_json,
                row.ip,
                row.created_at,
            ],
        )
        .context("insert audit_log")?;
        Ok(())
    })
    .await
    .context("join audit append")??;
    Ok(id)
}

/// Most recent entries for a workspace, newest first.
pub async fn list_recent(
    storage: Arc<Storage>,
    workspace_id: &str,
    limit: u32,
) -> anyhow::Result<Vec<AuditEntry>> {
    let workspace_id = workspace_id.to_string();
    let conn = storage.conn();
    tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<AuditEntry>> {
        let conn = conn.blocking_lock();
        let mut stmt = conn
            .prepare(
                "SELECT id, workspace_id, user_id, actor, action, target_kind, \
                 target_id, metadata_json, ip, created_at \
                 FROM audit_log \
                 WHERE workspace_id = ?1 \
                 ORDER BY created_at DESC LIMIT ?2",
            )
            .context("prepare audit list_recent")?;
        let rows = stmt
            .query_map(rusqlite::params![workspace_id, limit as i64], |r| {
                Ok(AuditEntry {
                    id: r.get(0)?,
                    workspace_id: r.get(1)?,
                    user_id: r.get(2)?,
                    actor: r.get(3)?,
                    action: r.get(4)?,
                    target_kind: r.get(5)?,
                    target_id: r.get(6)?,
                    metadata_json: r.get(7)?,
                    ip: r.get(8)?,
                    created_at: r.get(9)?,
                })
            })
            .context("query audit_log")?
            .collect::<Result<Vec<_>, _>>()
            .context("collect audit rows")?;
        Ok(rows)
    })
    .await
    .context("join audit list_recent")?
}

fn unix_now() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dashboard::repo::WS_DEFAULT;
    use tempfile::tempdir;

    #[tokio::test]
    async fn append_and_list_recent_roundtrip() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let storage = Arc::new(Storage::open(dir.path().join("test.db")).await?);

        let id = append(
            storage.clone(),
            AppendArgs {
                workspace_id: WS_DEFAULT,
                actor: "migration",
                action: "backfill.v1",
                ..Default::default()
            },
        )
        .await?;
        assert_eq!(id.len(), 26, "ULID is 26 chars in Crockford base32");

        let rows = list_recent(storage.clone(), WS_DEFAULT, 10).await?;
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].id, id);
        assert_eq!(rows[0].actor, "migration");
        assert_eq!(rows[0].action, "backfill.v1");
        Ok(())
    }
}
