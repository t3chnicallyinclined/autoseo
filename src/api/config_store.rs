use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokio::sync::RwLock;

/// Flat key-value config that maps to env var names.
/// Stored as JSON on disk at `{work_dir}/config.json`.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ConfigData {
    #[serde(flatten)]
    pub values: HashMap<String, serde_json::Value>,
}

/// Keys whose values are secrets and should be masked in GET responses.
const SECRET_KEYS: &[&str] = &[
    "OPENAI_API_KEY",
    "HF_API_KEY",
    "GOOGLE_CLIENT_SECRET",
    "GOOGLE_REFRESH_TOKEN",
    "BLUESKY_APP_PASSWORD",
    "AYRSHARE_API_KEY",
    "VLM_PREMIUM_API_KEY",
    "R2_SECRET_ACCESS_KEY",
];

/// Thread-safe config store backed by a JSON file.
pub struct ConfigStore {
    path: PathBuf,
    data: RwLock<ConfigData>,
}

impl ConfigStore {
    /// Load (or create) a config store from the given file path.
    pub async fn load(path: PathBuf) -> Result<Self> {
        let data = if path.exists() {
            let bytes = tokio::fs::read(&path)
                .await
                .with_context(|| format!("read config file {}", path.display()))?;
            serde_json::from_slice(&bytes)
                .with_context(|| format!("parse config file {}", path.display()))?
        } else {
            ConfigData::default()
        };
        Ok(Self {
            path,
            data: RwLock::new(data),
        })
    }

    /// Return the config file path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Get the full config data (unmasked). Used internally.
    pub async fn get_raw(&self) -> ConfigData {
        self.data.read().await.clone()
    }

    /// Get config with secrets masked for API responses.
    pub async fn get_masked(&self) -> ConfigData {
        let data = self.data.read().await;
        let mut masked = data.clone();
        for key in SECRET_KEYS {
            if let Some(val) = masked.values.get_mut(*key) {
                if let Some(s) = val.as_str() {
                    if !s.is_empty() {
                        let m = mask_secret(s);
                        *val = serde_json::Value::String(m);
                    }
                }
            }
        }
        masked
    }

    /// Merge partial updates into the config and persist to disk.
    ///
    /// For secret keys, an incoming string that still carries the mask
    /// sentinel (`•`) is ignored — that's the value the dashboard just got
    /// from `get_masked()` and is round-tripping back unchanged. Without
    /// this guard, the round-trip silently overwrites the real secret with
    /// its own display mask.
    pub async fn patch(&self, updates: HashMap<String, serde_json::Value>) -> Result<()> {
        let mut data = self.data.write().await;
        for (k, v) in updates {
            if v.is_null() {
                data.values.remove(&k);
                continue;
            }
            if SECRET_KEYS.contains(&k.as_str()) {
                if let Some(s) = v.as_str() {
                    if is_masked(s) {
                        // Keep the existing real value; the dashboard sent us
                        // back what we showed it.
                        continue;
                    }
                }
            }
            data.values.insert(k, v);
        }
        self.persist(&data).await
    }

    /// Get a single raw value by key.
    pub async fn get_value(&self, key: &str) -> Option<String> {
        let data = self.data.read().await;
        data.values.get(key).map(|v| match v {
            serde_json::Value::String(s) => s.clone(),
            other => other.to_string(),
        })
    }

    /// True when the user hasn't supplied any provider credentials yet.
    ///
    /// Checks both `config.json` (set via the wizard / Settings page) AND
    /// the process environment (set via `.env` sourced at launch). Either
    /// is sufficient — power users who manage everything in `.env` and
    /// never touch the dashboard's setup UI shouldn't be prompted to.
    pub async fn needs_setup(&self) -> bool {
        let data = self.data.read().await;
        !SECRET_KEYS.iter().any(|k| {
            let in_config = data
                .values
                .get(*k)
                .and_then(|v| v.as_str())
                .is_some_and(|s| !s.is_empty());
            let in_env = std::env::var(k)
                .map(|s| !s.is_empty())
                .unwrap_or(false);
            in_config || in_env
        })
    }

    /// Write the current data to disk with restricted permissions.
    async fn persist(&self, data: &ConfigData) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            tokio::fs::create_dir_all(parent).await.ok();
        }
        let json = serde_json::to_string_pretty(data)?;
        tokio::fs::write(&self.path, json.as_bytes()).await?;

        // Best-effort: set file permissions to 0600 on unix.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let perms = std::fs::Permissions::from_mode(0o600);
            tokio::fs::set_permissions(&self.path, perms).await.ok();
        }

        Ok(())
    }
}

/// Mask a secret string: show first 3 and last 4 chars with dots in between.
fn mask_secret(s: &str) -> String {
    let len = s.len();
    if len <= 8 {
        return "••••••••".to_string();
    }
    let prefix = &s[..3];
    let suffix = &s[len - 4..];
    format!("{prefix}••••{suffix}")
}

/// True if the string carries the mask sentinel — i.e. it came out of
/// `mask_secret()` and shouldn't be persisted back as if it were a real
/// secret. The `•` character (U+2022) is what we paint with, so its mere
/// presence is sufficient.
fn is_masked(s: &str) -> bool {
    s.contains('•')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mask_secret_short() {
        assert_eq!(mask_secret("abc"), "••••••••");
    }

    #[test]
    fn mask_secret_long() {
        assert_eq!(mask_secret("sk-1234567890abcdef"), "sk-••••cdef");
    }

    #[tokio::test]
    async fn config_store_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.json");

        let store = ConfigStore::load(path.clone()).await.unwrap();
        assert!(store.needs_setup().await);

        let mut updates = HashMap::new();
        updates.insert(
            "OPENAI_API_KEY".to_string(),
            serde_json::Value::String("sk-test123456789".to_string()),
        );
        updates.insert(
            "MODE".to_string(),
            serde_json::Value::String("clipper".to_string()),
        );
        store.patch(updates).await.unwrap();

        assert!(!store.needs_setup().await);

        // Masked response hides secret
        let masked = store.get_masked().await;
        let key = masked
            .values
            .get("OPENAI_API_KEY")
            .unwrap()
            .as_str()
            .unwrap();
        assert!(key.contains("••••"));
        assert!(!key.contains("test123456789"));

        // Raw response has the real value
        let raw = store.get_raw().await;
        assert_eq!(
            raw.values.get("OPENAI_API_KEY").unwrap().as_str().unwrap(),
            "sk-test123456789"
        );

        // Persisted to disk
        let store2 = ConfigStore::load(path).await.unwrap();
        let raw2 = store2.get_raw().await;
        assert_eq!(
            raw2.values.get("MODE").unwrap().as_str().unwrap(),
            "clipper"
        );
    }

    #[tokio::test]
    async fn patch_ignores_masked_secret_roundtrip() {
        // Reproduces the bug the user actually hit: the dashboard GETs a
        // masked secret, then PATCHes it back unchanged. Before the fix this
        // overwrote the real value with the mask string.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.json");

        let store = ConfigStore::load(path).await.unwrap();

        // Seed the real secret.
        let mut updates = HashMap::new();
        updates.insert(
            "OPENAI_API_KEY".to_string(),
            serde_json::Value::String("sk-real1234567890abcd".to_string()),
        );
        store.patch(updates).await.unwrap();

        // Simulate the dashboard sending the masked string back.
        let masked = store.get_masked().await;
        let masked_val = masked.values.get("OPENAI_API_KEY").unwrap().clone();
        assert!(masked_val.as_str().unwrap().contains("••••"));

        let mut updates = HashMap::new();
        updates.insert("OPENAI_API_KEY".to_string(), masked_val);
        store.patch(updates).await.unwrap();

        // Real value must still be intact.
        assert_eq!(
            store.get_value("OPENAI_API_KEY").await.as_deref(),
            Some("sk-real1234567890abcd"),
        );
    }

    #[tokio::test]
    async fn patch_accepts_new_secret_value() {
        // The mask guard must not block real updates.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.json");

        let store = ConfigStore::load(path).await.unwrap();
        let mut updates = HashMap::new();
        updates.insert(
            "OPENAI_API_KEY".to_string(),
            serde_json::Value::String("sk-aaa".to_string()),
        );
        store.patch(updates).await.unwrap();

        let mut updates = HashMap::new();
        updates.insert(
            "OPENAI_API_KEY".to_string(),
            serde_json::Value::String("sk-bbbbbbbbbbbbbbbb".to_string()),
        );
        store.patch(updates).await.unwrap();

        assert_eq!(
            store.get_value("OPENAI_API_KEY").await.as_deref(),
            Some("sk-bbbbbbbbbbbbbbbb"),
        );
    }

    #[tokio::test]
    async fn patch_null_removes_key() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.json");

        let store = ConfigStore::load(path).await.unwrap();
        let mut updates = HashMap::new();
        updates.insert(
            "FOO".to_string(),
            serde_json::Value::String("bar".to_string()),
        );
        store.patch(updates).await.unwrap();
        assert_eq!(store.get_value("FOO").await.as_deref(), Some("bar"));

        let mut updates = HashMap::new();
        updates.insert("FOO".to_string(), serde_json::Value::Null);
        store.patch(updates).await.unwrap();
        assert!(store.get_value("FOO").await.is_none());
    }
}
