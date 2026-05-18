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
    pub async fn patch(&self, updates: HashMap<String, serde_json::Value>) -> Result<()> {
        let mut data = self.data.write().await;
        for (k, v) in updates {
            if v.is_null() {
                data.values.remove(&k);
            } else {
                data.values.insert(k, v);
            }
        }
        self.persist(&data).await
    }

    /// Get a single raw value by key.
    pub async fn get_value(&self, key: &str) -> Option<String> {
        let data = self.data.read().await;
        data.values.get(key).and_then(|v| match v {
            serde_json::Value::String(s) => Some(s.clone()),
            other => Some(other.to_string()),
        })
    }

    /// Check if the config file exists and has any credentials set.
    pub async fn needs_setup(&self) -> bool {
        let data = self.data.read().await;
        // Needs setup if no secret keys have values
        !SECRET_KEYS.iter().any(|k| {
            data.values
                .get(*k)
                .and_then(|v| v.as_str())
                .is_some_and(|s| !s.is_empty())
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
        let key = masked.values.get("OPENAI_API_KEY").unwrap().as_str().unwrap();
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
