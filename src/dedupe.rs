use anyhow::Context;
use std::{collections::HashSet, path::Path};

#[derive(Debug)]
pub struct FileBackedDedupe {
    path: std::path::PathBuf,
    set: HashSet<String>,
}

impl FileBackedDedupe {
    pub async fn load(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut set = HashSet::new();

        if let Ok(contents) = tokio::fs::read_to_string(&path).await {
            for line in contents.lines() {
                let line = line.trim();
                if !line.is_empty() {
                    set.insert(line.to_string());
                }
            }
        }

        Ok(Self { path, set })
    }

    pub fn contains(&self, key: &str) -> bool {
        self.set.contains(key)
    }

    pub async fn insert(&mut self, key: String) -> anyhow::Result<()> {
        if self.set.insert(key.clone()) {
            if let Some(parent) = self.path.parent() {
                tokio::fs::create_dir_all(parent).await.ok();
            }
            let mut file = tokio::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.path)
                .await
                .with_context(|| format!("open dedupe file {}", self.path.display()))?;
            use tokio::io::AsyncWriteExt;
            file.write_all(key.as_bytes()).await?;
            file.write_all(b"\n").await?;
            file.flush().await.ok();
        }
        Ok(())
    }
}
