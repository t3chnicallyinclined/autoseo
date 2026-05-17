use anyhow::Context;
use futures_util::StreamExt;
use serde::Deserialize;

#[derive(Debug, Clone)]
pub struct DriveClient {
    http: reqwest::Client,
}

impl DriveClient {
    pub fn new() -> Self {
        Self {
            http: reqwest::Client::new(),
        }
    }

    pub async fn get_metadata(
        &self,
        access_token: &str,
        file_id: &str,
    ) -> anyhow::Result<DriveFile> {
        let url = format!("https://www.googleapis.com/drive/v3/files/{file_id}");
        let res = self
            .http
            .get(url)
            .bearer_auth(access_token)
            .query(&[("fields", "id,name,size,mimeType")])
            .send()
            .await
            .context("drive files.get")?;

        let status = res.status();
        if !status.is_success() {
            let body = res.text().await.unwrap_or_default();
            anyhow::bail!("drive metadata failed: {status} {body}");
        }

        res.json().await.context("parse drive metadata")
    }

    pub async fn download_to_path(
        &self,
        access_token: &str,
        file_id: &str,
        dest: &std::path::Path,
    ) -> anyhow::Result<()> {
        let url = format!("https://www.googleapis.com/drive/v3/files/{file_id}");
        let res = self
            .http
            .get(url)
            .bearer_auth(access_token)
            .query(&[("alt", "media")])
            .send()
            .await
            .context("drive download")?;

        let status = res.status();
        if !status.is_success() {
            let body = res.text().await.unwrap_or_default();
            anyhow::bail!("drive download failed: {status} {body}");
        }

        if let Some(parent) = dest.parent() {
            tokio::fs::create_dir_all(parent).await.ok();
        }

        let file_name = dest
            .file_name()
            .and_then(|s| s.to_str())
            .context("dest missing filename")?;
        let tmp = dest.with_file_name(format!("{file_name}.partial"));

        let mut file = tokio::fs::File::create(&tmp)
            .await
            .with_context(|| format!("create {}", tmp.display()))?;

        let mut stream = res.bytes_stream();
        use tokio::io::AsyncWriteExt;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.context("download stream chunk")?;
            file.write_all(&chunk).await?;
        }

        file.flush().await.ok();
        file.sync_all().await.ok();

        // Replace destination atomically (best-effort remove first for non-POSIX filesystems).
        let _ = tokio::fs::remove_file(dest).await;
        tokio::fs::rename(&tmp, dest)
            .await
            .with_context(|| format!("rename {} -> {}", tmp.display(), dest.display()))?;
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
pub struct DriveFile {
    #[allow(dead_code)]
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub size: Option<String>,
    #[serde(rename = "mimeType", default)]
    pub mime_type: String,
}
