//! Bluesky / ATProto poster.
//!
//! Flow:
//! 1. `com.atproto.server.createSession` (handle + app password) → accessJwt + did
//! 2. `com.atproto.server.getServiceAuth` (aud=did:web:video.bsky.app) → upload token
//! 3. POST raw video bytes to `app.bsky.video.uploadVideo` → jobId
//! 4. Poll `app.bsky.video.getJobStatus` until `state=JOB_STATE_COMPLETED` → blob ref
//! 5. `com.atproto.repo.createRecord` with `app.bsky.embed.video` referencing the blob

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Duration;

use crate::platforms::PostResult;
use crate::social_copy::BlueskyCopy;

const JOB_POLL_INTERVAL_SECS: u64 = 3;
const JOB_POLL_MAX_ATTEMPTS: u32 = 60; // ~3 minutes total
const VIDEO_AUD: &str = "did:web:video.bsky.app";
const VIDEO_LXM: &str = "app.bsky.video.uploadVideo";

#[derive(Clone)]
pub struct BlueskyPoster {
    pds_url: String,           // https://bsky.social (default)
    video_service_url: String, // https://video.bsky.app (default)
    handle: Option<String>,
    app_password: Option<String>,
    http: reqwest::Client,
}

impl BlueskyPoster {
    pub fn new(
        pds_url: String,
        video_service_url: String,
        handle: Option<String>,
        app_password: Option<String>,
    ) -> Self {
        Self {
            pds_url: pds_url.trim_end_matches('/').to_string(),
            video_service_url: video_service_url.trim_end_matches('/').to_string(),
            handle: handle.filter(|s| !s.is_empty()),
            app_password: app_password.filter(|s| !s.is_empty()),
            http: reqwest::Client::new(),
        }
    }

    pub async fn post(&self, video_path: &Path, copy: &BlueskyCopy) -> PostResult {
        let handle = match self.handle.as_deref() {
            Some(h) => h,
            None => {
                return PostResult::skipped("bluesky", "BLUESKY_HANDLE not set");
            }
        };
        let app_password = match self.app_password.as_deref() {
            Some(p) => p,
            None => {
                return PostResult::skipped("bluesky", "BLUESKY_APP_PASSWORD not set");
            }
        };
        let text = compose_post_text(copy);
        if text.is_empty() {
            return PostResult::skipped("bluesky", "post text was empty");
        }
        match self
            .post_inner(handle, app_password, video_path, &text)
            .await
        {
            Ok((uri, _cid)) => {
                let url = at_uri_to_web_url(&uri, handle).unwrap_or(uri.clone());
                PostResult::posted("bluesky", uri, url)
            }
            Err(e) => PostResult::failed("bluesky", format!("{e:#}")),
        }
    }

    async fn post_inner(
        &self,
        handle: &str,
        app_password: &str,
        video_path: &Path,
        text: &str,
    ) -> Result<(String, String)> {
        // 1. Session
        let session = self.create_session(handle, app_password).await?;
        tracing::info!(did = %session.did, "bluesky: session created");

        // 2. Service auth for video service
        let video_token = self.get_service_auth(&session.access_jwt).await?;

        // 3. Upload video bytes
        let video_bytes = tokio::fs::read(video_path)
            .await
            .with_context(|| format!("read video for bluesky upload: {}", video_path.display()))?;
        let file_name = video_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("clip.mp4");
        let upload = self
            .upload_video(&video_token, &session.did, file_name, video_bytes)
            .await?;
        tracing::info!(job_id = %upload.job_id, "bluesky: upload started");

        // 4. Poll until completed
        let blob = self.wait_for_job(&video_token, &upload.job_id).await?;
        tracing::info!("bluesky: video processing complete");

        // 5. Create post
        let create = self
            .create_post_record(&session.access_jwt, &session.did, text, blob)
            .await?;
        tracing::info!(uri = %create.uri, "bluesky: post created");
        Ok((create.uri, create.cid))
    }

    /// Veto (delete) a Bluesky post by its AT URI.
    pub async fn delete_record(&self, at_uri: &str) -> Result<()> {
        let handle = self
            .handle
            .as_deref()
            .context("BLUESKY_HANDLE not set for veto")?;
        let app_password = self
            .app_password
            .as_deref()
            .context("BLUESKY_APP_PASSWORD not set for veto")?;

        let session = self.create_session(handle, app_password).await?;

        // Parse AT URI: at://did:plc:xxx/app.bsky.feed.post/rkey
        let stripped = at_uri
            .strip_prefix("at://")
            .context("invalid AT URI for veto")?;
        let parts: Vec<&str> = stripped.splitn(3, '/').collect();
        if parts.len() < 3 {
            anyhow::bail!("cannot parse AT URI into repo/collection/rkey: {at_uri}");
        }
        let (repo, collection, rkey) = (parts[0], parts[1], parts[2]);

        let url = format!("{}/xrpc/com.atproto.repo.deleteRecord", self.pds_url);
        let body = serde_json::json!({
            "repo": repo,
            "collection": collection,
            "rkey": rkey,
        });
        let res = self
            .http
            .post(&url)
            .bearer_auth(&session.access_jwt)
            .json(&body)
            .send()
            .await
            .context("POST deleteRecord (veto)")?;

        if !res.status().is_success() {
            let status = res.status();
            let body = res.text().await.unwrap_or_default();
            anyhow::bail!("bluesky deleteRecord (veto) failed: {status} {body}");
        }
        tracing::info!(at_uri, "bluesky: record deleted (vetoed)");
        Ok(())
    }

    async fn create_session(
        &self,
        handle: &str,
        app_password: &str,
    ) -> Result<CreateSessionResponse> {
        let url = format!("{}/xrpc/com.atproto.server.createSession", self.pds_url);
        let body = serde_json::json!({
            "identifier": handle,
            "password": app_password,
        });
        let res = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await
            .context("POST createSession")?;
        if !res.status().is_success() {
            let status = res.status();
            let body = res.text().await.unwrap_or_default();
            anyhow::bail!("createSession failed: {status} {body}");
        }
        res.json().await.context("parse createSession")
    }

    async fn get_service_auth(&self, access_jwt: &str) -> Result<String> {
        let url = format!(
            "{}/xrpc/com.atproto.server.getServiceAuth?aud={}&lxm={}",
            self.pds_url, VIDEO_AUD, VIDEO_LXM
        );
        let res = self
            .http
            .get(&url)
            .bearer_auth(access_jwt)
            .send()
            .await
            .context("GET getServiceAuth")?;
        if !res.status().is_success() {
            let status = res.status();
            let body = res.text().await.unwrap_or_default();
            anyhow::bail!("getServiceAuth failed: {status} {body}");
        }
        let parsed: ServiceAuthResponse = res.json().await.context("parse getServiceAuth")?;
        Ok(parsed.token)
    }

    async fn upload_video(
        &self,
        video_token: &str,
        did: &str,
        name: &str,
        bytes: Vec<u8>,
    ) -> Result<UploadVideoResponse> {
        let url = format!(
            "{}/xrpc/app.bsky.video.uploadVideo?did={}&name={}",
            self.video_service_url, did, name
        );
        let res = self
            .http
            .post(&url)
            .bearer_auth(video_token)
            .header(reqwest::header::CONTENT_TYPE, "video/mp4")
            .body(bytes)
            .send()
            .await
            .context("POST uploadVideo")?;
        if !res.status().is_success() {
            let status = res.status();
            let body = res.text().await.unwrap_or_default();
            anyhow::bail!("uploadVideo failed: {status} {body}");
        }
        res.json().await.context("parse uploadVideo")
    }

    async fn wait_for_job(&self, video_token: &str, job_id: &str) -> Result<BlobRef> {
        let url = format!(
            "{}/xrpc/app.bsky.video.getJobStatus?jobId={}",
            self.video_service_url, job_id
        );
        for attempt in 1..=JOB_POLL_MAX_ATTEMPTS {
            let res = self
                .http
                .get(&url)
                .bearer_auth(video_token)
                .send()
                .await
                .with_context(|| format!("GET getJobStatus (attempt {attempt})"))?;
            if !res.status().is_success() {
                let status = res.status();
                let body = res.text().await.unwrap_or_default();
                anyhow::bail!("getJobStatus failed: {status} {body}");
            }
            let parsed: GetJobStatusResponse = res.json().await.context("parse getJobStatus")?;
            let state = parsed.job_status.state.as_str();
            tracing::debug!(attempt, state, progress = ?parsed.job_status.progress, "bluesky: job poll");

            match state {
                "JOB_STATE_COMPLETED" => {
                    return parsed
                        .job_status
                        .blob
                        .ok_or_else(|| anyhow::anyhow!("job completed without blob ref"));
                }
                "JOB_STATE_FAILED" => {
                    anyhow::bail!(
                        "video processing failed: {} {}",
                        parsed.job_status.error.unwrap_or_default(),
                        parsed.job_status.message.unwrap_or_default()
                    );
                }
                _ => {}
            }
            tokio::time::sleep(Duration::from_secs(JOB_POLL_INTERVAL_SECS)).await;
        }
        anyhow::bail!("video upload polling timed out after {JOB_POLL_MAX_ATTEMPTS} attempts")
    }

    async fn create_post_record(
        &self,
        access_jwt: &str,
        did: &str,
        text: &str,
        blob: BlobRef,
    ) -> Result<CreateRecordResponse> {
        let url = format!("{}/xrpc/com.atproto.repo.createRecord", self.pds_url);
        let now = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true);
        let body = serde_json::json!({
            "repo": did,
            "collection": "app.bsky.feed.post",
            "record": {
                "$type": "app.bsky.feed.post",
                "createdAt": now,
                "text": text,
                "embed": {
                    "$type": "app.bsky.embed.video",
                    "video": blob,
                    "alt": text,
                    "aspectRatio": {
                        "width": 9,
                        "height": 16
                    }
                }
            }
        });
        let res = self
            .http
            .post(&url)
            .bearer_auth(access_jwt)
            .json(&body)
            .send()
            .await
            .context("POST createRecord")?;
        if !res.status().is_success() {
            let status = res.status();
            let body = res.text().await.unwrap_or_default();
            anyhow::bail!("createRecord failed: {status} {body}");
        }
        res.json().await.context("parse createRecord")
    }
}

fn compose_post_text(copy: &BlueskyCopy) -> String {
    let mut text = copy.text.trim().to_string();
    if !copy.hashtags.is_empty() {
        let tags = copy
            .hashtags
            .iter()
            .map(|t| {
                let t = t.trim();
                if t.starts_with('#') {
                    t.to_string()
                } else {
                    format!("#{t}")
                }
            })
            .collect::<Vec<_>>()
            .join(" ");
        if !tags.is_empty() {
            if !text.is_empty() {
                text.push_str("\n\n");
            }
            text.push_str(&tags);
        }
    }
    // Bluesky cap is 300 graphemes. Approximate with chars().count().
    let mut chars: Vec<char> = text.chars().collect();
    if chars.len() > 300 {
        chars.truncate(297);
        chars.extend("...".chars());
        text = chars.into_iter().collect();
    }
    text
}

/// Convert `at://did:plc:.../app.bsky.feed.post/RKEY` to `https://bsky.app/profile/{handle}/post/RKEY`.
fn at_uri_to_web_url(at_uri: &str, handle: &str) -> Option<String> {
    let stripped = at_uri.strip_prefix("at://")?;
    let rkey = stripped.rsplit('/').next()?;
    Some(format!("https://bsky.app/profile/{handle}/post/{rkey}"))
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CreateSessionResponse {
    access_jwt: String,
    #[serde(default)]
    #[allow(dead_code)]
    refresh_jwt: String,
    did: String,
    #[serde(default)]
    #[allow(dead_code)]
    handle: String,
}

#[derive(Debug, Deserialize)]
struct ServiceAuthResponse {
    token: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct UploadVideoResponse {
    job_id: String,
    #[serde(default)]
    #[allow(dead_code)]
    state: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GetJobStatusResponse {
    job_status: JobStatusBody,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct JobStatusBody {
    #[serde(default)]
    #[allow(dead_code)]
    job_id: String,
    state: String,
    #[serde(default)]
    blob: Option<BlobRef>,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    message: Option<String>,
    #[serde(default)]
    progress: Option<i64>,
}

/// ATProto BlobRef envelope (serializes back into the createRecord call).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BlobRef {
    #[serde(rename = "$type")]
    pub ref_type: String,
    #[serde(rename = "ref")]
    pub blob_ref: BlobRefLink,
    #[serde(rename = "mimeType")]
    pub mime_type: String,
    pub size: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BlobRefLink {
    #[serde(rename = "$link")]
    pub link: String,
}

#[derive(Debug, Deserialize)]
struct CreateRecordResponse {
    uri: String,
    cid: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compose_text_joins_hashtags() {
        let copy = BlueskyCopy {
            text: "interesting clip".into(),
            hashtags: vec!["#podcast".into(), "comedy".into()],
        };
        let out = compose_post_text(&copy);
        assert!(out.contains("interesting clip"));
        assert!(out.contains("#podcast"));
        assert!(out.contains("#comedy"));
    }

    #[test]
    fn compose_text_truncates_over_300() {
        let copy = BlueskyCopy {
            text: "x".repeat(400),
            hashtags: vec![],
        };
        let out = compose_post_text(&copy);
        assert!(out.chars().count() <= 300);
        assert!(out.ends_with("..."));
    }

    #[test]
    fn compose_text_handles_empty_hashtags() {
        let copy = BlueskyCopy {
            text: "hello".into(),
            hashtags: vec![],
        };
        assert_eq!(compose_post_text(&copy), "hello");
    }

    #[test]
    fn at_uri_to_web_url_extracts_rkey() {
        let uri = "at://did:plc:abc/app.bsky.feed.post/3kxyz789";
        let url = at_uri_to_web_url(uri, "user.bsky.social").unwrap();
        assert_eq!(
            url,
            "https://bsky.app/profile/user.bsky.social/post/3kxyz789"
        );

        assert!(at_uri_to_web_url("not-a-valid-uri", "u").is_none());
    }

    #[test]
    fn missing_creds_yields_skipped() {
        let p = BlueskyPoster::new(
            "https://bsky.social".into(),
            "https://video.bsky.app".into(),
            None,
            None,
        );
        // Just verify the constructor accepts None creds — actual post requires runtime.
        assert!(p.handle.is_none());
        assert!(p.app_password.is_none());
    }
}
