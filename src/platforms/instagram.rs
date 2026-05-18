//! Instagram Reels poster via Meta's Graph API.
//!
//! Flow (resumable upload for local files):
//! 1. POST `/{ig-user-id}/media` with `media_type=REELS`, `upload_type=resumable`
//!    → container ID + upload URI
//! 2. Upload video bytes to the upload URI via POST with resumable headers
//! 3. Poll `GET /{container-id}?fields=status_code` until `FINISHED`
//! 4. POST `/{ig-user-id}/media_publish` with `creation_id={container-id}`
//!
//! Constraints: 9:16 aspect ratio, ≤60s, ≤100MB.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;
use std::time::Duration;

use crate::platforms::PostResult;
use crate::social_copy::InstagramReelsCopy;

const GRAPH_API_BASE: &str = "https://graph.facebook.com/v21.0";
const POLL_INTERVAL_SECS: u64 = 5;
const POLL_MAX_ATTEMPTS: u32 = 60; // ~5 minutes total
const CAPTION_MAX_CHARS: usize = 2200;

#[derive(Clone)]
pub struct InstagramPoster {
    access_token: Option<String>,
    user_id: Option<String>,
    http: reqwest::Client,
}

impl InstagramPoster {
    pub fn new(access_token: Option<String>, user_id: Option<String>) -> Self {
        Self {
            access_token: access_token.filter(|s| !s.is_empty()),
            user_id: user_id.filter(|s| !s.is_empty()),
            http: reqwest::Client::new(),
        }
    }

    pub async fn post(&self, video_path: &Path, copy: &InstagramReelsCopy) -> PostResult {
        let access_token = match self.access_token.as_deref() {
            Some(t) => t,
            None => return PostResult::skipped("instagram", "INSTAGRAM_ACCESS_TOKEN not set"),
        };
        let user_id = match self.user_id.as_deref() {
            Some(id) => id,
            None => return PostResult::skipped("instagram", "INSTAGRAM_USER_ID not set"),
        };
        let caption = compose_caption(copy);
        if caption.is_empty() {
            return PostResult::skipped("instagram", "caption was empty");
        }
        match self
            .post_inner(access_token, user_id, video_path, &caption)
            .await
        {
            Ok(media_id) => {
                let url = format!("https://www.instagram.com/reel/{media_id}/");
                PostResult::posted("instagram", media_id, url)
            }
            Err(e) => PostResult::failed("instagram", format!("{e:#}")),
        }
    }

    async fn post_inner(
        &self,
        access_token: &str,
        user_id: &str,
        video_path: &Path,
        caption: &str,
    ) -> Result<String> {
        let video_bytes = tokio::fs::read(video_path)
            .await
            .with_context(|| format!("read video for instagram upload: {}", video_path.display()))?;
        let file_size = video_bytes.len();

        // 1. Create resumable upload container
        let container = self
            .create_container(access_token, user_id, caption, file_size)
            .await?;
        tracing::info!(container_id = %container.id, "instagram: container created");

        let upload_uri = container
            .uri
            .as_deref()
            .context("instagram: no upload URI in container response")?;

        // 2. Upload video bytes
        self.upload_video(access_token, upload_uri, video_bytes, file_size)
            .await?;
        tracing::info!("instagram: video uploaded");

        // 3. Poll until container is FINISHED
        self.wait_for_container(access_token, &container.id).await?;
        tracing::info!("instagram: container ready");

        // 4. Publish
        let media_id = self
            .publish(access_token, user_id, &container.id)
            .await?;
        tracing::info!(media_id = %media_id, "instagram: reel published");

        Ok(media_id)
    }

    async fn create_container(
        &self,
        access_token: &str,
        user_id: &str,
        caption: &str,
        file_size: usize,
    ) -> Result<CreateContainerResponse> {
        let url = format!("{GRAPH_API_BASE}/{user_id}/media");
        let res = self
            .http
            .post(&url)
            .query(&[
                ("media_type", "REELS"),
                ("upload_type", "resumable"),
                ("caption", caption),
                ("share_to_feed", "true"),
                ("access_token", access_token),
            ])
            .header("Content-Length", "0")
            .send()
            .await
            .context("POST create container")?;

        if !res.status().is_success() {
            let status = res.status();
            let body = res.text().await.unwrap_or_default();
            anyhow::bail!("instagram create container failed: {status} {body}");
        }

        // The response includes the container ID and an upload URI in the header
        // For the resumable upload API, the URI comes in the response body
        let parsed: CreateContainerResponse = res.json().await.context("parse create container")?;

        // If no URI in response, construct the upload URI from the container ID
        if parsed.uri.is_none() {
            // Fallback: use the standard upload endpoint
            let upload_uri = format!(
                "https://rupload.facebook.com/ig-api/{}/media/upload",
                parsed.id
            );
            return Ok(CreateContainerResponse {
                id: parsed.id,
                uri: Some(upload_uri),
            });
        }

        Ok(parsed)
    }

    async fn upload_video(
        &self,
        access_token: &str,
        upload_uri: &str,
        video_bytes: Vec<u8>,
        file_size: usize,
    ) -> Result<()> {
        let offset = 0;
        let res = self
            .http
            .post(upload_uri)
            .header("Authorization", format!("OAuth {access_token}"))
            .header("offset", offset.to_string())
            .header("file_size", file_size.to_string())
            .header(reqwest::header::CONTENT_TYPE, "video/mp4")
            .body(video_bytes)
            .send()
            .await
            .context("POST upload video")?;

        if !res.status().is_success() {
            let status = res.status();
            let body = res.text().await.unwrap_or_default();
            anyhow::bail!("instagram video upload failed: {status} {body}");
        }
        Ok(())
    }

    async fn wait_for_container(&self, access_token: &str, container_id: &str) -> Result<()> {
        let url = format!("{GRAPH_API_BASE}/{container_id}");
        for attempt in 1..=POLL_MAX_ATTEMPTS {
            let res = self
                .http
                .get(&url)
                .query(&[
                    ("fields", "status_code"),
                    ("access_token", access_token),
                ])
                .send()
                .await
                .with_context(|| format!("GET container status (attempt {attempt})"))?;

            if !res.status().is_success() {
                let status = res.status();
                let body = res.text().await.unwrap_or_default();
                anyhow::bail!("instagram container status failed: {status} {body}");
            }
            let parsed: ContainerStatusResponse =
                res.json().await.context("parse container status")?;
            let status = parsed.status_code.as_str();
            tracing::debug!(attempt, status, "instagram: container poll");

            match status {
                "FINISHED" => return Ok(()),
                "ERROR" => {
                    anyhow::bail!("instagram container processing failed (status=ERROR)");
                }
                // IN_PROGRESS or PUBLISHED are transient
                _ => {}
            }
            tokio::time::sleep(Duration::from_secs(POLL_INTERVAL_SECS)).await;
        }
        anyhow::bail!(
            "instagram container polling timed out after {POLL_MAX_ATTEMPTS} attempts"
        )
    }

    async fn publish(
        &self,
        access_token: &str,
        user_id: &str,
        container_id: &str,
    ) -> Result<String> {
        let url = format!("{GRAPH_API_BASE}/{user_id}/media_publish");
        let res = self
            .http
            .post(&url)
            .query(&[
                ("creation_id", container_id),
                ("access_token", access_token),
            ])
            .send()
            .await
            .context("POST media_publish")?;

        if !res.status().is_success() {
            let status = res.status();
            let body = res.text().await.unwrap_or_default();
            anyhow::bail!("instagram media_publish failed: {status} {body}");
        }
        let parsed: PublishResponse = res.json().await.context("parse media_publish")?;
        Ok(parsed.id)
    }
}

fn compose_caption(copy: &InstagramReelsCopy) -> String {
    let mut text = copy.caption.trim().to_string();
    if !copy.hashtags.is_empty() {
        let tags: Vec<String> = copy
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
            .collect();

        // Only append hashtags not already in the caption
        let missing: Vec<&str> = tags
            .iter()
            .filter(|tag| !text.contains(tag.as_str()))
            .map(|s| s.as_str())
            .collect();
        if !missing.is_empty() {
            if !text.is_empty() {
                text.push_str("\n\n");
            }
            text.push_str(&missing.join(" "));
        }
    }
    // Instagram caption limit is 2200 chars
    if text.chars().count() > CAPTION_MAX_CHARS {
        let mut chars: Vec<char> = text.chars().collect();
        chars.truncate(CAPTION_MAX_CHARS - 3);
        chars.extend("...".chars());
        text = chars.into_iter().collect();
    }
    text
}

#[derive(Debug, Deserialize)]
struct CreateContainerResponse {
    id: String,
    #[serde(default)]
    uri: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ContainerStatusResponse {
    status_code: String,
}

#[derive(Debug, Deserialize)]
struct PublishResponse {
    id: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compose_caption_joins_hashtags() {
        let copy = InstagramReelsCopy {
            caption: "Great clip from the show".into(),
            hashtags: vec!["#reels".into(), "#podcast".into(), "comedy".into()],
        };
        let out = compose_caption(&copy);
        assert!(out.starts_with("Great clip from the show"));
        assert!(out.contains("#reels"));
        assert!(out.contains("#podcast"));
        assert!(out.contains("#comedy"));
    }

    #[test]
    fn compose_caption_skips_duplicate_hashtags() {
        let copy = InstagramReelsCopy {
            caption: "Watch this #reels".into(),
            hashtags: vec!["#reels".into(), "#comedy".into()],
        };
        let out = compose_caption(&copy);
        assert_eq!(out.matches("#reels").count(), 1);
        assert!(out.contains("#comedy"));
    }

    #[test]
    fn compose_caption_truncates_over_2200() {
        let copy = InstagramReelsCopy {
            caption: "x".repeat(2300),
            hashtags: vec![],
        };
        let out = compose_caption(&copy);
        assert!(out.chars().count() <= CAPTION_MAX_CHARS);
        assert!(out.ends_with("..."));
    }

    #[test]
    fn compose_caption_handles_empty() {
        let copy = InstagramReelsCopy {
            caption: "".into(),
            hashtags: vec![],
        };
        assert!(compose_caption(&copy).is_empty());
    }

    #[test]
    fn compose_caption_hashtags_only() {
        let copy = InstagramReelsCopy {
            caption: "".into(),
            hashtags: vec!["#reels".into(), "#fun".into()],
        };
        let out = compose_caption(&copy);
        assert!(out.contains("#reels"));
        assert!(out.contains("#fun"));
    }

    #[test]
    fn missing_creds_yields_skipped() {
        let p = InstagramPoster::new(None, None);
        assert!(p.access_token.is_none());
        assert!(p.user_id.is_none());
    }

    #[test]
    fn empty_string_creds_filtered() {
        let p = InstagramPoster::new(Some("".into()), Some("".into()));
        assert!(p.access_token.is_none());
        assert!(p.user_id.is_none());
    }

    #[test]
    fn valid_creds_retained() {
        let p = InstagramPoster::new(Some("tok123".into()), Some("12345".into()));
        assert_eq!(p.access_token.as_deref(), Some("tok123"));
        assert_eq!(p.user_id.as_deref(), Some("12345"));
    }
}
