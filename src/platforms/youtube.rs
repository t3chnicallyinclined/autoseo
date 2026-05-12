//! YouTube Shorts poster — uses the YouTube Data API v3 `videos.insert` endpoint
//! with the resumable upload protocol.
//!
//! Reuses the existing `GoogleAuth` token refresher (the same OAuth client the
//! seo-only Gmail/Drive path uses) — BUT the refresh token must have been
//! minted with the `https://www.googleapis.com/auth/youtube.upload` scope.
//! Re-mint at OAuth Playground if your existing token only has Gmail/Drive scopes.
//!
//! Privacy defaults to "unlisted" — anyone with the link can view, but the
//! video isn't surfaced in search. Flip via `YOUTUBE_PRIVACY_STATUS=public`.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::google_auth::GoogleAuth;
use crate::platforms::PostResult;
use crate::social_copy::YouTubeShortsCopy;

const INIT_URL: &str =
    "https://www.googleapis.com/upload/youtube/v3/videos?uploadType=resumable&part=snippet,status";

#[derive(Clone)]
pub struct YouTubePoster {
    privacy_status: String,
    category_id: String,
    google: Option<GoogleAuth>,
    http: reqwest::Client,
}

impl YouTubePoster {
    pub fn new(
        privacy_status: String,
        category_id: String,
        google: Option<GoogleAuth>,
    ) -> Self {
        Self {
            privacy_status: if privacy_status.is_empty() {
                "unlisted".to_string()
            } else {
                privacy_status
            },
            category_id: if category_id.is_empty() {
                "24".to_string() // Entertainment
            } else {
                category_id
            },
            google,
            http: reqwest::Client::new(),
        }
    }

    pub async fn post(&self, video_path: &Path, copy: &YouTubeShortsCopy) -> PostResult {
        let google = match self.google.as_ref() {
            Some(g) => g,
            None => return PostResult::skipped(
                "youtube",
                "Google creds missing (set GOOGLE_CLIENT_ID/SECRET/REFRESH_TOKEN)",
            ),
        };
        if copy.title.trim().is_empty() {
            return PostResult::skipped("youtube", "title was empty");
        }

        match self.post_inner(google, video_path, copy).await {
            Ok(video_id) => {
                let url = format!("https://www.youtube.com/shorts/{video_id}");
                PostResult::posted("youtube", video_id, url)
            }
            Err(e) => PostResult::failed("youtube", format!("{e:#}")),
        }
    }

    async fn post_inner(
        &self,
        google: &GoogleAuth,
        video_path: &Path,
        copy: &YouTubeShortsCopy,
    ) -> Result<String> {
        let token = google
            .access_token()
            .await
            .context("refresh Google access token")?;

        let video_bytes = tokio::fs::read(video_path)
            .await
            .with_context(|| format!("read video: {}", video_path.display()))?;
        let content_length = video_bytes.len() as u64;

        let metadata = self.build_metadata(copy);
        let metadata_json = serde_json::to_string(&metadata).context("serialize metadata")?;

        // 1. Initiate resumable upload — returns Location header pointing at the
        //    upload URL to PUT the bytes to.
        let init_res = self
            .http
            .post(INIT_URL)
            .bearer_auth(&token)
            .header(reqwest::header::CONTENT_TYPE, "application/json; charset=UTF-8")
            .header("X-Upload-Content-Length", content_length.to_string())
            .header("X-Upload-Content-Type", "video/mp4")
            .body(metadata_json)
            .send()
            .await
            .context("POST videos.insert (init)")?;

        if !init_res.status().is_success() {
            let status = init_res.status();
            let body = init_res.text().await.unwrap_or_default();
            // Common: 401/403 means the refresh token doesn't have the
            // youtube.upload scope. Surface a useful message.
            if status.as_u16() == 401 || status.as_u16() == 403 {
                anyhow::bail!(
                    "youtube auth failed ({status}). Your refresh token likely lacks the \
                     'https://www.googleapis.com/auth/youtube.upload' scope — re-mint at \
                     OAuth Playground with that scope added. Server said: {body}"
                );
            }
            anyhow::bail!("videos.insert init failed: {status} {body}");
        }

        let upload_url = init_res
            .headers()
            .get(reqwest::header::LOCATION)
            .and_then(|v| v.to_str().ok())
            .ok_or_else(|| anyhow::anyhow!("videos.insert init returned no Location header"))?
            .to_string();
        tracing::info!(content_length, "youtube: upload session opened");

        // 2. PUT the bytes.
        let upload_res = self
            .http
            .put(&upload_url)
            .header(reqwest::header::CONTENT_TYPE, "video/mp4")
            .header(reqwest::header::CONTENT_LENGTH, content_length.to_string())
            .body(video_bytes)
            .send()
            .await
            .context("PUT video bytes")?;

        if !upload_res.status().is_success() {
            let status = upload_res.status();
            let body = upload_res.text().await.unwrap_or_default();
            anyhow::bail!("video PUT failed: {status} {body}");
        }
        let parsed: VideoResource = upload_res.json().await.context("parse video resource")?;
        tracing::info!(id = %parsed.id, "youtube: upload complete");
        Ok(parsed.id)
    }

    fn build_metadata(&self, copy: &YouTubeShortsCopy) -> VideoInsertRequest {
        let title = clamp(&copy.title, 100);
        let description = build_description(copy);
        let tags: Vec<String> = copy
            .hashtags
            .iter()
            .map(|t| t.trim_start_matches('#').to_string())
            .filter(|t| !t.is_empty())
            .collect();
        VideoInsertRequest {
            snippet: VideoSnippet {
                title,
                description,
                tags,
                category_id: self.category_id.clone(),
            },
            status: VideoStatus {
                privacy_status: self.privacy_status.clone(),
                self_declared_made_for_kids: false,
                embeddable: true,
            },
        }
    }
}

fn build_description(copy: &YouTubeShortsCopy) -> String {
    let mut out = copy.description.trim().to_string();
    // The pinned-comment field isn't a description; it's intended for posting
    // separately. We don't auto-post comments (separate API call), so it's
    // dropped here. Could be wired later.
    if !out.to_lowercase().contains("#shorts") {
        if !out.is_empty() {
            out.push_str("\n");
        }
        out.push_str("#Shorts");
    }
    out
}

fn clamp(s: &str, max_chars: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= max_chars {
        return s.to_string();
    }
    chars.into_iter().take(max_chars).collect()
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct VideoInsertRequest {
    snippet: VideoSnippet,
    status: VideoStatus,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct VideoSnippet {
    title: String,
    description: String,
    tags: Vec<String>,
    category_id: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct VideoStatus {
    privacy_status: String,
    self_declared_made_for_kids: bool,
    embeddable: bool,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct VideoResource {
    id: String,
    #[serde(default)]
    kind: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::social_copy::YouTubeShortsCopy;

    fn dummy_copy() -> YouTubeShortsCopy {
        YouTubeShortsCopy {
            title: "He realizes the defense is hopeless".to_string(),
            description: "A clip from the show about the telephone defense.".to_string(),
            hashtags: vec!["#Shorts".into(), "#podcast".into(), "comedy".into()],
            pinned_comment: "What's your take?".to_string(),
        }
    }

    #[test]
    fn clamps_title_to_100() {
        let long = "x".repeat(200);
        assert_eq!(clamp(&long, 100).len(), 100);
        assert_eq!(clamp("short", 100), "short");
    }

    #[test]
    fn description_appends_shorts_if_missing() {
        let mut copy = dummy_copy();
        copy.description = "no shorts tag".into();
        let d = build_description(&copy);
        assert!(d.ends_with("#Shorts"));
    }

    #[test]
    fn description_keeps_existing_shorts_tag() {
        let mut copy = dummy_copy();
        copy.description = "already has #Shorts somewhere".into();
        let d = build_description(&copy);
        assert_eq!(d.matches("#Shorts").count(), 1);
    }

    #[test]
    fn metadata_strips_hash_from_tags() {
        let poster = YouTubePoster::new("unlisted".into(), "22".into(), None);
        let req = poster.build_metadata(&dummy_copy());
        assert_eq!(req.snippet.tags, vec!["Shorts", "podcast", "comedy"]);
        assert_eq!(req.status.privacy_status, "unlisted");
        assert_eq!(req.snippet.category_id, "22");
        assert!(!req.status.self_declared_made_for_kids);
    }

    #[test]
    fn empty_privacy_defaults_to_unlisted() {
        let poster = YouTubePoster::new(String::new(), String::new(), None);
        assert_eq!(poster.privacy_status, "unlisted");
        assert_eq!(poster.category_id, "24");
    }
}
