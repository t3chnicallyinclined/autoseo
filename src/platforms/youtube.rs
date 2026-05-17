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
//!
//! ## Retry & Quota
//! - Transient failures (HTTP 5xx, 429) are retried up to 3 times with exponential backoff.
//! - YouTube Data API v3 has a 10,000 unit daily quota; each `videos.insert` costs 1,600 units.
//! - The `QuotaTracker` tracks usage per UTC day and rejects uploads when exhausted.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;

use crate::google_auth::GoogleAuth;
use crate::platforms::PostResult;
use crate::social_copy::YouTubeShortsCopy;

const INIT_URL: &str =
    "https://www.googleapis.com/upload/youtube/v3/videos?uploadType=resumable&part=snippet,status";

/// Maximum retry attempts for transient errors.
const MAX_RETRIES: u32 = 3;
/// Base delay for exponential backoff (doubles each retry).
const BASE_DELAY: Duration = Duration::from_secs(2);

/// YouTube Data API v3 daily quota limit (units).
const DAILY_QUOTA_LIMIT: u64 = 10_000;
/// Cost of a single `videos.insert` call (units).
const UPLOAD_COST: u64 = 1_600;

#[derive(Clone)]
pub struct YouTubePoster {
    privacy_status: String,
    category_id: String,
    google: Option<GoogleAuth>,
    http: reqwest::Client,
    pub quota: Arc<Mutex<QuotaTracker>>,
}

/// Tracks YouTube API quota usage for the current UTC day.
#[derive(Debug, Clone)]
pub struct QuotaTracker {
    /// The UTC date string (YYYY-MM-DD) of the current tracking window.
    pub current_day: String,
    /// Units consumed so far today.
    pub used: u64,
    /// Daily limit in units.
    pub limit: u64,
}

impl QuotaTracker {
    pub fn new() -> Self {
        Self {
            current_day: today_utc(),
            used: 0,
            limit: DAILY_QUOTA_LIMIT,
        }
    }

    /// Returns true if there is enough quota remaining for one upload.
    pub fn can_upload(&mut self) -> bool {
        self.maybe_reset_day();
        self.used + UPLOAD_COST <= self.limit
    }

    /// Records one upload's cost. Call after a successful upload.
    pub fn record_upload(&mut self) {
        self.maybe_reset_day();
        self.used += UPLOAD_COST;
        tracing::info!(
            used = self.used,
            remaining = self.limit.saturating_sub(self.used),
            day = %self.current_day,
            "youtube quota: recorded upload"
        );
    }

    pub fn remaining(&self) -> u64 {
        self.limit.saturating_sub(self.used)
    }

    fn maybe_reset_day(&mut self) {
        let today = today_utc();
        if today != self.current_day {
            tracing::info!(
                old_day = %self.current_day,
                new_day = %today,
                old_used = self.used,
                "youtube quota: day rolled over, resetting"
            );
            self.current_day = today;
            self.used = 0;
        }
    }
}

fn today_utc() -> String {
    chrono::Utc::now().format("%Y-%m-%d").to_string()
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
            quota: Arc::new(Mutex::new(QuotaTracker::new())),
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

        // Check quota before attempting upload.
        {
            let mut quota = self.quota.lock().await;
            if !quota.can_upload() {
                return PostResult::failed(
                    "youtube",
                    format!(
                        "daily quota exhausted ({}/{} units used, upload costs {} units)",
                        quota.used, quota.limit, UPLOAD_COST
                    ),
                );
            }
        }

        match self.post_inner(google, video_path, copy).await {
            Ok(video_id) => {
                // Record quota usage on success.
                self.quota.lock().await.record_upload();
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

        // 1. Initiate resumable upload with retry on transient errors.
        let init_res = self
            .send_with_retry(|| {
                self.http
                    .post(INIT_URL)
                    .bearer_auth(&token)
                    .header(reqwest::header::CONTENT_TYPE, "application/json; charset=UTF-8")
                    .header("X-Upload-Content-Length", content_length.to_string())
                    .header("X-Upload-Content-Type", "video/mp4")
                    .body(metadata_json.clone())
            })
            .await
            .context("POST videos.insert (init)")?;

        if !init_res.status().is_success() {
            let status = init_res.status();
            let body = init_res.text().await.unwrap_or_default();
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

        // 2. PUT the bytes with retry on transient errors.
        let upload_res = self
            .send_with_retry(|| {
                self.http
                    .put(&upload_url)
                    .header(reqwest::header::CONTENT_TYPE, "video/mp4")
                    .header(reqwest::header::CONTENT_LENGTH, content_length.to_string())
                    .body(video_bytes.clone())
            })
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

    /// Send an HTTP request with exponential backoff retry on transient errors
    /// (5xx, 429). Returns the response on success or the last error after
    /// exhausting retries.
    async fn send_with_retry<F>(
        &self,
        build_request: F,
    ) -> Result<reqwest::Response>
    where
        F: Fn() -> reqwest::RequestBuilder,
    {
        let mut last_err: Option<anyhow::Error> = None;

        for attempt in 0..=MAX_RETRIES {
            let res = build_request().send().await;

            match res {
                Ok(response) => {
                    let status = response.status().as_u16();
                    if is_retryable(status) && attempt < MAX_RETRIES {
                        let delay = backoff_delay(attempt);
                        tracing::warn!(
                            status,
                            attempt = attempt + 1,
                            max = MAX_RETRIES,
                            delay_ms = delay.as_millis() as u64,
                            "youtube: transient error, retrying"
                        );
                        tokio::time::sleep(delay).await;
                        last_err = Some(anyhow::anyhow!(
                            "transient HTTP {status} on attempt {}",
                            attempt + 1
                        ));
                        continue;
                    }
                    return Ok(response);
                }
                Err(e) => {
                    if attempt < MAX_RETRIES {
                        let delay = backoff_delay(attempt);
                        tracing::warn!(
                            error = %e,
                            attempt = attempt + 1,
                            max = MAX_RETRIES,
                            delay_ms = delay.as_millis() as u64,
                            "youtube: network error, retrying"
                        );
                        tokio::time::sleep(delay).await;
                        last_err = Some(e.into());
                        continue;
                    }
                    return Err(e).context("request failed after retries");
                }
            }
        }

        Err(last_err.unwrap_or_else(|| anyhow::anyhow!("retry loop exhausted")))
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

/// Returns true for HTTP status codes that should be retried.
fn is_retryable(status: u16) -> bool {
    status == 429 || (500..=599).contains(&status)
}

/// Exponential backoff: BASE_DELAY * 2^attempt.
fn backoff_delay(attempt: u32) -> Duration {
    BASE_DELAY * 2u32.pow(attempt)
}

fn build_description(copy: &YouTubeShortsCopy) -> String {
    let mut out = copy.description.trim().to_string();
    if !out.to_lowercase().contains("#shorts") {
        if !out.is_empty() {
            out.push('\n');
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

    // --- Retry logic tests ---

    #[test]
    fn is_retryable_identifies_transient_codes() {
        assert!(is_retryable(429));
        assert!(is_retryable(500));
        assert!(is_retryable(502));
        assert!(is_retryable(503));
        assert!(is_retryable(599));
        assert!(!is_retryable(200));
        assert!(!is_retryable(400));
        assert!(!is_retryable(401));
        assert!(!is_retryable(403));
        assert!(!is_retryable(404));
    }

    #[test]
    fn backoff_delay_is_exponential() {
        assert_eq!(backoff_delay(0), Duration::from_secs(2));
        assert_eq!(backoff_delay(1), Duration::from_secs(4));
        assert_eq!(backoff_delay(2), Duration::from_secs(8));
        assert_eq!(backoff_delay(3), Duration::from_secs(16));
    }

    // --- Quota tracker tests ---

    #[test]
    fn quota_tracker_allows_uploads_within_limit() {
        let mut qt = QuotaTracker::new();
        assert!(qt.can_upload());
        assert_eq!(qt.remaining(), DAILY_QUOTA_LIMIT);

        qt.record_upload();
        assert_eq!(qt.used, UPLOAD_COST);
        assert_eq!(qt.remaining(), DAILY_QUOTA_LIMIT - UPLOAD_COST);
    }

    #[test]
    fn quota_tracker_denies_when_exhausted() {
        let mut qt = QuotaTracker::new();
        // 6 uploads = 9600 used. Next would be 11200 > 10000 limit.
        qt.used = UPLOAD_COST * 6;
        assert!(!qt.can_upload());
    }

    #[test]
    fn quota_tracker_blocks_at_threshold() {
        let mut qt = QuotaTracker::new();
        // 6 uploads = 9600 used. Next would bring to 11200 > 10000.
        qt.used = 9_600;
        assert!(!qt.can_upload());
    }

    #[test]
    fn quota_tracker_allows_exactly_at_limit() {
        let mut qt = QuotaTracker::new();
        // 5 uploads = 8000 used. Next brings to 9600 <= 10000.
        qt.used = 8_000;
        assert!(qt.can_upload());
        // At 8400, next = 10000 exactly — should be allowed.
        qt.used = 8_400;
        assert!(qt.can_upload());
    }

    #[test]
    fn quota_tracker_resets_on_new_day() {
        let mut qt = QuotaTracker::new();
        qt.used = 9_999;
        qt.current_day = "2020-01-01".to_string(); // old day
        // can_upload will call maybe_reset_day which sees today != 2020-01-01
        assert!(qt.can_upload());
        assert_eq!(qt.used, 0);
    }

    #[tokio::test]
    async fn post_rejects_when_quota_exhausted() {
        let poster = YouTubePoster::new("unlisted".into(), "24".into(), None);
        {
            let mut q = poster.quota.lock().await;
            q.used = DAILY_QUOTA_LIMIT;
        }
        let mut q = poster.quota.lock().await;
        assert!(!q.can_upload());
    }
}
