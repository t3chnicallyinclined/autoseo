//! Per-platform posting backends.
//!
//! Each backend takes a rendered video file + the platform-specific copy block
//! from `SocialCopy` and posts it via the platform's API. A `Platform` enum
//! dispatches to the right variant.
//!
//! M2 phase 1 ships YouTube + Bluesky (both free, no app-review gate). TikTok,
//! Instagram, Threads, LinkedIn, X land in subsequent phases (gated reviews
//! and/or paid tiers).

pub mod ayrshare;
pub mod bluesky;
pub mod instagram;
pub mod youtube;

use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::config::Config;
use crate::social_copy::SocialCopy;

/// Outcome of one post attempt to one platform.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostResult {
    pub platform: String,
    pub status: PostStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub external_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub external_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub posted_at_unix: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PostStatus {
    /// Successfully published to the platform.
    Posted,
    /// `POST_DRY_RUN=true` — nothing actually sent. Caller should not write to dedupe.
    DryRun,
    /// Platform disabled or required creds missing.
    Skipped,
    /// Attempt made but failed.
    Failed,
}

impl PostResult {
    pub fn skipped(platform: &str, reason: &str) -> Self {
        Self {
            platform: platform.to_string(),
            status: PostStatus::Skipped,
            external_id: None,
            external_url: None,
            posted_at_unix: None,
            error: Some(reason.to_string()),
        }
    }

    pub fn dry_run(platform: &str) -> Self {
        Self {
            platform: platform.to_string(),
            status: PostStatus::DryRun,
            external_id: None,
            external_url: None,
            posted_at_unix: None,
            error: None,
        }
    }

    pub fn failed(platform: &str, error: impl Into<String>) -> Self {
        Self {
            platform: platform.to_string(),
            status: PostStatus::Failed,
            external_id: None,
            external_url: None,
            posted_at_unix: None,
            error: Some(error.into()),
        }
    }

    pub fn posted(platform: &str, external_id: String, external_url: String) -> Self {
        Self {
            platform: platform.to_string(),
            status: PostStatus::Posted,
            external_id: Some(external_id),
            external_url: Some(external_url),
            posted_at_unix: Some(unix_now()),
            error: None,
        }
    }
}

/// One configured backend. Built once from config; reused across all clips.
pub enum Platform {
    YouTube(youtube::YouTubePoster),
    Bluesky(bluesky::BlueskyPoster),
    Instagram(instagram::InstagramPoster),
    Ayrshare(ayrshare::AyrsharePoster),
}

impl Platform {
    pub fn name(&self) -> &'static str {
        match self {
            Platform::YouTube(_) => "youtube",
            Platform::Bluesky(_) => "bluesky",
            Platform::Instagram(_) => "instagram",
            Platform::Ayrshare(_) => "ayrshare",
        }
    }

    /// Construct all enabled platforms from config. Each platform that is
    /// requested via `POST_ENABLED_PLATFORMS` but missing creds becomes a
    /// SKIPPED result at post-time rather than failing the whole run here.
    pub fn from_config(
        cfg: &Config,
        google: Option<&crate::google_auth::GoogleAuth>,
    ) -> Vec<Platform> {
        let enabled = parse_enabled_platforms(&cfg.post_enabled_platforms);
        let mut out = Vec::new();

        if enabled.contains(&"youtube") {
            out.push(Platform::YouTube(youtube::YouTubePoster::new(
                cfg.youtube_privacy_status.clone(),
                cfg.youtube_category_id.clone(),
                google.cloned(),
            )));
        }
        if enabled.contains(&"bluesky") {
            out.push(Platform::Bluesky(bluesky::BlueskyPoster::new(
                cfg.bluesky_pds_url.clone(),
                cfg.bluesky_video_service_url.clone(),
                cfg.bluesky_handle.clone(),
                cfg.bluesky_app_password.clone(),
            )));
        }
        if enabled.contains(&"instagram") {
            out.push(Platform::Instagram(instagram::InstagramPoster::new(
                cfg.instagram_access_token.clone(),
                cfg.instagram_user_id.clone(),
            )));
        }
        if enabled.contains(&"ayrshare") {
            out.push(Platform::Ayrshare(ayrshare::AyrsharePoster::new(
                cfg.ayrshare_api_key.clone(),
                cfg.ayrshare_platforms.clone(),
            )));
        }
        out
    }

    /// Veto (remove/unlist) a previously posted clip. Platform-specific:
    /// YouTube sets privacy to "private"; Bluesky deletes the record.
    pub async fn veto(&self, external_id: &str) -> anyhow::Result<()> {
        match self {
            Platform::YouTube(p) => p.set_private(external_id).await,
            Platform::Bluesky(p) => p.delete_record(external_id).await,
            Platform::Instagram(_) => {
                // Instagram Graph API does not expose a public delete endpoint
                // for Reels. Veto is a no-op; manual removal required.
                tracing::warn!(
                    media_id = external_id,
                    "instagram: veto not supported via API — remove manually"
                );
                Ok(())
            }
            Platform::Ayrshare(_) => {
                tracing::warn!(id = external_id, "ayrshare: veto not implemented");
                Ok(())
            }
        }
    }

    /// Post one clip to this platform. Caller selects which rendered variant
    /// (typically 9:16) to upload; platform extracts the right copy fields from
    /// `social`.
    pub async fn post_clip(
        &self,
        video_path: &Path,
        social: &SocialCopy,
        dry_run: bool,
    ) -> PostResult {
        if dry_run {
            tracing::info!(
                platform = self.name(),
                "POST_DRY_RUN — not actually posting"
            );
            return PostResult::dry_run(self.name());
        }
        match self {
            Platform::YouTube(p) => p.post(video_path, &social.youtube_shorts).await,
            Platform::Bluesky(p) => p.post(video_path, &social.bluesky).await,
            Platform::Instagram(p) => p.post(video_path, &social.instagram_reels).await,
            Platform::Ayrshare(p) => p.post(video_path, social).await,
        }
    }
}

fn parse_enabled_platforms(spec: &str) -> Vec<&'static str> {
    let mut out = Vec::new();
    for raw in spec.split(',') {
        let label = raw.trim().to_ascii_lowercase();
        match label.as_str() {
            "youtube" | "yt" | "shorts" => {
                if !out.contains(&"youtube") {
                    out.push("youtube");
                }
            }
            "bluesky" | "bsky" => {
                if !out.contains(&"bluesky") {
                    out.push("bluesky");
                }
            }
            "instagram" | "ig" | "reels" => {
                if !out.contains(&"instagram") {
                    out.push("instagram");
                }
            }
            "ayrshare" => {
                if !out.contains(&"ayrshare") {
                    out.push("ayrshare");
                }
            }
            "" => {}
            other => tracing::warn!(platform = other, "unknown platform; ignoring"),
        }
    }
    out
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

    #[test]
    fn parses_enabled_list_with_aliases() {
        assert_eq!(parse_enabled_platforms(""), Vec::<&str>::new());
        assert_eq!(parse_enabled_platforms("youtube"), vec!["youtube"]);
        assert_eq!(parse_enabled_platforms("bsky"), vec!["bluesky"]);
        assert_eq!(
            parse_enabled_platforms("youtube, bluesky"),
            vec!["youtube", "bluesky"]
        );
        // Dedupe.
        assert_eq!(
            parse_enabled_platforms("youtube,YT,shorts"),
            vec!["youtube"]
        );
        // Ayrshare.
        assert_eq!(
            parse_enabled_platforms("youtube,ayrshare"),
            vec!["youtube", "ayrshare"]
        );
        // Instagram with aliases.
        assert_eq!(parse_enabled_platforms("instagram"), vec!["instagram"]);
        assert_eq!(parse_enabled_platforms("ig"), vec!["instagram"]);
        assert_eq!(parse_enabled_platforms("reels"), vec!["instagram"]);
        assert_eq!(
            parse_enabled_platforms("instagram,IG,reels"),
            vec!["instagram"]
        );
        // Unknown ignored.
        assert_eq!(
            parse_enabled_platforms("youtube,tiktok,bluesky"),
            vec!["youtube", "bluesky"]
        );
    }

    #[test]
    fn post_result_constructors() {
        let r = PostResult::skipped("yt", "no creds");
        assert_eq!(r.status, PostStatus::Skipped);
        assert_eq!(r.error.as_deref(), Some("no creds"));
        assert!(r.external_id.is_none());

        let r = PostResult::dry_run("bluesky");
        assert_eq!(r.status, PostStatus::DryRun);

        let r = PostResult::posted("yt", "abc123".into(), "https://youtu.be/abc123".into());
        assert_eq!(r.status, PostStatus::Posted);
        assert!(r.posted_at_unix.is_some());
        assert_eq!(r.external_url.as_deref(), Some("https://youtu.be/abc123"));
    }
}
