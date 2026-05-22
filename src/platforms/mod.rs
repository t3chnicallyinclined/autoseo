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
pub mod browser;
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
    /// Identifies the account when a platform has multiple connected accounts
    /// (browser-backed posting). `None` for API-backed posters where there's
    /// only one account per platform.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub account_id: Option<String>,
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
            account_id: None,
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
            account_id: None,
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
            account_id: None,
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
            account_id: None,
            status: PostStatus::Posted,
            external_id: Some(external_id),
            external_url: Some(external_url),
            posted_at_unix: Some(unix_now()),
            error: None,
        }
    }

    /// Attach an account_id to a result. Used by browser-backed posters where
    /// the same platform has multiple connected accounts.
    pub fn with_account(mut self, account_id: impl Into<String>) -> Self {
        self.account_id = Some(account_id.into());
        self
    }
}

/// One configured backend. Built once from config; reused across all clips.
///
/// `Browser` represents a single `(platform, account)` pair routed through
/// the android-agent browser worker sidecar. Multiple accounts for the same platform
/// produce multiple `Platform::Browser` entries in the platforms vector;
/// `is_primary()` distinguishes the auto-post target from manual-only accounts.
pub enum Platform {
    YouTube(youtube::YouTubePoster),
    Bluesky(bluesky::BlueskyPoster),
    Instagram(instagram::InstagramPoster),
    Ayrshare(ayrshare::AyrsharePoster),
    Browser(browser::BrowserPoster),
}

impl Platform {
    pub fn name(&self) -> &'static str {
        match self {
            Platform::YouTube(_) => "youtube",
            Platform::Bluesky(_) => "bluesky",
            Platform::Instagram(_) => "instagram",
            Platform::Ayrshare(_) => "ayrshare",
            Platform::Browser(p) => p.platform_id(),
        }
    }

    /// Account identifier for browser-backed posters where multiple accounts
    /// may exist on the same platform. `None` for API-backed posters (always
    /// single-account by construction).
    pub fn account_id(&self) -> Option<&str> {
        match self {
            Platform::Browser(p) => Some(p.account_id()),
            _ => None,
        }
    }

    /// Whether this poster participates in the *automatic* post path. API-based
    /// posters are always primary (only one account); browser-backed posters
    /// flag a single primary per platform for auto-post fan-out.
    pub fn is_primary(&self) -> bool {
        match self {
            Platform::Browser(p) => p.is_primary,
            _ => true,
        }
    }

    /// Construct all enabled platforms from config. Each platform that is
    /// requested via `POST_ENABLED_PLATFORMS` but missing creds becomes a
    /// SKIPPED result at post-time rather than failing the whole run here.
    ///
    /// `POSTING_BACKEND` controls which posters are constructed:
    /// - `api` (default): only API-backed posters
    /// - `browser`: only browser-backed posters (one per `BROWSER_ACCOUNTS` entry)
    /// - `mixed`: both — useful while migrating one platform at a time
    pub fn from_config(
        cfg: &Config,
        google: Option<&crate::google_auth::GoogleAuth>,
    ) -> Vec<Platform> {
        let enabled = parse_enabled_platforms(&cfg.post_enabled_platforms);
        let backend = cfg.posting_backend.trim().to_ascii_lowercase();
        let want_api = backend == "api" || backend == "mixed";
        let want_browser = backend == "browser" || backend == "mixed";
        let mut out = Vec::new();

        if want_api {
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
        }

        if want_browser && cfg.browser_posting_enabled {
            let specs =
                browser::parse_accounts(&cfg.browser_accounts, &cfg.browser_primary_accounts);
            for spec in specs {
                let cap = browser::daily_cap_for(
                    spec.platform_id,
                    cfg.browser_post_daily_cap_default,
                );
                out.push(Platform::Browser(browser::BrowserPoster::new(
                    spec.platform_id,
                    spec.account_id,
                    spec.is_primary,
                    cap,
                    cfg.browser_humanize,
                    cfg.browser_worker_url.clone(),
                )));
            }
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
            Platform::Browser(p) => {
                // Browser-backed veto lands when each driver implements a
                // platform-specific delete flow. Phase 2 logs and no-ops.
                tracing::warn!(
                    platform = p.platform_id(),
                    account = %p.account_id(),
                    id = external_id,
                    "browser: veto not yet implemented for this platform"
                );
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
                account = self.account_id().unwrap_or(""),
                "POST_DRY_RUN — not actually posting"
            );
            let mut r = PostResult::dry_run(self.name());
            if let Some(a) = self.account_id() {
                r = r.with_account(a);
            }
            return r;
        }
        match self {
            Platform::YouTube(p) => p.post(video_path, &social.youtube_shorts).await,
            Platform::Bluesky(p) => p.post(video_path, &social.bluesky).await,
            Platform::Instagram(p) => p.post(video_path, &social.instagram_reels).await,
            Platform::Ayrshare(p) => p.post(video_path, social).await,
            Platform::Browser(p) => p.post(video_path, social).await,
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
