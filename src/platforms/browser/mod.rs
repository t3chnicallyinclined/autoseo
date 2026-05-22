//! CloakBrowser-backed poster.
//!
//! One `BrowserPoster` instance corresponds to a single `(platform_id,
//! account_id)` pair. `Platform::from_config` materializes one per pair listed
//! in `BROWSER_ACCOUNTS`. The auto-post pipeline filters to `is_primary == true`
//! before dispatching; manual dashboard posts can target any subset.
//!
//! Posting flow:
//!   1. Build a platform-appropriate caption from `SocialCopy`
//!   2. POST `{worker_url}/post` with the video path (already on the shared
//!      volume the worker mounts as `/work`) + caption
//!   3. Translate the worker's JSON response into a `PostResult`
//!
//! The worker enforces the daily cap (it owns the on-disk counter); we still
//! carry `daily_cap` here to surface it in the dashboard `/api/platforms` row.

use std::path::Path;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::platforms::PostResult;
use crate::social_copy::SocialCopy;

pub mod client;
pub mod config;

pub use config::{parse_accounts, daily_cap_for};

#[derive(Clone)]
pub struct BrowserPoster {
    pub platform_id: &'static str,
    pub account_id: String,
    pub is_primary: bool,
    pub daily_cap: u32,
    pub humanize: bool,
    worker_url: String,
    http: reqwest::Client,
}

impl BrowserPoster {
    pub fn new(
        platform_id: &'static str,
        account_id: String,
        is_primary: bool,
        daily_cap: u32,
        humanize: bool,
        worker_url: String,
    ) -> Self {
        let http = reqwest::Client::builder()
            // Posts can take a while (video upload + processing on platform
            // side). Cap at 5 minutes so a wedged worker doesn't pin a clip.
            .timeout(Duration::from_secs(300))
            .build()
            .expect("reqwest client build");
        Self {
            platform_id,
            account_id,
            is_primary,
            daily_cap,
            humanize,
            worker_url: worker_url.trim_end_matches('/').to_string(),
            http,
        }
    }

    pub async fn post(&self, video_path: &Path, social: &SocialCopy) -> PostResult {
        let caption = compose_caption(self.platform_id, social);
        if caption.trim().is_empty() {
            return PostResult::skipped(self.platform_id, "empty caption for platform")
                .with_account(&self.account_id);
        }

        let req = client::PostRequest {
            platform: self.platform_id.to_string(),
            account_id: self.account_id.clone(),
            video_path: video_path.to_string_lossy().to_string(),
            caption,
            dry_run: false,
            humanize: self.humanize,
            metadata: None,
        };

        match client::call_post(&self.http, &self.worker_url, &req).await {
            Ok(resp) => translate_response(self.platform_id, &self.account_id, resp),
            Err(e) => PostResult::failed(self.platform_id, format!("worker call failed: {e:#}"))
                .with_account(&self.account_id),
        }
    }

    pub fn platform_id(&self) -> &'static str {
        self.platform_id
    }

    pub fn account_id(&self) -> &str {
        &self.account_id
    }

    pub fn worker_url(&self) -> &str {
        &self.worker_url
    }
}

fn translate_response(
    platform_id: &'static str,
    account_id: &str,
    resp: client::PostResponse,
) -> PostResult {
    match resp.status.as_str() {
        "posted" => {
            let external_id = resp.external_id.unwrap_or_default();
            let external_url = resp.external_url.unwrap_or_default();
            PostResult::posted(platform_id, external_id, external_url).with_account(account_id)
        }
        "dry_run" => PostResult::dry_run(platform_id).with_account(account_id),
        "skipped" => PostResult::skipped(
            platform_id,
            resp.error.as_deref().unwrap_or("skipped by worker"),
        )
        .with_account(account_id),
        _ => PostResult::failed(
            platform_id,
            resp.error.unwrap_or_else(|| "unknown worker error".to_string()),
        )
        .with_account(account_id),
    }
}

/// Build the caption string the worker driver should paste into the platform's
/// composer. Each platform's `SocialCopy` block has a slightly different shape
/// — this fans out the difference here so the worker doesn't need to know.
pub fn compose_caption(platform_id: &str, social: &SocialCopy) -> String {
    match platform_id {
        "x" => compose_x(&social.x),
        "linkedin" => compose_linkedin(&social.linkedin),
        "threads" => compose_threads(&social.threads),
        "tiktok" => compose_with_hashtags(&social.tiktok.caption, &social.tiktok.hashtags),
        "instagram_browser" => {
            compose_with_hashtags(&social.instagram_reels.caption, &social.instagram_reels.hashtags)
        }
        "bluesky_browser" => compose_with_hashtags(&social.bluesky.text, &social.bluesky.hashtags),
        "youtube_browser" => {
            // YouTube needs richer fields than a flat caption (title + description
            // + pinned comment). Phase 2 ships only `x`; YouTube driver lands in
            // Phase 5 and will switch to a richer request schema then. For now,
            // fall back to "title\n\ndescription".
            let yt = &social.youtube_shorts;
            let body = if yt.description.is_empty() {
                yt.title.clone()
            } else {
                format!("{}\n\n{}", yt.title, yt.description)
            };
            compose_with_hashtags(&body, &yt.hashtags)
        }
        other => {
            tracing::warn!(platform = other, "unknown browser platform_id");
            String::new()
        }
    }
}

fn compose_x(copy: &crate::social_copy::XCopy) -> String {
    let raw = compose_with_hashtags(&copy.text, &copy.hashtags);
    // X tweet limit is 280 chars; rendered clips post as media so the limit
    // applies to the text. Truncate softly on whitespace.
    truncate_chars(&raw, 280)
}

fn compose_linkedin(copy: &crate::social_copy::LinkedInCopy) -> String {
    compose_with_hashtags(&copy.post_text, &copy.hashtags)
}

fn compose_threads(copy: &crate::social_copy::ThreadsCopy) -> String {
    // Threads caps at 500 chars on web composer.
    truncate_chars(&compose_with_hashtags(&copy.text, &copy.hashtags), 500)
}

fn compose_with_hashtags(body: &str, hashtags: &[String]) -> String {
    let body = body.trim();
    if hashtags.is_empty() {
        return body.to_string();
    }
    let tags = hashtags
        .iter()
        .map(|t| {
            let t = t.trim_start_matches('#').trim();
            if t.is_empty() {
                String::new()
            } else {
                format!("#{t}")
            }
        })
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    if tags.is_empty() {
        body.to_string()
    } else if body.is_empty() {
        tags
    } else {
        format!("{body}\n\n{tags}")
    }
}

fn truncate_chars(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let mut acc = String::new();
    for c in s.chars().take(max.saturating_sub(1)) {
        acc.push(c);
    }
    acc.push('…');
    acc
}

/// Wire payload (sent + received) shapes — re-exported from `client` for ergonomic use.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostMetadata {
    pub clip_id: Option<String>,
    pub rank: Option<i64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::social_copy::{LinkedInCopy, SocialCopy, ThreadsCopy, TikTokCopy, XCopy};

    fn social_with_x(text: &str, tags: Vec<&str>) -> SocialCopy {
        SocialCopy {
            x: XCopy {
                text: text.to_string(),
                hashtags: tags.into_iter().map(String::from).collect(),
            },
            ..Default::default()
        }
    }

    #[test]
    fn x_caption_joins_hashtags() {
        let s = social_with_x("hello world", vec!["one", "two"]);
        assert_eq!(compose_caption("x", &s), "hello world\n\n#one #two");
    }

    #[test]
    fn x_caption_truncates_to_280() {
        let long = "a".repeat(500);
        let s = social_with_x(&long, vec![]);
        let out = compose_caption("x", &s);
        assert_eq!(out.chars().count(), 280);
        assert!(out.ends_with('…'));
    }

    #[test]
    fn linkedin_uses_post_text() {
        let s = SocialCopy {
            linkedin: LinkedInCopy {
                post_text: "deep thoughts".into(),
                hashtags: vec!["growth".into()],
            },
            ..Default::default()
        };
        assert_eq!(compose_caption("linkedin", &s), "deep thoughts\n\n#growth");
    }

    #[test]
    fn threads_truncates_to_500() {
        let s = SocialCopy {
            threads: ThreadsCopy {
                text: "x".repeat(700),
                hashtags: vec![],
            },
            ..Default::default()
        };
        assert_eq!(compose_caption("threads", &s).chars().count(), 500);
    }

    #[test]
    fn tiktok_uses_caption_and_hashtags() {
        let s = SocialCopy {
            tiktok: TikTokCopy {
                caption: "watch this".into(),
                hashtags: vec!["fyp".into()],
            },
            ..Default::default()
        };
        assert_eq!(compose_caption("tiktok", &s), "watch this\n\n#fyp");
    }

    #[test]
    fn unknown_platform_returns_empty() {
        let s = SocialCopy::default();
        assert_eq!(compose_caption("mastodon", &s), "");
    }

    #[test]
    fn hashtag_leading_hash_is_stripped() {
        let s = social_with_x("hi", vec!["#one", "two", "#three"]);
        assert_eq!(compose_caption("x", &s), "hi\n\n#one #two #three");
    }
}
