//! Ayrshare posting shim.
//!
//! Ayrshare provides a single REST endpoint that fans out to multiple social
//! platforms (TikTok, Instagram, etc.) without needing direct API app review
//! approval from each platform. This module wraps their `/post` endpoint.
//!
//! Flow:
//! 1. Build caption from TikTok/Instagram social copy (pick first non-empty).
//! 2. Upload video via multipart POST to `https://app.ayrshare.com/api/post`.
//! 3. Parse per-platform results from the response.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::platforms::PostResult;
use crate::social_copy::SocialCopy;

const AYRSHARE_API_URL: &str = "https://app.ayrshare.com/api/post";

#[derive(Clone)]
pub struct AyrsharePoster {
    api_key: Option<String>,
    /// Which Ayrshare-supported platforms to post to (e.g. ["tiktok", "instagram"]).
    platforms: Vec<String>,
    http: reqwest::Client,
}

impl AyrsharePoster {
    pub fn new(api_key: Option<String>, platforms_csv: String) -> Self {
        let platforms: Vec<String> = platforms_csv
            .split(',')
            .map(|s| s.trim().to_ascii_lowercase())
            .filter(|s| !s.is_empty())
            .collect();
        Self {
            api_key: api_key.filter(|s| !s.is_empty()),
            platforms,
            http: reqwest::Client::new(),
        }
    }

    pub async fn post(&self, video_path: &Path, social: &SocialCopy) -> PostResult {
        let api_key = match self.api_key.as_deref() {
            Some(k) => k,
            None => return PostResult::skipped("ayrshare", "AYRSHARE_API_KEY not set"),
        };
        if self.platforms.is_empty() {
            return PostResult::skipped("ayrshare", "AYRSHARE_PLATFORMS is empty");
        }

        let caption = build_caption(social, &self.platforms);
        if caption.is_empty() {
            return PostResult::skipped("ayrshare", "no social copy available for caption");
        }

        match self.post_inner(api_key, video_path, &caption).await {
            Ok(resp) => {
                let id = resp.id.unwrap_or_default();
                let posted_platforms: Vec<&str> =
                    self.platforms.iter().map(|s| s.as_str()).collect();
                tracing::info!(
                    id = %id,
                    platforms = ?posted_platforms,
                    "ayrshare: post created"
                );
                PostResult::posted("ayrshare", id, resp.post_url.unwrap_or_default())
            }
            Err(e) => PostResult::failed("ayrshare", format!("{e:#}")),
        }
    }

    async fn post_inner(
        &self,
        api_key: &str,
        video_path: &Path,
        caption: &str,
    ) -> Result<AyrshareResponse> {
        let video_bytes = tokio::fs::read(video_path)
            .await
            .with_context(|| format!("read video for ayrshare: {}", video_path.display()))?;

        let file_name = video_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("clip.mp4");

        let file_part = reqwest::multipart::Part::bytes(video_bytes)
            .file_name(file_name.to_string())
            .mime_str("video/mp4")
            .context("set mime type")?;

        let platforms_json =
            serde_json::to_string(&self.platforms).context("serialize platforms")?;

        let form = reqwest::multipart::Form::new()
            .text("post", caption.to_string())
            .text("platforms", platforms_json)
            .part("file", file_part);

        let res = self
            .http
            .post(AYRSHARE_API_URL)
            .bearer_auth(api_key)
            .multipart(form)
            .send()
            .await
            .context("POST ayrshare /api/post")?;

        if !res.status().is_success() {
            let status = res.status();
            let body = res.text().await.unwrap_or_default();
            anyhow::bail!("ayrshare post failed: {status} {body}");
        }
        res.json().await.context("parse ayrshare response")
    }
}

/// Build a caption from the best available social copy.
/// Prefer TikTok copy if posting to TikTok, otherwise Instagram, then fallback.
fn build_caption(social: &SocialCopy, platforms: &[String]) -> String {
    let has_tiktok = platforms.iter().any(|p| p == "tiktok");
    let has_instagram = platforms.iter().any(|p| p == "instagram");

    let (text, hashtags) = if has_tiktok && !social.tiktok.caption.trim().is_empty() {
        (social.tiktok.caption.trim(), &social.tiktok.hashtags)
    } else if has_instagram && !social.instagram_reels.caption.trim().is_empty() {
        (
            social.instagram_reels.caption.trim(),
            &social.instagram_reels.hashtags,
        )
    } else if !social.tiktok.caption.trim().is_empty() {
        (social.tiktok.caption.trim(), &social.tiktok.hashtags)
    } else if !social.instagram_reels.caption.trim().is_empty() {
        (
            social.instagram_reels.caption.trim(),
            &social.instagram_reels.hashtags,
        )
    } else {
        // Last resort: use YouTube Shorts description
        (
            social.youtube_shorts.description.trim(),
            &social.youtube_shorts.hashtags,
        )
    };

    if text.is_empty() {
        return String::new();
    }

    let mut out = text.to_string();
    let tags: Vec<String> = hashtags
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
        .filter(|tag| !out.contains(tag.as_str()))
        .map(|s| s.as_str())
        .collect();
    if !missing.is_empty() {
        out.push_str("\n\n");
        out.push_str(&missing.join(" "));
    }

    // TikTok caption limit is ~2200 chars; Instagram is ~2200. Keep it safe.
    if out.chars().count() > 2200 {
        let mut chars: Vec<char> = out.chars().collect();
        chars.truncate(2197);
        chars.extend("...".chars());
        out = chars.into_iter().collect();
    }

    out
}

#[derive(Debug, Deserialize, Serialize)]
struct AyrshareResponse {
    #[serde(default)]
    id: Option<String>,
    #[serde(default, alias = "postUrl")]
    post_url: Option<String>,
    #[serde(default)]
    status: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::social_copy::{InstagramReelsCopy, TikTokCopy, YouTubeShortsCopy};

    #[test]
    fn build_caption_prefers_tiktok_when_posting_to_tiktok() {
        let mut social = SocialCopy::default();
        social.tiktok = TikTokCopy {
            caption: "TikTok caption here".into(),
            hashtags: vec!["#fyp".into(), "#comedy".into()],
        };
        social.instagram_reels = InstagramReelsCopy {
            caption: "IG caption".into(),
            hashtags: vec!["#reels".into()],
        };
        let platforms = vec!["tiktok".to_string(), "instagram".to_string()];
        let caption = build_caption(&social, &platforms);
        assert!(caption.starts_with("TikTok caption here"));
        assert!(caption.contains("#fyp"));
    }

    #[test]
    fn build_caption_prefers_instagram_when_only_instagram() {
        let mut social = SocialCopy::default();
        social.tiktok = TikTokCopy {
            caption: "TikTok caption".into(),
            hashtags: vec![],
        };
        social.instagram_reels = InstagramReelsCopy {
            caption: "IG caption".into(),
            hashtags: vec!["#reels".into()],
        };
        let platforms = vec!["instagram".to_string()];
        let caption = build_caption(&social, &platforms);
        assert!(caption.starts_with("IG caption"));
        assert!(caption.contains("#reels"));
    }

    #[test]
    fn build_caption_falls_back_to_youtube_description() {
        let mut social = SocialCopy::default();
        social.youtube_shorts = YouTubeShortsCopy {
            title: "Title".into(),
            description: "YT description text".into(),
            hashtags: vec!["#shorts".into()],
            pinned_comment: String::new(),
        };
        let platforms = vec!["tiktok".to_string()];
        let caption = build_caption(&social, &platforms);
        assert!(caption.starts_with("YT description text"));
    }

    #[test]
    fn build_caption_empty_when_no_copy() {
        let social = SocialCopy::default();
        let platforms = vec!["tiktok".to_string()];
        assert!(build_caption(&social, &platforms).is_empty());
    }

    #[test]
    fn build_caption_skips_duplicate_hashtags() {
        let mut social = SocialCopy::default();
        social.tiktok = TikTokCopy {
            caption: "Great clip #fyp".into(),
            hashtags: vec!["#fyp".into(), "#comedy".into()],
        };
        let platforms = vec!["tiktok".to_string()];
        let caption = build_caption(&social, &platforms);
        // #fyp should only appear once (in the caption itself)
        assert_eq!(caption.matches("#fyp").count(), 1);
        assert!(caption.contains("#comedy"));
    }

    #[test]
    fn build_caption_truncates_long_text() {
        let mut social = SocialCopy::default();
        social.tiktok = TikTokCopy {
            caption: "x".repeat(2300),
            hashtags: vec![],
        };
        let platforms = vec!["tiktok".to_string()];
        let caption = build_caption(&social, &platforms);
        assert!(caption.chars().count() <= 2200);
        assert!(caption.ends_with("..."));
    }

    #[test]
    fn constructor_filters_empty_key() {
        let p = AyrsharePoster::new(Some("".to_string()), "tiktok,instagram".to_string());
        assert!(p.api_key.is_none());
        assert_eq!(p.platforms, vec!["tiktok", "instagram"]);
    }

    #[test]
    fn constructor_parses_platforms_csv() {
        let p = AyrsharePoster::new(Some("key".into()), " TikTok , Instagram , ".to_string());
        assert_eq!(p.platforms, vec!["tiktok", "instagram"]);
    }

    #[test]
    fn constructor_handles_empty_platforms() {
        let p = AyrsharePoster::new(Some("key".into()), "".to_string());
        assert!(p.platforms.is_empty());
    }
}
