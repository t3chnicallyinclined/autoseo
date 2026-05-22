//! Posting orchestrator. Iterates the configured `Platform` list and posts
//! each clip's appropriate variant (currently the 9:16 render — both YouTube
//! Shorts and Bluesky's video embed are vertical-friendly).
//!
//! SQLite persistence of posts is deferred to M2 phase 2 (when we wire
//! retry-on-failure logic). For now the post results live in the manifest
//! and the in-memory list returned to clipper.rs.

use std::path::Path;

use crate::platforms::{Platform, PostResult, PostStatus};
use crate::social_copy::SocialCopy;

/// Post one clip to every configured platform. Returns one `PostResult` per
/// platform (Posted / DryRun / Skipped / Failed). Order matches `platforms`.
pub async fn post_one_clip(
    platforms: &[Platform],
    dry_run: bool,
    rank: usize,
    video_9x16: Option<&Path>,
    social: Option<&SocialCopy>,
) -> Vec<PostResult> {
    let mut results = Vec::with_capacity(platforms.len());
    if platforms.is_empty() {
        return results;
    }

    let Some(video) = video_9x16 else {
        for p in platforms {
            results.push(PostResult::skipped(p.name(), "9x16 variant missing"));
        }
        return results;
    };
    if !video.exists() {
        for p in platforms {
            results.push(PostResult::skipped(p.name(), "9x16 file not on disk"));
        }
        return results;
    }
    let Some(social) = social else {
        for p in platforms {
            results.push(PostResult::skipped(
                p.name(),
                "no social copy for this clip",
            ));
        }
        return results;
    };

    for platform in platforms {
        let name = platform.name();
        let r = platform.post_clip(video, social, dry_run).await;
        let acct = r.account_id.as_deref().unwrap_or("");
        match r.status {
            PostStatus::Posted => tracing::info!(
                clip = rank,
                platform = name,
                account = acct,
                url = r.external_url.as_deref().unwrap_or(""),
                "posted"
            ),
            PostStatus::DryRun => tracing::info!(
                clip = rank,
                platform = name,
                account = acct,
                "dry-run"
            ),
            PostStatus::Skipped => tracing::info!(
                clip = rank,
                platform = name,
                account = acct,
                reason = r.error.as_deref().unwrap_or(""),
                "skipped"
            ),
            PostStatus::Failed => tracing::warn!(
                clip = rank,
                platform = name,
                account = acct,
                error = r.error.as_deref().unwrap_or(""),
                "failed"
            ),
        }
        results.push(r);
    }
    results
}
