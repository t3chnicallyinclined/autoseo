//! Veto via Gmail reply.
//!
//! Polls Gmail for replies to digest emails containing `veto: clip_XX` commands.
//! For each vetoed clip, removes/unlists the post on each platform and updates
//! the posts table status to `vetoed`.

use anyhow::{Context, Result};
use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use regex::Regex;

use crate::gmail::GmailClient;
use crate::google_auth::GoogleAuth;
use crate::mime::build_mime_email;
use crate::platforms::Platform;
use crate::storage::Storage;

/// Parse all `veto: clip_XX` directives from an email body.
/// Returns a deduplicated list of clip labels like `["clip_03", "clip_07"]`.
pub fn parse_veto_directives(body: &str) -> Vec<String> {
    let re = Regex::new(r"(?i)\bveto:\s*(clip_\d+)").expect("valid veto regex");
    let mut labels: Vec<String> = re
        .captures_iter(body)
        .filter_map(|c| c.get(1).map(|m| m.as_str().to_lowercase()))
        .collect();
    labels.sort();
    labels.dedup();
    labels
}

/// Result of processing one veto directive.
#[derive(Debug)]
pub struct VetoResult {
    pub clip_label: String,
    pub outcomes: Vec<VetoPlatformOutcome>,
}

#[derive(Debug)]
pub struct VetoPlatformOutcome {
    pub platform: String,
    pub success: bool,
    pub detail: String,
}

/// Poll Gmail for veto replies and execute them.
/// Returns the count of clips vetoed.
pub async fn poll_and_process_vetoes(
    google: &GoogleAuth,
    gmail: &GmailClient,
    storage: &Storage,
    platforms: &[Platform],
    gmail_veto_query: &str,
    reply_to: &str,
    subject_prefix: &str,
) -> Result<Vec<VetoResult>> {
    let access_token = google
        .access_token()
        .await
        .context("refresh token for veto poll")?;

    let message_ids = gmail
        .list_message_ids(&access_token, gmail_veto_query, 20)
        .await
        .context("list veto candidate messages")?;

    if message_ids.is_empty() {
        tracing::debug!("veto: no candidate messages");
        return Ok(Vec::new());
    }

    let mut all_results = Vec::new();

    for message_id in &message_ids {
        // Skip already-processed veto messages.
        let veto_key = format!("veto:{message_id}");
        if storage.job_exists(&veto_key).await? {
            continue;
        }

        let msg = gmail
            .get_message_full(&access_token, message_id)
            .await
            .with_context(|| format!("fetch veto message {message_id}"))?;

        let body = GmailClient::extract_text_bodies(&msg);
        let directives = parse_veto_directives(&body);
        if directives.is_empty() {
            // Not a veto reply — mark to skip next time.
            storage.mark_processed(&veto_key).await?;
            continue;
        }

        tracing::info!(
            message_id,
            directives = ?directives,
            "veto: processing directives"
        );

        for label in &directives {
            let result = execute_veto(storage, platforms, label).await;
            all_results.push(result);
        }

        // Send confirmation email.
        let confirmation_body = build_confirmation(&all_results);
        let subject = format!("{subject_prefix} Veto confirmation");
        let raw_mime = build_mime_email("me", reply_to, &subject, &confirmation_body, &[]);
        let raw_b64url = URL_SAFE_NO_PAD.encode(raw_mime);
        match gmail.send_raw(&access_token, &raw_b64url).await {
            Ok(sent_id) => tracing::info!(sent_id, "veto: confirmation email sent"),
            Err(e) => tracing::warn!(error = %e, "veto: failed to send confirmation email"),
        }

        // Mark this veto message as processed.
        storage.mark_processed(&veto_key).await?;
    }

    Ok(all_results)
}

async fn execute_veto(storage: &Storage, platforms: &[Platform], clip_label: &str) -> VetoResult {
    let mut result = VetoResult {
        clip_label: clip_label.to_string(),
        outcomes: Vec::new(),
    };

    let clip_id = match storage.find_clip_by_rank_label(clip_label).await {
        Ok(Some((id, _job_id))) => id,
        Ok(None) => {
            result.outcomes.push(VetoPlatformOutcome {
                platform: "all".to_string(),
                success: false,
                detail: format!("clip '{clip_label}' not found in database"),
            });
            return result;
        }
        Err(e) => {
            result.outcomes.push(VetoPlatformOutcome {
                platform: "all".to_string(),
                success: false,
                detail: format!("db lookup failed: {e}"),
            });
            return result;
        }
    };

    let posts = match storage.get_posts_for_clip(&clip_id).await {
        Ok(p) => p,
        Err(e) => {
            result.outcomes.push(VetoPlatformOutcome {
                platform: "all".to_string(),
                success: false,
                detail: format!("failed to load posts: {e}"),
            });
            return result;
        }
    };

    if posts.is_empty() {
        result.outcomes.push(VetoPlatformOutcome {
            platform: "all".to_string(),
            success: false,
            detail: "no posts found for this clip".to_string(),
        });
        return result;
    }

    for post in &posts {
        if post.status == "vetoed" {
            result.outcomes.push(VetoPlatformOutcome {
                platform: post.platform.clone(),
                success: true,
                detail: "already vetoed".to_string(),
            });
            continue;
        }

        let external_id = match &post.external_id {
            Some(id) if !id.is_empty() => id.clone(),
            _ => {
                result.outcomes.push(VetoPlatformOutcome {
                    platform: post.platform.clone(),
                    success: false,
                    detail: "no external_id recorded".to_string(),
                });
                continue;
            }
        };

        // Find the matching platform backend.
        let platform = platforms.iter().find(|p| p.name() == post.platform);
        let platform = match platform {
            Some(p) => p,
            None => {
                result.outcomes.push(VetoPlatformOutcome {
                    platform: post.platform.clone(),
                    success: false,
                    detail: "platform not configured".to_string(),
                });
                continue;
            }
        };

        match platform.veto(&external_id).await {
            Ok(()) => {
                if let Err(e) = storage.veto_post(&clip_id, &post.platform).await {
                    tracing::warn!(error = %e, clip_id, platform = %post.platform, "failed to update post status to vetoed");
                }
                result.outcomes.push(VetoPlatformOutcome {
                    platform: post.platform.clone(),
                    success: true,
                    detail: "removed".to_string(),
                });
            }
            Err(e) => {
                result.outcomes.push(VetoPlatformOutcome {
                    platform: post.platform.clone(),
                    success: false,
                    detail: format!("{e:#}"),
                });
            }
        }
    }

    result
}

fn build_confirmation(results: &[VetoResult]) -> String {
    let mut out = String::from("Veto results:\n\n");
    for r in results {
        out.push_str(&format!("{}:\n", r.clip_label));
        for o in &r.outcomes {
            let icon = if o.success { "OK" } else { "FAIL" };
            out.push_str(&format!("  [{icon}] {}: {}\n", o.platform, o.detail));
        }
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_single_veto_directive() {
        let body = "Hey, please veto: clip_03. Thanks!";
        let directives = parse_veto_directives(body);
        assert_eq!(directives, vec!["clip_03"]);
    }

    #[test]
    fn parses_multiple_veto_directives() {
        let body = "veto: clip_01\nAlso veto: clip_07\nAnd veto: clip_03";
        let directives = parse_veto_directives(body);
        assert_eq!(directives, vec!["clip_01", "clip_03", "clip_07"]);
    }

    #[test]
    fn deduplicates_directives() {
        let body = "veto: clip_03\nveto: clip_03\nVETO: CLIP_03";
        let directives = parse_veto_directives(body);
        assert_eq!(directives, vec!["clip_03"]);
    }

    #[test]
    fn case_insensitive() {
        let body = "VETO: Clip_05";
        let directives = parse_veto_directives(body);
        assert_eq!(directives, vec!["clip_05"]);
    }

    #[test]
    fn no_match_returns_empty() {
        let body = "Thanks for the clips, they look great!";
        let directives = parse_veto_directives(body);
        assert!(directives.is_empty());
    }

    #[test]
    fn handles_whitespace_variations() {
        let body = "veto:  clip_10\nveto:clip_02";
        let directives = parse_veto_directives(body);
        assert_eq!(directives, vec!["clip_02", "clip_10"]);
    }

    #[test]
    fn confirmation_body_format() {
        let results = vec![VetoResult {
            clip_label: "clip_03".to_string(),
            outcomes: vec![
                VetoPlatformOutcome {
                    platform: "youtube".to_string(),
                    success: true,
                    detail: "removed".to_string(),
                },
                VetoPlatformOutcome {
                    platform: "bluesky".to_string(),
                    success: false,
                    detail: "not configured".to_string(),
                },
            ],
        }];
        let body = build_confirmation(&results);
        assert!(body.contains("clip_03"));
        assert!(body.contains("[OK] youtube: removed"));
        assert!(body.contains("[FAIL] bluesky: not configured"));
    }
}
