//! Parsers for the new `BROWSER_*` env-var-backed config fields.
//!
//! `BROWSER_ACCOUNTS` and `BROWSER_PRIMARY_ACCOUNTS` are comma-separated
//! `platform:account_id` pairs. `BROWSER_POST_DAILY_CAP_<PLATFORM_UPPER>`
//! overrides the per-platform cap; falls back to `BROWSER_POST_DAILY_CAP_DEFAULT`.

use std::collections::HashSet;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BrowserAccountSpec {
    pub platform_id: &'static str,
    pub account_id: String,
    pub is_primary: bool,
}

/// Map a user-supplied platform token to the canonical `&'static str` used in
/// `PostResult.platform`. Returns `None` for unknown tokens.
pub fn canonicalize_platform(raw: &str) -> Option<&'static str> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "x" | "twitter" => Some("x"),
        "linkedin" | "li" => Some("linkedin"),
        "threads" => Some("threads"),
        "tiktok" | "tt" => Some("tiktok"),
        "instagram_browser" | "ig_browser" | "instagram_b" => Some("instagram_browser"),
        "youtube_browser" | "yt_browser" => Some("youtube_browser"),
        "bluesky_browser" | "bsky_browser" => Some("bluesky_browser"),
        _ => None,
    }
}

/// Parse `BROWSER_ACCOUNTS` + `BROWSER_PRIMARY_ACCOUNTS` together into a list of
/// `BrowserAccountSpec`. If `primaries_csv` is empty, the first account per
/// platform in `accounts_csv` is treated as primary.
pub fn parse_accounts(accounts_csv: &str, primaries_csv: &str) -> Vec<BrowserAccountSpec> {
    let primaries: HashSet<(String, String)> = primaries_csv
        .split(',')
        .filter_map(|s| split_pair(s).map(|(p, a)| (p.to_string(), a.to_string())))
        .collect();
    let primaries_empty = primaries.is_empty();

    let mut seen_primary_for_platform: HashSet<&'static str> = HashSet::new();
    let mut out: Vec<BrowserAccountSpec> = Vec::new();

    for raw in accounts_csv.split(',') {
        let Some((platform_raw, account_id)) = split_pair(raw) else {
            continue;
        };
        let Some(platform_id) = canonicalize_platform(platform_raw) else {
            tracing::warn!(
                platform = platform_raw,
                "BROWSER_ACCOUNTS: unknown platform token, skipping"
            );
            continue;
        };

        let is_primary = if primaries_empty {
            seen_primary_for_platform.insert(platform_id)
        } else {
            // Explicit list — match canonical platform + verbatim account id.
            primaries.contains(&(platform_id.to_string(), account_id.to_string()))
                || primaries.contains(&(platform_raw.to_string(), account_id.to_string()))
        };

        out.push(BrowserAccountSpec {
            platform_id,
            account_id: account_id.to_string(),
            is_primary,
        });
    }

    out
}

/// Parse `BROWSER_PRIMARY_ACCOUNTS` into a set of canonical (platform, account)
/// pairs. Exposed for tests / dashboard reflection.
pub fn parse_primaries(primaries_csv: &str) -> HashSet<(&'static str, String)> {
    primaries_csv
        .split(',')
        .filter_map(|s| {
            let (p, a) = split_pair(s)?;
            canonicalize_platform(p).map(|cp| (cp, a.to_string()))
        })
        .collect()
}

/// Resolve the daily-post cap for a given platform: returns the value of
/// `BROWSER_POST_DAILY_CAP_<PLATFORM_UPPER>` from env if set, otherwise
/// `default`. The "platform upper" form maps `instagram_browser` → `INSTAGRAM`
/// etc., so users don't have to remember the canonical suffix.
pub fn daily_cap_for(platform_id: &str, default: u32) -> u32 {
    let key = format!("BROWSER_POST_DAILY_CAP_{}", platform_env_key(platform_id));
    match std::env::var(&key) {
        Ok(v) => v.trim().parse().unwrap_or(default),
        Err(_) => default,
    }
}

fn platform_env_key(platform_id: &str) -> String {
    // `instagram_browser` → `INSTAGRAM`, `youtube_browser` → `YOUTUBE`, etc.
    platform_id
        .trim_end_matches("_browser")
        .to_ascii_uppercase()
}

fn split_pair(raw: &str) -> Option<(&str, &str)> {
    let s = raw.trim();
    if s.is_empty() {
        return None;
    }
    let (p, a) = s.split_once(':')?;
    let p = p.trim();
    let a = a.trim();
    if p.is_empty() || a.is_empty() {
        return None;
    }
    Some((p, a))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonicalize_handles_aliases() {
        assert_eq!(canonicalize_platform("x"), Some("x"));
        assert_eq!(canonicalize_platform("twitter"), Some("x"));
        assert_eq!(canonicalize_platform("LI"), Some("linkedin"));
        assert_eq!(canonicalize_platform("yt_browser"), Some("youtube_browser"));
        assert_eq!(canonicalize_platform("mastodon"), None);
    }

    #[test]
    fn parse_accounts_promotes_first_when_primaries_empty() {
        let specs = parse_accounts("x:main,x:alt,linkedin:pro", "");
        assert_eq!(specs.len(), 3);
        assert!(specs[0].is_primary); // x:main is the only primary
        assert!(!specs[1].is_primary); // x:alt
        assert!(specs[2].is_primary); // linkedin:pro is first (and only) for linkedin
    }

    #[test]
    fn parse_accounts_respects_explicit_primaries() {
        let specs = parse_accounts("x:main,x:alt", "x:alt");
        assert_eq!(specs.len(), 2);
        assert!(!specs[0].is_primary, "x:main");
        assert!(specs[1].is_primary, "x:alt explicitly marked primary");
    }

    #[test]
    fn parse_accounts_ignores_unknown_platforms() {
        let specs = parse_accounts("x:main,mastodon:foo,linkedin:pro", "");
        assert_eq!(specs.len(), 2);
        assert_eq!(specs[0].platform_id, "x");
        assert_eq!(specs[1].platform_id, "linkedin");
    }

    #[test]
    fn parse_accounts_skips_malformed_entries() {
        let specs = parse_accounts("x:main,nope,:noplatform,noaccount:,x:alt", "");
        let names: Vec<_> = specs.iter().map(|s| s.account_id.as_str()).collect();
        assert_eq!(names, vec!["main", "alt"]);
    }

    #[test]
    fn platform_env_key_strips_browser_suffix() {
        assert_eq!(platform_env_key("x"), "X");
        assert_eq!(platform_env_key("instagram_browser"), "INSTAGRAM");
        assert_eq!(platform_env_key("youtube_browser"), "YOUTUBE");
    }

    #[test]
    fn daily_cap_for_falls_back_to_default() {
        // unset env var → default
        let key = "BROWSER_POST_DAILY_CAP_X_TEST_FALLBACK";
        unsafe {
            std::env::remove_var(key);
        }
        // Note: we don't actually exercise the env-set path here because tests
        // share env globally; the lookup logic is exercised in integration.
        assert_eq!(daily_cap_for("x_test_fallback", 7), 7);
        let _ = key;
    }
}
