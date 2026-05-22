//! Typed HTTP client for the android-agent browser worker sidecar.

use std::collections::HashMap;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize)]
pub struct PostRequest {
    pub platform: String,
    pub account_id: String,
    pub video_path: String,
    pub caption: String,
    pub dry_run: bool,
    pub humanize: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, String>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PostResponse {
    pub status: String,
    #[serde(default)]
    pub external_id: Option<String>,
    #[serde(default)]
    pub external_url: Option<String>,
    #[serde(default)]
    pub posted_at_unix: Option<i64>,
    #[serde(default)]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AccountInfo {
    pub platform: String,
    pub account_id: String,
    pub created_at: i64,
    #[serde(default)]
    pub last_used: Option<i64>,
    pub posts_today: u32,
    pub daily_cap: u32,
    pub cookies_valid: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct HealthResponse {
    pub ok: bool,
    pub cdp_connected: bool,
    pub cdp_url: String,
    pub drivers: Vec<String>,
    pub profiles_dir: String,
    pub work_dir: String,
}

pub async fn call_post(
    http: &reqwest::Client,
    base: &str,
    req: &PostRequest,
) -> Result<PostResponse> {
    let url = format!("{}/post", base.trim_end_matches('/'));
    let resp = http
        .post(&url)
        .json(req)
        .send()
        .await
        .with_context(|| format!("POST {url}"))?;
    let status = resp.status();
    let body = resp.text().await.unwrap_or_default();
    if !status.is_success() {
        bail!("worker /post returned {status}: {body}");
    }
    serde_json::from_str(&body).with_context(|| format!("parse /post response: {body}"))
}

pub async fn list_accounts(http: &reqwest::Client, base: &str) -> Result<Vec<AccountInfo>> {
    let url = format!("{}/accounts", base.trim_end_matches('/'));
    let resp = http.get(&url).send().await.with_context(|| format!("GET {url}"))?;
    if !resp.status().is_success() {
        bail!("worker /accounts returned {}", resp.status());
    }
    Ok(resp.json().await?)
}

pub async fn healthz(http: &reqwest::Client, base: &str) -> Result<HealthResponse> {
    let url = format!("{}/healthz", base.trim_end_matches('/'));
    Ok(http.get(&url).send().await?.error_for_status()?.json().await?)
}
