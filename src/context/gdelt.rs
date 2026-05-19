use anyhow::Context as _;
use serde::Deserialize;
use tracing::info;

use super::{ContextFetcher, TrendEntry};

/// GDELT GKG (Global Knowledge Graph) 15-minute feed poller.
/// Uses the GDELT DOC 2.0 API to fetch trending themes/topics.
pub struct GdeltFetcher {
    http: reqwest::Client,
}

/// GDELT DOC 2.0 API response structure.
#[derive(Debug, Deserialize)]
struct GdeltResponse {
    #[serde(default)]
    articles: Vec<GdeltArticle>,
}

#[derive(Debug, Deserialize)]
struct GdeltArticle {
    #[serde(default)]
    title: String,
    #[serde(default)]
    url: String,
    #[serde(default)]
    socialimage: String,
    #[serde(default)]
    domain: String,
    #[serde(default)]
    seendate: String,
}

const GDELT_API_URL: &str = "https://api.gdeltproject.org/api/v2/doc/doc?query=trending&mode=ArtList&maxrecords=30&format=json&sort=DateDesc&timespan=1h";

impl GdeltFetcher {
    pub fn new() -> Self {
        Self {
            http: reqwest::Client::new(),
        }
    }
}

impl ContextFetcher for GdeltFetcher {
    async fn fetch(&self) -> anyhow::Result<Vec<TrendEntry>> {
        info!("fetching GDELT trending articles");
        let resp = self
            .http
            .get(GDELT_API_URL)
            .timeout(std::time::Duration::from_secs(15))
            .send()
            .await
            .context("GDELT API request")?;

        let body = resp.text().await.context("GDELT response body")?;

        // GDELT may return empty or malformed JSON; handle gracefully.
        let parsed: GdeltResponse = match serde_json::from_str(&body) {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(error = %e, "GDELT response parse failed, returning empty");
                return Ok(Vec::new());
            }
        };

        let entries: Vec<TrendEntry> = parsed
            .articles
            .into_iter()
            .enumerate()
            .map(|(i, article)| {
                let topic_id = if article.url.is_empty() {
                    format!("gdelt_{i}")
                } else {
                    article.url.clone()
                };
                TrendEntry {
                    source: "gdelt".into(),
                    topic_id,
                    label: article.title,
                    // Score decays by position (top = 1.0, bottom ≈ 0.0).
                    score: 1.0 - (i as f64 / 30.0),
                }
            })
            .collect();

        info!(count = entries.len(), "GDELT articles fetched");
        Ok(entries)
    }

    fn source_name(&self) -> &'static str {
        "gdelt"
    }
}
