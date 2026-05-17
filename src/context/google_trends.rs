use anyhow::Context as _;
use serde::Deserialize;
use tracing::info;

use super::{ContextFetcher, TrendEntry};

/// Google Trends daily trending searches poller.
/// Uses the unofficial RSS-to-JSON endpoint for US daily trends.
pub struct GoogleTrendsFetcher {
    http: reqwest::Client,
    /// ISO country code (default "US").
    geo: String,
}

/// Google Trends daily RSS converted to JSON.
#[derive(Debug, Deserialize)]
struct TrendingResponse {
    #[serde(default, rename = "storySummaries")]
    story_summaries: Option<StorySummaries>,
    #[serde(default, rename = "featuredStoryIds")]
    featured_story_ids: Vec<serde_json::Value>,
    // Fallback: daily_searches feed format
    #[serde(default)]
    default: Option<DailySearches>,
}

#[derive(Debug, Deserialize)]
struct StorySummaries {
    #[serde(default, rename = "trendingStories")]
    trending_stories: Vec<TrendingStory>,
}

#[derive(Debug, Deserialize)]
struct TrendingStory {
    #[serde(default)]
    title: String,
    #[serde(default, rename = "entityNames")]
    entity_names: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct DailySearches {
    #[serde(default, rename = "trendingSearchesDays")]
    trending_searches_days: Vec<TrendingDay>,
}

#[derive(Debug, Deserialize)]
struct TrendingDay {
    #[serde(default, rename = "trendingSearches")]
    trending_searches: Vec<DailyTrend>,
}

#[derive(Debug, Deserialize)]
struct DailyTrend {
    title: DailyTrendTitle,
    #[serde(default, rename = "formattedTraffic")]
    formatted_traffic: String,
}

#[derive(Debug, Deserialize)]
struct DailyTrendTitle {
    #[serde(default)]
    query: String,
}

const TRENDS_API_URL: &str =
    "https://trends.google.com/trends/api/dailytrends?hl=en-US&tz=-300&ns=15";

impl GoogleTrendsFetcher {
    pub fn new() -> Self {
        Self::with_geo("US".to_string())
    }

    pub fn with_geo(geo: String) -> Self {
        Self {
            http: reqwest::Client::new(),
            geo,
        }
    }
}

impl ContextFetcher for GoogleTrendsFetcher {
    async fn fetch(&self) -> anyhow::Result<Vec<TrendEntry>> {
        info!(geo = %self.geo, "fetching Google Trends daily searches");

        let url = format!("{}&geo={}", TRENDS_API_URL, self.geo);
        let resp = self
            .http
            .get(&url)
            .timeout(std::time::Duration::from_secs(15))
            .send()
            .await
            .context("Google Trends API request")?;

        let body = resp.text().await.context("Google Trends response body")?;

        // Google Trends API prepends ")]}'" to the JSON response.
        let json_str = body
            .strip_prefix(")]}'")
            .map(|s| s.trim_start())
            .unwrap_or(&body);

        let parsed: TrendingResponse = match serde_json::from_str(json_str) {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(error = %e, "Google Trends parse failed, returning empty");
                return Ok(Vec::new());
            }
        };

        let mut entries = Vec::new();

        // Try daily searches format first.
        if let Some(daily) = parsed.default {
            for day in daily.trending_searches_days {
                for (i, trend) in day.trending_searches.into_iter().enumerate() {
                    entries.push(TrendEntry {
                        source: "google".into(),
                        topic_id: format!("google_{}", trend.title.query),
                        label: trend.title.query,
                        score: 1.0 - (i as f64 / 20.0).min(0.95),
                    });
                }
            }
        }

        // Also try story summaries format.
        if let Some(stories) = parsed.story_summaries {
            for (i, story) in stories.trending_stories.into_iter().enumerate() {
                let label = if story.title.is_empty() {
                    story.entity_names.join(", ")
                } else {
                    story.title
                };
                if !label.is_empty() {
                    entries.push(TrendEntry {
                        source: "google".into(),
                        topic_id: format!("google_story_{i}"),
                        label,
                        score: 1.0 - (i as f64 / 20.0).min(0.95),
                    });
                }
            }
        }

        info!(count = entries.len(), "Google Trends fetched");
        Ok(entries)
    }

    fn source_name(&self) -> &'static str {
        "google"
    }
}
