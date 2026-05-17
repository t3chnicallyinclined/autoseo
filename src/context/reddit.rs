use anyhow::Context as _;
use serde::Deserialize;
use tracing::info;

use super::{ContextFetcher, TrendEntry};

/// Reddit r/all hot poller — fetches the current top posts
/// from r/all via Reddit's public JSON API.
pub struct RedditFetcher {
    http: reqwest::Client,
}

#[derive(Debug, Deserialize)]
struct RedditListing {
    data: RedditListingData,
}

#[derive(Debug, Deserialize)]
struct RedditListingData {
    children: Vec<RedditChild>,
}

#[derive(Debug, Deserialize)]
struct RedditChild {
    data: RedditPost,
}

#[derive(Debug, Deserialize)]
struct RedditPost {
    #[serde(default)]
    id: String,
    #[serde(default)]
    title: String,
    #[serde(default)]
    subreddit: String,
    #[serde(default)]
    score: i64,
    #[serde(default)]
    ups: i64,
}

const REDDIT_HOT_URL: &str = "https://www.reddit.com/r/all/hot.json?limit=25";

impl RedditFetcher {
    pub fn new() -> Self {
        Self {
            http: reqwest::Client::builder()
                // Reddit blocks default reqwest UA.
                .user_agent("autoseo-clipper/0.1 (trend-context)")
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
        }
    }
}

impl ContextFetcher for RedditFetcher {
    async fn fetch(&self) -> anyhow::Result<Vec<TrendEntry>> {
        info!("fetching Reddit r/all hot posts");
        let resp = self
            .http
            .get(REDDIT_HOT_URL)
            .timeout(std::time::Duration::from_secs(15))
            .send()
            .await
            .context("Reddit API request")?;

        let listing: RedditListing = resp.json().await.context("Reddit response parse")?;

        let max_score = listing
            .data
            .children
            .iter()
            .map(|c| c.data.score)
            .max()
            .unwrap_or(1)
            .max(1) as f64;

        let entries: Vec<TrendEntry> = listing
            .data
            .children
            .into_iter()
            .map(|child| {
                let post = child.data;
                TrendEntry {
                    source: "reddit".into(),
                    topic_id: format!("reddit_{}", post.id),
                    label: format!("r/{}: {}", post.subreddit, post.title),
                    score: (post.score as f64 / max_score).clamp(0.0, 1.0),
                }
            })
            .collect();

        info!(count = entries.len(), "Reddit posts fetched");
        Ok(entries)
    }

    fn source_name(&self) -> &'static str {
        "reddit"
    }
}
