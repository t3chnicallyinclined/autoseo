pub mod cache;
pub mod gdelt;
pub mod google_trends;
pub mod reddit;

use serde::{Deserialize, Serialize};

/// A single trending topic from any source.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendEntry {
    /// Source identifier: "gdelt", "reddit", "google".
    pub source: String,
    /// Unique topic ID within the source.
    pub topic_id: String,
    /// Human-readable label / headline.
    pub label: String,
    /// Relevance score (0.0–1.0, higher = more trending).
    pub score: f64,
}

/// Trait that all trend pollers implement.
pub trait ContextFetcher: Send + Sync {
    /// Fetch current trends from this source.
    fn fetch(&self) -> impl std::future::Future<Output = anyhow::Result<Vec<TrendEntry>>> + Send;

    /// Source name used as the `source` column in the trends table.
    fn source_name(&self) -> &'static str;
}
