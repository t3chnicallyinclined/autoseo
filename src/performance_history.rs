//! Builds a performance-history context block for the ranker prompt.
//!
//! Queries the analytics table for past clip performance (CTR, views,
//! watch_pct), correlates with clip features (hook, score), and formats
//! the top-performing and worst-performing examples into a text block
//! that replaces the `{{performance_history}}` placeholder in the ranker
//! user prompt template.

use crate::storage::{ClipPerformanceRow, Storage};

/// How many top / worst examples to include in each section.
const TOP_N: usize = 5;
const WORST_N: usize = 3;

/// Build the `{{performance_history}}` replacement string from the DB.
/// Returns an empty string when there is no analytics data, making the
/// feature backward-compatible with fresh databases.
pub async fn build_performance_history(storage: &Storage) -> String {
    // Fetch enough rows to get both top and worst examples.
    // We fetch top by CTR desc, then reverse to get worst.
    let top = match storage
        .get_clip_performance_history(TOP_N + WORST_N + 10)
        .await
    {
        Ok(rows) => rows,
        Err(e) => {
            tracing::warn!(error = ?e, "failed to fetch performance history; proceeding without it");
            return String::new();
        }
    };

    if top.is_empty() {
        return String::new();
    }

    let best: Vec<&ClipPerformanceRow> = top.iter().take(TOP_N).collect();
    // Worst = lowest CTR from the fetched set.
    let worst: Vec<&ClipPerformanceRow> = top.iter().rev().take(WORST_N).collect();

    let mut out = String::new();
    out.push_str(
        "\n--- HISTORICAL PERFORMANCE DATA ---\n\
         Below are real metrics from previously posted clips. Use these to \
         calibrate your scoring — clips similar to the top performers should \
         score higher; patterns matching the worst performers should score lower.\n\n",
    );

    out.push_str("TOP PERFORMERS (highest CTR):\n");
    for (i, row) in best.iter().enumerate() {
        out.push_str(&format_row(i + 1, row));
    }

    out.push_str("\nWORST PERFORMERS (lowest CTR):\n");
    for (i, row) in worst.iter().enumerate() {
        out.push_str(&format_row(i + 1, row));
    }

    out.push_str("--- END HISTORICAL DATA ---\n");
    out
}

fn format_row(idx: usize, row: &ClipPerformanceRow) -> String {
    let hook = row.hook.as_deref().unwrap_or("(no hook)");
    let score = row
        .score
        .map(|s| format!("{s:.0}"))
        .unwrap_or_else(|| "?".into());
    let views = row
        .views
        .map(|v| v.to_string())
        .unwrap_or_else(|| "?".into());
    let ctr = row
        .ctr
        .map(|c| format!("{c:.2}%"))
        .unwrap_or_else(|| "?".into());
    let watch = row
        .watch_pct
        .map(|w| format!("{w:.1}%"))
        .unwrap_or_else(|| "?".into());
    let dur_s = (row.end_ms - row.start_ms) as f64 / 1000.0;

    format!(
        "  {idx}. hook=\"{hook}\" | ranker_score={score} | duration={dur_s:.0}s | \
         views={views} | CTR={ctr} | watch_pct={watch}\n",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_row_handles_missing_fields() {
        let row = ClipPerformanceRow {
            clip_id: "c1".into(),
            hook: None,
            score: None,
            rank: None,
            views: None,
            ctr: Some(3.45),
            watch_pct: None,
            start_ms: 10000,
            end_ms: 70000,
        };
        let s = format_row(1, &row);
        assert!(s.contains("(no hook)"));
        assert!(s.contains("ranker_score=?"));
        assert!(s.contains("CTR=3.45%"));
        assert!(s.contains("duration=60s"));
    }

    #[test]
    fn format_row_with_all_fields() {
        let row = ClipPerformanceRow {
            clip_id: "c2".into(),
            hook: Some("epic punchline".into()),
            score: Some(92.0),
            rank: Some(1),
            views: Some(15000),
            ctr: Some(8.2),
            watch_pct: Some(74.5),
            start_ms: 0,
            end_ms: 45000,
        };
        let s = format_row(1, &row);
        assert!(s.contains("epic punchline"));
        assert!(s.contains("ranker_score=92"));
        assert!(s.contains("views=15000"));
        assert!(s.contains("CTR=8.20%"));
        assert!(s.contains("watch_pct=74.5%"));
    }

    #[tokio::test]
    async fn empty_analytics_returns_empty_string() {
        let storage = Storage::open_in_memory_sync();
        let result = build_performance_history(&storage).await;
        assert!(
            result.is_empty(),
            "expected empty string for empty analytics table"
        );
    }

    #[tokio::test]
    async fn history_with_data_returns_formatted_block() {
        let storage = Storage::open_in_memory_sync();

        // Insert a job, clips, and analytics rows.
        {
            let conn = storage.conn_for_test();
            let conn = conn.lock().await;
            conn.execute(
                "INSERT INTO jobs (id, status, created_at, updated_at) VALUES ('j1', 'done', 0, 0)",
                [],
            )
            .unwrap();
            for i in 1..=8 {
                let clip_id = format!("c{i}");
                conn.execute(
                    "INSERT INTO clips (id, job_id, start_ms, end_ms, rank, score, hook) \
                     VALUES (?1, 'j1', ?2, ?3, ?4, ?5, ?6)",
                    rusqlite::params![
                        clip_id,
                        i * 10000,
                        i * 10000 + 60000,
                        i,
                        50.0 + (i as f64) * 5.0,
                        format!("hook_{i}"),
                    ],
                )
                .unwrap();
                conn.execute(
                    "INSERT INTO analytics (clip_id, platform, fetched_at, views, ctr, watch_pct) \
                     VALUES (?1, 'youtube', 1000, ?2, ?3, ?4)",
                    rusqlite::params![
                        clip_id,
                        i * 1000,
                        (i as f64) * 1.1,
                        50.0 + (i as f64) * 3.0,
                    ],
                ).unwrap();
            }
        }

        let result = build_performance_history(&storage).await;
        assert!(result.contains("TOP PERFORMERS"));
        assert!(result.contains("WORST PERFORMERS"));
        assert!(result.contains("hook_8")); // highest CTR
        assert!(result.contains("hook_1")); // lowest CTR
    }
}
