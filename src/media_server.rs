//! HTTP media server: serves rendered clips, thumbnails, and manifests from the
//! work directory. Designed to run alongside the main pipeline as a background
//! task when `API_PORT` is set.
//!
//! Endpoints:
//! - GET /api/media/video/:job_id/:clip_id/:variant — stream video (Range support)
//! - GET /api/media/thumb/:job_id/:clip_id          — serve thumbnail JPEG
//! - GET /api/media/manifest/:job_id                — serve manifest.json

use axum::{
    Router,
    extract::{Path, State},
    http::{HeaderMap, StatusCode, header},
    response::{IntoResponse, Response},
    routing::get,
};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::io::AsyncSeekExt;

#[derive(Clone)]
pub struct MediaServerState {
    pub work_dir: Arc<PathBuf>,
}

/// Build the axum router for the media server.
pub fn router(work_dir: PathBuf) -> Router {
    let state = MediaServerState {
        work_dir: Arc::new(work_dir),
    };
    Router::new()
        .route(
            "/api/media/video/{job_id}/{clip_id}/{variant}",
            get(serve_video),
        )
        .route("/api/media/thumb/{job_id}/{clip_id}", get(serve_thumb))
        .route("/api/media/manifest/{job_id}", get(serve_manifest))
        .with_state(state)
}

/// Validate a path segment: must be non-empty, contain only safe characters,
/// and not contain path traversal sequences.
fn is_safe_segment(s: &str) -> bool {
    if s.is_empty() || s == "." || s == ".." {
        return false;
    }
    s.chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.')
        && !s.contains("..")
}

/// Resolve a job directory from a job_id. Jobs live under:
///   work_dir/clipper/{sanitized_name}/{timestamp}/clips/
/// The job_id is the `{sanitized_name}/{timestamp}` portion.
///
/// We also search flat job dirs like `work_dir/clipper/{job_id}/clips/`.
fn resolve_job_clips_dir(work_dir: &std::path::Path, job_id: &str) -> Option<PathBuf> {
    // job_id could be a direct timestamp directory name under clipper/*/
    // We search for it under work_dir/clipper/
    let clipper_dir = work_dir.join("clipper");

    // Try: clipper/{job_id}/clips/
    let direct = clipper_dir.join(job_id).join("clips");
    if direct.is_dir() {
        return Some(direct);
    }

    // Try: clipper/*/{job_id}/clips/ (job_id is a timestamp inside a named show dir)
    if let Ok(entries) = std::fs::read_dir(&clipper_dir) {
        for entry in entries.flatten() {
            if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                let candidate = entry.path().join(job_id).join("clips");
                if candidate.is_dir() {
                    return Some(candidate);
                }
            }
        }
    }
    None
}

/// Serve a video clip file with Range header support for seeking.
async fn serve_video(
    State(state): State<MediaServerState>,
    Path((job_id, clip_id, variant)): Path<(String, String, String)>,
    headers: HeaderMap,
) -> Result<Response, StatusCode> {
    if !is_safe_segment(&job_id) || !is_safe_segment(&clip_id) || !is_safe_segment(&variant) {
        return Err(StatusCode::BAD_REQUEST);
    }

    let clips_dir = resolve_job_clips_dir(&state.work_dir, &job_id).ok_or(StatusCode::NOT_FOUND)?;

    // Find the video file: clip_{clip_id}_*_{variant}.mp4
    let filename = find_clip_file(&clips_dir, &clip_id, &variant).ok_or(StatusCode::NOT_FOUND)?;

    let path = clips_dir.join(&filename);
    if !path.is_file() {
        return Err(StatusCode::NOT_FOUND);
    }

    // Verify the resolved path is still under clips_dir (defense in depth)
    let canonical = path.canonicalize().map_err(|_| StatusCode::NOT_FOUND)?;
    let canonical_clips = clips_dir
        .canonicalize()
        .map_err(|_| StatusCode::NOT_FOUND)?;
    if !canonical.starts_with(&canonical_clips) {
        return Err(StatusCode::FORBIDDEN);
    }

    serve_file_with_range(&canonical, "video/mp4", &headers).await
}

/// Serve a thumbnail JPEG.
async fn serve_thumb(
    State(state): State<MediaServerState>,
    Path((job_id, clip_id)): Path<(String, String)>,
    _headers: HeaderMap,
) -> Result<Response, StatusCode> {
    if !is_safe_segment(&job_id) || !is_safe_segment(&clip_id) {
        return Err(StatusCode::BAD_REQUEST);
    }

    // Thumbnails live under work_dir/clipper/{show}/{ts}/thumbnails/
    // or directly work_dir/local/{show}/{ts}/thumbnails/
    let thumb_path =
        find_thumbnail(&state.work_dir, &job_id, &clip_id).ok_or(StatusCode::NOT_FOUND)?;

    let canonical = thumb_path
        .canonicalize()
        .map_err(|_| StatusCode::NOT_FOUND)?;
    let canonical_work = state
        .work_dir
        .canonicalize()
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    if !canonical.starts_with(&canonical_work) {
        return Err(StatusCode::FORBIDDEN);
    }

    let body = tokio::fs::read(&canonical)
        .await
        .map_err(|_| StatusCode::NOT_FOUND)?;
    Ok((
        StatusCode::OK,
        [
            (header::CONTENT_TYPE, "image/jpeg"),
            (header::CACHE_CONTROL, "public, max-age=86400"),
        ],
        body,
    )
        .into_response())
}

/// Serve manifest.json for a job.
async fn serve_manifest(
    State(state): State<MediaServerState>,
    Path(job_id): Path<String>,
) -> Result<Response, StatusCode> {
    if !is_safe_segment(&job_id) {
        return Err(StatusCode::BAD_REQUEST);
    }

    let clips_dir = resolve_job_clips_dir(&state.work_dir, &job_id).ok_or(StatusCode::NOT_FOUND)?;

    let manifest_path = clips_dir.join("manifest.json");
    if !manifest_path.is_file() {
        return Err(StatusCode::NOT_FOUND);
    }

    let canonical = manifest_path
        .canonicalize()
        .map_err(|_| StatusCode::NOT_FOUND)?;
    let canonical_clips = clips_dir
        .canonicalize()
        .map_err(|_| StatusCode::NOT_FOUND)?;
    if !canonical.starts_with(&canonical_clips) {
        return Err(StatusCode::FORBIDDEN);
    }

    let body = tokio::fs::read(&canonical)
        .await
        .map_err(|_| StatusCode::NOT_FOUND)?;
    Ok((
        StatusCode::OK,
        [
            (header::CONTENT_TYPE, "application/json"),
            (header::CACHE_CONTROL, "no-cache"),
        ],
        body,
    )
        .into_response())
}

/// Serve a file with HTTP Range header support (for video seeking).
async fn serve_file_with_range(
    path: &std::path::Path,
    content_type: &str,
    headers: &HeaderMap,
) -> Result<Response, StatusCode> {
    let metadata = tokio::fs::metadata(path)
        .await
        .map_err(|_| StatusCode::NOT_FOUND)?;
    let file_size = metadata.len();

    let range_header = headers.get(header::RANGE).and_then(|v| v.to_str().ok());

    match range_header {
        Some(range_str) => {
            let (start, end) =
                parse_range(range_str, file_size).ok_or(StatusCode::RANGE_NOT_SATISFIABLE)?;
            let length = end - start + 1;

            let mut file = tokio::fs::File::open(path)
                .await
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
            file.seek(std::io::SeekFrom::Start(start))
                .await
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

            let limited = tokio::io::AsyncReadExt::take(file, length);
            let stream = tokio_util::io::ReaderStream::new(limited);
            let body = axum::body::Body::from_stream(stream);

            Ok(Response::builder()
                .status(StatusCode::PARTIAL_CONTENT)
                .header(header::CONTENT_TYPE, content_type)
                .header(header::CONTENT_LENGTH, length.to_string())
                .header(
                    header::CONTENT_RANGE,
                    format!("bytes {start}-{end}/{file_size}"),
                )
                .header(header::ACCEPT_RANGES, "bytes")
                .header(header::CACHE_CONTROL, "public, max-age=86400")
                .body(body)
                .unwrap())
        }
        None => {
            let file = tokio::fs::File::open(path)
                .await
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
            let stream = tokio_util::io::ReaderStream::new(file);
            let body = axum::body::Body::from_stream(stream);

            Ok(Response::builder()
                .status(StatusCode::OK)
                .header(header::CONTENT_TYPE, content_type)
                .header(header::CONTENT_LENGTH, file_size.to_string())
                .header(header::ACCEPT_RANGES, "bytes")
                .header(header::CACHE_CONTROL, "public, max-age=86400")
                .body(body)
                .unwrap())
        }
    }
}

/// Parse an HTTP Range header value like `bytes=0-1023` or `bytes=1024-`.
fn parse_range(range_str: &str, file_size: u64) -> Option<(u64, u64)> {
    let range_str = range_str.strip_prefix("bytes=")?;
    let mut parts = range_str.splitn(2, '-');
    let start_str = parts.next()?.trim();
    let end_str = parts.next()?.trim();

    if start_str.is_empty() {
        // Suffix range: bytes=-500 means last 500 bytes
        let suffix_len: u64 = end_str.parse().ok()?;
        if suffix_len == 0 || suffix_len > file_size {
            return None;
        }
        let start = file_size - suffix_len;
        Some((start, file_size - 1))
    } else {
        let start: u64 = start_str.parse().ok()?;
        let end = if end_str.is_empty() {
            file_size - 1
        } else {
            end_str.parse().ok()?
        };
        if start > end || start >= file_size {
            return None;
        }
        let end = end.min(file_size - 1);
        Some((start, end))
    }
}

/// Find a clip video file in the clips directory matching the clip_id and variant.
/// Clips are named like `clip_01_00m30s-01m15s_9x16.mp4`.
fn find_clip_file(clips_dir: &std::path::Path, clip_id: &str, variant: &str) -> Option<String> {
    let prefix = format!("clip_{clip_id}_");
    let suffix = format!("_{variant}.mp4");
    let entries = std::fs::read_dir(clips_dir).ok()?;
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with(&prefix) && name.ends_with(&suffix) {
            return Some(name);
        }
    }
    // Also try exact filename match
    let exact = format!("{prefix}{suffix}");
    let exact_trimmed = exact.replace("__", "_");
    for candidate in [&exact, &exact_trimmed] {
        if clips_dir.join(candidate).is_file() {
            return Some(candidate.clone());
        }
    }
    None
}

/// Find a thumbnail for a given job and clip identifier.
fn find_thumbnail(work_dir: &std::path::Path, job_id: &str, clip_id: &str) -> Option<PathBuf> {
    // Search under clipper/{*}/{job_id}/thumbnails/ and clipper/{job_id}/thumbnails/
    let clipper_dir = work_dir.join("clipper");

    let search_dirs = |thumb_dir: PathBuf| -> Option<PathBuf> {
        if !thumb_dir.is_dir() {
            return None;
        }
        // Try exact match first
        let exact = thumb_dir.join(format!("{clip_id}.jpg"));
        if exact.is_file() {
            return Some(exact);
        }
        // Try prefix match (e.g., thumb_01.jpg matching clip_id "01")
        let entries = std::fs::read_dir(&thumb_dir).ok()?;
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.contains(clip_id) && (name.ends_with(".jpg") || name.ends_with(".jpeg")) {
                return Some(entry.path());
            }
        }
        None
    };

    // Direct: clipper/{job_id}/thumbnails/
    let result = search_dirs(clipper_dir.join(job_id).join("thumbnails"));
    if result.is_some() {
        return result;
    }

    // Nested: clipper/*/{job_id}/thumbnails/
    if let Ok(entries) = std::fs::read_dir(&clipper_dir) {
        for entry in entries.flatten() {
            if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                let result = search_dirs(entry.path().join(job_id).join("thumbnails"));
                if result.is_some() {
                    return result;
                }
            }
        }
    }

    // Also check under local/
    let local_dir = work_dir.join("local");
    if let Ok(entries) = std::fs::read_dir(&local_dir) {
        for entry in entries.flatten() {
            if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                let result = search_dirs(entry.path().join(job_id).join("thumbnails"));
                if result.is_some() {
                    return result;
                }
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safe_segment_rejects_traversal() {
        assert!(!is_safe_segment(".."));
        assert!(!is_safe_segment("."));
        assert!(!is_safe_segment(""));
        assert!(!is_safe_segment("foo/../bar"));
        assert!(!is_safe_segment("foo/bar"));
        assert!(!is_safe_segment("foo\\bar"));
    }

    #[test]
    fn safe_segment_accepts_valid_names() {
        assert!(is_safe_segment("1716134400"));
        assert!(is_safe_segment("clip_01"));
        assert!(is_safe_segment("9x16"));
        assert!(is_safe_segment("my-show_name.v2"));
    }

    #[test]
    fn parse_range_full() {
        assert_eq!(parse_range("bytes=0-999", 1000), Some((0, 999)));
    }

    #[test]
    fn parse_range_open_end() {
        assert_eq!(parse_range("bytes=500-", 1000), Some((500, 999)));
    }

    #[test]
    fn parse_range_suffix() {
        assert_eq!(parse_range("bytes=-200", 1000), Some((800, 999)));
    }

    #[test]
    fn parse_range_invalid() {
        assert_eq!(parse_range("bytes=1000-", 1000), None);
        assert_eq!(parse_range("bytes=500-400", 1000), None);
        assert_eq!(parse_range("bytes=-0", 1000), None);
    }

    #[test]
    fn parse_range_clamps_end() {
        // End beyond file size should be clamped
        assert_eq!(parse_range("bytes=0-9999", 1000), Some((0, 999)));
    }

    #[tokio::test]
    async fn serves_manifest_from_test_dir() {
        let dir = tempfile::tempdir().unwrap();
        let clips = dir.path().join("clipper").join("testjob").join("clips");
        std::fs::create_dir_all(&clips).unwrap();
        std::fs::write(clips.join("manifest.json"), r#"{"schema_version":1}"#).unwrap();

        let app = router(dir.path().to_path_buf());

        let req = axum::http::Request::builder()
            .uri("/api/media/manifest/testjob")
            .body(axum::body::Body::empty())
            .unwrap();

        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers().get("content-type").unwrap(),
            "application/json"
        );

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(parsed["schema_version"], 1);
    }

    #[tokio::test]
    async fn returns_404_for_missing_manifest() {
        let dir = tempfile::tempdir().unwrap();
        let app = router(dir.path().to_path_buf());

        let req = axum::http::Request::builder()
            .uri("/api/media/manifest/nonexistent")
            .body(axum::body::Body::empty())
            .unwrap();

        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn rejects_path_traversal_in_job_id() {
        let dir = tempfile::tempdir().unwrap();
        let app = router(dir.path().to_path_buf());

        let req = axum::http::Request::builder()
            .uri("/api/media/manifest/..%2F..%2Fetc")
            .body(axum::body::Body::empty())
            .unwrap();

        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert!(resp.status() == StatusCode::BAD_REQUEST || resp.status() == StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn serves_video_with_range() {
        let dir = tempfile::tempdir().unwrap();
        let clips = dir.path().join("clipper").join("testjob").join("clips");
        std::fs::create_dir_all(&clips).unwrap();

        // Create a fake video file (1000 bytes of zeros)
        let video_data = vec![0u8; 1000];
        std::fs::write(clips.join("clip_01_00m30s-01m15s_9x16.mp4"), &video_data).unwrap();

        let app = router(dir.path().to_path_buf());

        // Full request
        let req = axum::http::Request::builder()
            .uri("/api/media/video/testjob/01/9x16")
            .body(axum::body::Body::empty())
            .unwrap();

        let resp = tower::ServiceExt::oneshot(app.clone(), req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(resp.headers().get("content-type").unwrap(), "video/mp4");
        assert_eq!(resp.headers().get("accept-ranges").unwrap(), "bytes");
        assert_eq!(resp.headers().get("content-length").unwrap(), "1000");

        // Range request
        let req = axum::http::Request::builder()
            .uri("/api/media/video/testjob/01/9x16")
            .header("range", "bytes=0-499")
            .body(axum::body::Body::empty())
            .unwrap();

        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::PARTIAL_CONTENT);
        assert_eq!(resp.headers().get("content-length").unwrap(), "500");
        assert_eq!(
            resp.headers().get("content-range").unwrap(),
            "bytes 0-499/1000"
        );
    }

    #[tokio::test]
    async fn serves_thumbnail() {
        let dir = tempfile::tempdir().unwrap();
        let thumbs = dir
            .path()
            .join("clipper")
            .join("testjob")
            .join("thumbnails");
        std::fs::create_dir_all(&thumbs).unwrap();

        // Create a fake JPEG (just some bytes with JPEG header)
        let mut jpeg_data = vec![0xFF, 0xD8, 0xFF, 0xE0]; // JPEG magic bytes
        jpeg_data.extend_from_slice(&[0u8; 100]);
        std::fs::write(thumbs.join("thumb_01.jpg"), &jpeg_data).unwrap();

        let app = router(dir.path().to_path_buf());

        let req = axum::http::Request::builder()
            .uri("/api/media/thumb/testjob/01")
            .body(axum::body::Body::empty())
            .unwrap();

        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(resp.headers().get("content-type").unwrap(), "image/jpeg");
    }
}
