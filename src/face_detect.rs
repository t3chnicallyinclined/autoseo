//! SCRFD face detection via ONNX Runtime.
//!
//! Loads the `scrfd_10g_bnkps.onnx` model, runs inference on video frames
//! extracted via ffmpeg, and returns face bounding boxes with landmarks.
//! Includes a simple IoU-based tracker for frame-to-frame identity persistence.
//!
//! Model auto-downloads to `{WORK_DIR}/models/scrfd/` on first use.

use anyhow::{Context, Result};
use ndarray::Array4;
use ort::session::Session;
use std::path::Path;
use tokio::process::Command;

/// A single detected face in a frame.
#[derive(Debug, Clone)]
pub struct FaceDetection {
    /// Bounding box: (x1, y1, x2, y2) in pixel coordinates of the original frame.
    pub bbox: [f32; 4],
    /// Detection confidence in [0, 1].
    pub confidence: f32,
    /// Five facial landmarks: left eye, right eye, nose, left mouth, right mouth.
    /// Each is (x, y) in pixel coordinates. Empty if model doesn't output landmarks.
    pub landmarks: Vec<[f32; 2]>,
}

/// A tracked face across multiple frames.
#[derive(Debug, Clone)]
pub struct TrackedFace {
    pub id: u32,
    pub detection: FaceDetection,
}

/// SCRFD face detector wrapping an ORT session.
pub struct ScrfdDetector {
    session: Session,
    input_size: u32,
    conf_threshold: f32,
    nms_threshold: f32,
}

/// SCRFD model strides and corresponding anchor counts.
const STRIDES: [u32; 3] = [8, 16, 32];
const ANCHORS_PER_CELL: usize = 2;

impl ScrfdDetector {
    /// Load the SCRFD model from disk. If the model is missing and `model_url`
    /// is provided, download it first.
    pub async fn load(
        models_dir: &Path,
        model_url: Option<&str>,
        input_size: u32,
        conf_threshold: f32,
        nms_threshold: f32,
    ) -> Result<Self> {
        let scrfd_dir = models_dir.join("scrfd");
        let model_path = scrfd_dir.join("scrfd_10g_bnkps.onnx");

        if !model_path.exists() {
            if let Some(url) = model_url {
                tracing::info!(url, path = %model_path.display(), "SCRFD model not found; downloading");
                download_model(url, &model_path).await?;
            } else {
                anyhow::bail!(
                    "SCRFD model not found at {} and no download URL configured",
                    model_path.display()
                );
            }
        }

        let canonical = model_path
            .canonicalize()
            .with_context(|| format!("canonicalize SCRFD model path: {}", model_path.display()))?;
        let canonical_str = canonical
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("SCRFD model path is not valid UTF-8"))?;

        let session = Session::builder()
            .context("create ORT session builder")?
            .with_intra_threads(1)
            .context("set intra threads")?
            .commit_from_file(canonical_str)
            .with_context(|| format!("load SCRFD model from {canonical_str}"))?;

        Ok(Self {
            session,
            input_size,
            conf_threshold,
            nms_threshold,
        })
    }

    /// Detect faces in a single RGB frame.
    ///
    /// `frame_rgb` is raw RGB bytes in row-major order, `width` x `height` pixels.
    pub fn detect(&self, frame_rgb: &[u8], width: u32, height: u32) -> Result<Vec<FaceDetection>> {
        let sz = self.input_size as usize;
        let expected_len = (width * height * 3) as usize;
        anyhow::ensure!(
            frame_rgb.len() == expected_len,
            "frame_rgb length {} != expected {} ({}x{}x3)",
            frame_rgb.len(),
            expected_len,
            width,
            height
        );

        // Resize and normalize to model input: [1, 3, input_size, input_size]
        let rr = resize_rgb(frame_rgb, width, height, sz);
        let input = normalize_to_nchw(&rr.rgb, sz);

        let outputs = self.session.run(ort::inputs![
            "input.1" => input,
        ]?)?;

        let mut detections = Vec::new();

        // SCRFD 10g_bnkps outputs 9 tensors: for each of 3 strides,
        // score_stride, bbox_stride, kps_stride (in stride order: 8, 16, 32).
        for (stride_idx, &stride) in STRIDES.iter().enumerate() {
            let feat_h = sz / stride as usize;
            let feat_w = sz / stride as usize;

            let score_key = format!("score_{stride}");
            let bbox_key = format!("bbox_{stride}");
            let kps_key = format!("kps_{stride}");

            let scores = match outputs.get(&*score_key) {
                Some(v) => v
                    .try_extract_tensor::<f32>()
                    .with_context(|| format!("extract {score_key}"))?,
                None => {
                    // Fall back to positional indexing
                    outputs[stride_idx * 3]
                        .try_extract_tensor::<f32>()
                        .with_context(|| {
                            format!("extract score tensor at index {}", stride_idx * 3)
                        })?
                }
            };

            let bboxes = match outputs.get(&*bbox_key) {
                Some(v) => v
                    .try_extract_tensor::<f32>()
                    .with_context(|| format!("extract {bbox_key}"))?,
                None => outputs[stride_idx * 3 + 1]
                    .try_extract_tensor::<f32>()
                    .with_context(|| {
                        format!("extract bbox tensor at index {}", stride_idx * 3 + 1)
                    })?,
            };

            let kps = match outputs.get(&*kps_key) {
                Some(v) => v
                    .try_extract_tensor::<f32>()
                    .with_context(|| format!("extract {kps_key}"))?,
                None => outputs[stride_idx * 3 + 2]
                    .try_extract_tensor::<f32>()
                    .with_context(|| {
                        format!("extract kps tensor at index {}", stride_idx * 3 + 2)
                    })?,
            };

            for row in 0..feat_h {
                for col in 0..feat_w {
                    for anchor in 0..ANCHORS_PER_CELL {
                        let idx = (row * feat_w + col) * ANCHORS_PER_CELL + anchor;
                        let score = scores[[0, idx, 0]];

                        if score <= self.conf_threshold {
                            continue;
                        }

                        let anchor_cx = (col as f32 + 0.5) * stride as f32;
                        let anchor_cy = (row as f32 + 0.5) * stride as f32;

                        // bbox: distance from anchor center to left, top, right, bottom
                        let dl = bboxes[[0, idx, 0]] * stride as f32;
                        let dt = bboxes[[0, idx, 1]] * stride as f32;
                        let dr = bboxes[[0, idx, 2]] * stride as f32;
                        let db = bboxes[[0, idx, 3]] * stride as f32;

                        let x1 = (anchor_cx - dl - rr.pad_x) / rr.scale;
                        let y1 = (anchor_cy - dt - rr.pad_y) / rr.scale;
                        let x2 = (anchor_cx + dr - rr.pad_x) / rr.scale;
                        let y2 = (anchor_cy + db - rr.pad_y) / rr.scale;

                        // Clamp to original frame
                        let x1 = x1.clamp(0.0, width as f32);
                        let y1 = y1.clamp(0.0, height as f32);
                        let x2 = x2.clamp(0.0, width as f32);
                        let y2 = y2.clamp(0.0, height as f32);

                        // Landmarks (5 keypoints)
                        let mut landmarks = Vec::with_capacity(5);
                        for k in 0..5 {
                            let lx = (anchor_cx + kps[[0, idx, k * 2]] * stride as f32 - rr.pad_x)
                                / rr.scale;
                            let ly = (anchor_cy + kps[[0, idx, k * 2 + 1]] * stride as f32
                                - rr.pad_y)
                                / rr.scale;
                            landmarks
                                .push([lx.clamp(0.0, width as f32), ly.clamp(0.0, height as f32)]);
                        }

                        detections.push(FaceDetection {
                            bbox: [x1, y1, x2, y2],
                            confidence: score,
                            landmarks,
                        });
                    }
                }
            }
        }

        // NMS
        let detections = nms(&mut detections, self.nms_threshold);
        Ok(detections)
    }
}

/// Extract a single video frame as raw RGB bytes using ffmpeg.
pub async fn extract_frame_rgb(
    ffmpeg: &str,
    video_path: &Path,
    at_secs: f64,
) -> Result<(Vec<u8>, u32, u32)> {
    let ts = format!("{at_secs:.3}");

    // First, probe the frame dimensions
    let output = Command::new(ffmpeg)
        .args(["-hide_banner", "-loglevel", "error", "-nostats"])
        .args(["-ss", &ts])
        .arg("-i")
        .arg(video_path)
        .args(["-frames:v", "1"])
        .args(["-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
        .output()
        .await
        .with_context(|| {
            format!(
                "ffmpeg extract frame at {ts}s from {}",
                video_path.display()
            )
        })?;

    if !output.status.success() {
        let err = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("ffmpeg frame extraction failed: {err}");
    }

    // Probe dimensions separately
    let probe = Command::new(ffmpeg)
        .args(["-hide_banner", "-loglevel", "error"])
        .arg("-i")
        .arg(video_path)
        .args(["-frames:v", "1"])
        .args(["-vf", "showinfo"])
        .args(["-f", "null", "-"])
        .output()
        .await
        .context("ffmpeg showinfo probe")?;

    let stderr = String::from_utf8_lossy(&probe.stderr);
    let (width, height) = parse_frame_dimensions(&stderr, &output.stdout)?;

    Ok((output.stdout, width, height))
}

/// Extract multiple frames at specified timestamps.
pub async fn extract_frames_rgb(
    ffmpeg: &str,
    video_path: &Path,
    timestamps: &[f64],
) -> Result<Vec<(Vec<u8>, u32, u32)>> {
    let mut results = Vec::with_capacity(timestamps.len());
    for &ts in timestamps {
        let frame = extract_frame_rgb(ffmpeg, video_path, ts).await?;
        results.push(frame);
    }
    Ok(results)
}

/// Simple IoU-based face tracker across consecutive frames.
pub struct FaceTracker {
    next_id: u32,
    prev_faces: Vec<TrackedFace>,
    iou_threshold: f32,
}

impl FaceTracker {
    pub fn new(iou_threshold: f32) -> Self {
        Self {
            next_id: 0,
            prev_faces: Vec::new(),
            iou_threshold,
        }
    }

    /// Update the tracker with new detections and return tracked faces.
    /// Matches new detections to previous faces using IoU; unmatched detections
    /// get new IDs.
    pub fn update(&mut self, detections: &[FaceDetection]) -> Vec<TrackedFace> {
        if self.prev_faces.is_empty() {
            // First frame: assign new IDs to all detections
            let tracked: Vec<TrackedFace> = detections
                .iter()
                .map(|d| {
                    let id = self.next_id;
                    self.next_id += 1;
                    TrackedFace {
                        id,
                        detection: d.clone(),
                    }
                })
                .collect();
            self.prev_faces = tracked.clone();
            return tracked;
        }

        let n_prev = self.prev_faces.len();
        let n_det = detections.len();

        // Build IoU cost matrix
        let mut iou_matrix = vec![vec![0.0f32; n_det]; n_prev];
        for (i, prev) in self.prev_faces.iter().enumerate() {
            for (j, det) in detections.iter().enumerate() {
                iou_matrix[i][j] = compute_iou(&prev.detection.bbox, &det.bbox);
            }
        }

        // Greedy assignment: highest IoU first
        let mut used_prev = vec![false; n_prev];
        let mut used_det = vec![false; n_det];
        let mut tracked = Vec::with_capacity(n_det);

        // Collect all (iou, prev_idx, det_idx) and sort descending
        let mut pairs: Vec<(f32, usize, usize)> = Vec::new();
        for i in 0..n_prev {
            for j in 0..n_det {
                if iou_matrix[i][j] >= self.iou_threshold {
                    pairs.push((iou_matrix[i][j], i, j));
                }
            }
        }
        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        for (_, prev_idx, det_idx) in pairs {
            if used_prev[prev_idx] || used_det[det_idx] {
                continue;
            }
            used_prev[prev_idx] = true;
            used_det[det_idx] = true;
            tracked.push(TrackedFace {
                id: self.prev_faces[prev_idx].id,
                detection: detections[det_idx].clone(),
            });
        }

        // Assign new IDs to unmatched detections
        for (j, det) in detections.iter().enumerate() {
            if !used_det[j] {
                let id = self.next_id;
                self.next_id += 1;
                tracked.push(TrackedFace {
                    id,
                    detection: det.clone(),
                });
            }
        }

        self.prev_faces = tracked.clone();
        tracked
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Compute IoU between two bounding boxes [x1, y1, x2, y2].
fn compute_iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let inter_x1 = a[0].max(b[0]);
    let inter_y1 = a[1].max(b[1]);
    let inter_x2 = a[2].min(b[2]);
    let inter_y2 = a[3].min(b[3]);

    let inter_w = (inter_x2 - inter_x1).max(0.0);
    let inter_h = (inter_y2 - inter_y1).max(0.0);
    let inter_area = inter_w * inter_h;

    let area_a = (a[2] - a[0]) * (a[3] - a[1]);
    let area_b = (b[2] - b[0]) * (b[3] - b[1]);
    let union_area = area_a + area_b - inter_area;

    if union_area <= 0.0 {
        0.0
    } else {
        inter_area / union_area
    }
}

/// Result of resize + letterbox: the resized RGB image plus transform parameters.
struct ResizeResult {
    rgb: Vec<u8>,
    scale: f32,
    pad_x: f32,
    pad_y: f32,
}

/// Resize an RGB image to target_size x target_size with letterboxing.
fn resize_rgb(src: &[u8], src_w: u32, src_h: u32, target: usize) -> ResizeResult {
    let scale = (target as f32 / src_w as f32).min(target as f32 / src_h as f32);
    let new_w = (src_w as f32 * scale) as usize;
    let new_h = (src_h as f32 * scale) as usize;

    let pad_x = (target - new_w) / 2;
    let pad_y = (target - new_h) / 2;

    let mut dst = vec![0u8; target * target * 3];

    // Nearest-neighbor resize + pad
    for dy in 0..new_h {
        let sy = (dy as f32 / scale) as usize;
        let sy = sy.min(src_h as usize - 1);
        for dx in 0..new_w {
            let sx = (dx as f32 / scale) as usize;
            let sx = sx.min(src_w as usize - 1);
            let src_idx = (sy * src_w as usize + sx) * 3;
            let dst_idx = ((dy + pad_y) * target + dx + pad_x) * 3;
            dst[dst_idx] = src[src_idx];
            dst[dst_idx + 1] = src[src_idx + 1];
            dst[dst_idx + 2] = src[src_idx + 2];
        }
    }

    ResizeResult {
        rgb: dst,
        scale,
        pad_x: pad_x as f32,
        pad_y: pad_y as f32,
    }
}

/// Build NCHW f32 tensor from resized RGB image, with mean subtraction (ImageNet-style).
fn normalize_to_nchw(rgb: &[u8], size: usize) -> Array4<f32> {
    // SCRFD uses BGR order with mean subtraction: [104.0, 117.0, 123.0]
    let mut tensor = Array4::<f32>::zeros((1, 3, size, size));
    for y in 0..size {
        for x in 0..size {
            let idx = (y * size + x) * 3;
            let r = rgb[idx] as f32;
            let g = rgb[idx + 1] as f32;
            let b = rgb[idx + 2] as f32;
            // BGR order with mean subtraction
            tensor[[0, 0, y, x]] = b - 104.0;
            tensor[[0, 1, y, x]] = g - 117.0;
            tensor[[0, 2, y, x]] = r - 123.0;
        }
    }
    tensor
}

/// Greedy NMS (non-maximum suppression) sorted by confidence descending.
fn nms(detections: &mut [FaceDetection], threshold: f32) -> Vec<FaceDetection> {
    detections.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut keep = Vec::new();
    let mut suppressed = vec![false; detections.len()];

    for i in 0..detections.len() {
        if suppressed[i] {
            continue;
        }
        keep.push(detections[i].clone());
        for j in (i + 1)..detections.len() {
            if !suppressed[j] && compute_iou(&detections[i].bbox, &detections[j].bbox) > threshold {
                suppressed[j] = true;
            }
        }
    }

    keep
}

/// Parse frame dimensions from ffmpeg rawvideo output.
/// Falls back to computing from raw byte count assuming 16:9.
fn parse_frame_dimensions(stderr: &str, raw_bytes: &[u8]) -> Result<(u32, u32)> {
    // Try to parse from showinfo output: "n:   0 ... s:1920x1080 ..."
    let re = regex::Regex::new(r"s:(\d+)x(\d+)").expect("static regex");
    if let Some(caps) = re.captures(stderr) {
        if let (Some(w), Some(h)) = (caps.get(1), caps.get(2)) {
            if let (Ok(w), Ok(h)) = (w.as_str().parse::<u32>(), h.as_str().parse::<u32>()) {
                if w > 0 && h > 0 && raw_bytes.len() == (w * h * 3) as usize {
                    return Ok((w, h));
                }
            }
        }
    }

    // Fallback: try common resolutions
    let total_pixels = raw_bytes.len() / 3;
    let common: &[(u32, u32)] = &[
        (1920, 1080),
        (1280, 720),
        (3840, 2160),
        (640, 480),
        (640, 360),
        (320, 180),
        (854, 480),
        (1080, 1920),
        (720, 1280),
    ];
    for &(w, h) in common {
        if (w * h) as usize == total_pixels {
            return Ok((w, h));
        }
    }

    anyhow::bail!(
        "cannot determine frame dimensions from {} raw bytes",
        raw_bytes.len()
    );
}

/// Download a model file from a URL into `dest`, creating parent directories
/// as needed. Attaches `Authorization: Bearer <HF_API_KEY>` when present so
/// gated huggingface.co URLs (which SCRFD currently is) work.
///
/// Follows redirects (HF serves redirects to the actual blob CDN) and
/// returns a clear error for the common gated-model failure modes.
async fn download_model(url: &str, dest: &Path) -> Result<()> {
    if let Some(parent) = dest.parent() {
        tokio::fs::create_dir_all(parent).await.ok();
    }

    let mut req = reqwest::Client::new().get(url);
    if let Ok(token) = std::env::var("HF_API_KEY") {
        if !token.is_empty() {
            req = req.bearer_auth(token);
        }
    }
    let resp = req
        .send()
        .await
        .with_context(|| format!("GET {url}"))?;

    let status = resp.status();
    if !status.is_success() {
        // Specific guidance for the common gated-model 4xx codes.
        let hint = match status.as_u16() {
            401 => " — model is gated on HuggingFace; set HF_API_KEY (and accept terms on the model page)",
            403 => " — HF token rejected the model access; visit the model page and accept terms",
            404 => " — URL not found; the model repo may have restructured",
            _ => "",
        };
        anyhow::bail!("SCRFD model download failed: HTTP {status}{hint}");
    }

    let bytes = resp.bytes().await.context("read SCRFD model bytes")?;
    tokio::fs::write(dest, &bytes)
        .await
        .with_context(|| format!("write SCRFD model to {}", dest.display()))?;

    tracing::info!(
        path = %dest.display(),
        bytes = bytes.len(),
        "SCRFD model downloaded"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iou_identical_boxes() {
        let a = [10.0, 10.0, 50.0, 50.0];
        assert!((compute_iou(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn iou_no_overlap() {
        let a = [0.0, 0.0, 10.0, 10.0];
        let b = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(compute_iou(&a, &b), 0.0);
    }

    #[test]
    fn iou_partial_overlap() {
        let a = [0.0, 0.0, 10.0, 10.0]; // area = 100
        let b = [5.0, 0.0, 15.0, 10.0]; // area = 100, overlap = 5*10 = 50
        let iou = compute_iou(&a, &b);
        // union = 100 + 100 - 50 = 150, iou = 50/150 = 0.333...
        assert!((iou - 1.0 / 3.0).abs() < 1e-5);
    }

    #[test]
    fn nms_removes_overlapping() {
        let mut dets = vec![
            FaceDetection {
                bbox: [0.0, 0.0, 100.0, 100.0],
                confidence: 0.9,
                landmarks: vec![],
            },
            FaceDetection {
                bbox: [5.0, 5.0, 105.0, 105.0],
                confidence: 0.8,
                landmarks: vec![],
            },
            FaceDetection {
                bbox: [200.0, 200.0, 300.0, 300.0],
                confidence: 0.7,
                landmarks: vec![],
            },
        ];
        let kept = nms(&mut dets, 0.3);
        assert_eq!(kept.len(), 2, "NMS should keep 2 non-overlapping faces");
        assert!((kept[0].confidence - 0.9).abs() < 1e-6);
        assert!((kept[1].confidence - 0.7).abs() < 1e-6);
    }

    #[test]
    fn tracker_assigns_new_ids() {
        let mut tracker = FaceTracker::new(0.3);
        let dets = vec![
            FaceDetection {
                bbox: [10.0, 10.0, 50.0, 50.0],
                confidence: 0.9,
                landmarks: vec![],
            },
            FaceDetection {
                bbox: [100.0, 100.0, 150.0, 150.0],
                confidence: 0.8,
                landmarks: vec![],
            },
        ];
        let tracked = tracker.update(&dets);
        assert_eq!(tracked.len(), 2);
        assert_eq!(tracked[0].id, 0);
        assert_eq!(tracked[1].id, 1);
    }

    #[test]
    fn tracker_preserves_identity() {
        let mut tracker = FaceTracker::new(0.3);

        // Frame 1
        let dets1 = vec![FaceDetection {
            bbox: [10.0, 10.0, 50.0, 50.0],
            confidence: 0.9,
            landmarks: vec![],
        }];
        let tracked1 = tracker.update(&dets1);
        assert_eq!(tracked1[0].id, 0);

        // Frame 2: face moved slightly
        let dets2 = vec![FaceDetection {
            bbox: [12.0, 12.0, 52.0, 52.0],
            confidence: 0.85,
            landmarks: vec![],
        }];
        let tracked2 = tracker.update(&dets2);
        assert_eq!(tracked2[0].id, 0, "same face should keep same ID");
    }

    #[test]
    fn tracker_new_face_gets_new_id() {
        let mut tracker = FaceTracker::new(0.3);

        // Frame 1: one face
        let dets1 = vec![FaceDetection {
            bbox: [10.0, 10.0, 50.0, 50.0],
            confidence: 0.9,
            landmarks: vec![],
        }];
        tracker.update(&dets1);

        // Frame 2: original face + new face far away
        let dets2 = vec![
            FaceDetection {
                bbox: [12.0, 12.0, 52.0, 52.0],
                confidence: 0.85,
                landmarks: vec![],
            },
            FaceDetection {
                bbox: [300.0, 300.0, 400.0, 400.0],
                confidence: 0.7,
                landmarks: vec![],
            },
        ];
        let tracked2 = tracker.update(&dets2);
        assert_eq!(tracked2.len(), 2);
        // Original face keeps ID 0, new face gets ID 1
        let ids: Vec<u32> = tracked2.iter().map(|t| t.id).collect();
        assert!(ids.contains(&0));
        assert!(ids.contains(&1));
    }

    #[test]
    fn resize_rgb_preserves_size() {
        let w = 640u32;
        let h = 480u32;
        let src = vec![128u8; (w * h * 3) as usize];
        let target = 640;
        let rr = resize_rgb(&src, w, h, target);
        assert_eq!(rr.rgb.len(), target * target * 3);
    }

    #[test]
    fn normalize_produces_correct_shape() {
        let size = 64;
        let rgb = vec![128u8; size * size * 3];
        let tensor = normalize_to_nchw(&rgb, size);
        assert_eq!(tensor.shape(), &[1, 3, size, size]);
    }

    #[test]
    fn parse_dimensions_from_showinfo() {
        let stderr = "[Parsed_showinfo_0 @ 0x...] n:   0 pts:      0 pts_time:0 fmt:yuv420p sar:1/1 s:320x180 ...";
        let raw = vec![0u8; 320 * 180 * 3];
        let (w, h) = parse_frame_dimensions(stderr, &raw).unwrap();
        assert_eq!((w, h), (320, 180));
    }

    #[test]
    fn parse_dimensions_fallback_common() {
        let raw = vec![0u8; 1920 * 1080 * 3];
        let (w, h) = parse_frame_dimensions("no dimensions here", &raw).unwrap();
        assert_eq!((w, h), (1920, 1080));
    }

    #[tokio::test]
    async fn extract_frame_rgb_requires_ffmpeg() -> Result<()> {
        // Skip if ffmpeg not available
        let ok = Command::new("ffmpeg")
            .arg("-version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .await
            .map(|s| s.success())
            .unwrap_or(false);

        if !ok {
            eprintln!("skipping: ffmpeg not available on PATH");
            return Ok(());
        }

        let dir = tempfile::tempdir()?;
        let video = dir.path().join("test.mp4");

        // Synthesize a short test video
        let status = Command::new("ffmpeg")
            .arg("-y")
            .args(["-hide_banner", "-loglevel", "error"])
            .args([
                "-f",
                "lavfi",
                "-i",
                "color=red:size=320x180:duration=1:rate=15",
            ])
            .args(["-c:v", "libx264", "-pix_fmt", "yuv420p"])
            .arg(&video)
            .status()
            .await?;
        anyhow::ensure!(status.success(), "ffmpeg test video generation failed");

        let (rgb, w, h) = extract_frame_rgb("ffmpeg", &video, 0.5).await?;
        assert_eq!(w, 320);
        assert_eq!(h, 180);
        assert_eq!(rgb.len(), (320 * 180 * 3) as usize);
        // Red frame: most pixels should have high R, low G, low B
        // (ffmpeg yuv420p may not be perfectly pure, but R should dominate)
        let r = rgb[0] as u32;
        let g = rgb[1] as u32;
        let b = rgb[2] as u32;
        assert!(
            r > g && r > b,
            "expected reddish pixel, got R={r} G={g} B={b}"
        );

        Ok(())
    }
}
