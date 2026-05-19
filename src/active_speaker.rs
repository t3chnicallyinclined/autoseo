//! Rule-based active-speaker detection: face presence × VAD speech overlap.
//!
//! M3 fallback before a Python sidecar for Light-ASD. Given per-frame face
//! detections and VAD speech segments, scores each detected face by how much
//! of its on-screen time overlaps with speech. The highest-scoring face per
//! frame is elected as the active speaker and its bbox is output for the crop
//! trajectory.

use crate::vad::SpeechSegment;

// ── Types ───────────────────────────────────────────────────────────────────

/// Axis-aligned bounding box for a detected face, in pixel coordinates.
#[derive(Debug, Clone, PartialEq)]
pub struct FaceBbox {
    /// Left edge (x pixel).
    pub x: f64,
    /// Top edge (y pixel).
    pub y: f64,
    /// Width in pixels.
    pub w: f64,
    /// Height in pixels.
    pub h: f64,
    /// Detection confidence [0, 1].
    pub confidence: f64,
}

impl FaceBbox {
    pub fn center_x(&self) -> f64 {
        self.x + self.w / 2.0
    }

    pub fn center_y(&self) -> f64 {
        self.y + self.h / 2.0
    }

    pub fn area(&self) -> f64 {
        self.w * self.h
    }
}

/// All faces detected in a single video frame.
#[derive(Debug, Clone)]
pub struct FrameFaces {
    /// Timestamp of this frame in seconds (episode time).
    pub timestamp_secs: f64,
    /// Detected face bounding boxes.
    pub faces: Vec<FaceBbox>,
}

/// A face tracked across consecutive frames, identified by index in the first
/// frame it appeared. This is a simple positional tracker — not a re-ID model.
#[derive(Debug, Clone)]
pub struct TrackedFace {
    /// Index used to group detections across frames (assigned on first appearance).
    pub track_id: usize,
    /// Bounding box at this frame.
    pub bbox: FaceBbox,
    /// Timestamp of this observation.
    pub timestamp_secs: f64,
}

/// Per-face speech-overlap score across all frames it was observed.
#[derive(Debug, Clone)]
pub struct FaceSpeechScore {
    pub track_id: usize,
    /// Total seconds this face was on screen.
    pub presence_secs: f64,
    /// Total seconds this face overlapped with speech.
    pub speech_overlap_secs: f64,
}

impl FaceSpeechScore {
    /// Fraction of on-screen time that overlaps with speech [0.0, 1.0].
    pub fn overlap_ratio(&self) -> f64 {
        if self.presence_secs <= 0.0 {
            return 0.0;
        }
        (self.speech_overlap_secs / self.presence_secs).clamp(0.0, 1.0)
    }
}

/// Per-frame active speaker decision.
#[derive(Debug, Clone)]
pub struct ActiveSpeakerFrame {
    pub timestamp_secs: f64,
    /// The elected active-speaker bbox, or `None` if no face overlaps speech.
    pub active_speaker_bbox: Option<FaceBbox>,
    /// Track ID of the elected face (if any).
    pub track_id: Option<usize>,
}

// ── Core logic ──────────────────────────────────────────────────────────────

/// Returns `true` if `timestamp` falls within any speech segment.
fn is_speech_at(timestamp: f64, speech_segments: &[SpeechSegment]) -> bool {
    speech_segments
        .iter()
        .any(|seg| timestamp >= seg.start_secs && timestamp < seg.end_secs)
}

/// Compute seconds of overlap between a time range `[start, end)` and the
/// union of speech segments.
fn speech_overlap_secs(start: f64, end: f64, speech_segments: &[SpeechSegment]) -> f64 {
    let mut total = 0.0_f64;
    for seg in speech_segments {
        let overlap_start = start.max(seg.start_secs);
        let overlap_end = end.min(seg.end_secs);
        if overlap_end > overlap_start {
            total += overlap_end - overlap_start;
        }
    }
    total
}

/// Simple nearest-neighbor face tracker across frames.
///
/// Assigns a `track_id` to each detection by matching it to the closest
/// (Euclidean center distance) detection in the previous frame that hasn't
/// already been claimed. New detections that are farther than
/// `max_match_distance` pixels start a new track.
fn assign_tracks(frames: &[FrameFaces], max_match_distance: f64) -> Vec<Vec<TrackedFace>> {
    let mut next_id: usize = 0;
    // Previous frame's (track_id, center_x, center_y) for matching.
    let mut prev_tracks: Vec<(usize, f64, f64)> = Vec::new();
    let mut all_frames: Vec<Vec<TrackedFace>> = Vec::with_capacity(frames.len());

    for frame in frames {
        let mut frame_tracked: Vec<TrackedFace> = Vec::with_capacity(frame.faces.len());
        let mut claimed: Vec<bool> = vec![false; prev_tracks.len()];

        // Sort current faces by descending confidence so the best detections
        // claim tracks first.
        let mut indexed_faces: Vec<(usize, &FaceBbox)> = frame.faces.iter().enumerate().collect();
        indexed_faces.sort_by(|a, b| {
            b.1.confidence
                .partial_cmp(&a.1.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for (_orig_idx, face) in &indexed_faces {
            let cx = face.center_x();
            let cy = face.center_y();

            // Find closest unclaimed previous track.
            let mut best: Option<(usize, f64)> = None; // (prev_idx, dist)
            for (pi, &(_, px, py)) in prev_tracks.iter().enumerate() {
                if claimed[pi] {
                    continue;
                }
                let dist = ((cx - px).powi(2) + (cy - py).powi(2)).sqrt();
                if dist <= max_match_distance && (best.is_none() || dist < best.unwrap().1) {
                    best = Some((pi, dist));
                }
            }

            let track_id = if let Some((pi, _)) = best {
                claimed[pi] = true;
                prev_tracks[pi].0
            } else {
                let id = next_id;
                next_id += 1;
                id
            };

            frame_tracked.push(TrackedFace {
                track_id,
                bbox: (*face).clone(),
                timestamp_secs: frame.timestamp_secs,
            });
        }

        // Update prev_tracks for the next iteration.
        prev_tracks = frame_tracked
            .iter()
            .map(|t| (t.track_id, t.bbox.center_x(), t.bbox.center_y()))
            .collect();

        all_frames.push(frame_tracked);
    }

    all_frames
}

/// Score each tracked face by its speech overlap percentage.
///
/// For each unique `track_id`, sum up the on-screen duration (approximated as
/// the interval between consecutive frame timestamps where the face appears)
/// and the speech-overlapping portion of that duration.
fn score_faces(
    tracked_frames: &[Vec<TrackedFace>],
    speech_segments: &[SpeechSegment],
    frame_interval_secs: f64,
) -> Vec<FaceSpeechScore> {
    // Collect per-track timestamps.
    let mut track_timestamps: std::collections::HashMap<usize, Vec<f64>> =
        std::collections::HashMap::new();
    for frame in tracked_frames {
        for tf in frame {
            track_timestamps
                .entry(tf.track_id)
                .or_default()
                .push(tf.timestamp_secs);
        }
    }

    let mut scores: Vec<FaceSpeechScore> = Vec::new();
    for (&track_id, timestamps) in &track_timestamps {
        let mut ts = timestamps.clone();
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        ts.dedup();

        // Approximate presence: each frame contributes `frame_interval_secs`.
        let presence_secs = ts.len() as f64 * frame_interval_secs;

        // Sum speech overlap for each frame interval.
        let mut overlap = 0.0_f64;
        for &t in &ts {
            let seg_start = t;
            let seg_end = t + frame_interval_secs;
            overlap += speech_overlap_secs(seg_start, seg_end, speech_segments);
        }

        scores.push(FaceSpeechScore {
            track_id,
            presence_secs,
            speech_overlap_secs: overlap,
        });
    }

    // Sort descending by overlap ratio (best speaker first).
    scores.sort_by(|a, b| {
        b.overlap_ratio()
            .partial_cmp(&a.overlap_ratio())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    scores
}

/// Detect the active speaker per frame using face presence × VAD overlap.
///
/// # Arguments
/// * `frames` — Per-frame face detections (sorted by timestamp).
/// * `speech_segments` — VAD-derived speech intervals.
/// * `fps` — Video frame rate, used to estimate per-frame duration.
/// * `max_match_distance` — Max pixel distance for the nearest-neighbor tracker
///   to consider two detections the same face across frames (default: 150.0).
///
/// # Returns
/// A vec of [`ActiveSpeakerFrame`] aligned 1:1 with `frames`, each containing
/// the elected active-speaker bbox (highest speech-overlap face) or `None`.
pub fn detect_active_speaker(
    frames: &[FrameFaces],
    speech_segments: &[SpeechSegment],
    fps: f64,
    max_match_distance: f64,
) -> Vec<ActiveSpeakerFrame> {
    if frames.is_empty() {
        return Vec::new();
    }

    let frame_interval = if fps > 0.0 { 1.0 / fps } else { 1.0 / 30.0 };

    // Step 1: Track faces across frames.
    let tracked_frames = assign_tracks(frames, max_match_distance);

    // Step 2: Score each track by speech overlap.
    let face_scores = score_faces(&tracked_frames, speech_segments, frame_interval);

    // Build a quick lookup: track_id → overlap_ratio.
    let score_map: std::collections::HashMap<usize, f64> = face_scores
        .iter()
        .map(|s| (s.track_id, s.overlap_ratio()))
        .collect();

    // Step 3: For each frame, pick the face with the highest speech-overlap score.
    // Tie-break by detection confidence.
    let mut results: Vec<ActiveSpeakerFrame> = Vec::with_capacity(frames.len());
    for (fi, tracked) in tracked_frames.iter().enumerate() {
        let timestamp = frames[fi].timestamp_secs;

        // Only consider faces at timestamps where speech is happening.
        let speech_active = is_speech_at(timestamp, speech_segments);

        let elected = if speech_active {
            tracked
                .iter()
                .filter(|tf| score_map.get(&tf.track_id).copied().unwrap_or(0.0) > 0.0)
                .max_by(|a, b| {
                    let sa = score_map.get(&a.track_id).copied().unwrap_or(0.0);
                    let sb = score_map.get(&b.track_id).copied().unwrap_or(0.0);
                    // Compute overlap_ratio from sums of small floats, so
                    // numerically equal ratios can differ at ~1e-16. Treat
                    // anything inside the noise floor as a tie and fall through
                    // to the confidence tie-break.
                    let primary = if (sa - sb).abs() < 1e-9 {
                        std::cmp::Ordering::Equal
                    } else if sa > sb {
                        std::cmp::Ordering::Greater
                    } else {
                        std::cmp::Ordering::Less
                    };
                    primary.then_with(|| {
                        a.bbox
                            .confidence
                            .partial_cmp(&b.bbox.confidence)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                })
        } else {
            None
        };

        results.push(ActiveSpeakerFrame {
            timestamp_secs: timestamp,
            active_speaker_bbox: elected.map(|tf| tf.bbox.clone()),
            track_id: elected.map(|tf| tf.track_id),
        });
    }

    results
}

// ── Decision documentation ──────────────────────────────────────────────────

/// Documents what Light-ASD would improve over this rule-based approach.
///
/// This rule-based detector has known limitations:
///
/// 1. **No lip-motion signal**: We only check face *presence* during speech,
///    not whether the face is actually moving its mouth. In multi-speaker
///    scenarios where multiple faces are visible throughout, the face with
///    the most total on-screen time during speech wins — even if it's just
///    listening. Light-ASD uses audio-visual correlation (lip sync) to
///    distinguish the actual speaker from listeners.
///
/// 2. **No temporal smoothing of speaker identity**: We pick the best face
///    per-frame independently. Light-ASD maintains temporal context so the
///    speaker label doesn't flicker between faces.
///
/// 3. **Nearest-neighbor tracking is fragile**: Our simple center-distance
///    tracker can lose a face through occlusion, rapid motion, or shot
///    cuts. Light-ASD or a proper re-ID model handles this better.
///
/// 4. **Single-speaker assumption**: When two people talk simultaneously
///    (crosstalk), this heuristic cannot distinguish who is speaking. ASD
///    models handle this via audio-visual attention.
///
/// Recommendation: if multi-speaker podcasts or interviews are a primary use
/// case, invest in the Light-ASD Python sidecar (M3+). For single-speaker
/// talking-head content, this rule-based approach is sufficient.
pub const LIGHT_ASD_COMPARISON: &str = "See doc comment on this constant";

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vad::SpeechSegment;

    fn face(x: f64, y: f64, w: f64, h: f64, confidence: f64) -> FaceBbox {
        FaceBbox {
            x,
            y,
            w,
            h,
            confidence,
        }
    }

    #[test]
    fn is_speech_at_detects_overlap() {
        let segments = vec![
            SpeechSegment {
                start_secs: 1.0,
                end_secs: 3.0,
            },
            SpeechSegment {
                start_secs: 5.0,
                end_secs: 7.0,
            },
        ];
        assert!(!is_speech_at(0.5, &segments));
        assert!(is_speech_at(1.0, &segments));
        assert!(is_speech_at(2.0, &segments));
        assert!(!is_speech_at(3.0, &segments)); // half-open [start, end)
        assert!(!is_speech_at(4.0, &segments));
        assert!(is_speech_at(6.0, &segments));
    }

    #[test]
    fn speech_overlap_secs_computes_correctly() {
        let segments = vec![SpeechSegment {
            start_secs: 2.0,
            end_secs: 5.0,
        }];
        // Fully inside speech.
        assert!((speech_overlap_secs(2.5, 4.0, &segments) - 1.5).abs() < 1e-9);
        // Partial overlap at start.
        assert!((speech_overlap_secs(1.0, 3.0, &segments) - 1.0).abs() < 1e-9);
        // Partial overlap at end.
        assert!((speech_overlap_secs(4.0, 6.0, &segments) - 1.0).abs() < 1e-9);
        // No overlap.
        assert!((speech_overlap_secs(0.0, 1.0, &segments) - 0.0).abs() < 1e-9);
        // Spanning entire segment.
        assert!((speech_overlap_secs(0.0, 10.0, &segments) - 3.0).abs() < 1e-9);
    }

    #[test]
    fn speech_overlap_with_multiple_segments() {
        let segments = vec![
            SpeechSegment {
                start_secs: 1.0,
                end_secs: 2.0,
            },
            SpeechSegment {
                start_secs: 3.0,
                end_secs: 4.0,
            },
        ];
        // Range spanning both segments.
        assert!((speech_overlap_secs(0.0, 5.0, &segments) - 2.0).abs() < 1e-9);
        // Range between segments.
        assert!((speech_overlap_secs(2.0, 3.0, &segments) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn single_face_single_speaker() {
        let speech = vec![SpeechSegment {
            start_secs: 0.0,
            end_secs: 1.0,
        }];
        let frames: Vec<FrameFaces> = (0..10)
            .map(|i| FrameFaces {
                timestamp_secs: i as f64 * 0.1,
                faces: vec![face(100.0, 100.0, 50.0, 60.0, 0.95)],
            })
            .collect();

        let results = detect_active_speaker(&frames, &speech, 10.0, 150.0);
        assert_eq!(results.len(), 10);
        // All frames are during speech, face should be elected.
        for r in &results {
            assert!(r.active_speaker_bbox.is_some());
            assert_eq!(r.track_id, Some(0));
        }
    }

    #[test]
    fn no_face_during_speech_returns_none() {
        let speech = vec![SpeechSegment {
            start_secs: 0.0,
            end_secs: 1.0,
        }];
        let frames: Vec<FrameFaces> = (0..5)
            .map(|i| FrameFaces {
                timestamp_secs: i as f64 * 0.2,
                faces: vec![], // no faces detected
            })
            .collect();

        let results = detect_active_speaker(&frames, &speech, 5.0, 150.0);
        for r in &results {
            assert!(r.active_speaker_bbox.is_none());
        }
    }

    #[test]
    fn face_outside_speech_not_elected() {
        // Speech only in [0, 0.5), but face appears in frames at [0.5, 1.0).
        let speech = vec![SpeechSegment {
            start_secs: 0.0,
            end_secs: 0.5,
        }];
        let frames: Vec<FrameFaces> = (5..10)
            .map(|i| FrameFaces {
                timestamp_secs: i as f64 * 0.1, // 0.5, 0.6, ..., 0.9
                faces: vec![face(100.0, 100.0, 50.0, 60.0, 0.95)],
            })
            .collect();

        let results = detect_active_speaker(&frames, &speech, 10.0, 150.0);
        for r in &results {
            assert!(r.active_speaker_bbox.is_none());
        }
    }

    #[test]
    fn multi_face_picks_highest_speech_overlap() {
        // Face A is present for all 10 frames (0.0–1.0s). Speech at [0.0, 1.0).
        // Face B is present only for frames 0–4 (0.0–0.5s).
        // Face A has more speech overlap in total, so should win.
        let speech = vec![SpeechSegment {
            start_secs: 0.0,
            end_secs: 1.0,
        }];

        let mut frames: Vec<FrameFaces> = Vec::new();
        for i in 0..10 {
            let t = i as f64 * 0.1;
            let mut faces = vec![face(100.0, 100.0, 50.0, 60.0, 0.9)]; // Face A
            if i < 5 {
                faces.push(face(300.0, 100.0, 50.0, 60.0, 0.85)); // Face B
            }
            frames.push(FrameFaces {
                timestamp_secs: t,
                faces,
            });
        }

        let results = detect_active_speaker(&frames, &speech, 10.0, 150.0);

        // Face A (track 0) should be elected in all speech frames because it
        // has 100% speech overlap (present for all 10 frames, all during speech).
        // Face B also has 100% overlap but only in 5 frames. Both have ratio 1.0,
        // so tie-break by confidence: Face A (0.9) > Face B (0.85).
        for r in &results {
            assert!(r.active_speaker_bbox.is_some());
            assert_eq!(r.track_id, Some(0), "Face A should win");
        }
    }

    #[test]
    fn multi_face_speech_overlap_decides_winner() {
        // Face A is present in ALL frames but only during silence.
        // Face B is present only during speech frames.
        // Face B should have higher speech overlap ratio.
        let speech = vec![SpeechSegment {
            start_secs: 0.5,
            end_secs: 1.0,
        }];

        let mut frames: Vec<FrameFaces> = Vec::new();
        for i in 0..10 {
            let t = i as f64 * 0.1;
            let mut faces = vec![face(100.0, 100.0, 50.0, 60.0, 0.95)]; // Face A (always present)
            if i >= 5 {
                // Face B only appears during speech
                faces.push(face(300.0, 100.0, 50.0, 60.0, 0.80));
            }
            frames.push(FrameFaces {
                timestamp_secs: t,
                faces,
            });
        }

        let results = detect_active_speaker(&frames, &speech, 10.0, 150.0);

        // During speech frames (i >= 5), Face B has overlap_ratio = 1.0
        // (all its presence is during speech), Face A has overlap_ratio = 0.5
        // (only half its frames are during speech). Face B should win.
        for (i, r) in results.iter().enumerate() {
            if i >= 5 {
                assert!(r.active_speaker_bbox.is_some());
                assert_eq!(r.track_id, Some(1), "Face B should win during speech");
            } else {
                // Before speech, no face should be elected.
                assert!(r.active_speaker_bbox.is_none());
            }
        }
    }

    #[test]
    fn empty_frames_returns_empty() {
        let results = detect_active_speaker(&[], &[], 30.0, 150.0);
        assert!(results.is_empty());
    }

    #[test]
    fn face_bbox_helpers() {
        let f = face(10.0, 20.0, 100.0, 80.0, 0.9);
        assert!((f.center_x() - 60.0).abs() < 1e-9);
        assert!((f.center_y() - 60.0).abs() < 1e-9);
        assert!((f.area() - 8000.0).abs() < 1e-9);
    }

    #[test]
    fn face_speech_score_overlap_ratio() {
        let s = FaceSpeechScore {
            track_id: 0,
            presence_secs: 10.0,
            speech_overlap_secs: 7.5,
        };
        assert!((s.overlap_ratio() - 0.75).abs() < 1e-9);

        let zero = FaceSpeechScore {
            track_id: 1,
            presence_secs: 0.0,
            speech_overlap_secs: 0.0,
        };
        assert!((zero.overlap_ratio() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn tracker_assigns_consistent_ids() {
        // Two faces moving slightly between frames should keep their IDs.
        let frames = vec![
            FrameFaces {
                timestamp_secs: 0.0,
                faces: vec![
                    face(100.0, 100.0, 50.0, 60.0, 0.9),
                    face(300.0, 100.0, 50.0, 60.0, 0.8),
                ],
            },
            FrameFaces {
                timestamp_secs: 0.033,
                faces: vec![
                    face(102.0, 101.0, 50.0, 60.0, 0.9), // Face A moved slightly
                    face(298.0, 99.0, 50.0, 60.0, 0.8),  // Face B moved slightly
                ],
            },
        ];

        let tracked = assign_tracks(&frames, 150.0);
        assert_eq!(tracked.len(), 2);

        // Both frames should have 2 tracked faces.
        assert_eq!(tracked[0].len(), 2);
        assert_eq!(tracked[1].len(), 2);

        // IDs should be consistent across frames.
        let frame0_ids: Vec<usize> = tracked[0].iter().map(|t| t.track_id).collect();
        let frame1_ids: Vec<usize> = tracked[1].iter().map(|t| t.track_id).collect();
        assert_eq!(frame0_ids, frame1_ids);
    }

    #[test]
    fn tracker_creates_new_id_for_distant_face() {
        let frames = vec![
            FrameFaces {
                timestamp_secs: 0.0,
                faces: vec![face(100.0, 100.0, 50.0, 60.0, 0.9)],
            },
            FrameFaces {
                timestamp_secs: 0.033,
                faces: vec![
                    face(100.0, 100.0, 50.0, 60.0, 0.9), // Same face
                    face(800.0, 500.0, 50.0, 60.0, 0.7), // New face, far away
                ],
            },
        ];

        let tracked = assign_tracks(&frames, 150.0);
        assert_eq!(tracked[1].len(), 2);

        let ids: Vec<usize> = tracked[1].iter().map(|t| t.track_id).collect();
        assert!(ids.contains(&0)); // Original face
        assert!(ids.contains(&1)); // New face
    }
}
