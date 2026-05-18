//! Per-clip social media copy generator.
//!
//! For each top-K clip, makes one LLM call that returns ready-to-post copy for
//! every target platform (YouTube Shorts, TikTok, Instagram Reels, Threads,
//! LinkedIn, X, Bluesky) plus a short on-screen overlay hook for the first 1.5s
//! of the vertical render.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::ai_pipeline::ShowContext;
use crate::candidates::CandidateWindow;
use crate::openai::OpenAiClient;
use crate::ranker::RankedClip;

/// All per-platform copy + the overlay hook for one clip.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct SocialCopy {
    #[serde(default)]
    pub youtube_shorts: YouTubeShortsCopy,
    #[serde(default)]
    pub tiktok: TikTokCopy,
    #[serde(default)]
    pub instagram_reels: InstagramReelsCopy,
    #[serde(default)]
    pub threads: ThreadsCopy,
    #[serde(default)]
    pub linkedin: LinkedInCopy,
    #[serde(default)]
    pub x: XCopy,
    #[serde(default)]
    pub bluesky: BlueskyCopy,
    #[serde(default)]
    pub overlay_hook: String,
    /// Where the final overlay_hook text came from: "llm", "ranker", or "fallback".
    #[serde(default)]
    pub hook_source: String,
}

/// Which source to use for the on-screen overlay hook text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HookSource {
    /// Always use the LLM-generated hook from social copy.
    Llm,
    /// Always use the ranker-generated hook (truncated to ≤5 words).
    Ranker,
    /// Alternate between LLM and ranker per clip for A/B testing.
    AbTest,
}

impl HookSource {
    pub fn from_str(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "ranker" => Self::Ranker,
            "ab_test" | "ab" | "abtest" => Self::AbTest,
            _ => Self::Llm,
        }
    }
}

/// Maximum number of words allowed in an overlay hook.
const HOOK_MAX_WORDS: usize = 5;

/// Validate and sanitize an overlay hook: trim, enforce ≤5 words, strip
/// trailing punctuation that looks bad on-screen.
pub fn validate_hook(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    let words: Vec<&str> = trimmed.split_whitespace().collect();
    let capped: String = words.iter().take(HOOK_MAX_WORDS).copied().collect::<Vec<_>>().join(" ");
    // Strip trailing period — overlay text reads better without it.
    capped.trim_end_matches('.').to_string()
}

/// Build an overlay hook from the ranker's sentence-level hook by taking
/// the first 5 words. Used as fallback or when `HookSource::Ranker` is chosen.
pub fn hook_from_ranker(ranker_hook: &str) -> String {
    validate_hook(ranker_hook)
}

/// Resolve the final overlay hook for a clip given the LLM and ranker hooks,
/// the configured source strategy, and the clip index (for A/B alternation).
pub fn resolve_hook(
    social: &mut SocialCopy,
    ranker_hook: &str,
    source: HookSource,
    clip_index: usize,
) {
    let (hook, label) = match source {
        HookSource::Llm => {
            let validated = validate_hook(&social.overlay_hook);
            if validated.is_empty() {
                // Fallback to ranker if LLM returned empty.
                let fallback = hook_from_ranker(ranker_hook);
                (fallback, "fallback")
            } else {
                (validated, "llm")
            }
        }
        HookSource::Ranker => (hook_from_ranker(ranker_hook), "ranker"),
        HookSource::AbTest => {
            if clip_index % 2 == 0 {
                let validated = validate_hook(&social.overlay_hook);
                if validated.is_empty() {
                    (hook_from_ranker(ranker_hook), "fallback")
                } else {
                    (validated, "llm")
                }
            } else {
                (hook_from_ranker(ranker_hook), "ranker")
            }
        }
    };
    social.overlay_hook = hook;
    social.hook_source = label.to_string();
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct YouTubeShortsCopy {
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub hashtags: Vec<String>,
    #[serde(default)]
    pub pinned_comment: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct TikTokCopy {
    #[serde(default)]
    pub caption: String,
    #[serde(default)]
    pub hashtags: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct InstagramReelsCopy {
    #[serde(default)]
    pub caption: String,
    #[serde(default)]
    pub hashtags: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct ThreadsCopy {
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub hashtags: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct LinkedInCopy {
    #[serde(default)]
    pub post_text: String,
    #[serde(default)]
    pub hashtags: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct XCopy {
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub hashtags: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct BlueskyCopy {
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub hashtags: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct SocialCopyGenerator {
    pub openai: OpenAiClient,
    pub chat_model: String,
    pub system_prompt: String,
    pub user_prompt_template: String,
    pub transcript_char_budget: usize,
}

impl SocialCopyGenerator {
    pub fn new(
        openai: OpenAiClient,
        chat_model: String,
        system_prompt: String,
        user_prompt_template: String,
    ) -> Self {
        Self {
            openai,
            chat_model,
            system_prompt,
            user_prompt_template,
            transcript_char_budget: 8000,
        }
    }

    /// Generate copy for a single clip. The transcript comes from the original
    /// CandidateWindow (refined ranker boundaries are within ±5s; the original
    /// transcript is close enough for copy generation).
    pub async fn generate(
        &self,
        clip: &RankedClip,
        candidate: &CandidateWindow,
        show_context: Option<&ShowContext>,
    ) -> Result<SocialCopy> {
        let user = self.build_user_prompt(clip, candidate, show_context);
        let raw = self
            .openai
            .chat_json(&self.chat_model, &self.system_prompt, &user)
            .await
            .with_context(|| format!("social copy for clip {}", clip.candidate_index))?;
        let parsed: SocialCopy =
            serde_json::from_value(raw).context("parse social copy JSON")?;
        Ok(parsed)
    }

    fn build_user_prompt(
        &self,
        clip: &RankedClip,
        candidate: &CandidateWindow,
        show_context: Option<&ShowContext>,
    ) -> String {
        let (show_name, hosts, guest) = match show_context {
            Some(c) => (
                c.show_name.as_deref().unwrap_or("").to_string(),
                c.hosts.join(", "),
                c.guest.as_deref().unwrap_or("").to_string(),
            ),
            None => (String::new(), String::new(), String::new()),
        };
        let transcript = clamp_chars(&candidate.transcript, self.transcript_char_budget);
        let duration = (clip.end_secs - clip.start_secs).max(0.0) as u64;
        let time_range = format!(
            "{}-{}",
            fmt_mmss(clip.start_secs),
            fmt_mmss(clip.end_secs)
        );
        self.user_prompt_template
            .replace("{{show_name}}", &show_name)
            .replace("{{hosts}}", &hosts)
            .replace("{{guest}}", &guest)
            .replace("{{time_range}}", &time_range)
            .replace("{{duration_secs}}", &duration.to_string())
            .replace("{{hook}}", clip.hook.trim())
            .replace("{{reasoning}}", clip.reasoning.trim())
            .replace("{{transcript}}", transcript.trim())
    }
}

fn clamp_chars(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    s.chars().take(max).collect()
}

fn fmt_mmss(secs: f64) -> String {
    let s = secs.max(0.0) as i64;
    let m = s / 60;
    let r = s % 60;
    format!("{m:02}:{r:02}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linguistic_markers::LinguisticFeatures;

    fn dummy_clip() -> RankedClip {
        RankedClip {
            candidate_index: 5,
            start_secs: 130.0,
            end_secs: 185.0,
            score: 78,
            hook: "He says the old defense is hopeless.".to_string(),
            reasoning: "Big reaction; clean payoff.".to_string(),
            trend_match: None,
        }
    }

    fn dummy_candidate(transcript: &str) -> CandidateWindow {
        CandidateWindow {
            start_secs: 130.0,
            end_secs: 185.0,
            transcript: transcript.to_string(),
            word_count: transcript.split_whitespace().count(),
            linguistic: LinguisticFeatures::default(),
            rms_peak_db: Some(-12.0),
            rms_mean_db: Some(-22.0),
            f0_mean_hz: None,
            f0_variance_hz2: None,
            f0_peak_hz: None,
            speaking_rate_wps: Some(3.0),
            novelty_score: Some(0.6),
            audio_events: None,
        }
    }

    #[test]
    fn builds_user_prompt_with_substitutions() {
        let openai = OpenAiClient::new("https://example.com".into(), "x".into());
        let template = "show={{show_name}}|hosts={{hosts}}|time={{time_range}}|dur={{duration_secs}}|hook={{hook}}|why={{reasoning}}|tx={{transcript}}";
        let sc_gen = SocialCopyGenerator::new(
            openai,
            "model".into(),
            "sys".into(),
            template.into(),
        );
        let ctx = ShowContext {
            show_name: Some("TFATK".into()),
            hosts: vec!["Brendan".into(), "Bryan".into()],
            guest: None,
            evidence: vec![],
        };
        let prompt = sc_gen.build_user_prompt(&dummy_clip(), &dummy_candidate("test transcript"), Some(&ctx));
        assert!(prompt.contains("show=TFATK"));
        assert!(prompt.contains("hosts=Brendan, Bryan"));
        assert!(prompt.contains("time=02:10-03:05"));
        assert!(prompt.contains("dur=55"));
        assert!(prompt.contains("hook=He says the old defense is hopeless."));
        assert!(prompt.contains("tx=test transcript"));
    }

    #[test]
    fn missing_show_context_yields_empty_substitutions() {
        let openai = OpenAiClient::new("https://example.com".into(), "x".into());
        let template = "host={{hosts}}|show={{show_name}}|guest={{guest}}";
        let sc_gen = SocialCopyGenerator::new(
            openai,
            "model".into(),
            "sys".into(),
            template.into(),
        );
        let prompt = sc_gen.build_user_prompt(&dummy_clip(), &dummy_candidate("t"), None);
        assert!(prompt.contains("host=") && !prompt.contains("{{hosts}}"));
        assert!(prompt.contains("show=") && !prompt.contains("{{show_name}}"));
        assert!(prompt.contains("guest=") && !prompt.contains("{{guest}}"));
    }

    #[test]
    fn parses_full_social_copy_response() {
        let body = r##"{
          "youtube_shorts": {
            "title": "He realizes the defense is hopeless #Shorts",
            "description": "The old telephone defense gets brutally roasted in this clip from the show. Watch how fast the realization hits. #podcastclips #comedy #Shorts",
            "hashtags": ["#Shorts", "#podcastclips", "#comedy"],
            "pinned_comment": "Have you ever used this defense move?"
          },
          "tiktok": {
            "caption": "Wait until you see his reaction 😭 the telephone defense is dead",
            "hashtags": ["#fyp", "#mma", "#podcast"]
          },
          "instagram_reels": {
            "caption": "When the defense you've been using your whole life turns out to be useless. Watch his face.\n\n#reels #podcast #comedy #mma #grappling #bjj #martialarts #podcastclips",
            "hashtags": ["#reels", "#podcast", "#comedy", "#mma", "#grappling", "#bjj", "#martialarts", "#podcastclips"]
          },
          "threads": {
            "text": "the moment he realizes the telephone defense doesn't work",
            "hashtags": ["#podcast"]
          },
          "linkedin": {
            "post_text": "Sometimes the lesson lands fast.\n\nIn this short, the host realizes a defense he thought worked has been useless the whole time.\n\nWhat's a habit you found out wasn't doing what you thought?",
            "hashtags": ["#PodcastClips", "#LearningMoments"]
          },
          "x": {
            "text": "the moment your go-to defense gets exposed in real time",
            "hashtags": ["#mma"]
          },
          "bluesky": {
            "text": "watch the realization hit in real-time",
            "hashtags": []
          },
          "overlay_hook": "Wait for it"
        }"##;
        let parsed: SocialCopy = serde_json::from_str(body).expect("parse");
        assert!(parsed.youtube_shorts.title.contains("#Shorts"));
        assert_eq!(parsed.youtube_shorts.hashtags.len(), 3);
        assert!(parsed.tiktok.caption.contains("telephone defense"));
        assert!(parsed.instagram_reels.hashtags.len() >= 8);
        assert_eq!(parsed.linkedin.post_text.lines().count(), 5);
        assert!(parsed.x.text.len() <= 220);
        assert!(parsed.bluesky.text.len() <= 280);
        assert_eq!(parsed.overlay_hook, "Wait for it");
    }

    #[test]
    fn deserializes_with_missing_fields() {
        // Defensive: if LLM omits a platform block entirely, defaults should fill in.
        let body = r#"{"youtube_shorts": {"title": "a"}, "overlay_hook": "go"}"#;
        let parsed: SocialCopy = serde_json::from_str(body).expect("parse");
        assert_eq!(parsed.youtube_shorts.title, "a");
        assert_eq!(parsed.tiktok.caption, "");
        assert_eq!(parsed.overlay_hook, "go");
    }

    #[test]
    fn fmt_mmss_basic() {
        assert_eq!(fmt_mmss(0.0), "00:00");
        assert_eq!(fmt_mmss(125.5), "02:05");
        assert_eq!(fmt_mmss(3661.0), "61:01");
    }

    // ── Hook validation tests ──────────────────────────────────────────

    #[test]
    fn validate_hook_passes_short_text() {
        assert_eq!(validate_hook("Wait for it"), "Wait for it");
    }

    #[test]
    fn validate_hook_truncates_to_five_words() {
        assert_eq!(
            validate_hook("This is way too many words for a hook"),
            "This is way too many"
        );
    }

    #[test]
    fn validate_hook_strips_trailing_period() {
        assert_eq!(validate_hook("He was wrong."), "He was wrong");
    }

    #[test]
    fn validate_hook_returns_empty_for_blank() {
        assert_eq!(validate_hook(""), "");
        assert_eq!(validate_hook("   "), "");
    }

    #[test]
    fn hook_from_ranker_truncates_sentence() {
        let ranker = "He says the old defense is hopeless and nobody can stop it";
        let result = hook_from_ranker(ranker);
        assert_eq!(result.split_whitespace().count(), 5);
        assert_eq!(result, "He says the old defense");
    }

    #[test]
    fn resolve_hook_llm_preferred_when_valid() {
        let mut sc = SocialCopy {
            overlay_hook: "Wait for it".into(),
            ..Default::default()
        };
        resolve_hook(&mut sc, "ranker sentence hook", HookSource::Llm, 0);
        assert_eq!(sc.overlay_hook, "Wait for it");
        assert_eq!(sc.hook_source, "llm");
    }

    #[test]
    fn resolve_hook_falls_back_to_ranker_when_llm_empty() {
        let mut sc = SocialCopy::default(); // overlay_hook is ""
        resolve_hook(&mut sc, "He drops the bombshell live", HookSource::Llm, 0);
        assert_eq!(sc.overlay_hook, "He drops the bombshell live");
        assert_eq!(sc.hook_source, "fallback");
    }

    #[test]
    fn resolve_hook_ranker_mode_always_uses_ranker() {
        let mut sc = SocialCopy {
            overlay_hook: "LLM hook".into(),
            ..Default::default()
        };
        resolve_hook(&mut sc, "The ranker generated this long hook text", HookSource::Ranker, 0);
        assert_eq!(sc.overlay_hook, "The ranker generated this long");
        assert_eq!(sc.hook_source, "ranker");
    }

    #[test]
    fn resolve_hook_ab_test_alternates() {
        let mut sc0 = SocialCopy {
            overlay_hook: "LLM hook".into(),
            ..Default::default()
        };
        resolve_hook(&mut sc0, "Ranker hook text here", HookSource::AbTest, 0);
        assert_eq!(sc0.hook_source, "llm", "even index should use LLM");

        let mut sc1 = SocialCopy {
            overlay_hook: "LLM hook".into(),
            ..Default::default()
        };
        resolve_hook(&mut sc1, "Ranker hook text here", HookSource::AbTest, 1);
        assert_eq!(sc1.hook_source, "ranker", "odd index should use ranker");
    }

    #[test]
    fn hook_source_from_str_variants() {
        assert_eq!(HookSource::from_str("llm"), HookSource::Llm);
        assert_eq!(HookSource::from_str("ranker"), HookSource::Ranker);
        assert_eq!(HookSource::from_str("ab_test"), HookSource::AbTest);
        assert_eq!(HookSource::from_str("ab"), HookSource::AbTest);
        assert_eq!(HookSource::from_str("abtest"), HookSource::AbTest);
        assert_eq!(HookSource::from_str("unknown"), HookSource::Llm);
    }

    #[test]
    fn hook_source_serialized_in_social_copy() {
        let sc = SocialCopy {
            overlay_hook: "test".into(),
            hook_source: "llm".into(),
            ..Default::default()
        };
        let json = serde_json::to_value(&sc).expect("serialize");
        assert_eq!(json["hook_source"], "llm");
        assert_eq!(json["overlay_hook"], "test");
    }
}
