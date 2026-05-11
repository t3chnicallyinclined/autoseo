use anyhow::Context;
use std::path::{Path, PathBuf};

/// Named prompt slots that may be overridden per show.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptName {
    SeoSystem,
    SeoUser,
    SeoVariants,
    ThumbnailSystem,
    ThumbnailUser,
}

impl PromptName {
    /// Filename used inside `{shows_dir}/{slug}/`.
    pub fn filename(self) -> &'static str {
        match self {
            PromptName::SeoSystem => "seo_system.txt",
            PromptName::SeoUser => "seo_user.txt",
            PromptName::SeoVariants => "seo_variants.txt",
            PromptName::ThumbnailSystem => "thumbnail_system.txt",
            PromptName::ThumbnailUser => "thumbnail_user.txt",
        }
    }
}

/// Global prompt paths from `Config`. Used as fallback when no show override exists.
#[derive(Debug, Clone)]
pub struct GlobalPromptPaths {
    pub seo_system: PathBuf,
    pub seo_user: PathBuf,
    pub seo_variants: PathBuf,
    pub thumbnail_system: PathBuf,
    pub thumbnail_user: PathBuf,
}

impl GlobalPromptPaths {
    pub fn path_for(&self, name: PromptName) -> &Path {
        match name {
            PromptName::SeoSystem => &self.seo_system,
            PromptName::SeoUser => &self.seo_user,
            PromptName::SeoVariants => &self.seo_variants,
            PromptName::ThumbnailSystem => &self.thumbnail_system,
            PromptName::ThumbnailUser => &self.thumbnail_user,
        }
    }
}

/// Resolves prompt file paths, preferring per-show overrides where they exist.
#[derive(Debug, Clone)]
pub struct PromptLoader {
    shows_root: PathBuf,
    global: GlobalPromptPaths,
}

impl PromptLoader {
    pub fn new(shows_root: impl Into<PathBuf>, global: GlobalPromptPaths) -> Self {
        Self {
            shows_root: shows_root.into(),
            global,
        }
    }

    /// Return the effective path for a prompt. If `show_slug` is provided and a file exists at
    /// `{shows_root}/{slug}/{name.filename()}`, that path is returned. Otherwise the global path.
    pub async fn resolve(&self, name: PromptName, show_slug: Option<&str>) -> PathBuf {
        if let Some(slug) = show_slug.filter(|s| !s.is_empty()) {
            let candidate = self.shows_root.join(slug).join(name.filename());
            if tokio::fs::metadata(&candidate).await.is_ok() {
                return candidate;
            }
        }
        self.global.path_for(name).to_path_buf()
    }

    /// Read the resolved prompt to a trimmed `String`.
    pub async fn load(&self, name: PromptName, show_slug: Option<&str>) -> anyhow::Result<String> {
        let path = self.resolve(name, show_slug).await;
        let raw = tokio::fs::read_to_string(&path)
            .await
            .with_context(|| format!("read prompt {} at {}", name.filename(), path.display()))?;
        Ok(raw.trim().to_string())
    }
}

/// Slugify a free-form show name for use as a directory key.
/// ASCII-lowercase alnum, runs of non-alnum collapsed to `_`, trimmed of leading/trailing `_`.
pub fn slugify(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_underscore = true; // start as if we just emitted one, to suppress leading `_`
    for ch in s.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            prev_underscore = false;
        } else if !prev_underscore {
            out.push('_');
            prev_underscore = true;
        }
    }
    while out.ends_with('_') {
        out.pop();
    }
    out
}

/// Parsed pipeline mode. Validated from `cfg.mode` on startup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    SeoOnly,
    Clipper,
    Both,
}

impl Mode {
    pub fn parse(s: &str) -> anyhow::Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "seo-only" | "seo_only" | "seo" => Ok(Mode::SeoOnly),
            "clipper" | "clip" | "clips" => Ok(Mode::Clipper),
            "both" | "all" => Ok(Mode::Both),
            other => anyhow::bail!(
                "invalid MODE='{other}'; expected one of seo-only, clipper, both"
            ),
        }
    }

    pub fn produces_seo_emails(self) -> bool {
        matches!(self, Mode::SeoOnly | Mode::Both)
    }

    pub fn produces_clips(self) -> bool {
        matches!(self, Mode::Clipper | Mode::Both)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn paths_in(dir: &Path) -> GlobalPromptPaths {
        GlobalPromptPaths {
            seo_system: dir.join("seo_system.txt"),
            seo_user: dir.join("seo_user.txt"),
            seo_variants: dir.join("seo_variants.txt"),
            thumbnail_system: dir.join("thumbnail_system.txt"),
            thumbnail_user: dir.join("thumbnail_user.txt"),
        }
    }

    #[test]
    fn slugify_basic() {
        assert_eq!(slugify("The Fighter and the Kid"), "the_fighter_and_the_kid");
        assert_eq!(slugify("Hot Boxin' with Mike Tyson"), "hot_boxin_with_mike_tyson");
        assert_eq!(slugify("TFATK #1147 — Nick Simmons"), "tfatk_1147_nick_simmons");
        assert_eq!(slugify("   leading spaces"), "leading_spaces");
        assert_eq!(slugify("trailing___"), "trailing");
        assert_eq!(slugify(""), "");
        assert_eq!(slugify("!!!"), "");
    }

    #[test]
    fn mode_parses() {
        assert_eq!(Mode::parse("seo-only").unwrap(), Mode::SeoOnly);
        assert_eq!(Mode::parse("CLIPPER").unwrap(), Mode::Clipper);
        assert_eq!(Mode::parse(" both ").unwrap(), Mode::Both);
        assert!(Mode::parse("nonsense").is_err());

        assert!(Mode::SeoOnly.produces_seo_emails());
        assert!(!Mode::SeoOnly.produces_clips());
        assert!(Mode::Clipper.produces_clips());
        assert!(!Mode::Clipper.produces_seo_emails());
        assert!(Mode::Both.produces_seo_emails());
        assert!(Mode::Both.produces_clips());
    }

    #[tokio::test]
    async fn resolve_falls_back_to_global() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let globals_dir = dir.path().join("globals");
        let shows_root = dir.path().join("shows");
        tokio::fs::create_dir_all(&globals_dir).await?;
        tokio::fs::create_dir_all(&shows_root).await?;
        tokio::fs::write(globals_dir.join("seo_system.txt"), "GLOBAL").await?;

        let loader = PromptLoader::new(shows_root, paths_in(&globals_dir));
        let resolved = loader.resolve(PromptName::SeoSystem, Some("tfatk")).await;
        assert_eq!(resolved, globals_dir.join("seo_system.txt"));

        let text = loader.load(PromptName::SeoSystem, Some("tfatk")).await?;
        assert_eq!(text, "GLOBAL");
        Ok(())
    }

    #[tokio::test]
    async fn resolve_prefers_show_override() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let globals_dir = dir.path().join("globals");
        let shows_root = dir.path().join("shows");
        let tfatk_dir = shows_root.join("tfatk");
        tokio::fs::create_dir_all(&globals_dir).await?;
        tokio::fs::create_dir_all(&tfatk_dir).await?;
        tokio::fs::write(globals_dir.join("seo_system.txt"), "GLOBAL").await?;
        tokio::fs::write(tfatk_dir.join("seo_system.txt"), "TFATK-SPECIFIC").await?;

        let loader = PromptLoader::new(shows_root, paths_in(&globals_dir));
        let text = loader.load(PromptName::SeoSystem, Some("tfatk")).await?;
        assert_eq!(text, "TFATK-SPECIFIC");

        // Unrelated show falls back.
        let text2 = loader.load(PromptName::SeoSystem, Some("other_show")).await?;
        assert_eq!(text2, "GLOBAL");

        // No slug → global.
        let text3 = loader.load(PromptName::SeoSystem, None).await?;
        assert_eq!(text3, "GLOBAL");

        // Empty slug → global (defensive).
        let text4 = loader.load(PromptName::SeoSystem, Some("")).await?;
        assert_eq!(text4, "GLOBAL");
        Ok(())
    }
}
