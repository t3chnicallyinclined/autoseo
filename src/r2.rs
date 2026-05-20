//! S3-compatible object-storage uploader (Cloudflare R2, AWS S3, MinIO, etc.).
//!
//! Activated when these env vars are set:
//!
//! - `R2_ENDPOINT`        — e.g. `https://<accountid>.r2.cloudflarestorage.com`
//! - `R2_ACCESS_KEY_ID`   — API token access key
//! - `R2_SECRET_ACCESS_KEY` — API token secret
//! - `R2_BUCKET`          — bucket name (e.g. `autoseo-clips`)
//! - `R2_PUBLIC_BASE_URL` — public read URL prefix that maps to the bucket;
//!                          either `https://pub-<hash>.r2.dev` (R2's built-in
//!                          public dev URL — already includes the bucket root)
//!                          or `https://media.your-domain.com` (custom domain
//!                          mounted at the bucket root). The returned object
//!                          URL is `{R2_PUBLIC_BASE_URL}/{key}`.
//!
//! If any required var is unset, [`R2Uploader::from_env`] returns `Ok(None)`
//! and the clipper falls back to local-disk storage.

use anyhow::{Context, Result};
use aws_credential_types::Credentials;
use aws_sdk_s3::Client;
use aws_sdk_s3::config::Region;
use aws_sdk_s3::primitives::ByteStream;
use std::path::Path;

const ENV_ENDPOINT: &str = "R2_ENDPOINT";
const ENV_ACCESS_KEY: &str = "R2_ACCESS_KEY_ID";
const ENV_SECRET_KEY: &str = "R2_SECRET_ACCESS_KEY";
const ENV_BUCKET: &str = "R2_BUCKET";
const ENV_PUBLIC_BASE: &str = "R2_PUBLIC_BASE_URL";
const ENV_REGION: &str = "R2_REGION";
const ENV_KEY_PREFIX: &str = "R2_KEY_PREFIX";

#[derive(Clone)]
pub struct R2Uploader {
    client: Client,
    endpoint: String,
    bucket: String,
    public_base_url: String,
    key_prefix: String,
}

impl R2Uploader {
    /// Build an uploader from `R2_*` envs. Returns:
    ///   - `Ok(Some(u))` if all required vars are present and the SDK initialised
    ///   - `Ok(None)` if any required var is missing
    ///   - `Err(_)` if config was supplied but malformed
    pub async fn from_env() -> Result<Option<Self>> {
        let endpoint = match std::env::var(ENV_ENDPOINT) {
            Ok(v) if !v.is_empty() => v,
            _ => return Ok(None),
        };
        let access_key = match std::env::var(ENV_ACCESS_KEY) {
            Ok(v) if !v.is_empty() => v,
            _ => return Ok(None),
        };
        let secret_key = match std::env::var(ENV_SECRET_KEY) {
            Ok(v) if !v.is_empty() => v,
            _ => return Ok(None),
        };
        let bucket = match std::env::var(ENV_BUCKET) {
            Ok(v) if !v.is_empty() => v,
            _ => return Ok(None),
        };
        let public_base_url = match std::env::var(ENV_PUBLIC_BASE) {
            Ok(v) if !v.is_empty() => v.trim_end_matches('/').to_string(),
            _ => return Ok(None),
        };
        let region = std::env::var(ENV_REGION).unwrap_or_else(|_| "auto".to_string());
        let key_prefix = std::env::var(ENV_KEY_PREFIX)
            .unwrap_or_else(|_| "clipper".to_string())
            .trim_matches('/')
            .to_string();

        let creds = Credentials::new(access_key, secret_key, None, None, "autoseo-r2-env");

        let s3_config = aws_sdk_s3::config::Builder::new()
            .endpoint_url(&endpoint)
            .region(Region::new(region))
            .credentials_provider(creds)
            .force_path_style(true) // R2 + most non-AWS S3 work better in path style
            .behavior_version(aws_sdk_s3::config::BehaviorVersion::latest())
            .build();

        let client = Client::from_conf(s3_config);

        Ok(Some(Self {
            client,
            endpoint,
            bucket,
            public_base_url,
            key_prefix,
        }))
    }

    pub fn bucket(&self) -> &str {
        &self.bucket
    }

    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// Convert an absolute local clipper file path into a stable object key.
    /// Strips everything up to and including `/clipper/`, then prepends
    /// `R2_KEY_PREFIX` (default `clipper`) so all objects live under one
    /// logical root in the bucket.
    pub fn derive_key(&self, local_path: &str) -> String {
        let rel = local_path
            .split("/clipper/")
            .nth(1)
            .unwrap_or(local_path)
            .trim_start_matches('/');
        if self.key_prefix.is_empty() {
            rel.to_string()
        } else {
            format!("{}/{}", self.key_prefix, rel)
        }
    }

    /// Upload a file under `key` with the given content type. Returns the
    /// public URL the dashboard / posting workers should read from.
    pub async fn upload_file(
        &self,
        local_path: &Path,
        key: &str,
        content_type: &str,
    ) -> Result<String> {
        let body = ByteStream::from_path(local_path)
            .await
            .with_context(|| format!("open {} for upload", local_path.display()))?;

        self.client
            .put_object()
            .bucket(&self.bucket)
            .key(key)
            .content_type(content_type)
            .body(body)
            .send()
            .await
            .with_context(|| format!("put_object failed: bucket={} key={}", self.bucket, key))?;

        Ok(format!("{}/{}", self.public_base_url, key))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Saved-and-restored env guard so tests can't poison each other when
    /// run in parallel against the same `R2_*` keys.
    struct EnvGuard {
        saved: Vec<(&'static str, Option<String>)>,
    }
    impl EnvGuard {
        fn new(keys: &[&'static str]) -> Self {
            Self {
                saved: keys.iter().map(|k| (*k, std::env::var(*k).ok())).collect(),
            }
        }
    }
    impl Drop for EnvGuard {
        fn drop(&mut self) {
            for (k, v) in &self.saved {
                match v {
                    Some(v) => unsafe { std::env::set_var(k, v) },
                    None => unsafe { std::env::remove_var(k) },
                }
            }
        }
    }

    #[tokio::test]
    async fn from_env_returns_none_when_unset() {
        let _g = EnvGuard::new(&[
            ENV_ENDPOINT,
            ENV_ACCESS_KEY,
            ENV_SECRET_KEY,
            ENV_BUCKET,
            ENV_PUBLIC_BASE,
        ]);
        unsafe {
            std::env::remove_var(ENV_ENDPOINT);
            std::env::remove_var(ENV_ACCESS_KEY);
            std::env::remove_var(ENV_SECRET_KEY);
            std::env::remove_var(ENV_BUCKET);
            std::env::remove_var(ENV_PUBLIC_BASE);
        }
        assert!(R2Uploader::from_env().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn from_env_partial_returns_none() {
        let _g = EnvGuard::new(&[
            ENV_ENDPOINT,
            ENV_ACCESS_KEY,
            ENV_SECRET_KEY,
            ENV_BUCKET,
            ENV_PUBLIC_BASE,
        ]);
        unsafe {
            std::env::set_var(ENV_ENDPOINT, "https://example.r2.cloudflarestorage.com");
            std::env::set_var(ENV_ACCESS_KEY, "ak");
            std::env::set_var(ENV_SECRET_KEY, "sk");
            std::env::remove_var(ENV_BUCKET); // missing → None
            std::env::set_var(ENV_PUBLIC_BASE, "https://pub-xxx.r2.dev");
        }
        assert!(R2Uploader::from_env().await.unwrap().is_none());
    }

    #[test]
    fn derive_key_strips_local_root() {
        let u = R2Uploader {
            client: Client::from_conf(
                aws_sdk_s3::config::Builder::new()
                    .credentials_provider(Credentials::new("a", "b", None, None, "t"))
                    .region(Region::new("auto"))
                    .behavior_version(aws_sdk_s3::config::BehaviorVersion::latest())
                    .build(),
            ),
            endpoint: "https://x".to_string(),
            bucket: "autoseo-clips".to_string(),
            public_base_url: "https://pub-x.r2.dev".to_string(),
            key_prefix: "clipper".to_string(),
        };
        assert_eq!(
            u.derive_key("/home/op/work/clipper/show.mp4/12345/clips/clip_01_9x16.mp4"),
            "clipper/show.mp4/12345/clips/clip_01_9x16.mp4"
        );
        assert_eq!(
            u.derive_key("/home/op/work/clipper/show.mp4/12345/clips/clip_01_cover.jpg"),
            "clipper/show.mp4/12345/clips/clip_01_cover.jpg"
        );
    }
}
