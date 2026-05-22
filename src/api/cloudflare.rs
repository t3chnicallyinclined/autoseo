//! Cloudflare API client for one-shot R2 provisioning from the wizard.
//!
//! The user pastes a Cloudflare API token (created at
//! `dash.cloudflare.com/profile/api-tokens` with `Account:Read` + the API
//! Token permission so we can mint sub-tokens for R2); we use it to:
//!   1. discover their account id (`GET /accounts`),
//!   2. find an existing bucket whose name matches the requested one (or
//!      contains "autoseo") via `GET /accounts/{id}/r2/buckets`, else
//!      create it via `POST /accounts/{id}/r2/buckets`,
//!   3. enable the managed `pub-<hash>.r2.dev` domain via
//!      `PUT /accounts/{id}/r2/buckets/{name}/domains/managed`,
//!   4. mint long-lived S3-compatible credentials by creating an account-
//!      owned API token (`POST /accounts/{id}/tokens`) and deriving:
//!         access_key_id     = result.id
//!         secret_access_key = sha256_hex(result.value)
//!      per the Cloudflare R2 auth docs.
//!
//! References (verified against docs, not guessed):
//!   - https://developers.cloudflare.com/r2/api/tokens/
//!   - https://developers.cloudflare.com/api/resources/r2/subresources/buckets/methods/list/
//!   - https://developers.cloudflare.com/api/resources/accounts/subresources/tokens/methods/create/

use anyhow::{anyhow, Result};
use percent_encoding::{utf8_percent_encode, NON_ALPHANUMERIC};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

fn urlencode(s: &str) -> String {
    utf8_percent_encode(s, NON_ALPHANUMERIC).to_string()
}

const CF_API: &str = "https://api.cloudflare.com/client/v4";

/// Hex-encode the SHA-256 of a string. Per Cloudflare R2 auth docs:
///   "Secret Access Key is the SHA-256 hash of the API token value"
/// (https://developers.cloudflare.com/r2/api/tokens/).
fn sha256_hex(s: &str) -> String {
    let mut h = Sha256::new();
    h.update(s.as_bytes());
    let bytes = h.finalize();
    let mut out = String::with_capacity(64);
    for b in bytes {
        out.push_str(&format!("{b:02x}"));
    }
    out
}

/// Top-level outcome surfaced to the dashboard. `success: true` means the
/// bucket + endpoint + public URL were created. Whether S3 keys were also
/// minted is in `access_key_id` / `secret_access_key` — those are `None`
/// when the user still needs to mint keys manually.
#[derive(Debug, Serialize)]
pub struct ProvisionResult {
    pub account_id: String,
    pub account_name: String,
    pub bucket: String,
    pub endpoint: String,
    pub public_url: Option<String>,
    pub access_key_id: Option<String>,
    pub secret_access_key: Option<String>,
    /// Set when key creation didn't work; the wizard shows this URL so the
    /// user can mint a key manually.
    pub manual_key_url: Option<String>,
    /// Human-readable summary of what happened, suitable for showing inline.
    pub notes: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct CfAccount {
    id: String,
    name: String,
}

#[derive(Debug, Deserialize)]
struct CfBucket {
    name: String,
}

pub struct CloudflareClient {
    token: String,
    http: reqwest::Client,
}

impl CloudflareClient {
    pub fn new(token: String) -> Self {
        Self {
            token,
            http: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .expect("reqwest client build"),
        }
    }

    /// Run the full provisioning sequence. List-first so the operation is
    /// idempotent — if the requested bucket (or any bucket whose name
    /// contains "autoseo") already exists, we reuse it instead of creating
    /// a sibling. Returns partial success when S3 key minting fails so the
    /// wizard can surface a deep-link fallback.
    pub async fn provision(&self, bucket: &str) -> Result<ProvisionResult> {
        let mut notes = Vec::new();

        // 1. Discover account.
        let accounts = self.list_accounts().await?;
        let account = accounts
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("token has access to no accounts — check Account:Read scope"))?;
        notes.push(format!("Using account {} ({})", account.name, account.id));

        // 2. Look for an existing bucket first. Prefer exact-name match; fall
        //    back to any bucket whose name contains the literal "autoseo".
        //    Either way we resolve to a single `effective_bucket` string and
        //    skip the create call when it's already there.
        let existing = self.list_buckets(&account.id).await?;
        let effective_bucket = if existing.iter().any(|b| b.name == bucket) {
            notes.push(format!("Reusing existing bucket {bucket}"));
            bucket.to_string()
        } else if let Some(autoseo_match) = existing.iter().find(|b| b.name.contains("autoseo")) {
            notes.push(format!(
                "Reusing existing autoseo bucket {} (set explicit bucket name to create a separate one)",
                autoseo_match.name
            ));
            autoseo_match.name.clone()
        } else {
            self.create_bucket(&account.id, bucket).await?;
            notes.push(format!("Created bucket {bucket}"));
            bucket.to_string()
        };

        // 3. Enable managed r2.dev domain. PUT is idempotent. Non-fatal if
        //    it fails — user can enable manually in the dashboard.
        let public_url = match self.enable_managed_domain(&account.id, &effective_bucket).await {
            Ok(Some(url)) => {
                notes.push(format!("Public URL: {url}"));
                Some(url)
            }
            Ok(None) => {
                notes.push(
                    "Managed domain enabled but no URL returned; check Cloudflare dashboard"
                        .to_string(),
                );
                None
            }
            Err(e) => {
                notes.push(format!("Managed domain not enabled: {e}"));
                None
            }
        };

        // 4. Mint S3-compatible credentials by creating an account-owned API
        //    token scoped to this bucket; derive S3 keys from the response
        //    (access_key_id = result.id, secret_access_key = sha256(result.value)).
        let (access_key_id, secret_access_key, manual_key_url) =
            match self.create_r2_s3_token(&account.id, &effective_bucket).await {
                Ok((id, secret)) => {
                    notes.push("Minted S3 access keys via account-owned token".to_string());
                    (Some(id), Some(secret), None)
                }
                Err(e) => {
                    let url = format!(
                        "https://dash.cloudflare.com/{}/r2/api-tokens",
                        account.id
                    );
                    notes.push(format!(
                        "Couldn't auto-mint S3 access keys ({e}). Create one at {url} and paste below."
                    ));
                    (None, None, Some(url))
                }
            };

        let endpoint = format!(
            "https://{}.r2.cloudflarestorage.com/{}",
            account.id, effective_bucket
        );

        Ok(ProvisionResult {
            account_id: account.id,
            account_name: account.name,
            bucket: effective_bucket,
            endpoint,
            public_url,
            access_key_id,
            secret_access_key,
            manual_key_url,
            notes,
        })
    }

    /// GET /accounts/{id}/r2/buckets — verified shape:
    ///   { success, result: { buckets: [{ name, ... }] } }
    async fn list_buckets(&self, account_id: &str) -> Result<Vec<CfBucket>> {
        let url = format!("{CF_API}/accounts/{account_id}/r2/buckets");
        let body = self.get_json(&url).await?;
        if !body["success"].as_bool().unwrap_or(false) {
            return Err(anyhow!("list buckets: {}", summarize_errors(&body)));
        }
        let arr = body["result"]["buckets"]
            .as_array()
            .ok_or_else(|| anyhow!("list buckets: missing result.buckets"))?;
        Ok(arr
            .iter()
            .filter_map(|v| serde_json::from_value(v.clone()).ok())
            .collect())
    }

    async fn list_accounts(&self) -> Result<Vec<CfAccount>> {
        let url = format!("{CF_API}/accounts");
        let body = self.get_json(&url).await?;
        if !body["success"].as_bool().unwrap_or(false) {
            return Err(anyhow!("list accounts: {}", summarize_errors(&body)));
        }
        let arr = body["result"]
            .as_array()
            .ok_or_else(|| anyhow!("list accounts: missing result"))?;
        Ok(arr
            .iter()
            .filter_map(|v| serde_json::from_value(v.clone()).ok())
            .collect())
    }

    async fn create_bucket(&self, account_id: &str, name: &str) -> Result<()> {
        let url = format!("{CF_API}/accounts/{account_id}/r2/buckets");
        let body = self.post_json(&url, &json!({"name": name})).await?;
        if body["success"].as_bool().unwrap_or(false) {
            return Ok(());
        }
        Err(anyhow!("create bucket: {}", summarize_errors(&body)))
    }

    /// Enable the managed pub-<hash>.r2.dev domain. Returns the public URL on
    /// success. PUT is idempotent — enabling an already-enabled bucket is OK.
    async fn enable_managed_domain(
        &self,
        account_id: &str,
        bucket: &str,
    ) -> Result<Option<String>> {
        let url = format!("{CF_API}/accounts/{account_id}/r2/buckets/{bucket}/domains/managed");
        let resp = self
            .http
            .put(&url)
            .bearer_auth(&self.token)
            .json(&json!({"enabled": true}))
            .send()
            .await?;
        let body: Value = resp.json().await?;
        if !body["success"].as_bool().unwrap_or(false) {
            return Err(anyhow!(
                "enable managed domain: {}",
                summarize_errors(&body)
            ));
        }
        // Response shape: { result: { domain: "pub-<hash>.r2.dev", enabled: true }, ... }
        let domain = body["result"]["domain"].as_str();
        Ok(domain.map(|d| format!("https://{d}")))
    }

    /// Mint long-lived S3-compatible credentials.
    ///
    /// Per https://developers.cloudflare.com/r2/api/tokens/, S3 access keys
    /// are *derived* from a regular Cloudflare account-owned API token:
    ///     access_key_id     = result.id
    ///     secret_access_key = sha256_hex(result.value)
    /// We create the token via POST /accounts/{id}/tokens with a policy
    /// scoped to a single R2 bucket. The required permission group ID is
    /// looked up dynamically (it's an opaque UUID, not stable to hardcode)
    /// via GET /accounts/{id}/tokens/permission_groups?name=...
    async fn create_r2_s3_token(
        &self,
        account_id: &str,
        bucket: &str,
    ) -> Result<(String, String)> {
        // Look up the read/write permission group's UUID by exact name.
        let pg_name = "Workers R2 Storage Bucket Item Write";
        let pg_url = format!(
            "{CF_API}/accounts/{account_id}/tokens/permission_groups?name={}",
            urlencode(pg_name)
        );
        let pg_body = self.get_json(&pg_url).await?;
        if !pg_body["success"].as_bool().unwrap_or(false) {
            // 9109 = the calling token can't access the permission_groups
            // endpoint. That endpoint requires `API Tokens · Read` (which is
            // included by default when the token has `API Tokens · Write`).
            // Surface a specific hint so the user knows the exact permission
            // to add instead of going down a CF docs rabbit hole.
            let err = summarize_errors(&pg_body);
            if err.contains("9109") || err.to_lowercase().contains("unauthorized") {
                return Err(anyhow!(
                    "token is missing `API Tokens · Edit` permission \
                     (required to mint scoped S3 keys via the CF API). \
                     Re-mint the token with that permission added, or use \
                     the manual deep link below to mint S3 keys yourself."
                ));
            }
            return Err(anyhow!("lookup permission_groups: {err}"));
        }
        let pg_id = pg_body["result"]
            .as_array()
            .and_then(|a| a.iter().find(|p| p["name"].as_str() == Some(pg_name)))
            .and_then(|p| p["id"].as_str())
            .ok_or_else(|| {
                anyhow!(
                    "permission group {pg_name:?} not found — token may lack \
                     the necessary API Token permission"
                )
            })?
            .to_string();

        // POST /accounts/{id}/tokens — body shape per
        // https://developers.cloudflare.com/api/resources/accounts/subresources/tokens/methods/create/
        // The R2-bucket resource key format is documented at
        // https://developers.cloudflare.com/r2/api/tokens/ as:
        //   com.cloudflare.edge.r2.bucket.<ACCOUNT_ID>_<JURISDICTION>_<BUCKET>
        let resource_key = format!(
            "com.cloudflare.edge.r2.bucket.{account_id}_default_{bucket}"
        );
        let req_body = json!({
            "name": format!("autoseo-{bucket}"),
            "policies": [{
                "effect": "allow",
                "permission_groups": [{ "id": pg_id }],
                "resources": { resource_key: "*" }
            }]
        });
        let url = format!("{CF_API}/accounts/{account_id}/tokens");
        let body = self.post_json(&url, &req_body).await?;
        if !body["success"].as_bool().unwrap_or(false) {
            return Err(anyhow!("create token: {}", summarize_errors(&body)));
        }
        let token_id = body["result"]["id"]
            .as_str()
            .ok_or_else(|| anyhow!("create token: response missing result.id"))?
            .to_string();
        let token_value = body["result"]["value"]
            .as_str()
            .ok_or_else(|| anyhow!("create token: response missing result.value"))?;
        Ok((token_id, sha256_hex(token_value)))
    }

    async fn get_json(&self, url: &str) -> Result<Value> {
        let resp = self.http.get(url).bearer_auth(&self.token).send().await?;
        let body: Value = resp.json().await?;
        Ok(body)
    }

    async fn post_json(&self, url: &str, body: &Value) -> Result<Value> {
        let resp = self
            .http
            .post(url)
            .bearer_auth(&self.token)
            .json(body)
            .send()
            .await?;
        let body: Value = resp.json().await?;
        Ok(body)
    }
}

fn summarize_errors(body: &Value) -> String {
    if let Some(arr) = body["errors"].as_array() {
        let msgs: Vec<String> = arr
            .iter()
            .filter_map(|e| {
                let code = e["code"].as_i64();
                let msg = e["message"].as_str()?;
                Some(match code {
                    Some(c) => format!("[{c}] {msg}"),
                    None => msg.to_string(),
                })
            })
            .collect();
        if !msgs.is_empty() {
            return msgs.join("; ");
        }
    }
    body.to_string()
}
