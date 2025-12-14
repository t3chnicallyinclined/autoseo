use anyhow::Context;
use serde::Deserialize;

#[derive(Debug, Clone)]
pub struct GoogleAuth {
    client_id: String,
    client_secret: String,
    refresh_token: String,
    http: reqwest::Client,
}

#[derive(Debug, Deserialize)]
struct TokenResponse {
    access_token: String,
    #[allow(dead_code)]
    expires_in: Option<u64>,
    #[allow(dead_code)]
    token_type: Option<String>,
    #[allow(dead_code)]
    scope: Option<String>,
}

impl GoogleAuth {
    pub fn new(client_id: String, client_secret: String, refresh_token: String) -> Self {
        Self {
            client_id,
            client_secret,
            refresh_token,
            http: reqwest::Client::new(),
        }
    }

    pub async fn access_token(&self) -> anyhow::Result<String> {
        // OAuth refresh token exchange
        let params = [
            ("client_id", self.client_id.as_str()),
            ("client_secret", self.client_secret.as_str()),
            ("refresh_token", self.refresh_token.as_str()),
            ("grant_type", "refresh_token"),
        ];

        let res = self
            .http
            .post("https://oauth2.googleapis.com/token")
            .form(&params)
            .send()
            .await
            .context("POST oauth2 token")?;

        let status = res.status();
        if !status.is_success() {
            let body = res.text().await.unwrap_or_default();
            anyhow::bail!("oauth token refresh failed: {status} {body}");
        }

        let parsed: TokenResponse = res.json().await.context("parse token response")?;
        Ok(parsed.access_token)
    }
}
