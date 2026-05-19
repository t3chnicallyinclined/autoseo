use anyhow::Context;
use base64::{
    Engine as _, engine::general_purpose::URL_SAFE, engine::general_purpose::URL_SAFE_NO_PAD,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct GmailClient {
    http: reqwest::Client,
}

impl GmailClient {
    fn decode_b64url_lenient(data: &str) -> Option<Vec<u8>> {
        URL_SAFE_NO_PAD
            .decode(data.as_bytes())
            .ok()
            .or_else(|| URL_SAFE.decode(data.as_bytes()).ok())
    }

    pub fn new() -> Self {
        Self {
            http: reqwest::Client::new(),
        }
    }

    pub async fn list_message_ids(
        &self,
        access_token: &str,
        query: &str,
        max_results: u32,
    ) -> anyhow::Result<Vec<String>> {
        let url = "https://gmail.googleapis.com/gmail/v1/users/me/messages";
        let res = self
            .http
            .get(url)
            .bearer_auth(access_token)
            .query(&[("q", query), ("maxResults", &max_results.to_string())])
            .send()
            .await
            .context("gmail messages.list")?;

        let status = res.status();
        if !status.is_success() {
            let body = res.text().await.unwrap_or_default();
            anyhow::bail!("gmail list failed: {status} {body}");
        }

        let parsed: ListMessagesResponse = res.json().await.context("parse messages.list")?;
        Ok(parsed
            .messages
            .unwrap_or_default()
            .into_iter()
            .filter_map(|m| m.id)
            .collect())
    }

    pub async fn get_message_full(
        &self,
        access_token: &str,
        message_id: &str,
    ) -> anyhow::Result<Message> {
        let url = format!("https://gmail.googleapis.com/gmail/v1/users/me/messages/{message_id}");
        let res = self
            .http
            .get(url)
            .bearer_auth(access_token)
            .query(&[("format", "full")])
            .send()
            .await
            .context("gmail messages.get")?;

        let status = res.status();
        if !status.is_success() {
            let body = res.text().await.unwrap_or_default();
            anyhow::bail!("gmail get failed: {status} {body}");
        }

        res.json().await.context("parse messages.get")
    }

    pub async fn get_message_raw_rfc822(
        &self,
        access_token: &str,
        message_id: &str,
    ) -> anyhow::Result<String> {
        let url = format!("https://gmail.googleapis.com/gmail/v1/users/me/messages/{message_id}");
        let res = self
            .http
            .get(url)
            .bearer_auth(access_token)
            .query(&[("format", "raw")])
            .send()
            .await
            .context("gmail messages.get(raw)")?;

        let status = res.status();
        if !status.is_success() {
            let body = res.text().await.unwrap_or_default();
            anyhow::bail!("gmail get(raw) failed: {status} {body}");
        }

        let parsed: RawMessage = res.json().await.context("parse messages.get(raw)")?;
        let raw_b64 = parsed.raw.context("missing raw")?;
        let bytes = Self::decode_b64url_lenient(&raw_b64).context("base64url decode raw")?;
        Ok(String::from_utf8_lossy(&bytes).to_string())
    }

    pub fn extract_text_bodies(message: &Message) -> String {
        fn walk(part: &MessagePart, out: &mut String) {
            if let Some(body) = &part.body
                && let Some(data) = &body.data
                && let Some(bytes) = GmailClient::decode_b64url_lenient(data)
                && let Ok(s) = String::from_utf8(bytes)
            {
                out.push('\n');
                out.push_str(&s);
            }
            if let Some(parts) = &part.parts {
                for p in parts {
                    walk(p, out);
                }
            }
        }

        let mut out = String::new();
        if let Some(payload) = &message.payload {
            walk(payload, &mut out);
        }
        out
    }

    pub async fn extract_text_bodies_resolving_attachments(
        &self,
        access_token: &str,
        message_id: &str,
        message: &Message,
    ) -> anyhow::Result<String> {
        #[derive(Debug, Clone)]
        enum TextSource {
            InlineData(String),
            AttachmentId(String),
        }

        fn collect(part: &MessagePart, out: &mut Vec<TextSource>) {
            let mime = part.mime_type.as_deref().unwrap_or("");
            let is_text =
                mime.starts_with("text/plain") || mime.starts_with("text/html") || mime.is_empty();

            if let Some(body) = &part.body {
                if let Some(data) = &body.data {
                    out.push(TextSource::InlineData(data.clone()));
                } else if is_text && let Some(attachment_id) = &body.attachment_id {
                    out.push(TextSource::AttachmentId(attachment_id.clone()));
                }
            }

            if let Some(parts) = &part.parts {
                for p in parts {
                    collect(p, out);
                }
            }
        }

        let mut sources: Vec<TextSource> = Vec::new();
        if let Some(payload) = &message.payload {
            collect(payload, &mut sources);
        }

        let mut out = String::new();
        for src in sources {
            match src {
                TextSource::InlineData(data) => {
                    if let Some(bytes) = Self::decode_b64url_lenient(&data) {
                        let s = String::from_utf8_lossy(&bytes);
                        if !s.trim().is_empty() {
                            out.push('\n');
                            out.push_str(&s);
                        }
                    }
                }
                TextSource::AttachmentId(attachment_id) => {
                    let bytes = self
                        .get_attachment_bytes(access_token, message_id, &attachment_id)
                        .await
                        .with_context(|| {
                            format!("gmail attachments.get {message_id} {attachment_id}")
                        })?;
                    let s = String::from_utf8_lossy(&bytes);
                    if !s.trim().is_empty() {
                        out.push('\n');
                        out.push_str(&s);
                    }
                }
            }
        }

        Ok(out)
    }

    pub async fn get_attachment_bytes(
        &self,
        access_token: &str,
        message_id: &str,
        attachment_id: &str,
    ) -> anyhow::Result<Vec<u8>> {
        let url = format!(
            "https://gmail.googleapis.com/gmail/v1/users/me/messages/{message_id}/attachments/{attachment_id}"
        );
        let res = self
            .http
            .get(url)
            .bearer_auth(access_token)
            .send()
            .await
            .context("gmail attachments.get")?;

        let status = res.status();
        if !status.is_success() {
            let body = res.text().await.unwrap_or_default();
            anyhow::bail!("gmail attachments.get failed: {status} {body}");
        }

        let parsed: Attachment = res.json().await.context("parse attachments.get")?;
        let data_b64 = parsed.data.context("missing attachment data")?;
        let bytes =
            Self::decode_b64url_lenient(&data_b64).context("base64url decode attachment")?;
        Ok(bytes)
    }

    pub async fn send_raw(
        &self,
        access_token: &str,
        raw_base64url: &str,
    ) -> anyhow::Result<String> {
        let url = "https://gmail.googleapis.com/gmail/v1/users/me/messages/send";
        let body = serde_json::json!({"raw": raw_base64url});
        let res = self
            .http
            .post(url)
            .bearer_auth(access_token)
            .json(&body)
            .send()
            .await
            .context("gmail messages.send")?;

        let status = res.status();
        if !status.is_success() {
            let body = res.text().await.unwrap_or_default();
            anyhow::bail!("gmail send failed: {status} {body}");
        }

        let parsed: SendMessageResponse = res.json().await.context("parse messages.send")?;
        Ok(parsed.id.unwrap_or_default())
    }
}

#[derive(Debug, Deserialize)]
struct SendMessageResponse {
    #[serde(default)]
    id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RawMessage {
    #[serde(default)]
    raw: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Attachment {
    #[serde(default)]
    data: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ListMessagesResponse {
    #[serde(default)]
    messages: Option<Vec<MessageRef>>,
}

#[derive(Debug, Deserialize)]
struct MessageRef {
    #[serde(default)]
    id: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Message {
    #[allow(dead_code)]
    #[serde(default)]
    pub id: Option<String>,
    #[allow(dead_code)]
    #[serde(default)]
    pub thread_id: Option<String>,
    #[allow(dead_code)]
    #[serde(default)]
    pub snippet: Option<String>,
    #[serde(default)]
    pub payload: Option<MessagePart>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct MessagePart {
    #[allow(dead_code)]
    #[serde(default)]
    pub mime_type: Option<String>,
    #[allow(dead_code)]
    #[serde(default)]
    pub filename: Option<String>,
    #[allow(dead_code)]
    #[serde(default)]
    pub headers: Option<Vec<MessageHeader>>,
    #[serde(default)]
    pub body: Option<MessagePartBody>,
    #[serde(default)]
    pub parts: Option<Vec<MessagePart>>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct MessageHeader {
    #[allow(dead_code)]
    #[serde(default)]
    pub name: Option<String>,
    #[allow(dead_code)]
    #[serde(default)]
    pub value: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct MessagePartBody {
    #[allow(dead_code)]
    #[serde(default)]
    pub size: Option<u64>,
    #[serde(default, rename = "attachmentId")]
    pub attachment_id: Option<String>,
    #[serde(default)]
    pub data: Option<String>,
}
