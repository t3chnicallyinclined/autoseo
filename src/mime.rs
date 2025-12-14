use base64::{Engine as _, engine::general_purpose};

pub struct Attachment {
    pub filename: String,
    pub content_type: String,
    pub bytes: Vec<u8>,
}

pub fn build_mime_email(
    from: &str,
    to: &str,
    subject: &str,
    body_text: &str,
    attachments: &[Attachment],
) -> Vec<u8> {
    let boundary = format!("autoseo_{}", rand::random::<u64>());

    let mut out = String::new();
    out.push_str(&format!("From: {from}\r\n"));
    out.push_str(&format!("To: {to}\r\n"));
    out.push_str(&format!("Subject: {subject}\r\n"));
    out.push_str("MIME-Version: 1.0\r\n");

    if attachments.is_empty() {
        out.push_str("Content-Type: text/plain; charset=utf-8\r\n");
        out.push_str("Content-Transfer-Encoding: 7bit\r\n\r\n");
        out.push_str(body_text);
        out.push_str("\r\n");
        return out.into_bytes();
    }

    out.push_str(&format!(
        "Content-Type: multipart/mixed; boundary=\"{boundary}\"\r\n\r\n"
    ));

    // Text part
    out.push_str(&format!("--{boundary}\r\n"));
    out.push_str("Content-Type: text/plain; charset=utf-8\r\n");
    out.push_str("Content-Transfer-Encoding: 7bit\r\n\r\n");
    out.push_str(body_text);
    out.push_str("\r\n");

    for a in attachments {
        let b64 = general_purpose::STANDARD.encode(&a.bytes);
        out.push_str(&format!("--{boundary}\r\n"));
        out.push_str(&format!(
            "Content-Type: {}; name=\"{}\"\r\n",
            a.content_type, a.filename
        ));
        out.push_str("Content-Transfer-Encoding: base64\r\n");
        out.push_str(&format!(
            "Content-Disposition: attachment; filename=\"{}\"\r\n\r\n",
            a.filename
        ));

        // Wrap base64 at 76 chars per RFC.
        for chunk in b64.as_bytes().chunks(76) {
            out.push_str(std::str::from_utf8(chunk).unwrap_or(""));
            out.push_str("\r\n");
        }
    }

    out.push_str(&format!("--{boundary}--\r\n"));
    out.into_bytes()
}
