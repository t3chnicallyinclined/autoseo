use regex::Regex;

use percent_encoding::percent_decode_str;

pub fn extract_drive_file_ids(text: &str) -> Vec<String> {
    // Gmail often includes redirect wrappers like:
    //   https://www.google.com/url?q=https%3A%2F%2Fdrive.google.com%2Ffile%2Fd%2F<id>%2Fview...
    // So we:
    // 1) Search raw text for common patterns.
    // 2) Extract & decode any google.com/url?q=... targets and search those too.

    let normalized = text.replace("&amp;", "&");

    let mut candidates = Vec::new();
    candidates.push(normalized.clone());

    // Extract and decode google redirect URLs' q= parameter.
    // We keep this tolerant: it will just skip malformed items.
    let re_google_url =
        Regex::new(r#"https?://(?:www\.)?google\.com/url\?[^\s\"]*"#).expect("valid regex");
    let re_q = Regex::new(r#"(?:\?|&)q=([^&]+)"#).expect("valid regex");
    for m in re_google_url.find_iter(&normalized) {
        let url = m.as_str();
        if let Some(cap) = re_q.captures(url)
            && let Some(q) = cap.get(1)
        {
            let decoded = percent_decode_str(q.as_str())
                .decode_utf8_lossy()
                .to_string();
            candidates.push(decoded);
        }
    }

    let patterns = [
        // Explicit drive.google.com variants
        r"https?://drive\.google\.com/file/d/([a-zA-Z0-9_-]{10,})",
        r"https?://drive\.google\.com/drive/u/\d+/file/d/([a-zA-Z0-9_-]{10,})",
        r"https?://drive\.google\.com/open\?id=([a-zA-Z0-9_-]{10,})",
        r"https?://drive\.google\.com/uc\?id=([a-zA-Z0-9_-]{10,})",
        r"https?://drive\.google\.com/drive/(?:u/\d+/)?folders/([a-zA-Z0-9_-]{10,})",
        // Host-agnostic fallbacks (useful when links are partially stripped)
        r"/file/d/([a-zA-Z0-9_-]{10,})",
        r"[?&]id=([a-zA-Z0-9_-]{10,})",
    ];

    let mut out = Vec::new();
    for candidate in candidates {
        for pat in patterns {
            let re = Regex::new(pat).expect("valid regex");
            for cap in re.captures_iter(&candidate) {
                if let Some(m) = cap.get(1) {
                    out.push(m.as_str().to_string());
                }
            }
        }
    }

    // Deduplicate while keeping order.
    let mut seen = std::collections::HashSet::new();
    out.into_iter()
        .filter(|id| seen.insert(id.clone()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_file_d() {
        let s = "hey https://drive.google.com/file/d/1AbcDEFghIJkLmNoPq/view?usp=sharing";
        let ids = extract_drive_file_ids(s);
        assert_eq!(ids, vec!["1AbcDEFghIJkLmNoPq".to_string()]);
    }

    #[test]
    fn extracts_open_id() {
        let s = "https://drive.google.com/open?id=1XYZ_abc-123";
        let ids = extract_drive_file_ids(s);
        assert_eq!(ids, vec!["1XYZ_abc-123".to_string()]);
    }

    #[test]
    fn extracts_google_redirect_q() {
        let s = "https://www.google.com/url?q=https%3A%2F%2Fdrive.google.com%2Ffile%2Fd%2F1AbcDEFghIJkLmNoPq%2Fview%3Fusp%3Dsharing&sa=D";
        let ids = extract_drive_file_ids(s);
        assert_eq!(ids, vec!["1AbcDEFghIJkLmNoPq".to_string()]);
    }
}
