#!/usr/bin/env python3
"""HTTP viewer for autoseo clipper output.

Loads `manifest.json` from a clips directory and renders a rich HTML page:
each clip card shows all rendered aspect-ratio variants (with a format
switcher to swap the <video> source), and per-platform social-media copy
behind tabs with copy-to-clipboard buttons.

Usage:
    tools/serve_clips.py                      # auto-find latest run
    tools/serve_clips.py /path/to/clips_dir   # explicit path
    tools/serve_clips.py --port 8000 --host 0.0.0.0
"""
from __future__ import annotations

import argparse
import http.server
import json
import os
import socketserver
import sys
from html import escape
from pathlib import Path

CSS = r"""
*, *::before, *::after { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
  max-width: 1500px;
  margin: 2em auto;
  padding: 0 1.5em 4em;
  background: #0e0e10;
  color: #efeff1;
  line-height: 1.55;
}
h1 { margin: 0 0 0.2em; font-weight: 600; font-size: 1.6em; }
.top-meta { color: #adadb8; margin-bottom: 2em; font-size: 0.95em; }
.clip {
  display: grid;
  grid-template-columns: 380px 1fr;
  gap: 2em;
  margin-bottom: 2.5em;
  padding: 1.5em;
  background: #18181b;
  border-radius: 12px;
}
.video-col { display: flex; flex-direction: column; align-items: center; gap: 0.8em; }
video {
  width: 100%;
  max-height: 70vh;
  border-radius: 10px;
  background: black;
}
.format-tabs { display: flex; gap: 0.4em; flex-wrap: wrap; justify-content: center; }
.format-tab {
  padding: 0.3em 0.9em;
  border-radius: 6px;
  background: #1f1f23;
  color: #adadb8;
  cursor: pointer;
  font: 0.85em ui-monospace, SF Mono, monospace;
  user-select: none;
  border: 1px solid #2a2a2e;
}
.format-tab:hover { background: #2a2a2e; color: #efeff1; }
.format-tab.active {
  background: linear-gradient(135deg, #9146ff 0%, #6441a4 100%);
  color: white;
  border-color: #9146ff;
}
.info { min-width: 0; }
.rank-row { display: flex; align-items: center; flex-wrap: wrap; gap: 0.5em; margin-bottom: 0.7em; }
.rank { font: 600 1.3em ui-monospace, SF Mono, monospace; color: #888; }
.score {
  padding: 0.3em 0.75em;
  border-radius: 6px;
  background: linear-gradient(135deg, #9146ff 0%, #6441a4 100%);
  color: white;
  font-weight: 700;
  font-size: 1.1em;
}
.meta-pill {
  padding: 0.2em 0.6em;
  border-radius: 4px;
  background: #1f1f23;
  color: #adadb8;
  font: 0.85em ui-monospace, SF Mono, monospace;
}
.hook { font-size: 1.15em; margin: 0.4em 0 0.8em; color: #fff; font-weight: 500; }
.reasoning { color: #c8c8d0; font-size: 0.93em; margin-bottom: 1em; }
.reasoning > div { margin: 0.35em 0; }
.reasoning .llm b { color: #9eb4ff; }
.reasoning .vlm b { color: #ffb866; }
.overlay-hook {
  display: inline-block;
  margin-bottom: 1em;
  padding: 0.35em 0.85em;
  background: rgba(255,184,102,0.12);
  border-left: 3px solid #ffb866;
  border-radius: 4px;
  font-weight: 600;
  letter-spacing: 0.02em;
}
.platform-tabs { display: flex; gap: 0.3em; flex-wrap: wrap; margin: 1em 0 0.8em; }
.platform-tab {
  padding: 0.35em 0.8em;
  border-radius: 6px 6px 0 0;
  background: transparent;
  color: #888;
  cursor: pointer;
  font-size: 0.85em;
  user-select: none;
  border: 1px solid transparent;
  border-bottom: 0;
}
.platform-tab:hover { color: #efeff1; background: #1f1f23; }
.platform-tab.active {
  background: #0e0e10;
  color: #fff;
  border-color: #2a2a2e;
}
.platform-panel { display: none; padding: 1em; background: #0e0e10; border-radius: 0 8px 8px 8px; border: 1px solid #2a2a2e; }
.platform-panel.active { display: block; }
.field { display: flex; flex-direction: column; margin-bottom: 0.85em; }
.field-label { color: #888; font-size: 0.78em; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.3em; }
.field-body {
  display: flex;
  align-items: flex-start;
  gap: 0.5em;
}
.field-text {
  flex: 1;
  padding: 0.55em 0.75em;
  background: #1f1f23;
  border-radius: 4px;
  color: #efeff1;
  font: 0.93em -apple-system, BlinkMacSystemFont, system-ui;
  white-space: pre-wrap;
  word-break: break-word;
  min-height: 1em;
}
.copy-btn {
  padding: 0.45em 0.75em;
  border-radius: 4px;
  background: #2a2a2e;
  color: #efeff1;
  border: none;
  cursor: pointer;
  font-size: 0.8em;
  white-space: nowrap;
  height: fit-content;
  align-self: stretch;
}
.copy-btn:hover { background: #3a3a3e; }
.copy-btn.ok { background: #4caf50; color: white; }
.empty-platform { color: #666; font-style: italic; }
@media (max-width: 980px) {
  .clip { grid-template-columns: 1fr; }
  video { max-width: 360px; }
  .video-col { align-items: flex-start; }
}
"""

JS = r"""
function setFormat(clipId, label) {
  const card = document.getElementById('clip-' + clipId);
  if (!card) return;
  const video = card.querySelector('video');
  const sources = JSON.parse(card.dataset.sources || '{}');
  const url = sources[label];
  if (!url) return;
  const t = video.currentTime;
  const wasPlaying = !video.paused;
  video.src = url;
  video.addEventListener('loadedmetadata', () => {
    video.currentTime = Math.min(t, video.duration || t);
    if (wasPlaying) video.play().catch(() => {});
  }, { once: true });
  card.querySelectorAll('.format-tab').forEach(t => t.classList.toggle('active', t.dataset.label === label));
}
function setPlatform(clipId, platform) {
  const card = document.getElementById('clip-' + clipId);
  if (!card) return;
  card.querySelectorAll('.platform-tab').forEach(t => t.classList.toggle('active', t.dataset.platform === platform));
  card.querySelectorAll('.platform-panel').forEach(p => p.classList.toggle('active', p.dataset.platform === platform));
}
async function copyText(btn, sourceId) {
  const el = document.getElementById(sourceId);
  if (!el) return;
  const text = el.innerText;
  try {
    await navigator.clipboard.writeText(text);
    const orig = btn.textContent;
    btn.textContent = 'Copied!';
    btn.classList.add('ok');
    setTimeout(() => { btn.textContent = orig; btn.classList.remove('ok'); }, 1500);
  } catch (e) {
    btn.textContent = 'Failed';
  }
}
"""

PLATFORM_DEFS = [
    ("youtube_shorts", "YouTube Shorts", [
        ("title", "Title", "title"),
        ("description", "Description", "description"),
        ("hashtags", "Hashtags", "hashtags"),
        ("pinned_comment", "Pinned Comment", "pinned_comment"),
    ]),
    ("tiktok", "TikTok", [
        ("caption", "Caption", "caption"),
        ("hashtags", "Hashtags", "hashtags"),
    ]),
    ("instagram_reels", "Instagram Reels", [
        ("caption", "Caption", "caption"),
        ("hashtags", "Hashtags", "hashtags"),
    ]),
    ("threads", "Threads", [
        ("text", "Text", "text"),
        ("hashtags", "Hashtags", "hashtags"),
    ]),
    ("linkedin", "LinkedIn", [
        ("post_text", "Post", "post_text"),
        ("hashtags", "Hashtags", "hashtags"),
    ]),
    ("x", "X / Twitter", [
        ("text", "Text", "text"),
        ("hashtags", "Hashtags", "hashtags"),
    ]),
    ("bluesky", "Bluesky", [
        ("text", "Text", "text"),
        ("hashtags", "Hashtags", "hashtags"),
    ]),
]


def find_latest_clips_dir(work_dir: str = "./work/clipper") -> Path | None:
    work = Path(work_dir)
    if not work.exists():
        return None
    candidates: list[tuple[str, Path]] = []
    for media_dir in work.iterdir():
        if not media_dir.is_dir():
            continue
        for ts_dir in media_dir.iterdir():
            if not ts_dir.is_dir():
                continue
            clips_dir = ts_dir / "clips"
            if clips_dir.exists() and (clips_dir / "manifest.json").exists():
                candidates.append((ts_dir.name, clips_dir))
            elif clips_dir.exists() and (clips_dir / "digest.md").exists():
                # Older run without manifest — still usable in fallback mode.
                candidates.append((ts_dir.name, clips_dir))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def render_clip_card(clip: dict) -> str:
    rank = f"{clip['rank']:02d}"
    variants = clip.get("variants", [])
    if not variants:
        return ""

    sources_map = {v["label"]: v["filename"] for v in variants}
    sources_json = escape(json.dumps(sources_map), quote=True)
    default_label = variants[0]["label"]
    default_src = variants[0]["filename"]

    # Build format-switcher tabs.
    fmt_tabs = "\n".join(
        f'<span class="format-tab{" active" if v["label"] == default_label else ""}" '
        f'data-label="{escape(v["label"], quote=True)}" '
        f'onclick="setFormat({clip["rank"]}, \'{escape(v["label"], quote=True)}\')">'
        f'{escape(v["label"])} &middot; {v["width"]}x{v["height"]} &middot; '
        f'{v["bytes"]/1048576:.1f}MB</span>'
        for v in variants
    )

    # Reasoning split: the clipper joins LLM + VLM with " | vlm: ".
    reasoning = clip.get("reasoning", "")
    llm_why, vlm_why = reasoning, ""
    if " | vlm: " in reasoning:
        llm_why, vlm_why = reasoning.split(" | vlm: ", 1)

    overlay_hook = ""
    social = clip.get("social") or {}
    if isinstance(social, dict):
        overlay_hook = social.get("overlay_hook", "") or ""

    overlay_html = (
        f'<div class="overlay-hook">Overlay: "{escape(overlay_hook)}"</div>'
        if overlay_hook
        else ""
    )

    vlm_html = (
        f'<div class="vlm"><b>VLM:</b> {escape(vlm_why.strip())}</div>'
        if vlm_why.strip()
        else ""
    )

    # Per-platform tabs.
    platform_tabs = []
    platform_panels = []
    for i, (key, label, fields) in enumerate(PLATFORM_DEFS):
        active = " active" if i == 0 else ""
        platform_tabs.append(
            f'<span class="platform-tab{active}" data-platform="{key}" '
            f'onclick="setPlatform({clip["rank"]}, \'{key}\')">{escape(label)}</span>'
        )

        platform_data = social.get(key, {}) if isinstance(social, dict) else {}
        if not platform_data:
            panel_body = '<div class="empty-platform">No copy generated for this platform.</div>'
        else:
            field_blocks = []
            for fid, flabel, fkey in fields:
                value = platform_data.get(fkey, "")
                if isinstance(value, list):
                    value = " ".join(value)
                value = str(value or "")
                content_id = f"copy-{clip['rank']}-{key}-{fid}"
                field_blocks.append(
                    f'''
                    <div class="field">
                      <div class="field-label">{escape(flabel)}</div>
                      <div class="field-body">
                        <div class="field-text" id="{content_id}">{escape(value)}</div>
                        <button class="copy-btn" onclick="copyText(this, '{content_id}')">Copy</button>
                      </div>
                    </div>
                    '''
                )
            panel_body = "\n".join(field_blocks)

        platform_panels.append(
            f'<div class="platform-panel{active}" data-platform="{key}">{panel_body}</div>'
        )

    return f"""
    <div class="clip" id="clip-{clip['rank']}" data-sources='{sources_json}'>
      <div class="video-col">
        <video src="{escape(default_src, quote=True)}" controls preload="metadata" playsinline></video>
        <div class="format-tabs">{fmt_tabs}</div>
      </div>
      <div class="info">
        <div class="rank-row">
          <span class="rank">#{rank}</span>
          <span class="score">{clip['score']}</span>
          <span class="meta-pill">{escape(clip.get('time_range_mmss', ''))}</span>
          <span class="meta-pill">{int(clip.get('duration_secs', 0))}s</span>
        </div>
        <div class="hook">{escape(clip.get('hook', ''))}</div>
        {overlay_html}
        <div class="reasoning">
          <div class="llm"><b>LLM:</b> {escape(llm_why.strip())}</div>
          {vlm_html}
        </div>
        <div class="platform-tabs">{''.join(platform_tabs)}</div>
        {''.join(platform_panels)}
      </div>
    </div>
    """


def render_index(manifest: dict) -> str:
    episode = manifest.get("episode", "unknown")
    total = manifest.get("total_duration_secs", 0.0)
    clips = manifest.get("clips", [])

    if total >= 3600:
        h = int(total // 3600)
        m = int((total % 3600) // 60)
        s = int(total % 60)
        dur_str = f"{h}h {m:02d}m {s:02d}s"
    else:
        m = int(total // 60)
        s = int(total % 60)
        dur_str = f"{m}m {s:02d}s"

    cards = "\n".join(render_clip_card(c) for c in clips)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>autoseo clipper — {escape(episode)}</title>
<style>{CSS}</style>
</head>
<body>
<h1>autoseo clipper — {escape(episode)}</h1>
<div class="top-meta">{escape(dur_str)} &middot; {len(clips)} clips &middot; format switcher + platform copy below each video</div>
{cards}
<script>{JS}</script>
</body>
</html>
"""


def fallback_html(clips_dir: Path) -> str:
    """No manifest.json — render a minimal page explaining why."""
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>autoseo clipper</title>
<style>body {{ font-family: system-ui; max-width: 600px; margin: 4em auto; padding: 0 1em; color: #444; }}</style>
</head><body>
<h1>No manifest.json found</h1>
<p>This clips directory was produced by an older clipper run (before slice 6 added <code>manifest.json</code>).</p>
<p>Re-run the clipper to get the rich UI, or inspect the digest:</p>
<pre>{escape(str(clips_dir / "digest.md"))}</pre>
</body></html>
"""


class ReuseAddrServer(socketserver.TCPServer):
    allow_reuse_address = True


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("clips_dir", nargs="?", help="Path to a clips directory.")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="0.0.0.0")
    args = p.parse_args()

    clips_dir = Path(args.clips_dir) if args.clips_dir else find_latest_clips_dir()
    if not clips_dir or not clips_dir.is_dir():
        print(
            "No clips directory found. Pass one explicitly or ensure a run exists "
            "under ./work/clipper.",
            file=sys.stderr,
        )
        sys.exit(1)

    manifest_path = clips_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
            html = render_index(manifest)
        except Exception as e:
            print(f"failed to parse {manifest_path}: {e}", file=sys.stderr)
            html = fallback_html(clips_dir)
    else:
        html = fallback_html(clips_dir)

    (clips_dir / "index.html").write_text(html)
    os.chdir(clips_dir)

    handler = http.server.SimpleHTTPRequestHandler
    with ReuseAddrServer((args.host, args.port), handler) as httpd:
        bind_str = f"{args.host}:{args.port}"
        print(f"Serving {clips_dir}")
        print(f"  index:    http://{bind_str}/index.html")
        print(f"  local:    http://localhost:{args.port}/index.html")
        print(f"  ssh fwd:  ssh -L {args.port}:localhost:{args.port} <user>@<host>")
        print("Ctrl-C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")


if __name__ == "__main__":
    main()
