# syntax=docker/dockerfile:1

# ── Build stage ──────────────────────────────────────────────
FROM rust:1.88-bookworm AS builder
WORKDIR /app

# Copy source
COPY Cargo.toml Cargo.lock* ./
COPY src ./src
COPY prompts ./prompts

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/app/target \
    cargo build --release \
    && mkdir -p /out \
    && cp -a /app/target/release/autoseo /out/autoseo

# ── Runtime stage ────────────────────────────────────────────
FROM debian:bookworm-slim AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        ffmpeg \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /out/autoseo /usr/local/bin/autoseo
COPY --from=builder /app/prompts ./prompts

ENV WORK_DIR=/work \
    DEDUPE_FILE=/work/processed_message_ids.txt \
    CLIPPER_DB=/work/clipper.db \
    EMBED_MODEL_DIR=/work/models/fastembed \
    RUST_LOG=info

VOLUME ["/work"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["/usr/local/bin/autoseo", "--help"]

ENTRYPOINT ["/usr/local/bin/autoseo"]
