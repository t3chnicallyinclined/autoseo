# syntax=docker/dockerfile:1

FROM rust:1.85-bookworm AS builder
WORKDIR /app

# Build dependencies (helps incremental builds)
COPY Cargo.toml Cargo.lock* ./
COPY src ./src
COPY prompts ./prompts

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/app/target \
    cargo build --release \
    && mkdir -p /out \
    && cp -a /app/target/release/autoseo /out/autoseo

FROM debian:bookworm-slim AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /out/autoseo /usr/local/bin/autoseo
COPY --from=builder /app/prompts ./prompts

ENV WORK_DIR=/work \
    DEDUPE_FILE=/work/processed_message_ids.txt \
    RUST_LOG=info

VOLUME ["/work"]

ENTRYPOINT ["/usr/local/bin/autoseo"]
