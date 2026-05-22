//! axum-based WebSocket route at `/ws`.
//!
//! Replaces the previous standalone `tokio-tungstenite` server (formerly in
//! `crate::ws`). Multiplexing on the main HTTP port means a single
//! `cloudflared tunnel --url http://localhost:8080` covers the whole API
//! surface — no second tunnel for WS.
//!
//! Each connection subscribes to [`crate::events::EventBus`] and forwards
//! every pipeline event as a JSON text frame. The dashboard's
//! `WebSocketContext.tsx` consumes the wire shape defined in
//! [`crate::events::PipelineEvent`].

use std::sync::Arc;

use axum::{
    Router,
    extract::{
        State, WebSocketUpgrade,
        ws::{Message, WebSocket},
    },
    response::IntoResponse,
    routing::get,
};
use futures_util::{SinkExt, StreamExt};

use super::AppState;

pub fn router() -> Router<Arc<AppState>> {
    Router::new().route("/ws", get(ws_handler))
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

async fn handle_socket(socket: WebSocket, state: Arc<AppState>) {
    let (mut sink, mut incoming) = socket.split();
    let mut rx = state.event_bus.subscribe();

    // Replay the per-stage snapshot to this fresh client so it sees the
    // history of stages that already ran. Without this, opening the
    // dashboard mid-pipeline shows green stages with no inline detail —
    // the messages already fired and went into the void before this WS
    // connection existed. The reducer's PIPELINE_STAGE handler merges
    // these in just like a live event.
    for ev in state.event_bus.stage_snapshot() {
        let json = serde_json::to_string(&ev)
            .unwrap_or_else(|_| r#"{"type":"error","message":"serialization failed"}"#.to_string());
        if sink.send(Message::Text(json.into())).await.is_err() {
            return;
        }
    }

    loop {
        tokio::select! {
            // Server → client: forward broadcast events as JSON text frames.
            event = rx.recv() => {
                match event {
                    Ok(ev) => {
                        let json = serde_json::to_string(&ev)
                            .unwrap_or_else(|_| r#"{"type":"error","message":"serialization failed"}"#.to_string());
                        if sink.send(Message::Text(json.into())).await.is_err() {
                            break;
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                        tracing::warn!(skipped = n, "ws client lagged; emitting warning frame");
                        let warn = serde_json::json!({
                            "type": "warning",
                            "message": format!("dropped {n} events (slow consumer)"),
                        });
                        if sink.send(Message::Text(warn.to_string().into())).await.is_err() {
                            break;
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                }
            }
            // Client → server: ignore inbound text frames, handle Close/Ping.
            msg = incoming.next() => {
                match msg {
                    Some(Ok(Message::Ping(data))) => {
                        let _ = sink.send(Message::Pong(data)).await;
                    }
                    Some(Ok(Message::Close(_))) | None => break,
                    Some(Err(_)) => break,
                    _ => {}
                }
            }
        }
    }
}
