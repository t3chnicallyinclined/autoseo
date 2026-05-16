//! WebSocket server for streaming pipeline events to dashboard clients.
//!
//! Listens on a configurable TCP port and upgrades HTTP connections to
//! WebSocket. Each connected client receives all pipeline events from the
//! broadcast channel. Multiple clients are supported simultaneously.

use futures_util::{SinkExt, StreamExt};
use std::net::SocketAddr;
use tokio::net::{TcpListener, TcpStream};
use tokio_tungstenite::tungstenite::protocol::CloseFrame;
use tokio_tungstenite::tungstenite::protocol::frame::coding::CloseCode;
use tokio_tungstenite::tungstenite::Message;

use crate::events::EventBus;

/// Start the WebSocket server. Runs until the returned `JoinHandle` is dropped
/// or aborted. Designed to be spawned as a background task.
pub async fn serve(addr: SocketAddr, bus: EventBus) -> anyhow::Result<()> {
    let listener = TcpListener::bind(addr).await?;
    tracing::info!(%addr, "websocket server listening");

    loop {
        let (stream, peer) = listener.accept().await?;
        let bus = bus.clone();
        tokio::spawn(async move {
            if let Err(e) = handle_connection(stream, peer, bus).await {
                tracing::debug!(%peer, error = %e, "websocket connection ended");
            }
        });
    }
}

async fn handle_connection(
    stream: TcpStream,
    peer: SocketAddr,
    bus: EventBus,
) -> anyhow::Result<()> {
    let ws_stream = tokio_tungstenite::accept_async(stream).await?;
    tracing::info!(%peer, "websocket client connected");

    let (mut sink, mut incoming) = ws_stream.split();
    let mut rx = bus.subscribe();

    loop {
        tokio::select! {
            // Forward pipeline events to the client.
            event = rx.recv() => {
                match event {
                    Ok(ev) => {
                        let json = serde_json::to_string(&ev)
                            .unwrap_or_else(|_| r#"{"type":"error","data":{"message":"serialization failed"}}"#.to_string());
                        if sink.send(Message::Text(json.into())).await.is_err() {
                            break;
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                        tracing::warn!(%peer, skipped = n, "websocket client lagged");
                        let msg = serde_json::json!({
                            "type": "warning",
                            "data": { "message": format!("dropped {n} events (slow consumer)") }
                        });
                        let _ = sink.send(Message::Text(msg.to_string().into())).await;
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                        break;
                    }
                }
            }
            // Handle incoming messages (ping/pong/close).
            msg = incoming.next() => {
                match msg {
                    Some(Ok(Message::Ping(data))) => {
                        let _ = sink.send(Message::Pong(data)).await;
                    }
                    Some(Ok(Message::Close(_))) | None => {
                        break;
                    }
                    Some(Err(_)) => {
                        break;
                    }
                    _ => {}
                }
            }
        }
    }

    // Send close frame with reconnect hint.
    let close = CloseFrame {
        code: CloseCode::Normal,
        reason: "server closing — reconnect to resume".into(),
    };
    let _ = sink.send(Message::Close(Some(close))).await;
    tracing::info!(%peer, "websocket client disconnected");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::PipelineEvent;

    #[tokio::test]
    async fn ws_server_accepts_and_broadcasts() {
        let bus = EventBus::new();

        // Bind to an ephemeral port.
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);

        let bus_clone = bus.clone();
        let server = tokio::spawn(async move {
            serve(addr, bus_clone).await.ok();
        });

        // Give server time to start listening.
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // Connect a client.
        let url = format!("ws://{addr}");
        let (ws, _) = tokio_tungstenite::connect_async(&url).await.unwrap();
        let (mut _sink, mut stream) = ws.split();

        // Emit an event from the bus.
        bus.emit(PipelineEvent::JobComplete {
            job_id: "test-job".into(),
            clips_count: 3,
        });

        // Client should receive it.
        let msg = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            stream.next(),
        )
        .await
        .expect("timeout")
        .expect("stream ended")
        .expect("ws error");

        if let Message::Text(text) = msg {
            let v: serde_json::Value = serde_json::from_str(&text).unwrap();
            assert_eq!(v["type"], "job_complete");
            assert_eq!(v["data"]["clips_count"], 3);
        } else {
            panic!("expected text message, got: {:?}", msg);
        }

        server.abort();
    }
}
