use aniflive_tts_client::*;
use serde_json::json;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::sync::mpsc;
use std::time::Duration;

struct Reply {
    status: u16,
    media: &'static str,
    body: Vec<u8>,
    headers: String,
    delay: Duration,
}
impl Reply {
    fn json(value: serde_json::Value) -> Self {
        Self {
            status: 200,
            media: "application/json",
            body: serde_json::to_vec(&value).unwrap(),
            headers: String::new(),
            delay: Duration::ZERO,
        }
    }
}
/// Minimal test peer, never included in the SDK transport.
fn peer(replies: Vec<Reply>) -> (String, mpsc::Receiver<String>, std::thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = format!("http://{}/prefix/", listener.local_addr().unwrap());
    let (sender, receiver) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        let (mut socket, _) = listener.accept().unwrap();
        socket
            .set_read_timeout(Some(Duration::from_secs(3)))
            .unwrap();
        let mut reader = BufReader::new(socket.try_clone().unwrap());
        for reply in replies {
            let mut request = String::new();
            let mut length = 0;
            loop {
                let mut line = String::new();
                if reader.read_line(&mut line).unwrap_or(0) == 0 {
                    return;
                }
                if line.to_ascii_lowercase().starts_with("content-length:") {
                    length = line.split(':').nth(1).unwrap().trim().parse().unwrap();
                }
                request.push_str(&line);
                if line == "\r\n" {
                    break;
                }
            }
            let mut body = vec![0; length];
            reader.read_exact(&mut body).unwrap();
            request.push_str(&String::from_utf8(body).unwrap());
            let _ = sender.send(request);
            std::thread::sleep(reply.delay);
            let header = format!(
                "HTTP/1.1 {} Test\r\nContent-Type: {}\r\nContent-Length: {}\r\n{}\r\n",
                reply.status,
                reply.media,
                reply.body.len(),
                reply.headers
            );
            if socket.write_all(header.as_bytes()).is_err() {
                return;
            }
            if socket.write_all(&reply.body).is_err() {
                return;
            }
        }
    });
    (address, receiver, worker)
}

fn info(id: &str, state: &str) -> serde_json::Value {
    json!({"id":id,"object":"speech.session","state":state,"model":"m","voice_profile":"v","sample_rate":24000,"format":"pcm_s16le","continuity_policy":"Z",
    "context":{"mode":"future","policy":"Z","neural_state_continuity":true,"acoustic_latent_continuity":false,"committed_segments":0,"future_context":true},
    "audio_consumer_attached":false,"created_at":"now","updated_at":"now","segments":[],"future_session":{"nested":true}})
}
fn segment() -> serde_json::Value {
    json!({"segment_id":"s1","sequence":0,"characters":4,"language":"ja","paragraph_id":null,"pause_after_ms":0,"state":"queued","created_at":"now","updated_at":"now","error":null,"future_segment":true})
}
fn segment_request() -> SegmentRequest {
    SegmentRequest {
        segment_id: "s1".into(),
        text: "test".into(),
        language: Language::Ja,
        paragraph_id: None,
        pause_after_ms: None,
        expression: None,
        generation: Some(Generation {
            seed: Some(1234),
            ..Default::default()
        }),
    }
}
fn status(value: serde_json::Value, code: u16) -> Reply {
    let mut reply = Reply::json(value);
    reply.status = code;
    reply
}
fn pcm(id: &str) -> Reply {
    Reply {
        status: 200,
        media: "application/octet-stream",
        body: vec![1, 2, 3, 4],
        headers: format!(
            "X-TTS-Stream: pcm_s16le\r\nX-TTS-Sample-Format: s16le\r\nX-TTS-Sample-Rate: 24000\r\nX-TTS-Channels: 1\r\nX-TTS-Session-ID: {id}\r\n"
        ),
        delay: Duration::ZERO,
    }
}
#[tokio::test]
async fn finite_session_preserves_statuses_snapshots_duplicate_and_terminal_record() {
    let mut cancelled = info("sess_1", "cancelled");
    cancelled["active_audio_cancelled"] = json!(false);
    let (url, requests, worker) = peer(vec![
        status(info("sess_1", "open"), 201),
        status(segment(), 201),
        status(segment(), 200),
        Reply::json(info("sess_1", "flushing")),
        pcm("sess_1"),
        Reply::json(cancelled.clone()),
        Reply::json(cancelled),
    ]);
    let client = Client::new(ClientConfig::new(url).unwrap()).unwrap();
    let session = client
        .create_session(&SessionCreate::default())
        .await
        .unwrap();
    assert_eq!(
        session.initial_info().extra["future_session"]["nested"],
        true
    );
    assert_eq!(session.initial_info().context.extra["future_context"], true);
    assert_eq!(session.initial_info().continuity_policy, "Z");
    assert_eq!(
        session.append(&segment_request()).await.unwrap().extra["future_segment"],
        true
    );
    session.append(&segment_request()).await.unwrap();
    assert_eq!(session.flush().await.unwrap().state, "flushing");
    let audio = session.audio().await.unwrap();
    let mut bytes = Vec::new();
    while let Some(chunk) = audio.next_chunk().await.unwrap() {
        bytes.extend(chunk);
    }
    assert_eq!(bytes, vec![1, 2, 3, 4]);
    assert_eq!(
        session.close().await.unwrap().active_audio_cancelled,
        Some(false)
    );
    assert_eq!(session.get().await.unwrap().state, "cancelled");
    assert_eq!(session.initial_info().state, "open"); // snapshot is not a state cache
    let requests: Vec<_> = (0..7).map(|_| requests.recv().unwrap()).collect();
    assert!(requests[0].starts_with("POST /prefix/v1/sessions "));
    assert!(requests[0].ends_with("{}"));
    assert_eq!(
        requests[1].split("\r\n\r\n").nth(1),
        requests[2].split("\r\n\r\n").nth(1)
    );
    assert!(requests[3].starts_with("POST /prefix/v1/sessions/sess_1/flush "));
    assert!(requests[4].starts_with("GET /prefix/v1/sessions/sess_1/audio "));
    assert!(requests[5].starts_with("DELETE /prefix/v1/sessions/sess_1 "));
    assert!(requests[6].starts_with("GET /prefix/v1/sessions/sess_1 "));
    assert!(requests.iter().all(|r| !r.contains("/v1/audio/cancel")));
    worker.join().unwrap();
}
#[tokio::test]
async fn identifiers_are_one_encoded_segment_and_preserve_prefix() {
    let id = "sess/a?b#c%";
    let (url, requests, worker) = peer(vec![Reply::json(info(id, "open"))]);
    let client = Client::new(ClientConfig::new(url).unwrap()).unwrap();
    assert_eq!(client.get_session(id).await.unwrap().id(), id);
    assert!(
        requests
            .recv()
            .unwrap()
            .starts_with("GET /prefix/v1/sessions/sess%2Fa%3Fb%23c%25 ")
    );
    for id in ["", ".", ".."] {
        assert!(matches!(
            client.get_session(id).await,
            Err(ClientError::InvalidConfiguration)
        ));
    }
    worker.join().unwrap();
}
#[tokio::test]
async fn scoped_cancel_and_close_errors_are_not_hidden() {
    let (url, requests, worker) = peer(vec![
        Reply::json(info("s", "open")),
        status(json!({"code":409,"message":"conflict"}), 409),
        status(json!({"code":500,"message":"failed"}), 500),
        Reply::json(info("s", "cancelled")),
    ]);
    let client = Client::new(ClientConfig::new(url).unwrap()).unwrap();
    let session = client.get_session("s").await.unwrap();
    assert_eq!(
        session.append(&segment_request()).await.unwrap_err().kind(),
        "ConflictError"
    );
    assert_eq!(session.close().await.unwrap_err().kind(), "ServerError");
    assert_eq!(session.cancel().await.unwrap().state, "cancelled");
    let requests: Vec<_> = (0..4).map(|_| requests.recv().unwrap()).collect();
    assert!(requests[3].starts_with("POST /prefix/v1/sessions/s/cancel "));
    worker.join().unwrap();
}
#[tokio::test]
async fn malformed_or_mismatched_session_responses_fail() {
    for value in [json!({"id":"s"}), info("other", "open"), {
        let mut v = info("s", "open");
        v["sample_rate"] = json!(0);
        v
    }] {
        let (url, _, worker) = peer(vec![Reply::json(value)]);
        let client = Client::new(ClientConfig::new(url).unwrap()).unwrap();
        assert!(matches!(
            client.get_session("s").await,
            Err(ClientError::Protocol(_))
        ));
        worker.join().unwrap();
    }
    let (url, _, worker) = peer(vec![Reply::json(info("s", "open")), pcm("other")]);
    let client = Client::new(ClientConfig::new(url).unwrap()).unwrap();
    let session = client.get_session("s").await.unwrap();
    assert!(matches!(
        session.audio().await,
        Err(ClientError::Protocol(_))
    ));
    worker.join().unwrap();
}
#[tokio::test]
async fn only_last_session_owner_closes_local_audio() {
    let (url, _, worker) = peer(vec![Reply::json(info("s", "open")), pcm("s")]);
    let client = Client::new(ClientConfig::new(url).unwrap()).unwrap();
    let session = client.get_session("s").await.unwrap();
    let clone = session.clone();
    let audio = session.audio().await.unwrap();
    drop(session);
    assert!(!audio.is_closed());
    drop(clone);
    assert!(audio.is_closed());
    assert!(!client.is_closed());
    worker.join().unwrap();
}
#[tokio::test]
async fn session_create_requires_201_and_second_consumer_conflict_is_typed() {
    let (url, _, worker) = peer(vec![Reply::json(info("s", "open"))]);
    let client = Client::new(ClientConfig::new(url).unwrap()).unwrap();
    assert!(matches!(
        client.create_session(&SessionCreate::default()).await,
        Err(ClientError::Api(_))
    ));
    worker.join().unwrap();
    let (url, _, worker) = peer(vec![
        Reply::json(info("s", "open")),
        status(
            json!({"code":409,"message":"consumer already attached"}),
            409,
        ),
    ]);
    let client = Client::new(ClientConfig::new(url).unwrap()).unwrap();
    let session = client.get_session("s").await.unwrap();
    assert_eq!(session.audio().await.unwrap_err().kind(), "ConflictError");
    worker.join().unwrap();
}

async fn read_async_request(socket: &mut tokio::net::TcpStream) -> String {
    use tokio::io::AsyncReadExt;
    let mut bytes = Vec::new();
    let mut byte = [0u8; 1];
    while !bytes.ends_with(b"\r\n\r\n") {
        socket.read_exact(&mut byte).await.unwrap();
        bytes.push(byte[0]);
    }
    let head = String::from_utf8(bytes.clone())
        .unwrap()
        .to_ascii_lowercase();
    let length = head
        .lines()
        .find_map(|l| l.strip_prefix("content-length:"))
        .map(|v| v.trim().parse::<usize>().unwrap())
        .unwrap_or(0);
    let mut body = vec![0; length];
    socket.read_exact(&mut body).await.unwrap();
    bytes.extend(body);
    String::from_utf8(bytes).unwrap()
}
async fn write_async_json(
    socket: &mut tokio::net::TcpStream,
    value: serde_json::Value,
    status: u16,
) {
    use tokio::io::AsyncWriteExt;
    let body = serde_json::to_vec(&value).unwrap();
    socket.write_all(format!("HTTP/1.1 {status} Test\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",body.len()).as_bytes()).await.unwrap();
    socket.write_all(&body).await.unwrap();
}
#[tokio::test]
async fn live_audio_does_not_block_append_and_flush_on_shared_client() {
    use tokio::io::AsyncWriteExt;
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        let (mut audio, _) = listener.accept().await.unwrap();
        assert!(
            read_async_request(&mut audio)
                .await
                .starts_with("GET /v1/sessions/s ")
        );
        write_async_json(&mut audio, info("s", "open"), 200).await;
        assert!(
            read_async_request(&mut audio)
                .await
                .starts_with("GET /v1/sessions/s/audio ")
        );
        audio.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nTransfer-Encoding: chunked\r\nX-TTS-Stream: pcm_s16le\r\nX-TTS-Sample-Format: s16le\r\nX-TTS-Sample-Rate: 24000\r\nX-TTS-Channels: 1\r\nX-TTS-Session-ID: s\r\n\r\n").await.unwrap();
        let (mut control, _) = listener.accept().await.unwrap();
        assert!(
            read_async_request(&mut control)
                .await
                .starts_with("POST /v1/sessions/s/segments ")
        );
        write_async_json(&mut control, segment(), 201).await;
        audio.write_all(b"2\r\n\x01\x02\r\n").await.unwrap();
        assert!(
            read_async_request(&mut control)
                .await
                .starts_with("POST /v1/sessions/s/flush ")
        );
        write_async_json(&mut control, info("s", "flushing"), 200).await;
        audio
            .write_all(b"2\r\n\x03\x04\r\n0\r\n\r\n")
            .await
            .unwrap();
    });
    let client = Client::new(ClientConfig::new(url).unwrap()).unwrap();
    let session = client.get_session("s").await.unwrap();
    let audio = session.audio().await.unwrap();
    let reader = tokio::spawn(async move {
        let mut bytes = Vec::new();
        while let Some(chunk) = audio.next_chunk().await.unwrap() {
            bytes.extend(chunk);
        }
        bytes
    });
    tokio::time::timeout(Duration::from_secs(1), session.append(&segment_request()))
        .await
        .unwrap()
        .unwrap();
    tokio::time::timeout(Duration::from_secs(1), session.flush())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(1), reader)
            .await
            .unwrap()
            .unwrap(),
        vec![1, 2, 3, 4]
    );
    server.await.unwrap();
}
