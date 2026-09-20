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
fn health() -> serde_json::Value {
    json!({"service":"AnifLive-TTS","version":"9.0.0","ready":false,"status":"not_ready",
           "backend":"future-cpu","new_field":{"nested":true}})
}
fn caps() -> serde_json::Value {
    json!({"service":"AnifLive-TTS","version":"9.0.0","languages":["ja","future"],
           "streaming":{"pcm16":true,"wav":false},
           "expression":{"native":true,"controlled_profiles":false},
           "speech_sessions":{"committed_segments":true,"pcm_stream":true,"policies":{"Z":{}}}})
}
#[test]
fn canonical_requests_omit_absent_fields_and_preserve_expression_schema() {
    let mut request = SynthesisRequest::new("test", Language::Yue);
    request.generation = Some(Generation {
        seed: Some(1234),
        ..Generation::default()
    });
    request.expression = Some(Expression {
        enabled: Some(true),
        policy: Some(ConditioningPolicy::IdentityLock),
        ..Expression::default()
    });
    assert_eq!(
        serde_json::to_value(request).unwrap(),
        json!({
        "text":"test","language":"yue","generation":{"seed":1234},
        "expression":{"enabled":true,"policy":"identity-lock"}})
    );
    assert_eq!(
        serde_json::to_string(&ContinuityPolicy::F).unwrap(),
        "\"F\""
    );
}
#[tokio::test]
async fn probe_accepts_unready_future_backend_and_reuses_one_connection() {
    let (url, requests, worker) = peer(vec![Reply::json(health()), Reply::json(caps())]);
    let client = Client::new(ClientConfig::new(&url).unwrap()).unwrap();
    let info = client
        .probe(&[
            RequiredFeature::Expressions,
            RequiredFeature::SpeechSessions,
        ])
        .await
        .unwrap();
    assert!(!info.health.ready);
    assert!(!info.capabilities.expression.controlled_profiles);
    assert_eq!(info.health.extra["new_field"]["nested"], true);
    assert!(
        info.capabilities
            .speech_sessions
            .unwrap()
            .policies
            .contains_key("Z")
    );
    let first = requests.recv().unwrap().to_ascii_lowercase();
    assert!(first.starts_with("get /prefix/health "));
    assert!(first.contains("user-agent: aniflive-tts-python-sdk/"));
    assert!(
        requests
            .recv()
            .unwrap()
            .starts_with("GET /prefix/v1/capabilities ")
    );
    worker.join().unwrap(); // Both requests traversed the only accepted socket.
}
#[tokio::test]
async fn probe_checks_both_identities_and_requested_features() {
    let mut bad = caps();
    bad["service"] = json!("other");
    let (url, _, worker) = peer(vec![Reply::json(health()), Reply::json(bad)]);
    let error = Client::new(ClientConfig::new(url).unwrap())
        .unwrap()
        .probe(&[])
        .await
        .unwrap_err();
    assert_eq!(error.kind(), "CompatibilityError");
    worker.join().unwrap();
    let mut unsupported = caps();
    unsupported["streaming"]["pcm16"] = json!(false);
    let (url, _, worker) = peer(vec![Reply::json(health()), Reply::json(unsupported)]);
    let result = Client::new(ClientConfig::new(url).unwrap())
        .unwrap()
        .probe(&[RequiredFeature::PcmStreaming])
        .await;
    assert_eq!(result.unwrap_err().kind(), "CompatibilityError");
    worker.join().unwrap();
}
#[tokio::test]
async fn complete_wav_preserves_bytes_and_forces_canonical_mode() {
    let wav = b"RIFF\x04\x00\x00\x00WAVE".to_vec();
    let reply = Reply {
        status: 200,
        media: "audio/wav",
        body: wav.clone(),
        headers: "X-TTS-Sample-Rate: 32000\r\nX-TTS-Model: model\r\nSet-Cookie: secret\r\n".into(),
        delay: Duration::ZERO,
    };
    let (url, records, worker) = peer(vec![reply]);
    let result = Client::new(ClientConfig::new(url).unwrap())
        .unwrap()
        .synthesize(&SynthesisRequest::new("private text", Language::Ja))
        .await
        .unwrap();
    assert_eq!(result.audio, wav);
    assert_eq!(result.metadata.sample_rate, Some(32000));
    assert!(!result.metadata.headers.contains_key("set-cookie"));
    let record = records.recv().unwrap();
    let body: serde_json::Value =
        serde_json::from_str(record.split("\r\n\r\n").nth(1).unwrap()).unwrap();
    assert_eq!(body["stream"], false);
    assert_eq!(body["response_format"], "wav");
    assert!(!format!("{result:?}").contains("private"));
    worker.join().unwrap();
}
#[tokio::test]
async fn error_status_mapping_preserves_attributes_but_redacts_debug() {
    for (status, kind) in [
        (400, "BadRequestError"),
        (404, "NotFoundError"),
        (409, "ConflictError"),
        (413, "PayloadTooLargeError"),
        (429, "BusyError"),
        (500, "ServerError"),
        (401, "APIError"),
    ] {
        let mut reply = Reply::json(json!({"code":status,"message":"private secret"}));
        reply.status = status;
        reply.headers = "Retry-After: 1\r\n".into();
        let (url, _, worker) = peer(vec![reply]);
        let error = Client::new(ClientConfig::new(url).unwrap())
            .unwrap()
            .health()
            .await
            .unwrap_err();
        assert_eq!(error.kind(), kind);
        assert!(!format!("{error:?} {error}").contains("secret"));
        if let ClientError::Api(api) = error {
            assert_eq!(api.status_code, status);
            assert_eq!(api.retry_after.as_deref(), Some("1"));
            assert_eq!(api.message, "private secret");
        } else {
            panic!("expected API error");
        }
        worker.join().unwrap();
    }
}
#[tokio::test]
async fn malformed_json_and_body_limit_are_protocol_errors() {
    for body in [b"not-json".to_vec(), vec![b'x'; MAX_JSON_BYTES + 1]] {
        let reply = Reply {
            status: 200,
            media: "application/json",
            body,
            headers: String::new(),
            delay: Duration::ZERO,
        };
        let (url, _, worker) = peer(vec![reply]);
        let error = Client::new(ClientConfig::new(url).unwrap())
            .unwrap()
            .health()
            .await
            .unwrap_err();
        assert_eq!(error.kind(), "ProtocolError");
        worker.join().unwrap();
    }
}
#[tokio::test]
async fn redirects_are_not_followed_and_errors_are_bounded() {
    let reply = Reply {
        status: 302,
        media: "text/html",
        body: vec![b'x'; MAX_ERROR_BYTES + 100],
        headers: "Location: http://invalid.example/secret\r\n".into(),
        delay: Duration::ZERO,
    };
    let (url, _, worker) = peer(vec![reply]);
    let error = Client::new(ClientConfig::new(url).unwrap())
        .unwrap()
        .health()
        .await
        .unwrap_err();
    if let ClientError::Api(api) = error {
        assert_eq!(api.status_code, 302);
        assert_eq!(api.diagnostic.unwrap().len(), MAX_ERROR_BYTES);
    } else {
        panic!("expected redirect status");
    }
    worker.join().unwrap();
}
#[tokio::test]
async fn timeout_and_connection_failures_are_redacted() {
    let mut reply = Reply::json(health());
    reply.delay = Duration::from_millis(100);
    let (url, _, worker) = peer(vec![reply]);
    let config = ClientConfig::new(url)
        .unwrap()
        .with_request_timeout(Some(Duration::from_millis(10)));
    assert_eq!(
        Client::new(config)
            .unwrap()
            .health()
            .await
            .unwrap_err()
            .kind(),
        "TimeoutError"
    );
    worker.join().unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    drop(listener);
    let error = Client::new(ClientConfig::new(url).unwrap())
        .unwrap()
        .health()
        .await
        .unwrap_err();
    assert_eq!(error.kind(), "TransportError");
    assert!(!format!("{error:?}").contains("127.0.0.1"));
}
#[tokio::test]
async fn malformed_sample_rate_and_wav_limit_fail_closed() {
    for (headers, limit) in [("X-TTS-Sample-Rate: zero\r\n", 1024), ("", 4)] {
        let reply = Reply {
            status: 200,
            media: "audio/wav",
            body: b"RIFFxxxxWAVE".to_vec(),
            headers: headers.into(),
            delay: Duration::ZERO,
        };
        let (url, _, worker) = peer(vec![reply]);
        let client =
            Client::new(ClientConfig::new(url).unwrap().with_max_wav_bytes(limit)).unwrap();
        let error = client
            .synthesize(&SynthesisRequest::new("text", Language::Ja))
            .await
            .unwrap_err();
        assert_eq!(error.kind(), "ProtocolError");
        worker.join().unwrap();
    }
}
#[tokio::test]
async fn discovery_activation_and_cancel_use_typed_canonical_routes() {
    let replies = vec![
        Reply::json(
            json!({"object":"list","data":[{"id":"m","object":"model","active":true,"voice_profiles":["v"],"future":42}]}),
        ),
        Reply::json(
            json!({"object":"list","data":[{"id":"v","reference_language":"ja","reference_configured":true}]}),
        ),
        Reply::json(
            json!({"object":"list","enabled":false,"default":null,"profiles":[],"policies":[]}),
        ),
        Reply::json(json!({"code":0,"message":"Success","changed":false,"model":"m","voice":"v"})),
        Reply::json(json!({"cancelled":false,"model":"m"})),
    ];
    let (url, records, worker) = peer(replies);
    let client = Client::new(ClientConfig::new(url).unwrap()).unwrap();
    assert_eq!(client.models().await.unwrap().data[0].extra["future"], 42);
    assert_eq!(client.voices().await.unwrap().data[0].id, "v");
    assert!(!client.expressions().await.unwrap().enabled);
    assert!(!client.activate_model("m").await.unwrap().changed);
    assert!(!client.cancel().await.unwrap().cancelled);
    let paths: Vec<_> = records
        .try_iter()
        .map(|s| s.lines().next().unwrap().to_owned())
        .collect();
    assert_eq!(paths.len(), 5);
    assert!(paths[3].starts_with("POST /prefix/v1/models/activate "));
    assert!(paths[4].starts_with("POST /prefix/v1/audio/cancel "));
    worker.join().unwrap();
}

#[tokio::test]
async fn received_http_status_survives_stalled_error_diagnostics() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let worker = std::thread::spawn(move || {
        let (mut socket, _) = listener.accept().unwrap();
        socket
            .set_read_timeout(Some(Duration::from_secs(3)))
            .unwrap();
        let mut request = [0; 4096];
        let _ = socket.read(&mut request);
        socket.write_all(b"HTTP/1.1 429 Busy\r\nContent-Type: application/json\r\nContent-Length: 99\r\nRetry-After: 1\r\n\r\n").unwrap();
        std::thread::sleep(Duration::from_millis(150));
    });
    let config = ClientConfig::new(url)
        .unwrap()
        .with_request_timeout(Some(Duration::from_millis(50)));
    let error = Client::new(config).unwrap().health().await.unwrap_err();
    assert_eq!(error.kind(), "BusyError");
    if let ClientError::Api(api) = error {
        assert_eq!(api.retry_after.as_deref(), Some("1"));
    }
    worker.join().unwrap();
}
#[tokio::test]
async fn invalid_request_floats_fail_before_network() {
    let client = Client::new(ClientConfig::new("http://127.0.0.1:1").unwrap()).unwrap();
    let mut request = SynthesisRequest::new("text", Language::Ja);
    request.generation = Some(Generation {
        temperature: Some(f64::NAN),
        ..Generation::default()
    });
    assert!(matches!(
        client.synthesize(&request).await,
        Err(ClientError::InvalidConfiguration)
    ));
    request.generation = None;
    request.segments = Some(vec![]);
    assert!(matches!(
        client.synthesize(&request).await,
        Err(ClientError::InvalidConfiguration)
    ));
}
#[test]
fn required_response_fields_are_checked_but_future_fields_are_preserved() {
    assert!(serde_json::from_value::<Health>(json!({"service":"AnifLive-TTS"})).is_err());
    let catalog: ExpressionCatalogInfo = serde_json::from_value(json!({
        "object":"list","enabled":true,"default":"calm","policies":["future-policy"],
        "profiles":[{"id":"calm","intensity_levels":[0.5],"languages":["ja"],
            "reference_languages":["ja"],"preferred_policies":["future-policy"],"future":true}],
        "future":{"nested":true}
    }))
    .unwrap();
    assert_eq!(catalog.profiles[0].preferred_policies[0], "future-policy");
    assert_eq!(catalog.extra["future"]["nested"], true);
}

#[test]
fn expression_catalog_preserves_structured_playback_and_runtime_policies() {
    let value = serde_json::json!({
        "object": "list", "enabled": true, "default": null,
        "profiles": [], "policies": [],
        "playback_policy": {"short_prebuffer_ms": 120},
        "runtime_policy": {"first_context_tokens": 8}
    });
    let catalog: ExpressionCatalogInfo = serde_json::from_value(value).unwrap();
    assert_eq!(catalog.playback_policy.unwrap()["short_prebuffer_ms"], 120);
    assert_eq!(catalog.runtime_policy.unwrap()["first_context_tokens"], 8);
}
