use aniflive_tts_client::*;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::oneshot;
fn request() -> SynthesisRequest {
    SynthesisRequest::new("test", Language::Ja)
}
fn pcm(channels: u16) -> String {
    format!(
        "Content-Type: application/octet-stream\r\nX-TTS-Stream: pcm_s16le\r\nX-TTS-Sample-Format: s16le\r\nX-TTS-Sample-Rate: 24000\r\nX-TTS-Channels: {channels}\r\n"
    )
}
async fn peer(
    headers: String,
    chunks: Vec<(u64, Vec<u8>)>,
    header_delay: u64,
) -> (String, oneshot::Receiver<()>, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let (tx, rx) = oneshot::channel();
    let task = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut request = Vec::new();
        let mut buffer = [0u8; 1024];
        loop {
            let n = socket.read(&mut buffer).await.unwrap();
            if n == 0 {
                return;
            }
            request.extend_from_slice(&buffer[..n]);
            if let Some(end) = request.windows(4).position(|w| w == b"\r\n\r\n") {
                let head = String::from_utf8_lossy(&request[..end]).to_ascii_lowercase();
                let length = head
                    .lines()
                    .find_map(|line| line.strip_prefix("content-length:"))
                    .map(|n| n.trim().parse::<usize>().unwrap())
                    .unwrap_or(0);
                if request.len() >= end + 4 + length {
                    break;
                }
            }
        }
        let _ = tx.send(());
        tokio::time::sleep(Duration::from_millis(header_delay)).await;
        if socket
            .write_all(
                format!("HTTP/1.1 200 OK\r\n{headers}Transfer-Encoding: chunked\r\n\r\n")
                    .as_bytes(),
            )
            .await
            .is_err()
        {
            return;
        }
        for (delay, body) in chunks {
            tokio::time::sleep(Duration::from_millis(delay)).await;
            if socket
                .write_all(format!("{:x}\r\n", body.len()).as_bytes())
                .await
                .is_err()
            {
                return;
            }
            if socket.write_all(&body).await.is_err() {
                return;
            }
            if socket.write_all(b"\r\n").await.is_err() {
                return;
            }
        }
        let _ = socket.write_all(b"0\r\n\r\n").await;
    });
    (url, rx, task)
}
#[tokio::test]
async fn exact_bytes_are_incremental_bounded_and_multichannel() {
    let (url, _, server) = peer(
        pcm(2),
        vec![(0, vec![1, 2, 3]), (100, vec![4, 5, 6, 7, 8])],
        0,
    )
    .await;
    let client = Client::new(ClientConfig::new(url).unwrap().with_chunk_bytes(2)).unwrap();
    let stream = client.stream(&request()).await.unwrap();
    assert_eq!(stream.channels(), 2);
    assert_eq!(stream.sample_rate(), 24000);
    let first = tokio::time::timeout(Duration::from_millis(75), stream.next_chunk())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(!first.is_empty());
    let mut bytes = first;
    while let Some(chunk) = stream.next_chunk().await.unwrap() {
        assert!(chunk.len() <= 2);
        bytes.extend(chunk);
    }
    assert_eq!(bytes, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    assert!(stream.is_closed());
    server.await.unwrap();
}
#[tokio::test]
async fn consumer_delay_is_not_idle_timeout() {
    let (url, _, server) = peer(pcm(1), vec![(0, vec![1, 2]), (0, vec![3, 4])], 0).await;
    let client = Client::new(
        ClientConfig::new(url)
            .unwrap()
            .with_stream_idle_timeout(Some(Duration::from_millis(40))),
    )
    .unwrap();
    let stream = client.stream(&request()).await.unwrap();
    assert!(stream.next_chunk().await.unwrap().is_some());
    tokio::time::sleep(Duration::from_millis(100)).await;
    while stream.next_chunk().await.unwrap().is_some() {}
    server.await.unwrap();
}
#[tokio::test]
async fn network_idle_timeout_closes_stream() {
    let (url, _, server) = peer(pcm(1), vec![(300, vec![1, 2])], 0).await;
    let client = Client::new(
        ClientConfig::new(url)
            .unwrap()
            .with_stream_idle_timeout(Some(Duration::from_millis(30))),
    )
    .unwrap();
    let stream = client.stream(&request()).await.unwrap();
    assert!(matches!(
        stream.next_chunk().await,
        Err(ClientError::Timeout)
    ));
    assert!(stream.is_closed());
    assert!(stream.next_chunk().await.unwrap().is_none());
    server.abort();
}
#[tokio::test]
async fn close_interrupts_pending_read_and_rejects_concurrent_read() {
    let (url, _, server) = peer(pcm(1), vec![(1000, vec![1, 2])], 0).await;
    let client = Client::new(ClientConfig::new(url).unwrap()).unwrap();
    let stream = client.stream(&request()).await.unwrap();
    let reader = stream.clone();
    let task = tokio::spawn(async move { reader.next_chunk().await });
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert!(matches!(
        stream.next_chunk().await,
        Err(ClientError::ConcurrentRead)
    ));
    stream.close();
    assert!(
        tokio::time::timeout(Duration::from_millis(100), task)
            .await
            .unwrap()
            .unwrap()
            .unwrap()
            .is_none()
    );
    assert!(!client.is_closed());
    server.abort();
}
#[tokio::test]
async fn client_close_interrupts_read_and_closes_idle_stream() {
    for read_pending in [false, true] {
        let (url, _, server) = peer(pcm(1), vec![(1000, vec![1, 2])], 0).await;
        let client = Client::new(ClientConfig::new(url).unwrap()).unwrap();
        let stream = client.stream(&request()).await.unwrap();
        let task = if read_pending {
            let reader = stream.clone();
            Some(tokio::spawn(async move { reader.next_chunk().await }))
        } else {
            None
        };
        tokio::time::sleep(Duration::from_millis(10)).await;
        client.close();
        assert!(stream.is_closed());
        assert!(stream.next_chunk().await.unwrap().is_none());
        if let Some(task) = task {
            assert!(task.await.unwrap().unwrap().is_none());
        }
        assert!(matches!(client.health().await, Err(ClientError::Closed)));
        server.abort();
    }
}
#[tokio::test]
async fn client_close_interrupts_pending_health_and_wav() {
    for wav in [false, true] {
        let (url, received, server) = peer(
            "Content-Type: audio/wav\r\n".to_owned(),
            vec![(1000, vec![1, 2])],
            if wav { 0 } else { 1000 },
        )
        .await;
        let client = Client::new(ClientConfig::new(url).unwrap()).unwrap();
        let caller = client.clone();
        let task = tokio::spawn(async move {
            if wav {
                caller.synthesize(&request()).await.map(|_| ())
            } else {
                caller.health().await.map(|_| ())
            }
        });
        received.await.unwrap();
        if wav {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        client.close();
        assert!(matches!(
            tokio::time::timeout(Duration::from_millis(100), task)
                .await
                .unwrap()
                .unwrap(),
            Err(ClientError::Closed)
        ));
        server.abort();
    }
}
#[tokio::test]
async fn dropping_read_future_releases_stream() {
    let (url, _, server) = peer(pcm(1), vec![(1000, vec![1, 2])], 0).await;
    let client = Client::new(ClientConfig::new(url).unwrap()).unwrap();
    let stream = client.stream(&request()).await.unwrap();
    assert!(
        tokio::time::timeout(Duration::from_millis(20), stream.next_chunk())
            .await
            .is_err()
    );
    assert!(stream.is_closed());
    assert!(stream.next_chunk().await.unwrap().is_none());
    server.abort();
}
#[tokio::test]
async fn malformed_metadata_and_partial_final_frames_fail_closed() {
    for headers in [
        pcm(0),
        pcm(1).replace("24000", "invalid"),
        pcm(1).replace("s16le", "float32"),
        pcm(1).replace("application/octet-stream", "audio/wav"),
    ] {
        let (url, _, server) = peer(headers, vec![], 0).await;
        let client = Client::new(ClientConfig::new(url).unwrap()).unwrap();
        assert!(matches!(
            client.stream(&request()).await,
            Err(ClientError::Protocol(_))
        ));
        server.await.unwrap();
    }
    let (url, _, server) = peer(pcm(2), vec![(0, vec![1, 2, 3])], 0).await;
    let client = Client::new(ClientConfig::new(url).unwrap()).unwrap();
    let stream = client.stream(&request()).await.unwrap();
    assert_eq!(stream.next_chunk().await.unwrap().unwrap(), vec![1, 2, 3]);
    assert!(matches!(
        stream.next_chunk().await,
        Err(ClientError::Protocol(_))
    ));
    assert!(stream.is_closed());
    server.await.unwrap();
}

#[tokio::test]
async fn client_close_releases_idle_socket_without_remote_cancel() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut bytes = Vec::new();
        let mut buffer = [0u8; 1024];
        loop {
            let n = socket.read(&mut buffer).await.unwrap();
            bytes.extend_from_slice(&buffer[..n]);
            if let Some(end) = bytes.windows(4).position(|w| w == b"\r\n\r\n") {
                let head = String::from_utf8_lossy(&bytes[..end]).to_ascii_lowercase();
                let length = head
                    .lines()
                    .find_map(|line| line.strip_prefix("content-length:"))
                    .unwrap()
                    .trim()
                    .parse::<usize>()
                    .unwrap();
                if bytes.len() == end + 4 + length {
                    break;
                }
            }
        }
        assert!(bytes.starts_with(b"POST /v1/audio/speech "));
        socket
            .write_all(
                format!(
                    "HTTP/1.1 200 OK\r\n{}Transfer-Encoding: chunked\r\n\r\n",
                    pcm(1)
                )
                .as_bytes(),
            )
            .await
            .unwrap();
        // The unfinished response must be disconnected, with no HTTP cancel request.
        let n = tokio::time::timeout(Duration::from_secs(1), socket.read(&mut buffer))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(n, 0);
        assert!(
            tokio::time::timeout(Duration::from_millis(30), listener.accept())
                .await
                .is_err()
        );
    });
    let client = Client::new(ClientConfig::new(url).unwrap()).unwrap();
    let stream = client.stream(&request()).await.unwrap();
    client.close();
    assert!(stream.is_closed());
    server.await.unwrap();
}

#[tokio::test]
async fn closed_live_client_releases_idle_keepalive_pool() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut request = Vec::new();
        let mut buffer = [0u8; 1024];
        while !request.windows(4).any(|w| w == b"\r\n\r\n") {
            let n = socket.read(&mut buffer).await.unwrap();
            assert!(n > 0);
            request.extend_from_slice(&buffer[..n]);
        }
        let body =
            br#"{"service":"AnifLive-TTS","version":"future","ready":false,"status":"not_ready"}"#;
        socket.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",body.len()).as_bytes()).await.unwrap();
        socket.write_all(body).await.unwrap();
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(1), socket.read(&mut buffer))
                .await
                .unwrap()
                .unwrap(),
            0
        );
    });
    let client = Client::new(ClientConfig::new(url).unwrap()).unwrap();
    client.health().await.unwrap();
    client.close();
    server.await.unwrap();
    assert!(client.is_closed()); // Keep the closed handle alive until the peer observed EOF.
}
