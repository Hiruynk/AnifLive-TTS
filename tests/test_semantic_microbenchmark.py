from __future__ import annotations

from scripts.benchmark_semantic_micro import _int_header, _p95, _summarize


def test_p95_uses_nearest_rank() -> None:
    assert _p95([float(value) for value in range(1, 101)]) == 95.0


def test_headers_are_case_normalized_by_the_caller() -> None:
    headers = {"x-tts-semantic-nfe": "12"}
    assert _int_header(headers, "X-TTS-Semantic-NFE") == 12


def test_semantic_summary_keeps_backend_identity() -> None:
    row = {
        "wall_ms": 10.0,
        "server_ms": 9.0,
        "gpt_encoder_ms": 1.0,
        "gpt_decode_ms": 5.0,
        "first_preview_semantic_ms": 2.0,
        "semantic_tokens": 20,
        "semantic_nfe": 10,
        "semantic_tokens_per_nfe": 2.0,
        "semantic_tokens_per_second": 4000.0,
        "host_sync_count": 2,
        "host_sync_ms": 1.0,
        "attention_kv_bytes": 1024,
        "mamba_state_bytes": 0,
        "semantic_backend": "transformer",
    }
    summary = _summarize([row])
    assert summary["semantic_backends"] == ["transformer"]
    assert summary["metrics"]["semantic_tokens_per_nfe"]["p50"] == 2.0

