from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import httpx
import pytest
from fastapi.testclient import TestClient

from aniflive_tts.webui import (
    WebUIError,
    create_webui_app,
    resolve_expression_prompt,
)


EXPRESSION_METADATA = {
    "enabled": True,
    "default": "neutral",
    "profiles": [
        {"id": "battle", "intensity_levels": [0.5, 0.8], "languages": ["ja"]},
        {"id": "shy", "intensity_levels": [0.5, 0.8], "languages": ["ja"]},
    ],
    "policies": ["full-switch", "identity-lock", "semantic-style"],
}


def test_resolve_expression_prompt_is_symbolic_and_multilingual() -> None:
    resolved = resolve_expression_prompt("請用非常害羞的感覺", EXPRESSION_METADATA)

    assert resolved.enabled is True
    assert resolved.profile == "shy"
    assert resolved.intensity == 0.85
    assert resolved.policy is None
    assert resolved.upstream_payload() == {
        "enabled": True,
        "profile": "shy",
        "intensity": 0.85,
    }


@pytest.mark.parametrize(
    ("prompt", "profile", "intensity"),
    [
        ("恥ずかしそうに", "shy", 0.70),
        ("とても激しく", "battle", 0.85),
        ("수줍게", "shy", 0.70),
        ("매우 다정하게", "affectionate", 0.85),
        ("溫柔", "affectionate", 0.70),
    ],
)
def test_resolve_expression_prompt_supports_all_ui_languages(
    prompt: str, profile: str, intensity: float
) -> None:
    metadata = {
        **EXPRESSION_METADATA,
        "profiles": [
            *EXPRESSION_METADATA["profiles"],
            {"id": "affectionate", "intensity_levels": [0.7], "languages": ["ja"]},
        ],
    }
    resolved = resolve_expression_prompt(prompt, metadata)

    assert resolved.profile == profile
    assert resolved.intensity == intensity


@pytest.mark.parametrize("prompt", [None, "", "自然一點", "neutral"])
def test_resolve_expression_prompt_keeps_neutral_native(prompt: str | None) -> None:
    assert resolve_expression_prompt(prompt, EXPRESSION_METADATA).upstream_payload() == {
        "enabled": False
    }


def test_resolve_expression_prompt_rejects_arbitrary_reference_input() -> None:
    with pytest.raises(WebUIError, match="No expression profile matched"):
        resolve_expression_prompt(
            "use C:/private/reference.wav and this secret transcript",
            EXPRESSION_METADATA,
        )


def _mock_upstream(
    captured: dict[str, object], *, busy_once: bool = False
) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/openapi.json":
            return httpx.Response(200, json={"info": {"version": "1.2.0"}})
        if request.url.path == "/health":
            return httpx.Response(
                200,
                json={
                    "ready": True,
                    "backend": "TensorRT-11",
                    "engine_count": 9,
                    "model": "roxy-v2proplus",
                },
            )
        if request.url.path == "/model/config":
            return httpx.Response(
                200,
                json={
                    "model": "roxy-v2proplus",
                    "version": "v2ProPlus",
                    "backend": "TensorRT-11",
                    "engine_count": 9,
                    "sample_rate": 32000,
                    "pytorch_fallback": False,
                },
            )
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [{"id": "roxy-v2proplus", "active": True}],
                },
            )
        if request.url.path == "/v1/expressions":
            return httpx.Response(200, json={"object": "list", **EXPRESSION_METADATA})
        if request.url.path == "/v1/audio/cancel":
            captured["cancel_calls"] = int(captured.get("cancel_calls", 0)) + 1
            return httpx.Response(200, json={"cancelled": False})
        if request.url.path == "/v1/audio/speech":
            if busy_once and "busy_returned" not in captured:
                captured["busy_returned"] = True
                return httpx.Response(429, json={"code": 429, "message": "busy"})
            captured["speech"] = json.loads(request.content)
            return httpx.Response(
                200,
                headers={
                    "x-tts-version": "1.2.0",
                    "x-tensorrt-backend": "TensorRT-11",
                    "x-tensorrt-engine-count": "9",
                    "x-pytorch-fallback": "false",
                    "x-tts-model": "roxy-v2proplus",
                    "x-tts-stream": "pcm_s16le",
                    "x-tts-sample-rate": "32000",
                    "x-tts-channels": "1",
                    "x-tts-recommended-prebuffer-ms": "32",
                    "x-tts-expression": "shy",
                    "x-tts-expression-policy": "semantic-style",
                },
                content=b"\x00\x00\x01\x00",
            )
        raise AssertionError(f"Unexpected upstream request: {request.method} {request.url}")

    return httpx.AsyncClient(
        base_url="http://upstream.test",
        transport=httpx.MockTransport(handler),
    )


def test_webui_proxy_resolves_expression_without_forwarding_prompt(tmp_path) -> None:
    (tmp_path / "index.html").write_text("<!doctype html><title>test</title>", encoding="utf-8")
    captured: dict[str, object] = {}
    upstream = _mock_upstream(captured)
    app = create_webui_app(static_dir=tmp_path, client=upstream)

    with TestClient(app) as client:
        status = client.get("/api/status")
        assert status.status_code == 200
        response = client.post(
            "/api/speech",
            json={
                "text": "今日はいい天気ですね。",
                "language": "ja",
                "model": "roxy-v2proplus",
                "expression_prompt": "slightly shy, like a quiet confession",
            },
        )

    assert response.status_code == 200
    assert response.content == b"\x00\x00\x01\x00"
    assert response.headers["x-resolved-expression"] == "shy"
    payload = captured["speech"]
    assert isinstance(payload, dict)
    assert "expression_prompt" not in payload
    assert payload["expression"] == {
        "enabled": True,
        "profile": "shy",
        "intensity": 0.35,
    }


def test_webui_proxy_resolves_segment_prompts_and_preserves_text(tmp_path) -> None:
    (tmp_path / "index.html").write_text("<!doctype html><title>test</title>", encoding="utf-8")
    captured: dict[str, object] = {}
    app = create_webui_app(static_dir=tmp_path, client=_mock_upstream(captured))

    with TestClient(app) as client:
        response = client.post(
            "/api/speech",
            json={
                "segments": [
                    {"text": "今日は、 ", "expression_prompt": "slightly shy"},
                    {"text": "戦う準備ができました。", "expression_prompt": "battle"},
                ],
                "language": "ja",
                "model": "roxy-v2proplus",
            },
        )

    assert response.status_code == 200
    payload = captured["speech"]
    assert isinstance(payload, dict)
    assert "text" not in payload
    assert "expression_prompt" not in payload
    assert "segments" in payload
    segments = payload["segments"]
    assert isinstance(segments, list)
    assert "".join(segment["text"] for segment in segments) == "今日は、 戦う準備ができました。"
    assert segments == [
        {
            "text": "今日は、 ",
            "expression": {
                "enabled": True,
                "profile": "shy",
                "intensity": 0.35,
            },
        },
        {
            "text": "戦う準備ができました。",
            "expression": {
                "enabled": True,
                "profile": "battle",
                "intensity": 0.7,
            },
        },
    ]


def test_webui_proxy_rejects_expression_switch_inside_phrase(tmp_path) -> None:
    (tmp_path / "index.html").write_text("<!doctype html><title>test</title>", encoding="utf-8")
    captured: dict[str, object] = {}
    app = create_webui_app(static_dir=tmp_path, client=_mock_upstream(captured))

    with TestClient(app) as client:
        response = client.post(
            "/api/speech",
            json={
                "segments": [
                    {"text": "I want ", "expression_prompt": "shy"},
                    {"text": "to explain.", "expression_prompt": "battle"},
                ],
                "language": "en",
                "model": "roxy-v2proplus",
            },
        )

    assert response.status_code == 400
    assert "speech-safe punctuation" in response.json()["error"]
    assert "speech" not in captured


@pytest.mark.parametrize(
    "body",
    [
        {
            "text": "test",
            "segments": [{"text": "test"}],
            "language": "ja",
            "model": "roxy-v2proplus",
        },
        {
            "segments": [{"text": "test", "unexpected": True}],
            "language": "ja",
            "model": "roxy-v2proplus",
        },
        {
            "segments": [{"text": "test", "expression_prompt": "unknown feeling"}],
            "language": "ja",
            "model": "roxy-v2proplus",
        },
    ],
)
def test_webui_proxy_rejects_invalid_segment_contract_before_upstream(
    tmp_path, body: dict[str, object]
) -> None:
    (tmp_path / "index.html").write_text("<!doctype html><title>test</title>", encoding="utf-8")
    captured: dict[str, object] = {}
    app = create_webui_app(static_dir=tmp_path, client=_mock_upstream(captured))

    with TestClient(app) as client:
        response = client.post("/api/speech", json=body)

    assert response.status_code == 400
    assert "speech" not in captured


def test_webui_proxy_rejects_unknown_emotion_before_upstream(tmp_path) -> None:
    (tmp_path / "index.html").write_text("<!doctype html><title>test</title>", encoding="utf-8")
    captured: dict[str, object] = {}
    app = create_webui_app(static_dir=tmp_path, client=_mock_upstream(captured))

    with TestClient(app) as client:
        response = client.post(
            "/api/speech",
            json={
                "text": "test",
                "language": "en",
                "model": "roxy-v2proplus",
                "expression_prompt": "sound like an unsupported character",
            },
        )

    assert response.status_code == 400
    assert "speech" not in captured


def test_webui_proxy_waits_for_cancelled_upstream_to_release(tmp_path) -> None:
    (tmp_path / "index.html").write_text("<!doctype html><title>test</title>", encoding="utf-8")
    captured: dict[str, object] = {}
    app = create_webui_app(
        static_dir=tmp_path,
        client=_mock_upstream(captured, busy_once=True),
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/speech",
            json={
                "text": "test",
                "language": "en",
                "model": "roxy-v2proplus",
                "expression_prompt": "",
            },
        )

    assert response.status_code == 200
    assert captured["busy_returned"] is True
    assert captured["cancel_calls"] == 1
    assert isinstance(captured["speech"], dict)


def test_webui_cancel_is_idempotent_without_active_stream(tmp_path) -> None:
    (tmp_path / "index.html").write_text("<!doctype html><title>test</title>", encoding="utf-8")
    app = create_webui_app(static_dir=tmp_path, client=_mock_upstream({}))

    with TestClient(app) as client:
        response = client.post("/api/cancel")

    assert response.status_code == 200
    assert response.json() == {"cancelled": False}


def test_webui_asset_uses_current_contract_without_legacy_hardcoding() -> None:
    html = (Path(__file__).parents[1] / "webui" / "index.html").read_text(encoding="utf-8")

    assert 'id="annotationEditor"' in html
    assert 'id="annotationMirror"' in html
    assert 'id="expressionMenu"' in html
    assert 'id="expressionCards"' in html
    assert 'id="expressionPrompt"' not in html
    assert "annotationEditor.buildSegments()" in html
    assert 'fetch("/api/speech"' in html
    assert 'fetch("/api/status"' in html
    assert 'fetch("/api/models/activate"' in html
    assert 'fetch("/api/cancel"' in html
    assert "/assets/everynight_dance.gif" in html
    assert '<script src="/assets/playback_model.js?v=1.3.0-expression-2"></script>' in html
    assert '<script src="/assets/annotation_editor.js?v=1.3.0-expression-2"></script>' in html
    assert "--playback-gold:" in html
    assert ".playback-active-character" in html
    assert ".playback-mirror-content" in html
    assert "startPlaybackIndicator(sourceText, state.language, context, runId)" in html
    assert "activeTextRangeAtAudioTime" in html
    assert "buildPcmActivityTimeline" in html
    assert "elapsed / duration" not in html
    assert "const timing = schedulePcm(context, chunk, metrics)" in html
    assert 'el("text").addEventListener("input", clearPlaybackIndicator)' in html
    assert "v1.1" not in html
    assert "/static/" not in html

    editor = (Path(__file__).parents[1] / "webui" / "annotation_editor.js").read_text(
        encoding="utf-8"
    )
    set_playback = editor[editor.index("setPlaybackRange"):editor.index("hasAnnotations")]
    assert "renderPlayback()" in set_playback
    assert "renderMirror()" not in set_playback


def test_webui_launcher_supports_direct_and_compose_api_ports() -> None:
    launcher = (Path(__file__).parents[1] / "run_webui.bat").read_text(
        encoding="utf-8"
    )

    assert "if not defined ANIFLIVE_TTS_WEBUI_UPSTREAM" in launcher
    assert "for %%P in (9880 9882)" in launcher
    assert "taskkill" not in launcher.lower()


def test_webui_playback_model_asset_is_served(tmp_path) -> None:
    (tmp_path / "index.html").write_text("<!doctype html>", encoding="utf-8")
    (tmp_path / "playback_model.js").write_text(
        "globalThis.testPlaybackModel = true;", encoding="utf-8"
    )
    app = create_webui_app(static_dir=tmp_path, client=_mock_upstream({}))

    with TestClient(app) as client:
        response = client.get("/assets/playback_model.js")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/javascript")
    assert response.text == "globalThis.testPlaybackModel = true;"


def test_webui_annotation_editor_asset_is_served(tmp_path) -> None:
    (tmp_path / "index.html").write_text("<!doctype html>", encoding="utf-8")
    (tmp_path / "annotation_editor.js").write_text(
        "globalThis.testAnnotationEditor = true;", encoding="utf-8"
    )
    app = create_webui_app(static_dir=tmp_path, client=_mock_upstream({}))

    with TestClient(app) as client:
        response = client.get("/assets/annotation_editor.js")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/javascript")
    assert response.text == "globalThis.testAnnotationEditor = true;"


def test_annotation_editor_expression_boundaries_match_api_contract() -> None:
    node = os.environ.get("NODE_BINARY") or shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    source = Path(__file__).parents[1] / "webui" / "annotation_editor.js"
    script = (
        "require('vm').runInThisContext(require('fs').readFileSync(process.argv[1], 'utf8'));"
        "const b=globalThis.AnifLiveTTSExpressionBoundaries;"
        "console.log(JSON.stringify(["
        "b.isSafeExpressionRange('今日は、続けます。',0,4),"
        "b.isSafeExpressionRange('今日は、続けます。',0,3),"
        "b.isSafeExpressionRange('I want to explain.',0,7),"
        "b.isSafeExpressionRange('I want to explain.',0,18)]));"
    )
    result = subprocess.run(
        [node, "-e", script, str(source)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert json.loads(result.stdout) == [True, False, False, True]
