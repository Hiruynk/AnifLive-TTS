from __future__ import annotations

import re
import struct
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from aniflive_tts.cli import build_parser
from aniflive_tts.webui import create_webui_app
from aniflive_tts.workstation import WorkstationStore

ROOT = Path(__file__).resolve().parents[1]


def _offline_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url="http://upstream.invalid",
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(503, json={"ready": False})
        ),
    )


def _png_dimensions(path: Path) -> tuple[int, int]:
    payload = path.read_bytes()
    assert payload[:8] == b"\x89PNG\r\n\x1a\n"
    assert payload[12:16] == b"IHDR"
    return struct.unpack(">II", payload[16:24])


def test_studio_and_classic_surfaces_coexist_without_overwriting(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ANIFLIVE_TTS_WEBUI_TRUSTED_HOSTS", "testserver")
    client = _offline_client()
    studio = create_webui_app(
        static_dir=ROOT / "webui",
        client=client,
        workstation=WorkstationStore(tmp_path / "studio-state"),
        default_surface="studio",
    )
    classic = create_webui_app(
        static_dir=ROOT / "webui",
        client=client,
        workstation=WorkstationStore(tmp_path / "classic-state"),
        default_surface="classic",
    )

    with TestClient(studio) as app:
        assert "<title>AnifLive-TTS Studio</title>" in app.get("/").text
        assert "<title>AnifLive-TTS WebUI</title>" in app.get("/webui").text
        assert "<title>AnifLive-TTS WebUI</title>" in app.get("/synthesis").text
    with TestClient(classic) as app:
        assert "<title>AnifLive-TTS WebUI</title>" in app.get("/").text
        assert "<title>AnifLive-TTS Studio</title>" in app.get("/studio").text


def test_studio_pwa_manifest_icons_and_security_contract(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ANIFLIVE_TTS_WEBUI_TRUSTED_HOSTS", "testserver")
    app = create_webui_app(
        static_dir=ROOT / "webui",
        client=_offline_client(),
        workstation=WorkstationStore(tmp_path / "state"),
    )
    with TestClient(app) as client:
        manifest_response = client.get("/studio.webmanifest")
        worker_response = client.get("/studio-sw.js")
        offline_response = client.get("/offline.html")
        icon_192_response = client.get("/pwa/studio-icon-192.png")
        icon_512_response = client.get("/pwa/studio-icon-512.png")

    manifest = manifest_response.json()
    assert manifest_response.headers["content-type"].startswith(
        "application/manifest+json"
    )
    assert manifest["name"] == "AnifLive-TTS Studio"
    assert manifest["id"] == manifest["start_url"] == manifest["scope"] == "/"
    assert manifest["display"] == "standalone"
    assert [(icon["sizes"], icon["type"]) for icon in manifest["icons"]] == [
        ("192x192", "image/png"),
        ("512x512", "image/png"),
    ]
    assert _png_dimensions(ROOT / "webui/pwa/studio-icon-192.png") == (192, 192)
    assert _png_dimensions(ROOT / "webui/pwa/studio-icon-512.png") == (512, 512)
    assert icon_192_response.headers["content-type"].startswith("image/png")
    assert icon_512_response.headers["content-type"].startswith("image/png")
    assert worker_response.headers["service-worker-allowed"] == "/"
    assert worker_response.headers["cache-control"] == "no-cache"
    assert 'request.method !== "GET"' in worker_response.text
    assert 'request.headers.has("range")' in worker_response.text
    assert 'url.pathname.startsWith("/api/")' in worker_response.text
    assert 'url.pathname.startsWith("/v1/")' in worker_response.text
    assert 'url.pathname.startsWith("/media/")' in worker_response.text
    assert '"/assets/studio_i18n.js"' in worker_response.text
    assert "AnifLive-TTS Studio is offline" in offline_response.text


def test_studio_pwa_is_registered_and_has_discoverable_install_control() -> None:
    index = (ROOT / "webui/index.html").read_text(encoding="utf-8")
    script = (ROOT / "webui/studio.js").read_text(encoding="utf-8")

    assert '<link rel="manifest" href="/studio.webmanifest">' in index
    assert 'id="studioInstallButton"' in index
    assert "Install AnifLive-TTS Studio" in index
    assert 'navigator.serviceWorker.register("/studio-sw.js", { scope: "/" })' in script
    assert 'window.addEventListener("beforeinstallprompt"' in script
    assert 'window.addEventListener("appinstalled"' in script
    assert "deferredStudioInstallPrompt" in script


def test_launchers_and_cli_keep_classic_and_studio_lifecycles_separate() -> None:
    classic = (ROOT / "run_webui.bat").read_text(encoding="utf-8")
    studio = (ROOT / "run_studio.bat").read_text(encoding="utf-8")
    parser = build_parser()

    classic_args = parser.parse_args(["webui"])
    studio_args = parser.parse_args(["workstation"])
    assert classic_args.surface == "classic"
    assert classic_args.port == 9890
    assert studio_args.port == 9891
    assert "-m aniflive_tts webui" in classic
    assert "--surface classic" in classic
    assert "-m aniflive_tts workstation" not in classic
    assert "ANIFLIVE_TTS_WORKSTATION_DIR" not in classic
    assert "http://127.0.0.1:9890/" in classic
    assert "[AnifLive-TTS WebUI]" in classic
    assert "-m aniflive_tts workstation" in studio
    assert "--workstation-dir" in studio
    assert "http://127.0.0.1:9891/" in studio
    assert "--port 9891" in studio
    assert "[AnifLive-TTS Studio]" in studio
    assert "taskkill" not in (classic + studio).lower()


def test_studio_visible_branding_and_responsive_title_are_exact() -> None:
    index = (ROOT / "webui/index.html").read_text(encoding="utf-8")
    script = (ROOT / "webui/studio.js").read_text(encoding="utf-8")
    style = (ROOT / "webui/studio.css").read_text(encoding="utf-8")
    synthesis = (ROOT / "webui/synthesis.html").read_text(encoding="utf-8")

    assert "AnifLive-TTS Studio Studio" not in index + script
    assert "AnifLive-TTS Voice Workstation" not in index + script
    assert len(re.findall(r"AnifLive-TTS(?! Studio)", index)) == 1
    assert 'aria-label="AnifLive-TTS Studio"' in index
    assert '<span class="overview-title-core">AnifLive-TTS</span> <span' in index
    assert '<span class="overview-title-suffix">Studio</span>' in index
    assert ".overview-title-core,\n.overview-title-suffix" in style
    assert "white-space: nowrap;" in style
    assert "@media (max-width: 720px)" in style
    assert ".overview-title-suffix {\n    display: block;" in style
    assert re.search(r"AnifLive-TTS(?! Studio)", script) is None
    assert 'const SURFACE_NAME = document.body.classList.contains("embedded")' in synthesis
    assert '? "AnifLive-TTS Studio"' in synthesis
    assert ': "AnifLive-TTS WebUI"' in synthesis
    assert synthesis.count('statusApiReady: "AnifLive-TTS v1.4') == 3
    assert 'statusApiReady: "{surface} v1.4' not in synthesis


def test_overview_activity_timeline_uses_the_wine_brand_color() -> None:
    style = (ROOT / "webui/studio.css").read_text(encoding="utf-8")

    assert re.search(r"\.activity-mark\s*\{[^}]*background:\s*var\(--wine\)", style)
    assert re.search(r"\.activity-mark::after\s*\{[^}]*background:\s*var\(--wine\)", style)


def test_evaluation_chain_uses_verified_package_dependency_in_forward_order() -> None:
    index = (ROOT / "webui/index.html").read_text(encoding="utf-8")
    script = (ROOT / "webui/studio.js").read_text(encoding="utf-8")

    assert "Upstream package dependency" in index
    assert "Queue engine → package → evaluation" in index
    selector = script[script.index("function updateEvaluationSelectors"):script.index("function evaluationMetric")]
    queue = script[script.index("async function queueEvaluationRun"):script.index("function selectedTseProject")]
    assert 'job.type === "model.package"' in selector
    assert 'job.type === "training.prepare"' not in selector
    assert queue.index('createWorkstationJob("engine.prepare"') < queue.index(
        'createWorkstationJob("model.package"'
    ) < queue.index('createWorkstationJob("evaluation.prepare"')


def test_editorial_heading_boxes_reserve_descender_space() -> None:
    style = (ROOT / "webui/studio.css").read_text(encoding="utf-8")

    final_heading = style[style.rindex(".module-header h1 {"):]
    assert "line-height: 1.14;" in style
    assert "padding-bottom: .14em;" in style
    assert "overflow: hidden" not in final_heading.split("}", 1)[0]
