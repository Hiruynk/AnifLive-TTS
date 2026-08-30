from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
import json
import logging
import os
from pathlib import Path
import re
import time
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from .expression import has_safe_expression_boundary


LOGGER = logging.getLogger("aniflive_tts.webui")
LANGUAGES = frozenset({"zh", "yue", "en", "ja", "ko"})
MAX_TEXT_CHARS = 1000
MAX_EXPRESSION_PROMPT_CHARS = 160
MAX_EXPRESSION_SEGMENTS = 64
EXPECTED_BACKEND = "TensorRT-11"
EXPECTED_ENGINE_COUNT = 9
EXPECTED_SAMPLE_RATE = 32000

_EXPRESSION_ALIASES: dict[str, tuple[str, ...]] = {
    "affectionate": (
        "affectionate",
        "loving",
        "tender",
        "sweet",
        "撒嬌",
        "撒娇",
        "溫柔",
        "温柔",
        "親密",
        "亲密",
        "寵溺",
        "宠溺",
        "愛情を込めて",
        "優しく",
        "다정하게",
        "애정",
    ),
    "aggrieved": (
        "aggrieved",
        "hurt",
        "wronged",
        "委屈",
        "難過",
        "难过",
        "受傷",
        "受伤",
        "悔しげ",
        "傷ついた",
        "억울하게",
        "상처받은",
    ),
    "battle": (
        "battle",
        "fierce",
        "heroic",
        "戰鬥",
        "战斗",
        "激昂",
        "強勢",
        "强势",
        "熱血",
        "热血",
        "激しく",
        "勇ましく",
        "격앙되게",
        "용감하게",
    ),
    "languid": (
        "languid",
        "lazy",
        "sleepy",
        "tired",
        "慵懶",
        "慵懒",
        "疲倦",
        "睏倦",
        "困倦",
        "気だるく",
        "眠そうに",
        "나른하게",
        "졸린 듯",
    ),
    "relieved": (
        "relieved",
        "reassured",
        "relaxed",
        "欣慰",
        "放心",
        "放鬆",
        "放松",
        "釋然",
        "释然",
        "安堵して",
        "安心して",
        "안도하며",
        "안심하며",
    ),
    "reproachful": (
        "reproachful",
        "scolding",
        "angry",
        "責備",
        "责备",
        "斥責",
        "斥责",
        "生氣",
        "生气",
        "責めるように",
        "怒って",
        "책망하듯",
        "화난 듯",
    ),
    "self-deprecating": (
        "self-deprecating",
        "self deprecating",
        "wry",
        "自嘲",
        "苦笑",
        "自嘲気味に",
        "자조적으로",
        "쓴웃음으로",
    ),
    "shy": (
        "shy",
        "bashful",
        "embarrassed",
        "害羞",
        "羞澀",
        "羞涩",
        "靦腆",
        "腼腆",
        "恥ずかしそうに",
        "照れながら",
        "수줍게",
        "부끄러운 듯",
    ),
}
_NEUTRAL_ALIASES = (
    "neutral",
    "natural",
    "normal",
    "中性",
    "自然",
    "普通",
    "原生",
    "默認",
    "默认",
    "穏やか",
    "차분하게",
)
_LOW_INTENSITY = (
    "slightly", "subtle", "lightly", "輕微", "轻微", "稍微", "淡淡",
    "少し", "少々", "やや", "조금", "약하게",
)
_HIGH_INTENSITY = (
    "strongly",
    "intense",
    "very",
    "extremely",
    "非常",
    "強烈",
    "强烈",
    "極度",
    "极度",
    "とても",
    "非常に",
    "強く",
    "매우",
    "아주",
    "강하게",
)
_INTENSITY_NUMBER = re.compile(r"(?<!\d)(?:0(?:\.\d+)?|1(?:\.0+)?|\d{1,3}%)(?!\d)")


class WebUIError(ValueError):
    pass


class UpstreamContractError(RuntimeError):
    pass


@dataclass(frozen=True)
class ResolvedExpression:
    enabled: bool
    profile: str | None = None
    intensity: float = 0.5
    policy: str | None = None

    def upstream_payload(self) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False}
        payload = {
            "enabled": True,
            "profile": self.profile,
            "intensity": self.intensity,
        }
        if self.policy is not None:
            payload["policy"] = self.policy
        return payload


def _normalise_prompt(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _contains_alias(prompt: str, alias: str) -> bool:
    if not alias.isascii() or " " in alias or "-" in alias:
        return alias in prompt
    return re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", prompt) is not None


def _requested_intensity(prompt: str) -> float:
    match = _INTENSITY_NUMBER.search(prompt)
    if match:
        raw = match.group(0)
        value = float(raw[:-1]) / 100.0 if raw.endswith("%") else float(raw)
        return min(1.0, max(0.0, value))
    if any(_contains_alias(prompt, item) for item in _LOW_INTENSITY):
        return 0.35
    if any(_contains_alias(prompt, item) for item in _HIGH_INTENSITY):
        return 0.85
    return 0.70


def resolve_expression_prompt(
    prompt: str | None,
    expression_metadata: Mapping[str, Any],
) -> ResolvedExpression:
    """Resolve free-form UI text to a package-owned symbolic expression only."""

    if prompt is None or not prompt.strip():
        return ResolvedExpression(enabled=False)
    if len(prompt) > MAX_EXPRESSION_PROMPT_CHARS:
        raise WebUIError(
            f"expression_prompt is limited to {MAX_EXPRESSION_PROMPT_CHARS} characters"
        )
    normalised = _normalise_prompt(prompt)
    if any(_contains_alias(normalised, alias) for alias in _NEUTRAL_ALIASES):
        return ResolvedExpression(enabled=False)
    if expression_metadata.get("enabled") is not True:
        raise WebUIError("The active model package does not provide controlled expressions")

    records = expression_metadata.get("profiles")
    if not isinstance(records, list):
        raise WebUIError("The active model returned invalid expression metadata")
    available = {
        str(record.get("id"))
        for record in records
        if isinstance(record, Mapping) and isinstance(record.get("id"), str)
    }
    matches: list[str] = []
    for profile in sorted(available):
        aliases = (profile, profile.replace("-", " "), *_EXPRESSION_ALIASES.get(profile, ()))
        if any(_contains_alias(normalised, alias) for alias in aliases):
            matches.append(profile)
    if len(matches) > 1:
        raise WebUIError("The expression request is ambiguous: " + ", ".join(matches))
    if not matches:
        choices = ", ".join(sorted(available)) or "none"
        raise WebUIError(f"No expression profile matched the request; available: {choices}")

    policies = expression_metadata.get("policies")
    if not isinstance(policies, list) or not policies:
        raise WebUIError("The active model returned no expression conditioning policies")
    return ResolvedExpression(
        enabled=True,
        profile=matches[0],
        intensity=_requested_intensity(normalised),
        policy=None,
    )


def _version_tuple(value: str) -> tuple[int, int, int]:
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", value)
    if not match:
        raise UpstreamContractError(f"Unsupported AnifLive-TTS API version: {value!r}")
    return tuple(int(part) for part in match.groups())


async def _read_json(client: httpx.AsyncClient, path: str) -> dict[str, Any]:
    response = await client.get(path)
    response.raise_for_status()
    value = response.json()
    if not isinstance(value, dict):
        raise UpstreamContractError(f"{path} did not return a JSON object")
    return value


async def verify_upstream(client: httpx.AsyncClient) -> dict[str, Any]:
    openapi, health, config, models, expressions = await asyncio.gather(
        _read_json(client, "/openapi.json"),
        _read_json(client, "/health"),
        _read_json(client, "/model/config"),
        _read_json(client, "/v1/models"),
        _read_json(client, "/v1/expressions"),
    )
    version = str(openapi.get("info", {}).get("version", ""))
    if _version_tuple(version) < (1, 2, 0):
        raise UpstreamContractError(f"AnifLive-TTS API {version!r} is older than v1.2.0")
    records = models.get("data")
    if not isinstance(records, list) or not records:
        raise UpstreamContractError("/v1/models did not return any local models")
    active = [record for record in records if isinstance(record, Mapping) and record.get("active")]
    active_model = health.get("model")
    checks = {
        "health ready": (health.get("ready"), True),
        "health backend": (health.get("backend"), EXPECTED_BACKEND),
        "health engine count": (health.get("engine_count"), EXPECTED_ENGINE_COUNT),
        "config model": (config.get("model"), active_model),
        "config version": (config.get("version"), "v2ProPlus"),
        "config backend": (config.get("backend"), EXPECTED_BACKEND),
        "config engine count": (config.get("engine_count"), EXPECTED_ENGINE_COUNT),
        "config sample rate": (config.get("sample_rate"), EXPECTED_SAMPLE_RATE),
        "config PyTorch fallback": (config.get("pytorch_fallback"), False),
    }
    failures = [
        f"{name}: got {actual!r}, expected {expected!r}"
        for name, (actual, expected) in checks.items()
        if actual != expected
    ]
    if len(active) != 1 or active[0].get("id") != active_model:
        failures.append("model registry must identify exactly one active health model")
    if failures:
        raise UpstreamContractError("AnifLive-TTS preflight failed: " + "; ".join(failures))
    return {
        "ready": True,
        "api_version": version,
        "health": health,
        "config": config,
        "models": records,
        "expressions": expressions,
    }


def _speech_payload(
    *,
    text: str | None,
    segments: list[dict[str, Any]] | None = None,
    language: str,
    model: str,
    expression: ResolvedExpression,
) -> dict[str, Any]:
    if (text is None) == (segments is None):
        raise WebUIError("Exactly one of text or segments must be provided")
    payload: dict[str, Any] = {
        "model": model,
        "voice_profile": "default",
        "language": language,
        "stream": True,
        "response_format": "pcm",
        "pause_length": 0.440,
        "expression": expression.upstream_payload(),
        "generation": {
            "top_k": 15,
            "top_p": 1.0,
            "temperature": 1.0,
            "seed": 1234,
            "noise_scale": 0.5,
            "speed": 1.0,
        },
    }
    if segments is not None:
        payload["segments"] = segments
    else:
        payload["text"] = text
    return payload


def _resolve_webui_segments(
    value: Any,
    expression_metadata: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[ResolvedExpression], str]:
    if not isinstance(value, list) or not value:
        raise WebUIError("segments must be a non-empty array")
    if len(value) > MAX_EXPRESSION_SEGMENTS:
        raise WebUIError(f"segments is limited to {MAX_EXPRESSION_SEGMENTS} items")

    upstream: list[dict[str, Any]] = []
    resolved_items: list[ResolvedExpression] = []
    full_text_parts: list[str] = []
    pending_whitespace = ""
    total_characters = 0
    for index, item in enumerate(value):
        label = f"segments[{index}]"
        if not isinstance(item, Mapping):
            raise WebUIError(f"{label} must be an object")
        unknown = set(item) - {"text", "expression_prompt"}
        if unknown:
            raise WebUIError(
                f"{label} contains unsupported fields: {', '.join(sorted(unknown))}"
            )
        segment_text = item.get("text")
        if not isinstance(segment_text, str) or not segment_text:
            raise WebUIError(f"{label}.text must not be empty")
        expression_prompt = item.get("expression_prompt")
        if expression_prompt is not None and not isinstance(expression_prompt, str):
            raise WebUIError(f"{label}.expression_prompt must be a string")
        total_characters += len(segment_text)
        if total_characters > MAX_TEXT_CHARS:
            raise WebUIError(f"segments text is limited to {MAX_TEXT_CHARS} characters")
        full_text_parts.append(segment_text)

        resolved = resolve_expression_prompt(expression_prompt, expression_metadata)
        if not segment_text.strip():
            if resolved.enabled:
                raise WebUIError(f"{label} cannot apply an expression to whitespace only")
            if upstream:
                upstream[-1]["text"] += segment_text
            else:
                pending_whitespace += segment_text
            continue

        upstream.append(
            {
                "text": pending_whitespace + segment_text,
                "expression": resolved.upstream_payload(),
            }
        )
        pending_whitespace = ""
        resolved_items.append(resolved)

    if not upstream:
        raise WebUIError("segments text must contain visible characters")
    if pending_whitespace:
        upstream[-1]["text"] += pending_whitespace

    coalesced: list[dict[str, Any]] = []
    for segment in upstream:
        if coalesced and coalesced[-1]["expression"] == segment["expression"]:
            coalesced[-1]["text"] += segment["text"]
        else:
            coalesced.append(segment)

    for index, (current, following) in enumerate(zip(coalesced, coalesced[1:])):
        if current["expression"] == following["expression"]:
            continue
        if has_safe_expression_boundary(str(current["text"])):
            continue
        raise WebUIError(
            "Expression changes must follow a speech-safe punctuation boundary; "
            f"segments[{index}] ends inside a phrase"
        )

    return coalesced, resolved_items, "".join(full_text_parts)


def _static_root(configured: Path | None = None) -> Path:
    if configured is not None:
        return configured.resolve()
    env_value = os.environ.get("ANIFLIVE_TTS_WEBUI_DIR")
    if env_value:
        return Path(env_value).expanduser().resolve()
    candidates = (Path("/app/webui"), Path(__file__).resolve().parents[2] / "webui")
    for candidate in candidates:
        if (candidate / "index.html").is_file():
            return candidate.resolve()
    return candidates[-1].resolve()


def _json_error(message: str, status_code: int) -> JSONResponse:
    return JSONResponse(
        {"error": message},
        status_code=status_code,
        headers={"Cache-Control": "no-store"},
    )


def create_webui_app(
    *,
    upstream: str | None = None,
    static_dir: Path | None = None,
    client: httpx.AsyncClient | None = None,
) -> FastAPI:
    upstream_url = (upstream or os.environ.get("ANIFLIVE_TTS_WEBUI_UPSTREAM") or "http://127.0.0.1:9880").rstrip("/")
    root = _static_root(static_dir)
    owns_client = client is None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not (root / "index.html").is_file():
            raise FileNotFoundError(f"WebUI asset is missing: {root / 'index.html'}")
        if client is None:
            timeout = httpx.Timeout(connect=5.0, read=300.0, write=10.0, pool=5.0)
            app.state.client = httpx.AsyncClient(
                base_url=upstream_url,
                timeout=timeout,
                follow_redirects=False,
                trust_env=False,
                limits=httpx.Limits(max_connections=2, max_keepalive_connections=1),
            )
        else:
            app.state.client = client
        app.state.speech_lock = asyncio.Lock()
        app.state.active_upstream_guard = asyncio.Lock()
        app.state.active_upstream = None
        app.state.status = await verify_upstream(app.state.client)
        try:
            yield
        finally:
            if owns_client:
                await app.state.client.aclose()

    app = FastAPI(title="AnifLive-TTS WebUI", version="1.3.0", lifespan=lifespan)

    async def cancel_active_upstream() -> bool:
        async with app.state.active_upstream_guard:
            active = app.state.active_upstream
            app.state.active_upstream = None
        core_cancelled = False
        try:
            response = await app.state.client.post("/v1/audio/cancel")
            if response.status_code == 200:
                payload = response.json()
                core_cancelled = bool(payload.get("cancelled")) if isinstance(payload, dict) else False
        except (httpx.HTTPError, ValueError):
            LOGGER.debug("Upstream explicit stream cancellation was unavailable", exc_info=True)
        if active is None:
            return core_cancelled
        await active.aclose()
        return True

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("Cache-Control", "no-store")
        return response

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(root / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/assets/everynight_dance.gif")
    async def project_gif() -> FileResponse:
        asset = root.parent / "assets" / "everynight_dance.gif"
        if not asset.is_file():
            return FileResponse(root / "transparent.gif")
        return FileResponse(asset, headers={"Cache-Control": "public, max-age=3600"})

    @app.get("/assets/playback_model.js")
    async def playback_model_js() -> FileResponse:
        return FileResponse(
            root / "playback_model.js",
            media_type="text/javascript",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/assets/annotation_editor.js")
    async def annotation_editor_js() -> FileResponse:
        return FileResponse(
            root / "annotation_editor.js",
            media_type="text/javascript",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/status")
    async def status(request: Request) -> JSONResponse:
        try:
            current = await verify_upstream(request.app.state.client)
            request.app.state.status = current
            return JSONResponse(current, headers={"Cache-Control": "no-store"})
        except Exception as error:
            return _json_error(str(error), 503)

    @app.post("/api/resolve-expression")
    async def resolve_expression(request: Request) -> JSONResponse:
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise WebUIError("Request body must be a JSON object")
            prompt = body.get("prompt")
            if prompt is not None and not isinstance(prompt, str):
                raise WebUIError("prompt must be a string")
            result = resolve_expression_prompt(
                prompt,
                request.app.state.status.get("expressions", {}),
            )
            return JSONResponse(asdict(result), headers={"Cache-Control": "no-store"})
        except (json.JSONDecodeError, WebUIError) as error:
            return _json_error(str(error), 400)

    @app.post("/api/cancel")
    async def cancel() -> JSONResponse:
        cancelled = await cancel_active_upstream()
        return JSONResponse(
            {"cancelled": cancelled},
            headers={"Cache-Control": "no-store"},
        )

    @app.post("/api/models/activate")
    async def activate_model(request: Request) -> JSONResponse:
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise WebUIError("Request body must be a JSON object")
            model = body.get("model")
            if not isinstance(model, str) or not model.strip():
                raise WebUIError("model must not be empty")
        except (json.JSONDecodeError, WebUIError) as error:
            return _json_error(str(error), 400)

        lock: asyncio.Lock = request.app.state.speech_lock
        try:
            await asyncio.wait_for(lock.acquire(), timeout=2.0)
        except TimeoutError:
            return _json_error("AnifLive-TTS is still finishing an active request", 409)
        try:
            response = await request.app.state.client.post(
                "/v1/models/activate", json={"model": model.strip()}
            )
            if response.status_code != 200:
                return _json_error("AnifLive-TTS rejected the model switch", response.status_code)
            current = await verify_upstream(request.app.state.client)
            request.app.state.status = current
            return JSONResponse(current, headers={"Cache-Control": "no-store"})
        except httpx.HTTPError as error:
            LOGGER.warning("Model activation failed: %s", error)
            return _json_error("AnifLive-TTS model activation connection failed", 502)
        finally:
            lock.release()

    async def upstream_error(response: httpx.Response) -> JSONResponse:
        body = await response.aread()
        try:
            detail: Any = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            detail = body.decode("utf-8", errors="replace")[:2000]
        await response.aclose()
        return JSONResponse(
            {
                "error": "AnifLive-TTS rejected the request",
                "upstream_status": response.status_code,
                "detail": detail,
            },
            status_code=response.status_code,
            headers={"Cache-Control": "no-store"},
        )

    @app.post("/api/speech")
    async def speech(request: Request):
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise WebUIError("Request body must be a JSON object")
            text = body.get("text")
            raw_segments = body.get("segments")
            language = body.get("language", "ja")
            model = body.get("model")
            expression_prompt = body.get("expression_prompt")
            if (text is None) == (raw_segments is None):
                raise WebUIError("Exactly one of text or segments must be provided")
            if raw_segments is not None and expression_prompt is not None:
                raise WebUIError("expression_prompt cannot be combined with segments")
            if language not in LANGUAGES:
                raise WebUIError("language must be one of zh, yue, en, ja, ko")
            if not isinstance(model, str) or not model.strip():
                raise WebUIError("model must not be empty")
            known_models = {
                record.get("id")
                for record in request.app.state.status.get("models", [])
                if isinstance(record, Mapping)
            }
            if model not in known_models:
                raise WebUIError("model is not available locally")
            expression_metadata = request.app.state.status.get("expressions", {})
            if raw_segments is not None:
                upstream_segments, resolved_items, text = _resolve_webui_segments(
                    raw_segments,
                    expression_metadata,
                )
                resolved = ResolvedExpression(enabled=False)
            else:
                if not isinstance(text, str) or not text.strip():
                    raise WebUIError("text must not be empty")
                text = text.strip()
                if len(text) > MAX_TEXT_CHARS:
                    raise WebUIError(f"text is limited to {MAX_TEXT_CHARS} characters")
                if expression_prompt is not None and not isinstance(expression_prompt, str):
                    raise WebUIError("expression_prompt must be a string")
                resolved = resolve_expression_prompt(expression_prompt, expression_metadata)
                resolved_items = [resolved]
                upstream_segments = None
        except (json.JSONDecodeError, WebUIError) as error:
            return _json_error(str(error), 400)

        payload = _speech_payload(
            text=text if upstream_segments is None else None,
            segments=upstream_segments,
            language=language,
            model=model,
            expression=resolved,
        )
        lock: asyncio.Lock = request.app.state.speech_lock
        if lock.locked():
            await cancel_active_upstream()
        try:
            await asyncio.wait_for(lock.acquire(), timeout=0.75)
        except TimeoutError:
            return _json_error("AnifLive-TTS is still cancelling the previous request", 409)

        started = time.perf_counter()
        busy_deadline = started + 1.5
        try:
            cancellation_sent = False
            while True:
                upstream_response = await request.app.state.client.send(
                    request.app.state.client.build_request(
                        "POST", "/v1/audio/speech", json=payload
                    ),
                    stream=True,
                )
                if upstream_response.status_code != 429:
                    break
                await upstream_response.aread()
                await upstream_response.aclose()
                if not cancellation_sent:
                    await cancel_active_upstream()
                    cancellation_sent = True
                if time.perf_counter() >= busy_deadline:
                    lock.release()
                    return _json_error(
                        "AnifLive-TTS is still cancelling the previous request",
                        409,
                    )
                await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            lock.release()
            raise
        except httpx.HTTPError as error:
            lock.release()
            LOGGER.warning("Speech request failed: %s", error)
            return _json_error("AnifLive-TTS upstream connection failed", 502)
        if upstream_response.status_code != 200:
            response = await upstream_error(upstream_response)
            lock.release()
            return response

        stream = upstream_response.aiter_bytes()
        try:
            first = await stream.__anext__()
            while not first:
                first = await stream.__anext__()
        except asyncio.CancelledError:
            await upstream_response.aclose()
            lock.release()
            raise
        except (httpx.HTTPError, StopAsyncIteration):
            await upstream_response.aclose()
            lock.release()
            return _json_error("AnifLive-TTS stream ended before first PCM audio", 502)

        upstream_ttfa_ms = (time.perf_counter() - started) * 1000.0
        required = {
            "x-tensorrt-backend": EXPECTED_BACKEND,
            "x-tensorrt-engine-count": str(EXPECTED_ENGINE_COUNT),
            "x-pytorch-fallback": "false",
            "x-tts-model": model,
            "x-tts-stream": "pcm_s16le",
            "x-tts-sample-rate": str(EXPECTED_SAMPLE_RATE),
            "x-tts-channels": "1",
        }
        mismatches = [
            f"{name}={upstream_response.headers.get(name)!r}"
            for name, expected in required.items()
            if upstream_response.headers.get(name) != expected
        ]
        if mismatches:
            await upstream_response.aclose()
            lock.release()
            return JSONResponse(
                {"error": "AnifLive-TTS stream headers failed validation", "detail": mismatches},
                status_code=502,
            )
        try:
            prebuffer = int(upstream_response.headers["x-tts-recommended-prebuffer-ms"])
        except (KeyError, TypeError, ValueError):
            prebuffer = -1
        if not 0 <= prebuffer <= 250:
            await upstream_response.aclose()
            lock.release()
            return _json_error("AnifLive-TTS returned an invalid prebuffer recommendation", 502)

        async with app.state.active_upstream_guard:
            app.state.active_upstream = upstream_response

        async def chunks() -> AsyncIterator[bytes]:
            try:
                yield first
                async for chunk in stream:
                    if chunk:
                        yield chunk
            finally:
                async with app.state.active_upstream_guard:
                    if app.state.active_upstream is upstream_response:
                        app.state.active_upstream = None
                await upstream_response.aclose()
                if lock.locked():
                    lock.release()

        headers = {
            name: upstream_response.headers[name]
            for name in required
        }
        for name in (
            "x-tts-version",
            "x-tts-expression",
            "x-tts-expression-policy",
            "x-tts-first-packet-seconds",
            "x-tts-first-audio-seconds",
        ):
            if name in upstream_response.headers:
                headers[name] = upstream_response.headers[name]
        headers.update(
            {
                "X-TTS-Recommended-Prebuffer-Ms": str(prebuffer),
                "X-TTS-Sample-Format": "s16le",
                "X-Upstream-TTFA-Ms": f"{upstream_ttfa_ms:.3f}",
                "X-Resolved-Expression": ",".join(
                    dict.fromkeys(item.profile or "native" for item in resolved_items)
                ),
                "X-Resolved-Expression-Intensity": f"{sum(item.intensity for item in resolved_items) / len(resolved_items):.3f}",
                "Cache-Control": "no-store",
                "X-Accel-Buffering": "no",
            }
        )
        return StreamingResponse(
            chunks(),
            media_type="application/octet-stream",
            headers=headers,
        )

    return app


def run_webui(*, host: str, port: int, upstream: str) -> None:
    import uvicorn

    uvicorn.run(
        create_webui_app(upstream=upstream),
        host=host,
        port=port,
        workers=1,
        access_log=False,
    )
