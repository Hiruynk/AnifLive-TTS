from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import logging
import math
import os
import re
import shutil
import time
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from urllib.parse import urlsplit
from uuid import NAMESPACE_URL, uuid5

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

from .dataset_acquisition import DatasetAcquisitionConfig, DatasetAcquisitionError
from .dataset_acquisition_worker import TARGET_FINALIZE_SCHEMA
from .dataset_factory import DatasetFactory, DatasetFactoryError
from .dataset_factory_api import create_dataset_factory_router
from .expression import has_safe_expression_boundary
from .voice_acquisition_report import build_voice_acquisition_report
from .workstation import (
    PROJECT_KINDS,
    WorkstationError,
    WorkstationStore,
    validated_artifact_relative_path,
)
from .workstation_assets import WorkstationAssetError, WorkstationAssetManager
from .workstation_path_picker import (
    WorkstationPathPickerError,
    pick_workstation_paths,
)

LOGGER = logging.getLogger("aniflive_tts.webui")
LANGUAGES = frozenset({"zh", "yue", "en", "ja", "ko"})
MAX_TEXT_CHARS = 1000
MAX_EXPRESSION_PROMPT_CHARS = 160
MAX_EXPRESSION_SEGMENTS = 64
EXPECTED_BACKEND = "TensorRT-11"
EXPECTED_ENGINE_COUNT = 9
EXPECTED_SAMPLE_RATE = 32000
_MUTATION_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
_JSON_MUTATION_ROUTES = frozenset(
    {
        ("POST", "/api/workstation/projects"),
        ("POST", "/api/workstation/components/install"),
        ("POST", "/api/workstation/components/download"),
        ("POST", "/api/workstation/components/import"),
        ("POST", "/api/workstation/components/export"),
        ("POST", "/api/workstation/jobs"),
        ("POST", "/api/workstation/artifacts"),
        ("POST", "/api/workstation/qualifications/compose"),
        ("POST", "/api/workstation/qualifications/import"),
        ("POST", "/api/workstation/expression-drafts"),
        ("PATCH", "/api/workstation/settings"),
        ("POST", "/api/workstation/ui-history/clear"),
        ("POST", "/api/workstation/path-picker"),
        ("POST", "/api/resolve-expression"),
        ("POST", "/api/models/activate"),
        ("POST", "/api/speech"),
        ("POST", "/api/sessions"),
    }
)
_SESSION_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")


def _deployment_model_id(name: str, dataset_id: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip()).strip("._-")
    if not normalized or not normalized[0].isalnum() or not normalized[-1].isalnum():
        normalized = f"voice-{dataset_id.rsplit('_', 1)[-1][:12]}"
    return normalized[:64].rstrip("._-")


def _register_training_dataset_artifact(
    store: WorkstationStore,
    *,
    dataset_id: str,
    dataset_name: str,
    bundle: Mapping[str, Any],
) -> dict[str, Any]:
    """Register the frozen training input as an immutable lineage root."""

    frozen_sha256 = bundle.get("frozen_manifest_sha256")
    training_list_sha256 = bundle.get("training_list_sha256")
    examples = bundle.get("examples")
    if (
        not isinstance(frozen_sha256, str)
        or not isinstance(training_list_sha256, str)
        or not isinstance(examples, int)
        or isinstance(examples, bool)
        or examples < 2
    ):
        raise WorkstationError("Training bundle lineage metadata is malformed")
    payload = {
        "schema": "aniflive-frozen-training-dataset-artifact-v1",
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "examples": examples,
        "frozen_manifest_sha256": frozen_sha256,
        "training_list_sha256": training_list_sha256,
    }
    content = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    digest = hashlib.sha256(content).hexdigest()
    relative = (
        Path("dataset")
        / dataset_id
        / frozen_sha256
        / "training-dataset-lineage.json"
    )
    destination = store.artifact_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file() or destination.read_bytes() != content:
            raise WorkstationError("Existing training dataset lineage artifact is invalid")
    else:
        temporary = destination.with_name(
            f".{destination.name}.{os.getpid()}.{time.time_ns()}.tmp"
        )
        try:
            with temporary.open("xb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    artifact_id = "artifact_" + str(
        uuid5(
            NAMESPACE_URL,
            f"aniflive-frozen-training-dataset:{dataset_id}:{frozen_sha256}:"
            f"{training_list_sha256}",
        )
    )
    metadata = {
        "schema": payload["schema"],
        "dataset_id": dataset_id,
        "examples": examples,
        "frozen_manifest_sha256": frozen_sha256,
        "training_list_sha256": training_list_sha256,
    }
    try:
        artifact = store.get_artifact(artifact_id)
    except WorkstationError as error:
        if "not found" not in str(error).lower():
            raise
        artifact = store.register_artifact(
            artifact_type="dataset",
            name=f"{dataset_name} frozen training dataset",
            status="ready",
            project_id=dataset_id,
            local_path=destination,
            sha256=digest,
            metadata=metadata,
            artifact_id=artifact_id,
        )
    if (
        artifact.get("type") != "dataset"
        or artifact.get("status") != "ready"
        or artifact.get("sha256") != digest
        or artifact.get("metadata") != metadata
        or artifact.get("parent_artifact_ids") != []
    ):
        raise WorkstationError("Training dataset lineage artifact does not match")
    return artifact

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


def _authority_parts(authority: str) -> tuple[str, int | None]:
    if not isinstance(authority, str) or not authority.strip():
        raise WebUIError("HTTP Host header is required")
    parsed = urlsplit(f"//{authority.strip()}")
    if (
        parsed.username is not None
        or parsed.password is not None
        or parsed.hostname is None
        or parsed.path
        or parsed.query
        or parsed.fragment
    ):
        raise WebUIError("HTTP Host header is malformed")
    try:
        port = parsed.port
    except ValueError as error:
        raise WebUIError("HTTP Host header is malformed") from error
    return parsed.hostname.rstrip(".").lower(), port


def _loopback_hostname(hostname: str) -> bool:
    if hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _trusted_webui_hostnames() -> frozenset[str]:
    configured = os.environ.get("ANIFLIVE_TTS_WEBUI_TRUSTED_HOSTS", "")
    trusted: set[str] = set()
    for value in configured.split(","):
        if not value.strip():
            continue
        hostname, _ = _authority_parts(value)
        trusted.add(hostname)
    return frozenset(trusted)


def _origin_matches_request(origin: str, request: Request) -> bool:
    parsed = urlsplit(origin)
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.hostname is None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        return False
    try:
        origin_port = parsed.port
    except ValueError:
        return False
    try:
        request_host, request_port = _authority_parts(request.headers.get("host", ""))
    except WebUIError:
        return False
    default_port = 443 if parsed.scheme == "https" else 80
    request_default_port = 443 if request.url.scheme == "https" else 80
    return (
        parsed.scheme == request.url.scheme
        and parsed.hostname.rstrip(".").lower() == request_host
        and (origin_port or default_port) == (request_port or request_default_port)
    )


def _requires_json_content_type(method: str, path: str) -> bool:
    return (method, path) in _JSON_MUTATION_ROUTES or (
        method == "PATCH"
        and path.startswith("/api/workstation/expression-drafts/")
    ) or (
        method == "PATCH"
        and path.startswith("/api/workstation/projects/")
    ) or (
        method in {"POST", "PATCH"}
        and path.startswith("/api/workstation/datasets/")
    ) or (
        method == "POST"
        and path.startswith("/api/workstation/training/")
    ) or (
        method == "POST"
        and re.fullmatch(
            r"/api/sessions/[A-Za-z0-9][A-Za-z0-9._:-]{0,127}/"
            r"(?:segments|flush|cancel)",
            path,
        )
    ) or (
        method == "POST"
        and path.startswith("/api/workstation/artifacts/")
        and path.endswith("/promote")
    ) or (
        method == "POST"
        and path.startswith("/api/workstation/expression-drafts/")
        and path.endswith(("/promote", "/analyze-reference"))
    )


def _webui_session_id(value: Any, field: str = "session_id") -> str:
    if not isinstance(value, str) or _SESSION_ID_PATTERN.fullmatch(value) is None:
        raise WebUIError(
            f"{field} must contain 1-128 letters, numbers, dots, underscores, "
            "colons or hyphens"
        )
    return value


class UpstreamContractError(RuntimeError):
    pass


class _UpstreamSpeechOwner:
    """Own one prefetched upstream stream and its WebUI serialization lock."""

    def __init__(
        self,
        *,
        app: FastAPI,
        response: httpx.Response,
        speech_lock: asyncio.Lock,
        cancel_path: str = "/v1/audio/cancel",
    ) -> None:
        self._app = app
        self._response = response
        self._speech_lock = speech_lock
        self._cancel_path = cancel_path
        self._close_task: asyncio.Task[None] | None = None
        self._closed = False

    @property
    def response(self) -> httpx.Response:
        return self._response

    @property
    def cancel_path(self) -> str:
        return self._cancel_path

    async def aclose(self) -> None:
        """Release each owned resource exactly once, even across cancel races."""

        task = self._close_task
        if task is None:
            # Cleanup runs in its own task so cancellation of the ASGI response
            # cannot interrupt it after marking the owner closed but before
            # releasing the upstream socket and speech_lock.
            task = asyncio.create_task(self._close())
            self._close_task = task
        await asyncio.shield(task)

    async def _close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            async with self._app.state.active_upstream_guard:
                if self._app.state.active_upstream is self:
                    self._app.state.active_upstream = None
        finally:
            try:
                await self._response.aclose()
            finally:
                # This owner is constructed only after this request has
                # acquired speech_lock. No other request can acquire it until
                # this exact owner releases it.
                self._speech_lock.release()


class _OwnedStreamingResponse(StreamingResponse):
    """A streaming response whose resources do not depend on body iteration."""

    def __init__(self, *args: Any, owner: _UpstreamSpeechOwner, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._owner = owner

    async def aclose(self) -> None:
        await self._owner.aclose()

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            # Starlette may observe a disconnect before it advances the body
            # iterator. Response-level ownership closes that otherwise-leaky
            # window after the first PCM chunk has already been prefetched.
            await self.aclose()


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


async def _read_expression_metadata(client: httpx.AsyncClient) -> dict[str, Any]:
    response = await client.get("/v1/expressions")
    if response.status_code == 404:
        return {
            "object": "list",
            "enabled": False,
            "default": None,
            "profiles": [],
            "policies": [],
        }
    response.raise_for_status()
    value = response.json()
    if not isinstance(value, dict):
        raise UpstreamContractError("/v1/expressions did not return a JSON object")
    return value


async def verify_upstream(client: httpx.AsyncClient) -> dict[str, Any]:
    openapi, health, config, models, expressions = await asyncio.gather(
        _read_json(client, "/openapi.json"),
        _read_json(client, "/health"),
        _read_json(client, "/model/config"),
        _read_json(client, "/v1/models"),
        _read_expression_metadata(client),
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
    generation: Mapping[str, int | float] | None = None,
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
        "generation": dict(generation or _validated_generation(None)),
    }
    if segments is not None:
        payload["segments"] = segments
    else:
        payload["text"] = text
    return payload


def _validated_generation(value: Any) -> dict[str, int | float]:
    defaults: dict[str, int | float] = {
        "top_k": 15,
        "top_p": 1.0,
        "temperature": 1.0,
        "seed": 1234,
        "noise_scale": 0.5,
        "speed": 1.0,
    }
    if value is None:
        return defaults
    if not isinstance(value, Mapping):
        raise WebUIError("generation must be a JSON object")
    allowed = {"top_k", "temperature", "noise_scale", "seed", "speed"}
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise WebUIError(f"generation contains unsupported fields: {', '.join(unknown)}")

    result = dict(defaults)
    top_k = value.get("top_k", defaults["top_k"])
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 50:
        raise WebUIError("generation.top_k must be an integer between 1 and 50")
    result["top_k"] = top_k

    seed = value.get("seed", defaults["seed"])
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < -1:
        raise WebUIError("generation.seed must be an integer greater than or equal to -1")
    result["seed"] = seed

    for field, minimum, maximum in (
        ("temperature", 0.0, None),
        ("noise_scale", 0.0, 10.0),
        ("speed", 1.0, 1.0),
    ):
        raw = value.get(field, defaults[field])
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise WebUIError(f"generation.{field} must be a finite number")
        parsed = float(raw)
        if not math.isfinite(parsed):
            raise WebUIError(f"generation.{field} must be a finite number")
        if field == "temperature" and parsed <= minimum:
            raise WebUIError("generation.temperature must be greater than 0")
        if field == "noise_scale" and not minimum <= parsed <= float(maximum):
            raise WebUIError("generation.noise_scale must be between 0 and 10")
        if field == "speed" and parsed != 1.0:
            raise WebUIError("generation.speed must be 1.0")
        result[field] = parsed
    return result


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


def _offline_status() -> dict[str, Any]:
    return {
        "ready": False,
        "api_version": None,
        "health": {
            "ready": False,
            "status": "offline",
            "model": None,
            "backend": None,
            "engine_count": 0,
            "gpu": {},
        },
        "config": {},
        "models": [],
        "expressions": {
            "enabled": False,
            "profiles": [],
            "policies": [],
        },
    }


def create_webui_app(
    *,
    upstream: str | None = None,
    static_dir: Path | None = None,
    client: httpx.AsyncClient | None = None,
    workstation: WorkstationStore | None = None,
    dataset_factory: DatasetFactory | None = None,
    default_surface: Literal["studio", "classic"] = "studio",
) -> FastAPI:
    if default_surface not in {"studio", "classic"}:
        raise WebUIError("WebUI surface must be 'studio' or 'classic'")
    surface_name = (
        "AnifLive-TTS Studio"
        if default_surface == "studio"
        else "AnifLive-TTS WebUI"
    )
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
        app.state.expression_analysis_lock = asyncio.Lock()
        app.state.active_upstream_guard = asyncio.Lock()
        app.state.active_upstream = None
        app.state.workstation = workstation or WorkstationStore()
        app.state.dataset_factory = dataset_factory or DatasetFactory(
            app.state.workstation.root / "dataset-factory",
            allowed_source_roots=(
                *app.state.workstation.allowed_import_roots(),
                app.state.workstation.artifact_root,
            ),
        )
        app.state.workstation_assets = WorkstationAssetManager(
            app.state.workstation.root / "components",
            lock_path=Path(__file__).with_name("workstation_assets_lock.json"),
        )
        app.state.workstation_assets.replacement_allowed = lambda: not any(
            job.get("status") == "running" and job.get("resource_class") == "gpu-exclusive"
            for job in app.state.workstation.list_jobs()
        )
        try:
            app.state.status = await verify_upstream(app.state.client)
        except Exception:
            LOGGER.info(
                "AnifLive-TTS inference API is offline; workstation modules remain available"
            )
            app.state.status = _offline_status()
        try:
            yield
        finally:
            async with app.state.active_upstream_guard:
                active = app.state.active_upstream
                app.state.active_upstream = None
            if active is not None:
                await active.aclose()
            if owns_client:
                await app.state.client.aclose()

    app = FastAPI(
        title=surface_name,
        version="1.4.0",
        lifespan=lifespan,
    )

    def visible_records(
        store: WorkstationStore,
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        cutoff = store.get_visible_history_cutoff()
        if cutoff is None:
            return records
        return [
            record
            for record in records
            if str(record.get("created_at") or record.get("updated_at") or "") > cutoff
        ]

    def validate_dataset_project(dataset_id: str) -> None:
        try:
            project = app.state.workstation.get_project(dataset_id)
        except WorkstationError as error:
            raise DatasetFactoryError("Dataset project was not found") from error
        if project.get("kind") != "dataset":
            raise DatasetFactoryError("Dataset project was not found")

    def verified_artifact_path(
        store: WorkstationStore, artifact: Mapping[str, Any]
    ) -> Path:
        relative = validated_artifact_relative_path(
            artifact.get("local_path"), field="Worker artifact path"
        )
        root_path = store.artifact_root.resolve(strict=True)
        path = root_path.joinpath(*relative.parts).resolve(strict=True)
        if root_path not in path.parents or not path.is_file() or path.is_symlink():
            raise WorkstationError("Worker artifact escaped the artifact store")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
        if digest.hexdigest() != artifact.get("sha256"):
            raise WorkstationError("Worker artifact failed checksum validation")
        return path

    def component_root(component_id: str) -> Path:
        manager: WorkstationAssetManager = app.state.workstation_assets
        status = manager.status(component_id)["components"][0]
        if not status.get("ready") or not isinstance(status.get("root"), str):
            reason = status.get("reason") or status.get("state") or "not-ready"
            raise WorkstationAssetError(
                f"AI component {component_id} is not ready: {reason}"
            )
        return Path(status["root"]).resolve(strict=True)

    def presented_projects(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Overlay Dataset Factory lifecycle without erasing historical job evidence."""

        result: list[dict[str, Any]] = []
        for record in records:
            presented = dict(record)
            if record.get("kind") == "dataset":
                try:
                    lifecycle = app.state.dataset_factory.project_state(record["id"])
                except DatasetFactoryError:
                    lifecycle = None
                if isinstance(lifecycle, Mapping):
                    metrics = dict(presented.get("metrics", {}))
                    metrics["dataset_lifecycle"] = {
                        "stage": lifecycle.get("lifecycle_stage"),
                        "frozen": bool(lifecycle.get("frozen")),
                    }
                    presented["metrics"] = metrics
                    if (
                        lifecycle.get("lifecycle_stage") == "ready"
                        and lifecycle.get("frozen") is True
                    ):
                        presented["status"] = "ready"
                        presented["progress"] = 1.0
            result.append(presented)
        return result

    def component_bundle_destination(store: WorkstationStore, value: Any) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise WorkstationError("Component bundle destination is required")
        candidate = Path(value).expanduser().absolute()
        if candidate.suffix.casefold() != ".zip":
            raise WorkstationError("Component bundle destination must use .zip")
        parent = candidate.parent.resolve(strict=True)
        roots = store.allowed_import_roots()
        if not any(parent == root or parent.is_relative_to(root) for root in roots):
            raise WorkstationError(
                "Component bundle destination is outside the configured import roots"
            )
        if candidate.exists() and (candidate.is_dir() or candidate.is_symlink()):
            raise WorkstationError("Component bundle destination is unsafe")
        return candidate

    app.include_router(
        create_dataset_factory_router(
            lambda: app.state.dataset_factory,
            dataset_validator=validate_dataset_project,
        )
    )

    @app.post("/api/workstation/datasets/{dataset_id}/prepare-standard")
    async def prepare_standard_dataset(
        dataset_id: str, request: Request
    ) -> JSONResponse:
        store: WorkstationStore = app.state.workstation
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise WorkstationError("Request body must be a JSON object")
            if body:
                raise WorkstationError(
                    "Standard preparation uses the validated Dataset project config"
                )
            validate_dataset_project(dataset_id)
            project = store.get_project(dataset_id)
            config = DatasetAcquisitionConfig.from_mapping(project["config"])
            if config.acquisition_mode != "standard":
                raise WorkstationError(
                    "This Dataset project does not use standard acquisition"
                )
            sources = store.resolve_dataset_media_paths(
                config.sources, maximum_files=2_048
            )
            asr = component_root(config.asr_component)
            vad = component_root(config.vad_component)
            jobs: list[dict[str, Any]] = []
            for source in sources:
                jobs.append(
                    store.create_job(
                        job_type="dataset.process",
                        project_id=dataset_id,
                        parameters={
                            "acquisition_mode": "standard",
                            "source": str(source),
                            "asr_model": str(asr),
                            "asr_backend": config.asr_component,
                            "asr_component": config.asr_component,
                            "vad_model": str(vad),
                            "vad_component": config.vad_component,
                            "declared_language": config.declared_language,
                        },
                    )
                )
            dataset_state = app.state.dataset_factory.project_state(dataset_id)
            if dataset_state["lifecycle_stage"] in {"source", "speaker", "clean"}:
                app.state.dataset_factory.advance_project_stage(dataset_id, "clean")
        except (
            json.JSONDecodeError,
            DatasetAcquisitionError,
            DatasetFactoryError,
            OSError,
            WorkstationAssetError,
            WorkstationError,
        ) as error:
            return _json_error(str(error), 409)
        return JSONResponse(
            {
                "schema": "aniflive-standard-dataset-job-graph-v1",
                "dataset_id": dataset_id,
                "jobs": jobs,
                "count": len(jobs),
            },
            status_code=201,
        )

    @app.post("/api/workstation/datasets/{dataset_id}/import-worker-job/{job_id}")
    def import_dataset_worker_segments(dataset_id: str, job_id: str) -> JSONResponse:
        store: WorkstationStore = app.state.workstation
        try:
            validate_dataset_project(dataset_id)
            job = store.get_job(job_id)
            if (
                job.get("project_id") != dataset_id
                or job.get("type") != "dataset.process"
            ):
                raise WorkstationError("Dataset processing job does not belong to this project")
            if job.get("status") != "succeeded":
                raise WorkstationError("Dataset processing job has not succeeded")
            result = job.get("result", {})
            artifact_ids = (
                result.get("registered_artifact_ids", [])
                if isinstance(result, Mapping)
                else []
            )
            if isinstance(artifact_ids, (str, bytes)) or not isinstance(
                artifact_ids, list
            ):
                raise WorkstationError("Dataset processing artifacts are malformed")

            segment_paths: list[Path] = []
            imported_artifact_ids: list[str] = []
            by_worker_path: dict[str, tuple[dict[str, Any], Path]] = {}
            report_artifact: dict[str, Any] | None = None
            report_path: Path | None = None
            for artifact_id in artifact_ids:
                artifact = store.get_artifact(artifact_id)
                metadata = artifact.get("metadata", {})
                worker_relative = (
                    metadata.get("worker_relative_path")
                    if isinstance(metadata, Mapping)
                    else None
                )
                if (
                    artifact.get("type") != "dataset"
                    or artifact.get("status") != "ready"
                    or artifact.get("project_id") != dataset_id
                    or metadata.get("job_id") != job_id
                ):
                    continue
                if not isinstance(worker_relative, str):
                    continue
                path = verified_artifact_path(store, artifact)
                by_worker_path[worker_relative] = (artifact, path)
                if worker_relative == "dataset-pipeline/dataset-report.json":
                    report_artifact, report_path = artifact, path
                elif (
                    worker_relative.startswith("dataset-pipeline/segments/")
                    and worker_relative.lower().endswith(".wav")
                ):
                    segment_paths.append(path)
                    imported_artifact_ids.append(str(artifact_id))
            if not segment_paths:
                raise WorkstationError("Dataset processing job produced no verified speech segments")

            imported: dict[str, Any] | None = None
            if report_artifact is not None and report_path is not None:
                report = json.loads(report_path.read_text(encoding="utf-8"))
                if (
                    not isinstance(report, Mapping)
                    or report.get("schema") != "aniflive-dataset-pipeline-v1"
                ):
                    raise WorkstationError("Dataset processing report schema is unsupported")
                source_value = report.get("input")
                output_value = report.get("output")
                asr_value = report.get("asr")
                source_record = dict(source_value) if isinstance(source_value, Mapping) else {}
                output_record = dict(output_value) if isinstance(output_value, Mapping) else {}
                asr_record = dict(asr_value) if isinstance(asr_value, Mapping) else {}
                segment_values = output_record.get("segments")
                if not isinstance(segment_values, list) or not segment_values:
                    raise WorkstationError("Dataset processing report contains no clips")
                records: list[dict[str, Any]] = []
                job_parameters = job.get("parameters")
                source_path = (
                    job_parameters.get("source")
                    if isinstance(job_parameters, Mapping)
                    else None
                )
                for index, value in enumerate(segment_values):
                    if not isinstance(value, Mapping):
                        raise WorkstationError("Dataset processing clip record is malformed")
                    worker_relative = "dataset-pipeline/" + str(value.get("path", ""))
                    pair = by_worker_path.get(worker_relative)
                    if pair is None:
                        raise WorkstationError(
                            "Dataset processing clip is missing from the artifact registry"
                        )
                    artifact, path = pair
                    region_value = value.get("region")
                    region = dict(region_value) if isinstance(region_value, Mapping) else {}
                    transcript_value = value.get("transcript")
                    transcript = (
                        dict(transcript_value)
                        if isinstance(transcript_value, Mapping)
                        else {}
                    )
                    records.append(
                        {
                            "path": str(path),
                            "sha256": artifact.get("sha256"),
                            "position": value.get("position", index),
                            "source_sha256": source_record.get("sha256"),
                            "source_start_sample": region.get("start_frame"),
                            "source_end_sample": region.get("end_frame"),
                            "source_path": source_path or source_record.get("name"),
                            "quality": value.get("quality"),
                            "annotations": {
                                "transcript": transcript.get("text"),
                                "language": transcript.get("language"),
                                "expression_suggestion": transcript.get(
                                    "emotion_suggestion"
                                ),
                                "asr_backend": asr_record.get("backend"),
                                "authoritative": False,
                                "review_required": True,
                            },
                        }
                    )
                imported = app.state.dataset_factory.import_standard_clips(
                    dataset_id,
                    records,
                    parent_artifact_id=str(report_artifact["id"]),
                )
                items = imported["items"]
            else:
                # Backward-compatible import for pre-v1.4 worker artifacts that
                # did not publish a dataset report.
                items = app.state.dataset_factory.ingest(
                    dataset_id, segment_paths, recursive=False
                )
        except (
            json.JSONDecodeError,
            OSError,
            DatasetFactoryError,
            WorkstationError,
        ) as error:
            message = str(error)
            status_code = 404 if "not found" in message.lower() else 409
            return _json_error(message, status_code)
        return JSONResponse(
            {
                "dataset_id": dataset_id,
                "job_id": job_id,
                "artifact_ids": imported_artifact_ids,
                "items": items,
                "count": len(items),
                "import": imported,
            }
        )

    @app.post(
        "/api/workstation/datasets/{dataset_id}/prepare-target-speaker"
    )
    async def prepare_target_speaker_dataset(
        dataset_id: str, request: Request
    ) -> JSONResponse:
        store: WorkstationStore = app.state.workstation
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise WorkstationError("Request body must be a JSON object")
            if body:
                raise WorkstationError(
                    "Target-speaker preparation uses the validated Dataset project config"
                )
            validate_dataset_project(dataset_id)
            project = store.get_project(dataset_id)
            config = DatasetAcquisitionConfig.from_mapping(project["config"])
            if config.acquisition_mode != "target-speaker":
                raise WorkstationError(
                    "This Dataset project does not use target-speaker acquisition"
                )
            reference = store.validate_import_path(config.reference_audio or "")
            if not reference.is_file():
                raise WorkstationError("Target reference must be a local audio file")
            sources = store.resolve_dataset_media_paths(
                config.sources, maximum_files=2_048
            )
            speaker = component_root(config.speaker_component)
            vad = component_root(config.vad_component)
            diarization = component_root(config.diarization_component)
            separator = component_root(config.separation_component)
            asr = component_root(config.asr_component)
            chains: list[dict[str, Any]] = []
            common = config.as_dict()
            for source in sources:
                routing = store.create_job(
                    job_type="dataset.target-speaker",
                    project_id=dataset_id,
                    parameters={
                        **common,
                        "source": str(source),
                        "reference": str(reference),
                        "speaker_engine": str(speaker),
                        "vad_model": str(vad),
                        "diarization_model": str(diarization),
                    },
                )
                separation = store.create_job(
                    job_type="dataset.separate",
                    project_id=dataset_id,
                    parameters={
                        **common,
                        "reference": str(reference),
                        "speaker_engine": str(speaker),
                        "separation_model": str(separator),
                    },
                    depends_on=[routing["id"]],
                )
                transcription = store.create_job(
                    job_type="dataset.transcribe",
                    project_id=dataset_id,
                    parameters={
                        **common,
                        "asr_model": str(asr),
                        "asr_backend": config.asr_component,
                    },
                    depends_on=[separation["id"]],
                )
                finalize = store.create_job(
                    job_type="dataset.finalize",
                    project_id=dataset_id,
                    parameters=common,
                    depends_on=[transcription["id"]],
                )
                chains.append(
                    {
                        "source": str(source),
                        "routing": routing,
                        "separation": separation,
                        "transcription": transcription,
                        "finalize": finalize,
                    }
                )
            app.state.dataset_factory.advance_project_stage(dataset_id, "speaker")
        except (
            json.JSONDecodeError,
            DatasetAcquisitionError,
            DatasetFactoryError,
            OSError,
            WorkstationAssetError,
            WorkstationError,
        ) as error:
            return _json_error(str(error), 409)
        return JSONResponse(
            {
                "schema": "aniflive-target-speaker-job-graph-v1",
                "dataset_id": dataset_id,
                "chains": chains,
                "count": len(chains),
            },
            status_code=201,
        )

    @app.post(
        "/api/workstation/datasets/{dataset_id}/import-acquisition-job/{job_id}"
    )
    def import_dataset_acquisition_job(
        dataset_id: str, job_id: str
    ) -> JSONResponse:
        store: WorkstationStore = app.state.workstation
        try:
            validate_dataset_project(dataset_id)
            job = store.get_job(job_id)
            if (
                job.get("project_id") != dataset_id
                or job.get("type") != "dataset.finalize"
            ):
                raise WorkstationError(
                    "Dataset acquisition job does not belong to this project"
                )
            if job.get("status") != "succeeded":
                raise WorkstationError("Dataset acquisition job has not succeeded")
            result = job.get("result", {})
            artifact_ids = (
                result.get("registered_artifact_ids", [])
                if isinstance(result, Mapping)
                else []
            )
            if (
                isinstance(artifact_ids, (str, bytes))
                or not isinstance(artifact_ids, list)
                or not artifact_ids
            ):
                raise WorkstationError("Dataset acquisition artifacts are malformed")
            artifacts = [store.get_artifact(value) for value in artifact_ids]
            by_worker_path: dict[str, tuple[dict[str, Any], Path]] = {}
            report_artifact: dict[str, Any] | None = None
            report_path: Path | None = None
            for artifact in artifacts:
                metadata = artifact.get("metadata", {})
                worker_path = (
                    metadata.get("worker_relative_path")
                    if isinstance(metadata, Mapping)
                    else None
                )
                if not isinstance(worker_path, str):
                    continue
                path = verified_artifact_path(store, artifact)
                by_worker_path[worker_path] = (artifact, path)
                if worker_path == "dataset-acquisition/dataset-acquisition-report.json":
                    report_artifact, report_path = artifact, path
            if report_artifact is None or report_path is None:
                raise WorkstationError("Dataset acquisition report is missing")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if not isinstance(report, dict) or report.get("schema") != TARGET_FINALIZE_SCHEMA:
                raise WorkstationError("Dataset acquisition report schema is unsupported")
            values = report.get("records")
            if not isinstance(values, list) or not values:
                raise WorkstationError("Dataset acquisition report contains no clips")
            records: list[dict[str, Any]] = []
            clip_artifact_ids: list[str] = []
            for value in values:
                if not isinstance(value, Mapping):
                    raise WorkstationError("Dataset acquisition clip record is malformed")
                worker_path = value.get("path")
                pair = by_worker_path.get(str(worker_path))
                if pair is None:
                    raise WorkstationError(
                        "Dataset acquisition clip is missing from the artifact registry"
                    )
                artifact, path = pair
                record = dict(value)
                record["path"] = str(path)
                acquisition = dict(record.get("acquisition") or {})
                audio_value = record.get("audio")
                if isinstance(audio_value, Mapping):
                    audio_artifacts: dict[str, Any] = {
                        "selected": audio_value.get("selected", "original")
                    }
                    for kind in ("original", "processed"):
                        worker_audio_path = audio_value.get(f"{kind}_path")
                        if worker_audio_path is None:
                            audio_artifacts[kind] = None
                            continue
                        audio_pair = by_worker_path.get(str(worker_audio_path))
                        if audio_pair is None:
                            raise WorkstationError(
                                f"Dataset acquisition {kind} audio is missing from the artifact registry"
                            )
                        audio_artifact, _audio_path = audio_pair
                        audio_artifacts[kind] = {
                            "artifact_id": str(audio_artifact["id"]),
                            "sha256": audio_artifact.get("sha256"),
                            "worker_relative_path": str(worker_audio_path),
                        }
                    acquisition["audio_artifacts"] = audio_artifacts
                if isinstance(record.get("annotations"), Mapping):
                    acquisition["asr_suggestion"] = dict(record["annotations"])
                record["acquisition"] = acquisition
                records.append(record)
                clip_artifact_ids.append(str(artifact["id"]))
            imported = app.state.dataset_factory.import_target_speaker_clips(
                dataset_id,
                records,
                parent_artifact_id=str(report_artifact["id"]),
            )
            state = app.state.dataset_factory.advance_project_stage(
                dataset_id, "review"
            )
        except (
            DatasetFactoryError,
            json.JSONDecodeError,
            OSError,
            WorkstationError,
        ) as error:
            return _json_error(str(error), 409)
        return JSONResponse(
            {
                "dataset_id": dataset_id,
                "job_id": job_id,
                "report_artifact_id": report_artifact["id"],
                "clip_artifact_ids": clip_artifact_ids,
                "import": imported,
                "state": state,
            }
        )

    @app.post(
        "/api/workstation/datasets/{dataset_id}/expression-candidates/{item_id}/draft"
    )
    async def create_dataset_expression_candidate_draft(
        dataset_id: str, item_id: str, request: Request
    ) -> JSONResponse:
        store: WorkstationStore = app.state.workstation
        factory: DatasetFactory = app.state.dataset_factory
        try:
            body = await request.json()
            if not isinstance(body, dict) or set(body) - {"intensity"}:
                raise WorkstationError(
                    "Expression candidate draft accepts only an optional intensity"
                )
            requested_intensity = body.get("intensity")
            if requested_intensity is not None and (
                isinstance(requested_intensity, bool)
                or not isinstance(requested_intensity, (int, float))
                or not math.isfinite(float(requested_intensity))
                or not 0.0 <= float(requested_intensity) <= 1.0
            ):
                raise WorkstationError("Expression candidate intensity must be from 0 to 1")
            validate_dataset_project(dataset_id)
            item = factory.get_item(item_id)
            if item.get("dataset_id") != dataset_id or item.get("kind") != "segment":
                raise WorkstationError("Expression candidate does not belong to this Dataset")
            if item.get("review_status") != "accepted":
                raise WorkstationError("Expression candidate must be accepted by human review")
            annotations = item.get("annotations")
            if not isinstance(annotations, Mapping):
                raise WorkstationError("Expression candidate annotations are missing")
            expression = annotations.get("expression")
            language = annotations.get("language")
            transcript = annotations.get("transcript")
            if not all(isinstance(value, str) and value.strip() for value in (expression, language, transcript)):
                raise WorkstationError(
                    "Expression candidate requires reviewed expression, language and transcript"
                )
            expression = expression.strip()
            language = language.strip().casefold()
            transcript = transcript.strip()
            annotated_intensity = annotations.get("expression_intensity")
            intensity = (
                float(requested_intensity)
                if requested_intensity is not None
                else float(annotated_intensity)
                if isinstance(annotated_intensity, (int, float))
                and not isinstance(annotated_intensity, bool)
                else 0.7
            )
            style_description = annotations.get("style_description")
            descriptions = (
                [style_description.strip()]
                if isinstance(style_description, str) and style_description.strip()
                else [f"Human-reviewed reference candidate for {expression}"]
            )
            vad = {
                field: float(value)
                for field in ("valence", "arousal", "dominance")
                if isinstance((value := annotations.get(field)), (int, float))
                and not isinstance(value, bool)
            }
            slug = re.sub(r"[^a-z0-9._-]+", "-", expression.casefold()).strip("-._")
            if not slug or not slug[0].isalnum():
                slug = "style-" + hashlib.sha256(expression.encode("utf-8")).hexdigest()[:10]
            slug = slug[:64].rstrip("-._")
            existing = [
                value
                for value in store.list_expression_drafts(model_id=dataset_id)
                if value.get("profile_id") == slug
            ]
            if existing:
                return JSONResponse(
                    {
                        "schema": "aniflive-dataset-expression-candidate-draft-v1",
                        "dataset_id": dataset_id,
                        "item_id": item_id,
                        "reused": True,
                        "draft": existing[0],
                    }
                )
            source = factory.verified_audio_path(item_id)
            source_digest = hashlib.sha256()
            with source.open("rb") as stream:
                while block := stream.read(1024 * 1024):
                    source_digest.update(block)
            sha256 = source_digest.hexdigest()
            relative = PurePosixPath(
                "dataset-expression-candidates",
                dataset_id,
                f"{item_id}-{sha256[:12]}.wav",
            )
            destination = store.artifact_root.joinpath(*relative.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if destination.is_symlink() or not destination.is_file():
                    raise WorkstationError("Expression candidate destination is unsafe")
                digest = hashlib.sha256(destination.read_bytes()).hexdigest()
                if digest != sha256:
                    raise WorkstationError("Expression candidate destination collided")
            else:
                temporary = destination.with_name(f".{destination.name}.tmp")
                try:
                    with source.open("rb") as reader, temporary.open("xb") as writer:
                        shutil.copyfileobj(reader, writer, length=1024 * 1024)
                        writer.flush()
                        os.fsync(writer.fileno())
                    if hashlib.sha256(temporary.read_bytes()).hexdigest() != sha256:
                        raise WorkstationError("Expression candidate copy failed verification")
                    os.replace(temporary, destination)
                finally:
                    temporary.unlink(missing_ok=True)
            lineage = item.get("metadata", {}).get("lineage", {})
            parent_id = lineage.get("parent_artifact_id") if isinstance(lineage, Mapping) else None
            parent_ids: list[str] = []
            if isinstance(parent_id, str):
                store.get_artifact(parent_id)
                parent_ids.append(parent_id)
            artifact_id = "artifact_" + str(
                uuid5(NAMESPACE_URL, f"aniflive-dataset-expression:{dataset_id}:{item_id}:{sha256}")
            )
            try:
                artifact = store.get_artifact(artifact_id)
                if artifact.get("sha256") != sha256:
                    raise WorkstationError("Expression candidate artifact identity collided")
            except WorkstationError as error:
                if "not found" not in str(error).lower():
                    raise
                artifact = store.register_artifact(
                    artifact_type="expression-bank",
                    name=f"{expression} reference candidate",
                    status="planned",
                    project_id=dataset_id,
                    local_path=relative.as_posix(),
                    sha256=sha256,
                    metadata={
                        "schema": "aniflive-expression-reference-candidate-v1",
                        "dataset_id": dataset_id,
                        "dataset_item_id": item_id,
                        "stable_clip_id": item.get("metadata", {}).get("stable_clip_id"),
                        "expression": expression,
                        "language": language,
                        "human_reviewed": True,
                    },
                    parent_artifact_ids=parent_ids,
                    artifact_id=artifact_id,
                )
            candidate_groups = factory.expression_reference_candidates(dataset_id).get(
                "expressions", {}
            )
            candidate_score = None
            for value in candidate_groups.get(expression, []):
                if value.get("item_id") == item_id:
                    candidate_score = value.get("score")
                    break
            draft = store.create_expression_draft(
                name=f"{expression} · Dataset candidate",
                profile_id=slug,
                model_id=dataset_id,
                reference_path=destination,
                language=language,
                emotion=expression,
                intensity=intensity,
                descriptions=descriptions,
                vad=vad,
                prosody={
                    "reference_transcript": transcript,
                    "dataset_candidate": {
                        "dataset_id": dataset_id,
                        "item_id": item_id,
                        "artifact_id": artifact["id"],
                        "candidate_score": candidate_score,
                    },
                },
            )
        except (
            DatasetFactoryError,
            json.JSONDecodeError,
            OSError,
            WorkstationError,
        ) as error:
            return _json_error(str(error), 409)
        return JSONResponse(
            {
                "schema": "aniflive-dataset-expression-candidate-draft-v1",
                "dataset_id": dataset_id,
                "item_id": item_id,
                "reused": False,
                "artifact": artifact,
                "draft": draft,
            },
            status_code=201,
        )

    @app.post("/api/workstation/datasets/{dataset_id}/continue-to-training")
    async def continue_dataset_to_training(
        dataset_id: str, request: Request
    ) -> JSONResponse:
        store: WorkstationStore = app.state.workstation
        try:
            body = await request.json()
            if not isinstance(body, dict) or set(body) - {"name", "model_id"}:
                raise WorkstationError(
                    "Training handoff accepts only optional project name and model_id"
                )
            requested_model_id = body.get("model_id")
            if requested_model_id is not None and (
                not isinstance(requested_model_id, str)
                or not requested_model_id.strip()
            ):
                raise WorkstationError("Training handoff model_id must be non-empty text")
            validate_dataset_project(dataset_id)
            dataset_project = store.get_project(dataset_id)
            manifest = app.state.dataset_factory.verified_frozen_manifest_path(
                dataset_id
            )
            bundle = app.state.dataset_factory.materialize_training_bundle(dataset_id)
            dataset_artifact = _register_training_dataset_artifact(
                store,
                dataset_id=dataset_id,
                dataset_name=dataset_project["name"],
                bundle=bundle,
            )
            training_base = component_root("gpt-sovits-v2proplus-training")
            evaluation_asr_model: str | None = None
            try:
                evaluation_asr_model = str(component_root("faster-whisper-small"))
            except WorkstationAssetError:
                # Checkpoint selection and training do not depend on this optional
                # evaluation component. Canonical evaluation remains fail-closed.
                pass
            name = body.get("name") or f"{dataset_project['name']} Training"
            training_config = {
                "dataset": bundle["path"],
                "shared_dir": str(training_base),
                "pretrained_gpt": str(training_base / "s1v3.ckpt"),
                "pretrained_sovits_g": str(
                    training_base / "v2Pro" / "s2Gv2ProPlus.pth"
                ),
                "pretrained_sovits_d": str(
                    training_base / "v2Pro" / "s2Dv2ProPlus.pth"
                ),
                "preset": store.get_settings()["default_training_preset"],
                "stage": "both",
                "experiment_name": dataset_project["name"],
                "source_dataset_id": dataset_id,
                "source_dataset_manifest_sha256": hashlib.sha256(
                    manifest.read_bytes()
                ).hexdigest(),
                "source_dataset_artifact_id": dataset_artifact["id"],
                "training_bundle_sha256": bundle["training_list_sha256"],
                "speaker_component": str(
                    component_root("eres2netv2-speaker-verifier")
                ),
                "reference_selection_policy": (
                    "reviewed-quality-speaker-centroid-v2"
                ),
                "reference_status": "pending-checkpoint-selection",
                "model_id": _deployment_model_id(
                    requested_model_id or dataset_project["name"], dataset_id
                ),
                "voice_profile": "default",
                "auto_build_production": True,
            }
            if evaluation_asr_model is not None:
                training_config["asr_model"] = evaluation_asr_model
            training = store.create_project(
                kind="training",
                name=name,
                config=training_config,
            )
        except (
            DatasetFactoryError,
            json.JSONDecodeError,
            OSError,
            WorkstationError,
            WorkstationAssetError,
        ) as error:
            return _json_error(str(error), 409)
        return JSONResponse(training, status_code=201)

    @app.post("/api/workstation/training/{project_id}/reference-selection/lock")
    async def lock_training_reference(project_id: str, request: Request) -> JSONResponse:
        store: WorkstationStore = app.state.workstation
        try:
            body = await request.json()
            if not isinstance(body, dict) or set(body) - {"item_id", "decision"}:
                raise WorkstationError("Reference lock accepts item_id and decision")
            item_id = body.get("item_id")
            decision = body.get("decision")
            if decision not in {
                "preferred",
                "confirm-auto-winner",
                "no-preference",
                "all-poor",
            }:
                raise WorkstationError(
                    "Reference decision must be preferred, confirm-auto-winner, "
                    "no-preference, or all-poor"
                )
            if decision in {"preferred", "confirm-auto-winner"} and (
                not isinstance(item_id, str) or not item_id
            ):
                raise WorkstationError("Reference item_id is required")
            if decision in {"no-preference", "all-poor"} and item_id is not None:
                raise WorkstationError(f"{decision} must not select a reference item")
            project = store.get_project(project_id)
            if project.get("kind") != "training":
                raise WorkstationError("Training project was not found")
            config = project.get("config", {})
            if not isinstance(config, Mapping):
                raise WorkstationError("Training project config is malformed")
            superseded_reference_evidence: list[str] = []
            superseded_final_decision: str | None = None
            if config.get("reference_status") == "all-poor":
                if decision == "all-poor":
                    return JSONResponse(project)
                if config.get("test_split_accessed") is True:
                    raise WorkstationError(
                        "Reference rejection cannot be revised after test access"
                    )
                downstream_types = {
                    "holdout.evaluate",
                    "engine.prepare",
                    "conversion.parity",
                    "model.package",
                    "evaluation.prepare",
                }
                if any(
                    job.get("project_id") == project_id
                    and job.get("type") in downstream_types
                    for job in store.list_jobs()
                ):
                    raise WorkstationError(
                        "Reference rejection cannot be revised after downstream work"
                    )
                for field in (
                    "reference_rejection_artifact_id",
                    "reference_rejection_diagnostic_artifact_id",
                ):
                    artifact_id = config.get(field)
                    if isinstance(artifact_id, str) and artifact_id:
                        store.get_artifact(artifact_id)
                        superseded_reference_evidence.append(artifact_id)
                final_decision_id = config.get("final_decision_artifact_id")
                if isinstance(final_decision_id, str) and final_decision_id:
                    store.get_artifact(final_decision_id)
                    superseded_final_decision = final_decision_id
            if config.get("reference_status") == "human-locked":
                if (
                    decision == "no-preference"
                    and config.get("reference_human_decision") == "no-preference"
                ):
                    return JSONResponse(project)
                if config.get("reference_item_id") != item_id:
                    raise WorkstationError("Deployment reference is already locked")
                return JSONResponse(project)
            source_dataset_id = config.get("source_dataset_id")
            if not isinstance(source_dataset_id, str):
                raise WorkstationError("Training project has no source dataset")
            reference_jobs = [
                job
                for job in store.list_jobs()
                if job.get("project_id") == project_id
                and job.get("type") == "reference.select"
                and job.get("status") == "succeeded"
            ]
            if not reference_jobs:
                raise WorkstationError("Reference sweep has not completed")
            reference_job = max(reference_jobs, key=lambda job: str(job.get("updated_at", "")))
            checkpoint_ids = [
                dependency_id
                for dependency_id in reference_job.get("depends_on", [])
                if store.get_job(str(dependency_id)).get("type") == "checkpoint.select"
            ]
            if len(checkpoint_ids) != 1:
                raise WorkstationError("Reference sweep has no locked checkpoint dependency")
            result = reference_job.get("result", {})
            artifact_ids = (
                result.get("registered_artifact_ids", [])
                if isinstance(result, Mapping)
                else []
            )
            report: dict[str, Any] | None = None
            report_artifact_id: str | None = None
            report_sha256: str | None = None
            for artifact_id in artifact_ids:
                artifact = store.get_artifact(str(artifact_id))
                metadata = artifact.get("metadata", {})
                if (
                    isinstance(metadata, Mapping)
                    and metadata.get("worker_relative_path")
                    == "reference-selection-report.json"
                ):
                    report = json.loads(
                        verified_artifact_path(store, artifact).read_text(encoding="utf-8")
                    )
                    report_artifact_id = str(artifact_id)
                    report_sha256 = str(artifact.get("sha256"))
                    break
            if (
                not isinstance(report, dict)
                or report_artifact_id is None
                or not isinstance(report_sha256, str)
            ):
                raise WorkstationError("Reference selection report was not published")
            candidates = report.get("top_candidates")
            if not isinstance(candidates, list):
                raise WorkstationError("Reference candidates are malformed")
            if decision == "all-poor":
                payload = {
                    "schema": "aniflive-tts-v2proplus-reference-rejection-v1",
                    "status": "all-poor",
                    "dataset_id": source_dataset_id,
                    "automatic_recommendation": report.get(
                        "automatic_recommendation"
                    ),
                    "human_decision": "all-poor",
                    "selection_policy": report.get("policy"),
                    "reference_selection_report_sha256": report_sha256,
                }
                destination = (
                    store.artifact_root
                    / "human-evidence"
                    / project_id
                    / "reference-rejection.json"
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(".tmp")
                temporary.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
                    + "\n",
                    encoding="utf-8",
                )
                os.replace(temporary, destination)
                artifact = store.register_artifact(
                    artifact_type="reference",
                    name=f"{project['name']} rejected reference sweep",
                    project_id=project_id,
                    status="rejected",
                    local_path=destination,
                    metadata={
                        "human_decision": "all-poor",
                        "reference_job_id": reference_job["id"],
                    },
                    parent_artifact_ids=(report_artifact_id,),
                )
                updated = store.update_project(
                    project_id,
                    config={
                        **dict(config),
                        "reference_status": "all-poor",
                        "reference_rejection": str(destination),
                        "reference_rejection_artifact_id": artifact["id"],
                    },
                )
                return JSONResponse(
                    {
                        "schema": "aniflive-tts-reference-lock-v1",
                        "project": updated,
                        "reference": payload,
                        "artifact": artifact,
                        "holdout_job": None,
                    },
                    status_code=201,
                )
            if decision == "no-preference":
                item_id = report.get("automatic_recommendation")
                if not isinstance(item_id, str) or not item_id:
                    raise WorkstationError(
                        "Reference report has no automatic recommendation"
                    )
            selected = next(
                (
                    candidate
                    for candidate in candidates
                    if isinstance(candidate, Mapping) and candidate.get("item_id") == item_id
                ),
                None,
            )
            if selected is None:
                raise WorkstationError("Reference item is not in the qualified Top 5")
            if decision in {"confirm-auto-winner", "no-preference"} and (
                report.get("automatic_recommendation") != item_id
            ):
                raise WorkstationError("Selected item is not the automatic recommendation")
            item = app.state.dataset_factory.get_item(item_id)
            if item.get("dataset_id") != source_dataset_id:
                raise WorkstationError("Reference item belongs to another dataset")
            audio = app.state.dataset_factory.verified_audio_path(item_id)
            annotations = item.get("annotations", {})
            payload = {
                "schema": "aniflive-tts-v2proplus-deployment-reference-v2",
                "status": "human-locked",
                "dataset_id": source_dataset_id,
                "item_id": item_id,
                "audio_path": str(audio),
                "audio_sha256": item["sha256"],
                "text": annotations["transcript"],
                "language": annotations["language"],
                "automatic_recommendation": report["automatic_recommendation"],
                "human_decision": decision,
                "selection_policy": report["policy"],
                "reference_selection_report_sha256": report_sha256,
            }
            if superseded_reference_evidence:
                payload["supersedes_reference_evidence"] = list(
                    superseded_reference_evidence
                )
            if superseded_final_decision is not None:
                payload["supersedes_final_decision"] = superseded_final_decision
            destination = (
                store.artifact_root
                / "human-evidence"
                / project_id
                / "deployment-reference.json"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, destination)
            artifact = store.register_artifact(
                artifact_type="reference",
                name=f"{project['name']} deployment reference",
                project_id=project_id,
                status="ready",
                local_path=destination,
                metadata={
                    "human_decision": decision,
                    "source_item_id": item_id,
                    "reference_job_id": reference_job["id"],
                    **(
                        {
                            "supersedes_reference_evidence": list(
                                superseded_reference_evidence
                            )
                        }
                        if superseded_reference_evidence
                        else {}
                    ),
                    **(
                        {"supersedes_final_decision": superseded_final_decision}
                        if superseded_final_decision is not None
                        else {}
                    ),
                },
                parent_artifact_ids=tuple(
                    dict.fromkeys(
                        (
                            report_artifact_id,
                            *superseded_reference_evidence,
                            *(
                                (superseded_final_decision,)
                                if superseded_final_decision is not None
                                else ()
                            ),
                        )
                    )
                ),
            )
            next_config = dict(config)
            if superseded_reference_evidence or superseded_final_decision is not None:
                for field in (
                    "reference_rejection",
                    "reference_rejection_artifact_id",
                    "reference_rejection_diagnosis",
                    "reference_rejection_diagnostic_artifact_id",
                    "reference_rejection_diagnostic_sha256",
                    "production_status",
                    "final_decision",
                    "final_decision_artifact_id",
                    "final_decision_sha256",
                ):
                    next_config.pop(field, None)
                next_config["reference_decision_supersedes_artifact_ids"] = list(
                    dict.fromkeys(
                        (
                            *superseded_reference_evidence,
                            *(
                                (superseded_final_decision,)
                                if superseded_final_decision is not None
                                else ()
                            ),
                        )
                    )
                )
            updated = store.update_project(
                project_id,
                config={
                    **next_config,
                    "reference_status": "human-locked",
                    "reference": str(audio),
                    "reference_text": annotations["transcript"],
                    "reference_language": annotations["language"],
                    "reference_item_id": item_id,
                    "reference_sha256": item["sha256"],
                    "reference_human_decision": decision,
                    "deployment_reference": str(destination),
                    "deployment_reference_artifact_id": artifact["id"],
                },
            )
            holdout = store.create_job(
                job_type="holdout.evaluate",
                project_id=project_id,
                depends_on=(checkpoint_ids[0], str(reference_job["id"])),
                priority=int(reference_job.get("priority", 0)),
                parameters={"parent_artifact_ids": [artifact["id"]]},
            )
        except (
            DatasetFactoryError,
            json.JSONDecodeError,
            KeyError,
            OSError,
            WorkstationError,
        ) as error:
            return _json_error(str(error), 409)
        return JSONResponse(
            {
                "schema": "aniflive-tts-reference-lock-v1",
                "project": updated,
                "reference": payload,
                "artifact": artifact,
                "holdout_job": holdout,
            },
            status_code=201,
        )

    @app.get("/api/workstation/datasets/{dataset_id}/qualification-report")
    def dataset_voice_acquisition_report(dataset_id: str) -> JSONResponse:
        try:
            validate_dataset_project(dataset_id)
            report = build_voice_acquisition_report(
                app.state.workstation,
                app.state.dataset_factory,
                dataset_id,
            )
        except (DatasetFactoryError, ValueError, WorkstationError) as error:
            return _json_error(str(error), 409)
        return JSONResponse(report)

    async def cancel_active_upstream() -> bool:
        async with app.state.active_upstream_guard:
            active = app.state.active_upstream
            app.state.active_upstream = None
        core_cancelled = False
        try:
            try:
                cancel_path = active.cancel_path if active is not None else "/v1/audio/cancel"
                response = await app.state.client.post(cancel_path)
                if response.status_code == 200:
                    payload = response.json()
                    core_cancelled = (
                        bool(payload.get("cancelled")) if isinstance(payload, dict) else False
                    )
            except (httpx.HTTPError, ValueError):
                LOGGER.debug("Upstream explicit stream cancellation was unavailable", exc_info=True)
        finally:
            # The core cancellation request is allowed to be cancelled, but the
            # already-captured response owner must still release its upstream
            # socket and speech_lock. Owner cleanup runs in a shielded task.
            if active is not None:
                await active.aclose()
        return active is not None or core_cancelled

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        try:
            hostname, _ = _authority_parts(request.headers.get("host", ""))
        except WebUIError as error:
            return _json_error(str(error), 400)
        if not _loopback_hostname(hostname) and hostname not in _trusted_webui_hostnames():
            return _json_error("HTTP Host is not trusted by the local WebUI", 400)

        method = request.method.upper()
        if method in _MUTATION_METHODS:
            fetch_site = request.headers.get("sec-fetch-site", "").strip().lower()
            if fetch_site == "cross-site":
                return _json_error("Cross-site WebUI mutations are not allowed", 403)
            origin = request.headers.get("origin")
            if origin is not None and not _origin_matches_request(origin, request):
                return _json_error("Cross-origin WebUI mutations are not allowed", 403)
            if _requires_json_content_type(method, request.url.path):
                media_type = request.headers.get("content-type", "").partition(";")[0]
                if media_type.strip().lower() != "application/json":
                    return _json_error(
                        "This WebUI mutation requires application/json", 415
                    )

        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("Cache-Control", "no-store")
        return response

    @app.get("/")
    async def index() -> FileResponse:
        asset = root / (
            "index.html" if default_surface == "studio" else "synthesis.html"
        )
        return FileResponse(asset, headers={"Cache-Control": "no-store"})

    @app.get("/studio")
    async def studio() -> FileResponse:
        return FileResponse(root / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/webui")
    @app.get("/webui/")
    async def classic_webui() -> FileResponse:
        return FileResponse(
            root / "synthesis.html", headers={"Cache-Control": "no-store"}
        )

    @app.get("/synthesis")
    async def synthesis() -> FileResponse:
        asset = root / "synthesis.html"
        if not asset.is_file():
            return FileResponse(root / "index.html", headers={"Cache-Control": "no-store"})
        return FileResponse(asset, headers={"Cache-Control": "no-store"})

    @app.get("/assets/everynight_dance.gif")
    async def project_gif() -> Response:
        asset = root.parent / "assets" / "everynight_dance.gif"
        if not asset.is_file():
            import base64

            return Response(
                base64.b64decode("R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"),
                media_type="image/gif",
                headers={"Cache-Control": "public, max-age=3600"},
            )
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

    @app.get("/assets/studio.css")
    async def studio_css() -> FileResponse:
        return FileResponse(
            root / "studio.css",
            media_type="text/css",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/assets/styled_select.css")
    async def styled_select_css() -> FileResponse:
        return FileResponse(
            root / "styled_select.css",
            media_type="text/css",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/assets/styled_select.js")
    async def styled_select_js() -> FileResponse:
        return FileResponse(
            root / "styled_select.js",
            media_type="text/javascript",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/assets/studio.js")
    async def studio_js() -> FileResponse:
        return FileResponse(
            root / "studio.js",
            media_type="text/javascript",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/assets/studio_i18n.js")
    async def studio_i18n_js() -> FileResponse:
        return FileResponse(
            root / "studio_i18n.js",
            media_type="text/javascript",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/assets/dataset_factory_model.js")
    async def dataset_factory_model_js() -> FileResponse:
        return FileResponse(
            root / "dataset_factory_model.js",
            media_type="text/javascript",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/assets/tse_workstation_model.js")
    async def tse_workstation_model_js() -> FileResponse:
        return FileResponse(
            root / "tse_workstation_model.js",
            media_type="text/javascript",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/assets/job_controls.js")
    async def job_controls_js() -> FileResponse:
        return FileResponse(
            root / "job_controls.js",
            media_type="text/javascript",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/assets/lucide.min.js")
    async def lucide_js() -> FileResponse:
        return FileResponse(
            root / "lucide.min.js",
            media_type="text/javascript",
            headers={"Cache-Control": "public, max-age=31536000, immutable"},
        )

    @app.get("/studio.webmanifest")
    async def studio_manifest() -> FileResponse:
        return FileResponse(
            root / "studio.webmanifest",
            media_type="application/manifest+json",
            headers={"Cache-Control": "no-cache"},
        )

    @app.get("/studio-sw.js")
    async def studio_service_worker() -> FileResponse:
        return FileResponse(
            root / "studio-sw.js",
            media_type="text/javascript",
            headers={
                "Cache-Control": "no-cache",
                "Service-Worker-Allowed": "/",
            },
        )

    @app.get("/offline.html")
    async def studio_offline_shell() -> FileResponse:
        return FileResponse(
            root / "offline.html",
            media_type="text/html",
            headers={"Cache-Control": "public, max-age=3600"},
        )

    @app.get("/pwa/studio-icon.svg")
    async def studio_icon() -> FileResponse:
        return FileResponse(
            root / "pwa" / "studio-icon.svg",
            media_type="image/svg+xml",
            headers={"Cache-Control": "public, max-age=31536000, immutable"},
        )

    @app.get("/pwa/studio-icon-192.png")
    async def studio_icon_192() -> FileResponse:
        return FileResponse(
            root / "pwa" / "studio-icon-192.png",
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=31536000, immutable"},
        )

    @app.get("/pwa/studio-icon-512.png")
    async def studio_icon_512() -> FileResponse:
        return FileResponse(
            root / "pwa" / "studio-icon-512.png",
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=31536000, immutable"},
        )

    @app.get("/media/voice-workstation-background.mp4")
    async def voice_workstation_background() -> FileResponse:
        return FileResponse(
            root / "media" / "voice-workstation-background.mp4",
            media_type="video/mp4",
            headers={"Cache-Control": "public, max-age=31536000, immutable"},
        )

    @app.get("/media/voice-workstation-poster.jpg")
    async def voice_workstation_poster() -> FileResponse:
        return FileResponse(
            root / "media" / "voice-workstation-poster.jpg",
            media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=31536000, immutable"},
        )

    @app.get("/api/workstation/overview")
    async def workstation_overview(request: Request) -> JSONResponse:
        store: WorkstationStore = request.app.state.workstation
        snapshot = store.snapshot()
        projects = presented_projects(
            visible_records(store, list(snapshot.projects))
        )
        jobs = visible_records(store, list(snapshot.jobs))
        running = [record for record in jobs if record.get("status") == "running"]
        queued = [record for record in jobs if record.get("status") == "queued"]
        health = request.app.state.status.get("health", {})
        return JSONResponse(
            {
                "product": surface_name,
                "workspace_version": "1.4.0",
                "active_model": health.get("model"),
                "gpu": health.get("gpu", {}),
                "backend": health.get("backend"),
                "ready": bool(health.get("ready")),
                "project_counts": {
                    kind: sum(1 for record in projects if record.get("kind") == kind)
                    for kind in sorted(PROJECT_KINDS)
                },
                "running_jobs": running,
                "queued_jobs": len(queued),
                "recent_projects": sorted(
                    projects,
                    key=lambda record: str(record.get("updated_at", "")),
                    reverse=True,
                )[:5],
            },
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/workstation/projects")
    async def workstation_projects(request: Request) -> JSONResponse:
        store: WorkstationStore = request.app.state.workstation
        try:
            records = presented_projects(
                visible_records(
                    store,
                    store.list_projects(kind=request.query_params.get("kind")),
                )
            )
        except WorkstationError as error:
            return _json_error(str(error), 400)
        return JSONResponse({"object": "list", "data": records})

    @app.get("/api/workstation/settings")
    async def workstation_settings(request: Request) -> JSONResponse:
        store: WorkstationStore = request.app.state.workstation
        try:
            settings = store.get_settings()
            visible_history_cleared_at = store.get_visible_history_cutoff()
        except WorkstationError as error:
            return _json_error(str(error), 500)
        return JSONResponse(
            {
                "object": "workstation.settings",
                "data": settings,
                "visible_history_cleared_at": visible_history_cleared_at,
            }
        )

    @app.post("/api/workstation/path-picker")
    async def workstation_path_picker(request: Request) -> JSONResponse:
        store: WorkstationStore = request.app.state.workstation
        filters = {
            "all": "All files (*.*)|*.*",
            "audio": "Audio files (*.wav;*.flac;*.mp3;*.m4a;*.ogg)|*.wav;*.flac;*.mp3;*.m4a;*.ogg|All files (*.*)|*.*",
            "media": "Audio and video (*.wav;*.flac;*.mp3;*.m4a;*.ogg;*.mp4;*.mkv;*.mov;*.webm)|*.wav;*.flac;*.mp3;*.m4a;*.ogg;*.mp4;*.mkv;*.mov;*.webm|All files (*.*)|*.*",
            "checkpoint": "Model checkpoints (*.ckpt;*.pth;*.pt;*.safetensors)|*.ckpt;*.pth;*.pt;*.safetensors|All files (*.*)|*.*",
            "json": "JSON reports (*.json)|*.json|All files (*.*)|*.*",
            "zip": "ZIP bundles (*.zip)|*.zip|All files (*.*)|*.*",
            "list": "GPT-SoVITS lists (*.list)|*.list|All files (*.*)|*.*",
        }
        try:
            body = await request.json()
            if not isinstance(body, dict) or set(body) - {"kind", "title", "accept"}:
                raise WorkstationPathPickerError(
                    "Path picker accepts kind, title and optional accept"
                )
            kind = body.get("kind")
            title = body.get("title")
            accept = body.get("accept", "all")
            if kind not in {"file", "files", "directory", "save-file"}:
                raise WorkstationPathPickerError("Unsupported path picker kind")
            if accept not in filters:
                raise WorkstationPathPickerError("Unsupported path picker filter")
            roots = store.allowed_import_roots()
            initial_directory = roots[0] if roots else Path.home()
            paths = await asyncio.to_thread(
                pick_workstation_paths,
                kind=kind,
                title=title,
                initial_directory=initial_directory,
                file_filter=filters[accept],
            )
        except (
            json.JSONDecodeError,
            OSError,
            WorkstationError,
            WorkstationPathPickerError,
        ) as error:
            return _json_error(str(error), 409)
        return JSONResponse(
            {
                "object": "workstation.path-selection",
                "cancelled": not paths,
                "paths": paths,
            }
        )

    @app.get("/api/workstation/components")
    async def workstation_components(request: Request) -> JSONResponse:
        manager: WorkstationAssetManager = request.app.state.workstation_assets
        try:
            return JSONResponse(await asyncio.to_thread(manager.status_snapshot))
        except WorkstationAssetError as error:
            return _json_error(str(error), 500)

    @app.post("/api/workstation/components/install")
    async def install_workstation_component(request: Request) -> JSONResponse:
        store: WorkstationStore = request.app.state.workstation
        manager: WorkstationAssetManager = request.app.state.workstation_assets
        try:
            body = await request.json()
            if not isinstance(body, dict) or set(body) != {"component_id", "source"}:
                raise WorkstationAssetError(
                    "Component install requires component_id and source"
                )
            source = store.validate_import_path(body["source"])
            if not source.is_dir():
                raise WorkstationAssetError("Component source must be a directory")
            record = await asyncio.to_thread(
                manager.install_from_directory, body["component_id"], source
            )
        except (
            json.JSONDecodeError,
            OSError,
            WorkstationAssetError,
            WorkstationError,
        ) as error:
            return _json_error(str(error), 409)
        return JSONResponse(record)

    @app.post("/api/workstation/components/download")
    async def download_workstation_components(request: Request) -> JSONResponse:
        """Run the explicit online setup action outside all offline workers."""

        manager: WorkstationAssetManager = request.app.state.workstation_assets
        try:
            body = await request.json()
            if not isinstance(body, dict) or set(body) - {"component_ids"}:
                raise WorkstationAssetError(
                    "Component download accepts only optional component_ids"
                )
            component_ids = body.get("component_ids")
            if component_ids is None:
                status = await asyncio.to_thread(manager.status)
                component_ids = [
                    record["id"]
                    for record in status["components"]
                    if record.get("required")
                    and not record.get("ready")
                    and record.get("online_installable")
                ]
            if (
                isinstance(component_ids, (str, bytes))
                or not isinstance(component_ids, list)
                or len(component_ids) > 16
                or any(not isinstance(value, str) for value in component_ids)
            ):
                raise WorkstationAssetError("component_ids must be a short string list")
            if not component_ids:
                raise WorkstationAssetError(
                    "No missing online-installable AI Components were selected"
                )
            installed = []
            for component_id in component_ids:
                installed.append(
                    await asyncio.to_thread(manager.download_component, component_id)
                )
        except (
            json.JSONDecodeError,
            OSError,
            WorkstationAssetError,
        ) as error:
            return _json_error(str(error), 409)
        return JSONResponse(
            {
                "schema": "aniflive-workstation-component-download-v1",
                "installed": installed,
                "worker_network_required": False,
            }
        )

    @app.post("/api/workstation/components/import")
    async def import_workstation_components(request: Request) -> JSONResponse:
        store: WorkstationStore = request.app.state.workstation
        manager: WorkstationAssetManager = request.app.state.workstation_assets
        try:
            body = await request.json()
            if not isinstance(body, dict) or set(body) != {"path"}:
                raise WorkstationAssetError("Component import requires path")
            bundle = store.validate_import_path(body["path"])
            result = await asyncio.to_thread(manager.import_bundle, bundle)
        except (
            json.JSONDecodeError,
            OSError,
            WorkstationAssetError,
            WorkstationError,
        ) as error:
            return _json_error(str(error), 409)
        return JSONResponse(result)

    @app.post("/api/workstation/components/export")
    async def export_workstation_components(request: Request) -> JSONResponse:
        store: WorkstationStore = request.app.state.workstation
        manager: WorkstationAssetManager = request.app.state.workstation_assets
        try:
            body = await request.json()
            if not isinstance(body, dict) or not set(body).issubset(
                {"path", "component_ids"}
            ) or "path" not in body:
                raise WorkstationAssetError(
                    "Component export requires path and optional component_ids"
                )
            component_ids = body.get("component_ids")
            if component_ids is not None and (
                isinstance(component_ids, (str, bytes))
                or not isinstance(component_ids, list)
            ):
                raise WorkstationAssetError("component_ids must be a list")
            destination = component_bundle_destination(store, body["path"])
            exported = await asyncio.to_thread(manager.export_bundle, destination, component_ids)
        except (
            json.JSONDecodeError,
            OSError,
            WorkstationAssetError,
            WorkstationError,
        ) as error:
            return _json_error(str(error), 409)
        return JSONResponse({"path": str(exported), "network_required": False})

    @app.patch("/api/workstation/settings")
    async def update_workstation_settings(request: Request) -> JSONResponse:
        store: WorkstationStore = request.app.state.workstation
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise WorkstationError("Request body must be a JSON object")
            settings = store.update_settings(body)
        except (json.JSONDecodeError, WorkstationError) as error:
            return _json_error(str(error), 400)
        return JSONResponse({"object": "workstation.settings", "data": settings})

    @app.post("/api/workstation/ui-history/clear")
    async def clear_workstation_visible_history(request: Request) -> JSONResponse:
        store: WorkstationStore = request.app.state.workstation
        try:
            cutoff = store.clear_visible_history()
        except WorkstationError as error:
            return _json_error(str(error), 500)
        return JSONResponse(
            {
                "object": "workstation.visible-history",
                "cleared_at": cutoff,
                "records_deleted": 0,
                "assets_deleted": 0,
            }
        )

    @app.post("/api/workstation/projects")
    async def create_workstation_project(request: Request) -> JSONResponse:
        store: WorkstationStore = request.app.state.workstation
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise WorkstationError("Request body must be a JSON object")
            record = store.create_project(
                kind=body.get("kind"),
                name=body.get("name"),
                config=body.get("config"),
            )
            if record.get("kind") == "dataset":
                request.app.state.dataset_factory.ensure_project(
                    record["id"], record["config"]
                )
        except (json.JSONDecodeError, WorkstationError) as error:
            return _json_error(str(error), 400)
        except DatasetFactoryError as error:
            return _json_error(str(error), 409)
        return JSONResponse(record, status_code=201)

    @app.patch("/api/workstation/projects/{project_id}")
    async def update_workstation_project(
        project_id: str, request: Request
    ) -> JSONResponse:
        store: WorkstationStore = request.app.state.workstation
        try:
            body = await request.json()
            if (
                not isinstance(body, dict)
                or not body
                or not set(body).issubset({"name", "config"})
            ):
                raise WorkstationError(
                    "Project update requires name and/or config"
                )
            record = store.update_project(
                project_id,
                name=body.get("name"),
                config=body.get("config") if "config" in body else None,
            )
            if record.get("kind") == "dataset":
                request.app.state.dataset_factory.ensure_project(
                    record["id"], record["config"]
                )
        except (
            DatasetFactoryError,
            json.JSONDecodeError,
            WorkstationError,
        ) as error:
            status_code = 404 if "not found" in str(error).lower() else 409
            return _json_error(str(error), status_code)
        return JSONResponse(record)

    @app.get("/api/workstation/jobs")
    async def workstation_jobs(request: Request) -> JSONResponse:
        store: WorkstationStore = request.app.state.workstation
        records = visible_records(store, store.list_jobs())
        return JSONResponse({"object": "list", "data": records})

    @app.get("/api/workstation/jobs/{job_id}")
    async def workstation_job(job_id: str, request: Request) -> JSONResponse:
        store: WorkstationStore = request.app.state.workstation
        try:
            record = store.get_job(job_id)
            logs = store.list_job_logs(job_id)
        except WorkstationError as error:
            return _json_error(str(error), 404)
        return JSONResponse({"job": record, "logs": logs})

    async def evaluation_preflight(store, project_id, parameters):
        from .evaluation_preflight import preflight_workstation_evaluation
        return await asyncio.to_thread(
            preflight_workstation_evaluation, store, project_id, parameters,
        )

    @app.post("/api/workstation/evaluation/preflight")
    async def preflight_workstation_evaluation(request: Request) -> JSONResponse:
        try:
            body = await request.json()
            if not isinstance(body, dict) or set(body) - {"project_id", "parameters"}:
                raise WorkstationError("Evaluation preflight requires project_id and optional parameters")
            parameters = body.get("parameters", {})
            if not isinstance(parameters, dict):
                raise WorkstationError("Job parameters must be a JSON object")
            result = await evaluation_preflight(
                request.app.state.workstation, body.get("project_id"), parameters,
            )
        except (json.JSONDecodeError, WorkstationError) as error:
            return _json_error(str(error), 400)
        return JSONResponse(result)

    @app.post("/api/workstation/jobs")
    async def create_workstation_job(request: Request) -> JSONResponse:
        store: WorkstationStore = request.app.state.workstation
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise WorkstationError("Request body must be a JSON object")
            parameters = body.get("parameters")
            if parameters is None:
                parameters = {}
            if not isinstance(parameters, Mapping):
                raise WorkstationError("Job parameters must be a JSON object")
            parameters = dict(parameters)
            if body.get("type") == "evaluation.prepare":
                await evaluation_preflight(store, body.get("project_id"), parameters)
            if body.get("type") == "training.prepare" and isinstance(
                body.get("project_id"), str
            ):
                project = store.get_project(body["project_id"])
                source_artifact_id = project.get("config", {}).get(
                    "source_dataset_artifact_id"
                )
                if source_artifact_id is not None:
                    store.get_artifact(source_artifact_id)
                    parents = parameters.get("parent_artifact_ids", [])
                    if isinstance(parents, (str, bytes)) or not isinstance(
                        parents, list
                    ):
                        raise WorkstationError(
                            "parent_artifact_ids must be a list of artifact IDs"
                        )
                    parameters["parent_artifact_ids"] = list(
                        dict.fromkeys([source_artifact_id, *parents])
                    )
            record = store.create_job(
                job_type=body.get("type"),
                project_id=body.get("project_id"),
                parameters=parameters,
                depends_on=body.get("depends_on"),
                priority=body.get("priority", 0),
            )
        except (json.JSONDecodeError, WorkstationError) as error:
            return _json_error(str(error), 400)
        return JSONResponse(record, status_code=201)

    @app.post("/api/workstation/jobs/{job_id}/run")
    async def run_workstation_job(job_id: str, request: Request) -> JSONResponse:
        store: WorkstationStore = request.app.state.workstation
        jobs = {record["id"]: record for record in store.list_jobs()}
        record = jobs.get(job_id)
        if record is None:
            return _json_error("Job was not found", 404)
        if record.get("type") != "dataset.inventory":
            return _json_error(
                "This workflow adapter is not available in the v1.4 development build",
                409,
            )
        try:
            result = await asyncio.to_thread(store.run_dataset_inventory, job_id)
        except WorkstationError as error:
            return _json_error(str(error), 400)
        return JSONResponse(result)

    @app.post("/api/workstation/jobs/{job_id}/cancel")
    async def cancel_workstation_job(job_id: str, request: Request) -> JSONResponse:
        store: WorkstationStore = request.app.state.workstation
        try:
            result = store.cancel_job(job_id)
        except WorkstationError as error:
            return _json_error(str(error), 404)
        return JSONResponse(result)

    @app.post("/api/workstation/jobs/{job_id}/pause")
    async def pause_workstation_job(job_id: str, request: Request) -> JSONResponse:
        store: WorkstationStore = request.app.state.workstation
        try:
            result = store.pause_job(job_id)
        except WorkstationError as error:
            message = str(error)
            return _json_error(message, 404 if "not found" in message.lower() else 409)
        return JSONResponse(result)

    @app.post("/api/workstation/jobs/{job_id}/resume")
    async def resume_workstation_job(job_id: str, request: Request) -> JSONResponse:
        store: WorkstationStore = request.app.state.workstation
        try:
            source = store.get_job(job_id)
            if source["type"] == "evaluation.prepare":
                await evaluation_preflight(store, source["project_id"], source["parameters"])
            result = await asyncio.to_thread(store.resume_job, job_id)
        except WorkstationError as error:
            message = str(error)
            return _json_error(message, 404 if "not found" in message.lower() else 409)
        return JSONResponse(result)

    @app.post("/api/workstation/jobs/{job_id}/retry")
    async def retry_workstation_job(job_id: str, request: Request) -> JSONResponse:
        store: WorkstationStore = request.app.state.workstation
        try:
            source = store.get_job(job_id)
            if source["type"] == "evaluation.prepare":
                await evaluation_preflight(store, source["project_id"], source["parameters"])
            result = store.retry_job(job_id)
        except WorkstationError as error:
            message = str(error)
            return _json_error(message, 404 if "not found" in message.lower() else 409)
        return JSONResponse(result, status_code=201)

    @app.get("/api/workstation/artifacts")
    async def workstation_artifacts(request: Request) -> JSONResponse:
        store: WorkstationStore = request.app.state.workstation
        try:
            records = visible_records(
                store,
                store.list_artifacts(
                    artifact_type=request.query_params.get("type"),
                    status=request.query_params.get("status"),
                    project_id=request.query_params.get("project_id"),
                ),
            )
        except WorkstationError as error:
            return _json_error(str(error), 400)
        return JSONResponse({"object": "list", "data": records})

    @app.post("/api/workstation/artifacts")
    async def register_workstation_artifact(request: Request) -> JSONResponse:
        store: WorkstationStore = request.app.state.workstation
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise WorkstationError("Request body must be a JSON object")
            record = store.register_artifact(
                artifact_type=body.get("type"),
                name=body.get("name"),
                status=body.get("status", "planned"),
                project_id=body.get("project_id"),
                local_path=body.get("local_path"),
                sha256=body.get("sha256"),
                metadata=body.get("metadata"),
                parent_artifact_ids=body.get("parent_artifact_ids"),
            )
        except (json.JSONDecodeError, WorkstationError) as error:
            return _json_error(str(error), 400)
        return JSONResponse(record, status_code=201)

    @app.get("/api/workstation/artifacts/{artifact_id}")
    async def workstation_artifact_details(
        artifact_id: str, request: Request
    ) -> JSONResponse:
        store: WorkstationStore = request.app.state.workstation
        try:
            result = store.artifact_details(artifact_id)
        except WorkstationError as error:
            message = str(error)
            return _json_error(message, 404 if "not found" in message.lower() else 400)
        return JSONResponse(result)

    @app.get(
        "/api/workstation/artifacts/{artifact_id}/content",
        response_model=None,
    )
    async def workstation_artifact_content(
        artifact_id: str, request: Request
    ) -> FileResponse | JSONResponse:
        store: WorkstationStore = request.app.state.workstation
        try:
            artifact = store.get_artifact(artifact_id)
            if artifact.get("status") != "ready":
                raise WorkstationError("Only ready artifacts can be read")
            relative = validated_artifact_relative_path(
                artifact.get("local_path"), field="artifact local_path"
            )
            artifact_root = store.artifact_root.resolve(strict=True)
            path = artifact_root.joinpath(*PurePosixPath(relative).parts).resolve(
                strict=True
            )
            if not path.is_file() or artifact_root not in path.parents:
                raise WorkstationError("Artifact content escaped the artifact store")
        except (OSError, WorkstationError) as error:
            message = str(error)
            status_code = 404 if "not found" in message.lower() else 409
            return _json_error(message, status_code)

        media_types = {
            ".json": "application/json",
            ".wav": "audio/wav",
            ".flac": "audio/flac",
            ".mp3": "audio/mpeg",
            ".ogg": "audio/ogg",
        }
        download = request.query_params.get("download", "").lower() in {
            "1",
            "true",
            "yes",
        }
        return FileResponse(
            path,
            filename=str(artifact.get("name") or path.name) if download else None,
            media_type=media_types.get(path.suffix.lower(), "application/octet-stream"),
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/workstation/qualifications")
    async def workstation_qualifications(request: Request) -> JSONResponse:
        store: WorkstationStore = request.app.state.workstation
        try:
            records = visible_records(
                store,
                store.list_qualifications(
                    subject_kind=request.query_params.get("subject_kind"),
                    subject_id=request.query_params.get("subject_id"),
                    evaluation_artifact_id=request.query_params.get(
                        "evaluation_artifact_id"
                    ),
                ),
            )
        except WorkstationError as error:
            return _json_error(str(error), 400)
        return JSONResponse({"object": "list", "data": records})

    @app.post("/api/workstation/qualifications/import")
    async def import_workstation_qualification(request: Request) -> JSONResponse:
        store: WorkstationStore = request.app.state.workstation
        try:
            body = await request.json()
            if not isinstance(body, dict) or set(body) != {
                "evaluation_artifact_id", "subject_kind", "subject_id"
            }:
                raise WorkstationError(
                    "Qualification import requires evaluation_artifact_id, "
                    "subject_kind, and subject_id"
                )
            record = store.record_qualification(
                evaluation_artifact_id=body["evaluation_artifact_id"],
                subject_kind=body["subject_kind"],
                subject_id=body["subject_id"],
            )
        except (json.JSONDecodeError, WorkstationError) as error:
            message = str(error)
            status_code = 404 if "not found" in message.lower() else 409
            return _json_error(message, status_code)
        return JSONResponse(record, status_code=201)

    @app.post("/api/workstation/qualifications/compose")
    async def compose_workstation_qualification(request: Request) -> JSONResponse:
        store: WorkstationStore = request.app.state.workstation
        required = {
            "subject_kind",
            "subject_id",
            "automated_evaluation_artifact_id",
            "long_form_evidence_artifact_id",
            "security_evidence_artifact_id",
        }
        try:
            body = await request.json()
            if (not isinstance(body, dict) or not required.issubset(body)
                    or set(body) - required - {"content_evidence_artifact_id", "expression_evidence_artifact_id"}):
                raise WorkstationError(
                    "Qualification composition requires subject_kind, subject_id, "
                    "automated_evaluation_artifact_id, long_form_evidence_artifact_id, "
                    "and security_evidence_artifact_id"
                )
            if body.get("expression_evidence_artifact_id") is None or body.get("expression_evidence_artifact_id") == "":
                body.pop("expression_evidence_artifact_id", None)
            result = await asyncio.to_thread(store.compose_qualification, **body)
        except (json.JSONDecodeError, WorkstationError) as error:
            message = str(error)
            status_code = 404 if "not found" in message.lower() else 409
            return _json_error(message, status_code)
        return JSONResponse(result, status_code=201)

    @app.post("/api/workstation/artifacts/{artifact_id}/promote")
    async def promote_workstation_artifact(
        artifact_id: str, request: Request
    ) -> JSONResponse:
        store: WorkstationStore = request.app.state.workstation
        try:
            body = await request.json()
            if not isinstance(body, dict) or set(body) != {"qualification_id"}:
                raise WorkstationError("Promotion requires qualification_id")
            from .workstation_deployment import promote_and_publish
            record = await asyncio.to_thread(
                promote_and_publish, store, artifact_id, body["qualification_id"]
            )
        except (json.JSONDecodeError, WorkstationError) as error:
            message = str(error)
            status_code = 404 if "not found" in message.lower() else 409
            return _json_error(message, status_code)
        return JSONResponse(record)

    @app.get("/api/workstation/models")
    async def workstation_models(request: Request) -> JSONResponse:
        records = request.app.state.status.get("models", [])
        return JSONResponse({"object": "list", "data": records})

    @app.get("/api/workstation/expression-bank")
    async def workstation_expression_bank(request: Request) -> JSONResponse:
        metadata = request.app.state.status.get("expressions", {})
        profiles = metadata.get("profiles", []) if isinstance(metadata, Mapping) else []
        return JSONResponse(
            {
                "object": "list",
                "source": "active-model-package",
                "data": profiles if isinstance(profiles, list) else [],
            }
        )

    @app.get("/api/workstation/expression-drafts")
    async def workstation_expression_drafts(request: Request) -> JSONResponse:
        store: WorkstationStore = request.app.state.workstation
        try:
            records = visible_records(
                store,
                store.list_expression_drafts(
                    model_id=request.query_params.get("model_id"),
                    language=request.query_params.get("language"),
                    emotion=request.query_params.get("emotion"),
                    qualification_status=request.query_params.get("qualification_status"),
                ),
            )
        except WorkstationError as error:
            return _json_error(str(error), 400)
        return JSONResponse(
            {"object": "list", "source": "local-workstation", "data": records}
        )

    @app.post("/api/workstation/expression-drafts")
    async def create_workstation_expression_draft(request: Request) -> JSONResponse:
        store: WorkstationStore = request.app.state.workstation
        allowed = {
            "name", "profile_id", "model_id", "reference_path", "language",
            "emotion", "intensity", "descriptions", "vad", "prosody",
            "qualification_status",
        }
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise WorkstationError("Request body must be a JSON object")
            unknown = set(body) - allowed
            if unknown:
                raise WorkstationError(
                    "Request body contains unsupported fields: "
                    + ", ".join(sorted(unknown))
                )
            record = store.create_expression_draft(
                name=body.get("name"),
                profile_id=body.get("profile_id"),
                model_id=body.get("model_id"),
                reference_path=body.get("reference_path"),
                language=body.get("language"),
                emotion=body.get("emotion"),
                intensity=body.get("intensity"),
                descriptions=body.get("descriptions"),
                vad=body.get("vad"),
                prosody=body.get("prosody"),
                qualification_status=body.get("qualification_status", "draft"),
            )
        except (json.JSONDecodeError, WorkstationError) as error:
            status_code = 409 if "already exists" in str(error).lower() else 400
            return _json_error(str(error), status_code)
        return JSONResponse(record, status_code=201)

    @app.get("/api/workstation/expression-drafts/{expression_id}")
    async def workstation_expression_draft(
        expression_id: str, request: Request
    ) -> JSONResponse:
        store: WorkstationStore = request.app.state.workstation
        try:
            record = store.get_expression_draft(expression_id)
        except WorkstationError as error:
            status_code = 404 if "not found" in str(error).lower() else 400
            return _json_error(str(error), status_code)
        return JSONResponse(record)

    @app.patch("/api/workstation/expression-drafts/{expression_id}")
    async def update_workstation_expression_draft(
        expression_id: str, request: Request
    ) -> JSONResponse:
        store: WorkstationStore = request.app.state.workstation
        mutable = {
            "name", "language", "emotion", "intensity", "descriptions", "vad",
            "prosody", "qualification_status",
        }
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise WorkstationError("Request body must be a JSON object")
            unknown = set(body) - mutable
            if unknown:
                raise WorkstationError(
                    "Expression source identity is immutable; unsupported fields: "
                    + ", ".join(sorted(unknown))
                )
            record = store.update_expression_draft(expression_id, **body)
        except (json.JSONDecodeError, WorkstationError) as error:
            status_code = 404 if "not found" in str(error).lower() else 400
            return _json_error(str(error), status_code)
        return JSONResponse(record)

    @app.post("/api/workstation/expression-drafts/{expression_id}/promote")
    async def promote_workstation_expression(
        expression_id: str, request: Request
    ) -> JSONResponse:
        store: WorkstationStore = request.app.state.workstation
        try:
            body = await request.json()
            if not isinstance(body, dict) or set(body) != {"qualification_id"}:
                raise WorkstationError("Promotion requires qualification_id")
            record = store.promote_expression(
                expression_id, qualification_id=body["qualification_id"]
            )
        except (json.JSONDecodeError, WorkstationError) as error:
            message = str(error)
            status_code = 404 if "not found" in message.lower() else 409
            return _json_error(message, status_code)
        return JSONResponse(record)

    @app.post(
        "/api/workstation/expression-drafts/{expression_id}/analyze-reference"
    )
    async def analyze_workstation_expression_reference(
        expression_id: str, request: Request
    ) -> JSONResponse:
        store: WorkstationStore = request.app.state.workstation
        try:
            body = await request.json()
            if not isinstance(body, dict) or body:
                raise WorkstationError(
                    "Reference analysis accepts an empty JSON object"
                )
            async with request.app.state.expression_analysis_lock:
                record = await asyncio.to_thread(
                    store.analyze_expression_reference, expression_id
                )
        except (json.JSONDecodeError, WorkstationError) as error:
            message = str(error)
            status_code = 404 if "not found" in message.lower() else 400
            return _json_error(message, status_code)
        return JSONResponse(
            {
                "expression": record,
                "analysis": record["prosody"].get("reference_analysis"),
            }
        )

    @app.delete("/api/workstation/expression-drafts/{expression_id}")
    async def delete_workstation_expression_draft(
        expression_id: str, request: Request
    ) -> JSONResponse:
        store: WorkstationStore = request.app.state.workstation
        try:
            record = store.delete_expression_draft(expression_id)
        except WorkstationError as error:
            status_code = 404 if "not found" in str(error).lower() else 400
            return _json_error(str(error), status_code)
        return JSONResponse({"deleted": True, "expression": record})

    def public_handoff_state(store: WorkstationStore) -> dict[str, Any]:
        record = store.get_runtime_handoff()
        if not record:
            return {"phase": "idle", "job_id": None, "message": None}
        messages = {
            "draining": "Finishing existing speech before the GPU workspace job",
            "stopping": "Releasing the GPU for the workspace job",
            "ready-for-job": "The GPU workspace job is preparing to start",
            "job-running": "The GPU is running a workspace job; synthesis will resume automatically",
            "restoring": "Restoring the previous voice model",
            "failed": "GPU recovery needs attention; inspect the workspace job logs",
        }
        phase = record.get("phase", "failed")
        return {
            "phase": phase, "job_id": record.get("job_id"),
            "message": messages.get(phase),
        }

    @app.get("/api/workstation/runtime-handoff")
    async def runtime_handoff_status() -> JSONResponse:
        return JSONResponse(
            public_handoff_state(app.state.workstation), headers={"Cache-Control": "no-store"}
        )

    @app.get("/api/status")
    async def status(request: Request) -> JSONResponse:
        handoff = public_handoff_state(request.app.state.workstation)
        if handoff["phase"] != "idle":
            return JSONResponse(
                {"error": handoff["message"], "runtime_handoff": handoff},
                status_code=503, headers={"Cache-Control": "no-store", "Retry-After": "5"},
            )
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
            return _json_error(f"{surface_name} is still finishing an active request", 409)
        try:
            response = await request.app.state.client.post(
                "/v1/models/activate", json={"model": model.strip()}
            )
            if response.status_code != 200:
                return _json_error(f"{surface_name} rejected the model switch", response.status_code)
            current = await verify_upstream(request.app.state.client)
            request.app.state.status = current
            return JSONResponse(current, headers={"Cache-Control": "no-store"})
        except httpx.HTTPError as error:
            LOGGER.warning("Model activation failed: %s", error)
            return _json_error(f"{surface_name} model activation connection failed", 502)
        finally:
            lock.release()

    async def upstream_error(response: httpx.Response) -> JSONResponse:
        body = await response.aread()
        try:
            detail: Any = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            detail = body.decode("utf-8", errors="replace")[:2000]
        return JSONResponse(
            {
                "error": f"{surface_name} rejected the request",
                "upstream_status": response.status_code,
                "detail": detail,
            },
            status_code=response.status_code,
            headers={"Cache-Control": "no-store"},
        )

    async def session_json_response(response: httpx.Response) -> JSONResponse:
        if not 200 <= response.status_code < 300:
            return await upstream_error(response)
        try:
            payload = response.json()
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return _json_error(f"{surface_name} returned invalid session metadata", 502)
        if not isinstance(payload, dict):
            return _json_error(f"{surface_name} returned invalid session metadata", 502)
        return JSONResponse(
            payload,
            status_code=response.status_code,
            headers={"Cache-Control": "no-store"},
        )

    @app.post("/api/sessions")
    async def create_webui_speech_session(request: Request) -> JSONResponse:
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise WebUIError("Request body must be a JSON object")
            unknown = set(body) - {"model", "voice_profile", "continuity_policy"}
            if unknown:
                raise WebUIError(
                    "Session request contains unsupported fields: "
                    + ", ".join(sorted(unknown))
                )
            model = body.get("model")
            if not isinstance(model, str) or not model.strip():
                raise WebUIError("model must not be empty")
            model = model.strip()
            known_models = {
                record.get("id")
                for record in request.app.state.status.get("models", [])
                if isinstance(record, Mapping)
            }
            if model not in known_models:
                raise WebUIError("model is not available locally")
            voice_profile = body.get("voice_profile", "default")
            if voice_profile != "default":
                raise WebUIError("voice_profile must be default")
            continuity_policy = body.get("continuity_policy", "A")
            if not isinstance(continuity_policy, str):
                raise WebUIError("continuity_policy must be one of A, B, C, D, E or F")
            continuity_policy = continuity_policy.strip().upper()
            if continuity_policy not in {"A", "B", "C", "D", "E", "F"}:
                raise WebUIError("continuity_policy must be one of A, B, C, D, E or F")
        except (json.JSONDecodeError, WebUIError) as error:
            return _json_error(str(error), 400)
        try:
            response = await request.app.state.client.post(
                "/v1/sessions",
                json={
                    "model": model,
                    "voice_profile": voice_profile,
                    "continuity_policy": continuity_policy,
                },
            )
            return await session_json_response(response)
        except httpx.HTTPError as error:
            LOGGER.warning("Speech session creation failed: %s", error)
            return _json_error(f"{surface_name} session connection failed", 502)

    @app.post("/api/sessions/{session_id}/segments")
    async def append_webui_speech_session_segment(
        session_id: str, request: Request
    ) -> JSONResponse:
        try:
            safe_session_id = _webui_session_id(session_id)
            body = await request.json()
            if not isinstance(body, dict):
                raise WebUIError("Request body must be a JSON object")
            allowed = {
                "segment_id",
                "text",
                "language",
                "expression_prompt",
                "generation",
                "paragraph_id",
                "pause_after_ms",
            }
            unknown = set(body) - allowed
            if unknown:
                raise WebUIError(
                    "Session segment contains unsupported fields: "
                    + ", ".join(sorted(unknown))
                )
            segment_id = _webui_session_id(body.get("segment_id"), "segment_id")
            text = body.get("text")
            if not isinstance(text, str) or not text.strip():
                raise WebUIError("text must not be empty")
            if len(text) > MAX_TEXT_CHARS:
                raise WebUIError(f"text is limited to {MAX_TEXT_CHARS} characters")
            language = body.get("language")
            if language not in LANGUAGES:
                raise WebUIError("language must be one of zh, yue, en, ja, ko")
            expression_prompt = body.get("expression_prompt")
            if expression_prompt is not None and not isinstance(expression_prompt, str):
                raise WebUIError("expression_prompt must be a string")
            paragraph_id = body.get("paragraph_id")
            if paragraph_id is not None:
                paragraph_id = _webui_session_id(paragraph_id, "paragraph_id")
            pause_after_ms = body.get("pause_after_ms", 0)
            if (
                isinstance(pause_after_ms, bool)
                or not isinstance(pause_after_ms, int)
                or not 0 <= pause_after_ms <= 10000
            ):
                raise WebUIError("pause_after_ms must be an integer between 0 and 10000")
            generation = _validated_generation(body.get("generation"))
            resolved = resolve_expression_prompt(
                expression_prompt,
                request.app.state.status.get("expressions", {}),
            )
        except (json.JSONDecodeError, WebUIError) as error:
            return _json_error(str(error), 400)

        payload: dict[str, Any] = {
            "segment_id": segment_id,
            "text": text,
            "language": language,
            "expression": resolved.upstream_payload(),
            "generation": generation,
            "pause_after_ms": pause_after_ms,
        }
        if paragraph_id is not None:
            payload["paragraph_id"] = paragraph_id
        try:
            response = await request.app.state.client.post(
                f"/v1/sessions/{safe_session_id}/segments",
                json=payload,
            )
            return await session_json_response(response)
        except httpx.HTTPError as error:
            LOGGER.warning("Speech session append failed: %s", error)
            return _json_error(f"{surface_name} session connection failed", 502)

    @app.post("/api/sessions/{session_id}/flush")
    async def flush_webui_speech_session(
        session_id: str, request: Request
    ) -> JSONResponse:
        try:
            safe_session_id = _webui_session_id(session_id)
            body = await request.json()
            if not isinstance(body, dict) or body:
                raise WebUIError("Flush request body must be an empty JSON object")
        except (json.JSONDecodeError, WebUIError) as error:
            return _json_error(str(error), 400)
        try:
            response = await request.app.state.client.post(
                f"/v1/sessions/{safe_session_id}/flush"
            )
            return await session_json_response(response)
        except httpx.HTTPError as error:
            LOGGER.warning("Speech session flush failed: %s", error)
            return _json_error(f"{surface_name} session connection failed", 502)

    @app.post("/api/sessions/{session_id}/cancel")
    async def cancel_webui_speech_session(
        session_id: str, request: Request
    ) -> JSONResponse:
        try:
            safe_session_id = _webui_session_id(session_id)
            body = await request.json()
            if not isinstance(body, dict) or body:
                raise WebUIError("Cancel request body must be an empty JSON object")
        except (json.JSONDecodeError, WebUIError) as error:
            return _json_error(str(error), 400)
        cancel_path = f"/v1/sessions/{safe_session_id}/cancel"
        try:
            response = await request.app.state.client.post(cancel_path)
            result = await session_json_response(response)
        except httpx.HTTPError as error:
            LOGGER.warning("Speech session cancellation failed: %s", error)
            return _json_error(f"{surface_name} session connection failed", 502)
        async with app.state.active_upstream_guard:
            active = app.state.active_upstream
            if active is not None and active.cancel_path == cancel_path:
                app.state.active_upstream = None
            else:
                active = None
        if active is not None:
            await active.aclose()
        return result

    @app.get("/api/sessions/{session_id}/audio")
    async def stream_webui_speech_session_audio(session_id: str, request: Request):
        try:
            safe_session_id = _webui_session_id(session_id)
        except WebUIError as error:
            return _json_error(str(error), 400)
        lock: asyncio.Lock = request.app.state.speech_lock
        if lock.locked():
            await cancel_active_upstream()
        try:
            await asyncio.wait_for(lock.acquire(), timeout=0.75)
        except TimeoutError:
            return _json_error(f"{surface_name} is still cancelling the previous request", 409)

        started = time.perf_counter()
        try:
            upstream_response = await request.app.state.client.send(
                request.app.state.client.build_request(
                    "GET", f"/v1/sessions/{safe_session_id}/audio"
                ),
                stream=True,
            )
        except asyncio.CancelledError:
            lock.release()
            raise
        except httpx.HTTPError as error:
            lock.release()
            LOGGER.warning("Speech session audio request failed: %s", error)
            return _json_error(f"{surface_name} session connection failed", 502)
        owner = _UpstreamSpeechOwner(
            app=app,
            response=upstream_response,
            speech_lock=lock,
            cancel_path=f"/v1/sessions/{safe_session_id}/cancel",
        )
        if upstream_response.status_code != 200:
            try:
                return await upstream_error(upstream_response)
            finally:
                await owner.aclose()

        expected_model = request.app.state.status.get("health", {}).get("model")
        required = {
            "x-tensorrt-backend": EXPECTED_BACKEND,
            "x-tensorrt-engine-count": str(EXPECTED_ENGINE_COUNT),
            "x-pytorch-fallback": "false",
            "x-tts-model": expected_model,
            "x-tts-stream": "pcm_s16le",
            "x-tts-sample-rate": str(EXPECTED_SAMPLE_RATE),
            "x-tts-channels": "1",
            "x-tts-session-id": safe_session_id,
            "x-tts-session-context": "committed-neural-v1",
            "x-tts-acoustic-latent-continuity": "false",
            "x-tts-continuity-qualification": "experimental-unqualified",
        }
        mismatches = [
            f"{name}={upstream_response.headers.get(name)!r}"
            for name, expected in required.items()
            if not isinstance(expected, str)
            or upstream_response.headers.get(name) != expected
        ]
        if mismatches:
            await owner.aclose()
            return JSONResponse(
                {
                    "error": f"{surface_name} session stream headers failed validation",
                    "detail": mismatches,
                },
                status_code=502,
            )
        context_policy = upstream_response.headers.get(
            "x-tts-session-context-policy"
        )
        neural_continuity = upstream_response.headers.get(
            "x-tts-neural-state-continuity"
        )
        if context_policy not in {"A", "B", "C", "D", "E", "F"} or (
            neural_continuity != str(context_policy != "A").lower()
        ):
            await owner.aclose()
            return _json_error(
                f"{surface_name} returned inconsistent session continuity headers",
                502,
            )
        try:
            async with app.state.active_upstream_guard:
                app.state.active_upstream = owner
        except BaseException:
            await owner.aclose()
            raise

        stream = upstream_response.aiter_bytes()
        try:
            first = await stream.__anext__()
            while not first:
                first = await stream.__anext__()
        except asyncio.CancelledError:
            await owner.aclose()
            raise
        except (httpx.HTTPError, StopAsyncIteration):
            await owner.aclose()
            return _json_error(f"{surface_name} session ended before first PCM audio", 502)

        upstream_ttfa_ms = (time.perf_counter() - started) * 1000.0

        async def session_chunks() -> AsyncIterator[bytes]:
            try:
                yield first
                async for chunk in stream:
                    if chunk:
                        yield chunk
            finally:
                await owner.aclose()

        headers = {name: upstream_response.headers[name] for name in required}
        headers.update(
            {
                "X-TTS-Recommended-Prebuffer-Ms": "32",
                "X-TTS-Sample-Format": "s16le",
                "X-Upstream-TTFA-Ms": f"{upstream_ttfa_ms:.3f}",
                "X-TTS-Session-Context-Policy": context_policy,
                "X-TTS-Neural-State-Continuity": neural_continuity,
                "Cache-Control": "no-store",
                "X-Accel-Buffering": "no",
            }
        )
        return _OwnedStreamingResponse(
            session_chunks(),
            owner=owner,
            media_type="application/octet-stream",
            headers=headers,
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
            generation = _validated_generation(body.get("generation"))
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
            generation=generation,
        )
        lock: asyncio.Lock = request.app.state.speech_lock
        if lock.locked():
            await cancel_active_upstream()
        try:
            await asyncio.wait_for(lock.acquire(), timeout=0.75)
        except TimeoutError:
            return _json_error(f"{surface_name} is still cancelling the previous request", 409)

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
                        f"{surface_name} is still cancelling the previous request",
                        409,
                    )
                await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            lock.release()
            raise
        except httpx.HTTPError as error:
            lock.release()
            LOGGER.warning("Speech request failed: %s", error)
            return _json_error(f"{surface_name} upstream connection failed", 502)
        owner = _UpstreamSpeechOwner(
            app=app,
            response=upstream_response,
            speech_lock=lock,
        )
        if upstream_response.status_code != 200:
            try:
                return await upstream_error(upstream_response)
            finally:
                # aread(), response parsing, request cancellation, and aclose()
                # failures must all converge on the same idempotent owner cleanup.
                await owner.aclose()

        stream = upstream_response.aiter_bytes()
        try:
            first = await stream.__anext__()
            while not first:
                first = await stream.__anext__()
        except asyncio.CancelledError:
            await owner.aclose()
            raise
        except (httpx.HTTPError, StopAsyncIteration):
            await owner.aclose()
            return _json_error(f"{surface_name} stream ended before first PCM audio", 502)

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
            await owner.aclose()
            return JSONResponse(
                {"error": f"{surface_name} stream headers failed validation", "detail": mismatches},
                status_code=502,
            )
        try:
            prebuffer = int(upstream_response.headers["x-tts-recommended-prebuffer-ms"])
        except (KeyError, TypeError, ValueError):
            prebuffer = -1
        if not 0 <= prebuffer <= 250:
            await owner.aclose()
            return _json_error(f"{surface_name} returned an invalid prebuffer recommendation", 502)

        try:
            async with app.state.active_upstream_guard:
                app.state.active_upstream = owner
        except BaseException:
            await owner.aclose()
            raise

        async def chunks() -> AsyncIterator[bytes]:
            try:
                yield first
                async for chunk in stream:
                    if chunk:
                        yield chunk
            finally:
                await owner.aclose()

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
        return _OwnedStreamingResponse(
            chunks(),
            owner=owner,
            media_type="application/octet-stream",
            headers=headers,
        )

    return app


def validate_webui_bind_host(host: str, *, allow_non_loopback: bool = False) -> str:
    """Return a normalized bind host after enforcing the local-only default."""

    if not isinstance(host, str) or not host.strip():
        raise WebUIError("WebUI bind host must not be empty")
    normalized = host.strip()
    if allow_non_loopback:
        return normalized
    if normalized.lower() == "localhost":
        return normalized
    try:
        address = ipaddress.ip_address(normalized)
    except ValueError as error:
        raise WebUIError(
            "WebUI binds to loopback only by default; use --allow-non-loopback "
            "only behind a trusted access layer"
        ) from error
    if not address.is_loopback:
        raise WebUIError(
            "WebUI binds to loopback only by default; use --allow-non-loopback "
            "only behind a trusted access layer"
        )
    return normalized


def run_webui(
    *,
    host: str,
    port: int,
    upstream: str,
    allow_non_loopback: bool = False,
    surface: Literal["studio", "classic"] = "classic",
) -> None:
    import uvicorn

    bind_host = validate_webui_bind_host(
        host, allow_non_loopback=allow_non_loopback
    )
    uvicorn.run(
        create_webui_app(upstream=upstream, default_surface=surface),
        host=bind_host,
        port=port,
        workers=1,
        access_log=False,
    )
