#!/usr/bin/env python3
"""Minimal final functional acceptance for a native macOS service (one process).

Runs against a shared-base service whose voice library already holds a voice
with several emotion references. Checks that health/capabilities/model-config
and the voice expression catalog agree with the loaded worker, A->B->A
reference selection without cross-contamination, one mixed conditioning
policy, one segmented expression request, one non-default request sampling,
one complete WAV and one 48 kHz mono PCM16 stream, Session A
append/flush/play/cancel/recovery, in-flight complete-WAV cancellation (409)
and recovery, and one sync plus one async SDK call. Every conditioning and
sampling claim is read back from /health last_request_conditioning (what the
worker reported), not assumed. Audio is kept locally; no listening verdict.
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import threading
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path

from aniflive_tts.sdk import AsyncClient, Client, Expression, ExpressionSegment, Generation

SEED = 20261112
TEXT = "今日はいい天気ですね。"
LONG = ("昔々、ある山の奥に小さな村がありました。村の人々は毎朝早く起きて、畑を耕し、川で水を汲み、"
        "夕方には皆で集まって一日の出来事を話し合いました。ある日、旅の商人が村を訪れ、遠い町の不思議な話を"
        "たくさん聞かせてくれました。子供たちは目を輝かせて、その話に夢中になりました。")


class Run:
    def __init__(self, base: str, out: Path):
        self.base, self.out, self.records = base, out, []

    def check(self, name: str, ok: bool, **detail) -> None:
        self.records.append({"check": name, "passed": bool(ok), **detail})
        print(("PASS " if ok else "FAIL ") + name, json.dumps(detail, ensure_ascii=False, default=str)[:400], flush=True)

    def get(self, path: str) -> dict:
        with urllib.request.urlopen(self.base + path, timeout=60) as r:
            return json.load(r)

    def post(self, path: str, body: dict, timeout: float = 600) -> tuple[int, bytes, dict]:
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, r.read(), {k.lower(): v for k, v in r.headers.items()}
        except urllib.error.HTTPError as e:
            return e.code, e.read(), {k.lower(): v for k, v in e.headers.items()}

    def last(self) -> list[dict]:
        return self.get("/health").get("last_request_conditioning") or []

    def save(self, name: str, data: bytes) -> None:
        (self.out / "audio").mkdir(parents=True, exist_ok=True)
        (self.out / "audio" / name).write_bytes(data)


def pcm(wav_bytes: bytes) -> bytes:
    with wave.open(io.BytesIO(wav_bytes)) as w:
        return w.readframes(w.getnframes())


def wav_format(wav_bytes: bytes) -> tuple[int, int, int]:
    with wave.open(io.BytesIO(wav_bytes)) as w:
        return w.getframerate(), w.getnchannels(), w.getsampwidth()


def keys(item: dict) -> tuple[str, str, str]:
    c = item["worker_conditioning"]
    return c["semantic_reference_key"], c["style_reference_key"], c["acoustic_prompt_reference_key"]


def contract(run: Run, voice: str) -> None:
    health, caps, config = run.get("/health"), run.get("/v1/capabilities"), run.get("/model/config")
    catalog = run.get(f"/v1/expressions?voice_profile={voice}")
    models = run.get("/v1/models")["data"]
    active = [m["id"] for m in models if m["active"]]
    precision = {health.get("precision"), caps.get("precision"), config.get("precision")}
    run.check("contract: health, capabilities and model-config name one active model and precision",
              len(precision) == 1 and active == [health["model"]] == [config["model"]],
              precision=sorted(p for p in precision if p), model=health["model"], active=active)
    worker = caps.get("native_worker") or {}
    expression = caps.get("expression") or {}
    run.check("contract: the loaded worker reports separate conditioning components and request sampling",
              worker.get("separate_conditioning_components") is True and worker.get("request_sampling") is True
              and caps.get("sampling", {}).get("request_sampling") is True
              and len(expression.get("policies", [])) == 5 and expression.get("complete_policies") == expression.get("policies"),
              native_worker=worker, policies=expression.get("policies"))
    sessions = caps.get("speech_sessions") or {}
    run.check("contract: Session A advertised as committed-neural-v1 / qualified-policy-a, B-F not offered",
              sessions.get("context_mode") == "committed-neural-v1" and sessions.get("qualification") == "qualified-policy-a"
              and list(sessions.get("policies", {})) == ["A"], sessions={k: sessions.get(k) for k in ("context_mode", "qualification", "policies")})
    run.check("contract: product flags stay scoped (not prototype; overall quality not claimed qualified)",
              caps.get("prototype") is False and health.get("prototype") is False and config.get("prototype") is False
              and caps.get("quality_qualified") is False and config.get("quality_qualified") is False
              and caps.get("qualification", {}).get("identity_style_equivalence") == "passed_by_explicit_owner_acceptance",
              qualification_scope=caps.get("quality_qualification_scope"))
    refs = {r["id"] for p in catalog.get("profiles", []) for r in p.get("references", [])}
    run.check("contract: /v1/expressions lists the voice's references and only backend-supported policies",
              catalog.get("enabled") is True and {"calm", "happy10", "sad05"} <= refs
              and catalog.get("policies") == expression.get("policies")
              and catalog.get("preferred_policy") in catalog.get("policies", []),
              references=sorted(refs), default=catalog.get("default"), preferred_policy=catalog.get("preferred_policy"),
              policies=catalog.get("policies"))


def selection(run: Run, client: Client, voice: str) -> None:
    kw = {"voice_profile": voice, "language": "ja"}
    neutral = client.synthesize(text=TEXT, generation=Generation(seed=SEED), **kw).audio
    identity = keys(run.last()[0])
    run.save("neutral.wav", neutral)
    outputs, ks = [], []
    for tag, profile, intensity in (("A", "happy", 1.0), ("B", "sad", 0.5), ("A2", "happy", 1.0)):
        audio = client.synthesize(text=TEXT, generation=Generation(seed=SEED),
                                  expression=Expression(enabled=True, profile=profile, intensity=intensity, policy="full-switch"), **kw).audio
        item = run.last()[0]
        outputs.append(audio)
        ks.append((item["expression_id"], keys(item)))
        run.save(f"select-{tag}.wav", audio)
    (a_id, a_keys), (b_id, b_keys), (a2_id, a2_keys) = ks
    run.check("selection: A->B->A returns to A's conditioning and output without B mixed in",
              a_id == a2_id == "happy10" and b_id == "sad05" and a_keys == a2_keys and not (set(a_keys) & set(b_keys))
              and outputs[0] == outputs[2] and outputs[0] != outputs[1], ids=[a_id, b_id, a2_id])
    mixed = client.synthesize(text=TEXT, generation=Generation(seed=SEED),
                              expression=Expression(enabled=True, profile="happy", intensity=1.0, policy="semantic-style"), **kw).audio
    item = run.last()[0]
    run.save("mixed-semantic-style.wav", mixed)
    run.check("selection: semantic-style takes semantic from the emotion reference, style/acoustic prompt from the identity",
              item["policy"] == "semantic-style" and keys(item) == (a_keys[0], identity[1], identity[2]),
              composed=keys(item))
    segs = [ExpressionSegment(text="今日はいい天気ですね。", expression=Expression(enabled=False)),
            ExpressionSegment(text="一緒に散歩に行きましょう！"),
            ExpressionSegment(text="でも、雨が降りそうです。", expression=Expression(enabled=True, profile="sad", intensity=0.5))]
    seg = client.synthesize(segments=segs, generation=Generation(seed=SEED),
                            expression=Expression(enabled=True, profile="happy", intensity=1.0, policy="full-switch"), **kw).audio
    items = run.last()
    run.save("segmented.wav", seg)
    run.check("selection: segmented request keeps order, inherits and overrides expressions",
              [x["expression_id"] for x in items] == ["calm", "happy10", "sad05"] and [x["segment_index"] for x in items] == [0, 1, 2],
              ids=[x["expression_id"] for x in items])
    again = client.synthesize(text=TEXT, generation=Generation(seed=SEED), **kw).audio
    run.check("selection: a later neutral request is unchanged", again == neutral)


def sampling_and_formats(run: Run, client: Client, voice: str) -> None:
    kw = {"voice_profile": voice, "language": "ja"}
    default = client.synthesize(text=TEXT, generation=Generation(seed=SEED), **kw).audio
    gen = Generation(seed=SEED, top_k=20, top_p=0.9, temperature=0.8)
    first = client.synthesize(text=TEXT, generation=gen, **kw).audio
    reported = run.last()[0]["sampling"]
    second = client.synthesize(text=TEXT, generation=gen, **kw).audio
    run.save("sampling-k20-p09-t08.wav", first)
    run.check("sampling: worker reports and applies top_k=20, top_p=0.9, temperature=0.8 deterministically",
              reported["top_k"] == 20 and abs(reported["top_p"] - 0.9) < 1e-6 and abs(reported["temperature"] - 0.8) < 1e-6
              and first == second and pcm(first) != pcm(default), reported=reported)
    status, body, headers = run.post("/v1/audio/speech", {"text": TEXT, "language": "ja", "voice_profile": voice,
                                                          "generation": {"seed": SEED}})
    run.check("format: complete WAV is 48 kHz mono PCM16", status == 200 and body[:4] == b"RIFF" and wav_format(body) == (48000, 1, 2),
              status=status, format=wav_format(body) if body[:4] == b"RIFF" else None)
    status, stream, headers = run.post("/v1/audio/speech", {"text": TEXT, "language": "ja", "voice_profile": voice,
                                                            "generation": {"seed": SEED}, "stream": True})
    run.save("stream.wav", wav_bytes(stream))
    run.check("format: PCM stream is s16le 48 kHz mono with the same budget as the complete WAV (onset trim only)",
              status == 200 and headers.get("x-tts-stream") == "pcm_s16le" and headers.get("x-tts-sample-rate") == "48000"
              and headers.get("x-tts-channels") == "1" and 0 <= (len(pcm(body)) - len(stream)) // 2 <= 2880,
              status=status, samples=len(stream) // 2, complete_samples=len(pcm(body)) // 2)


def wav_bytes(data: bytes) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(48000)
        w.writeframes(data)
    return out.getvalue()


def session_a(run: Run, client: Client, voice: str) -> None:
    def run_session() -> tuple[bytes, object]:
        s = client.create_session(voice_profile=voice)
        s.append(segment_id="s1", text=TEXT, language="ja", pause_after_ms=300, generation=Generation(seed=SEED))
        s.append(segment_id="s2", text="一緒に散歩に行きましょう！", language="ja", generation=Generation(seed=SEED),
                 expression=Expression(enabled=True, profile="happy", intensity=1.0, policy="full-switch"))
        s.flush()
        return b"".join(s.audio()), s
    data, session = run_session()
    info = session.get()
    run.save("session-a.wav", wav_bytes(data))
    run.check("session: policy A append/flush/play completes both segments", info.continuity_policy == "A" and len(data) > 48000
              and [s.state for s in info.segments] == ["completed", "completed"], states=[s.state for s in info.segments])
    raw = client.create_session(voice_profile=voice)
    raw.append(segment_id="h", text=TEXT, language="ja", generation=Generation(seed=SEED))
    raw.flush()
    with urllib.request.urlopen(f"{run.base}/v1/sessions/{raw.initial_info.id}/audio", timeout=600) as r:
        headers = {k.lower(): v for k, v in r.headers.items()}
        r.read()
    run.check("session: audio headers match the shared WebUI contract",
              headers.get("x-tts-session-context") == "committed-neural-v1" and headers.get("x-tts-session-context-policy") == "A"
              and headers.get("x-tts-continuity-qualification") == "qualified-policy-a",
              headers={k: headers.get(k) for k in ("x-tts-session-context", "x-tts-session-context-policy", "x-tts-continuity-qualification")})
    s2 = client.create_session(voice_profile=voice)
    s2.append(segment_id="a", text="これはキャンセルのテストです。長めの文章を読み上げます。", language="ja", generation=Generation(seed=SEED))
    s2.append(segment_id="b", text="この部分は読まれません。", language="ja")
    s2.flush()
    audio = iter(s2.audio())
    first = next(audio)
    s2.cancel()
    try:
        for _ in audio:
            pass
    except Exception:  # noqa: BLE001 - the stream may end with a cancellation error
        pass
    final = s2.get()
    run.check("session: cancel after first PCM stops the session", bool(first) and str(final.state).endswith("cancelled"), state=final.state)
    again, _ = run_session()
    run.check("session: a new session after cancel recovers exactly", again == data)


def complete_cancel(run: Run, client: Client, voice: str) -> None:
    base = client.synthesize(text=TEXT, language="ja", voice_profile=voice, generation=Generation(seed=SEED)).audio
    result: dict = {}

    def single() -> None:
        status, body, _ = run.post("/v1/audio/speech", {"text": LONG, "language": "ja", "voice_profile": voice,
                                                        "generation": {"seed": SEED}})
        result.update(status=status, body=body)
    thread = threading.Thread(target=single)
    thread.start()
    time.sleep(0.6)
    cancelled = client.cancel()
    thread.join(600)
    body = result.get("body", b"")
    run.check("cancel: in-flight complete WAV returns 409 with no audio", cancelled.cancelled is True and result.get("status") == 409
              and body[:4] != b"RIFF", status=result.get("status"), message=body[:160].decode("utf-8", "replace"))
    after = client.synthesize(text=TEXT, language="ja", voice_profile=voice, generation=Generation(seed=SEED)).audio
    run.check("cancel: the next synthesis recovers exactly", after == base)


def async_sdk(run: Run, voice: str) -> None:
    async def calls() -> tuple[bytes, bytes]:
        async with AsyncClient(run.base, request_timeout=600, stream_idle_timeout=240) as client:
            r = await client.synthesize(text=TEXT, language="ja", voice_profile=voice, generation=Generation(seed=SEED))
            s = await client.stream(text=TEXT, language="ja", voice_profile=voice, generation=Generation(seed=SEED))
            return r.audio, b"".join([c async for c in s])
    audio, stream = asyncio.run(calls())
    run.check("sdk: async complete equals the sync result; async stream produced PCM",
              audio == (run.out / "audio" / "neutral.wav").read_bytes() and len(stream) > 48000, stream_samples=len(stream) // 2)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", required=True)
    p.add_argument("--voice", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    run = Run(a.base_url, a.output)
    try:
        contract(run, a.voice)
        with Client(a.base_url, request_timeout=600, stream_idle_timeout=240) as client:
            selection(run, client, a.voice)
            sampling_and_formats(run, client, a.voice)
            session_a(run, client, a.voice)
            complete_cancel(run, client, a.voice)
        async_sdk(run, a.voice)
    except Exception as error:  # noqa: BLE001
        run.check("suite completed without exception", False, error=f"{type(error).__name__}: {error}"[:500])
    failed = [r["check"] for r in run.records if not r["passed"]]
    health = run.get("/health")
    report = {"kind": "macos_final_functional_acceptance", "seed": SEED, "model": health.get("model"),
              "precision": health.get("precision"), "engine_fingerprint": health.get("engine_fingerprint"),
              "passed": not failed, "failed": failed, "checks": run.records,
              "listening": "audio kept locally; no listening verdict is claimed"}
    (a.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n")
    print(json.dumps({"passed": not failed, "failed": failed}, ensure_ascii=False))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
