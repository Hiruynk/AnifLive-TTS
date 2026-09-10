from __future__ import annotations

from html.parser import HTMLParser
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SYNTHESIS = ROOT / "webui" / "synthesis.html"


class _InlineScriptParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.scripts: list[str] = []
        self._inline = False
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "script" and not dict(attrs).get("src"):
            self._inline = True
            self._parts = []

    def handle_data(self, data: str) -> None:
        if self._inline:
            self._parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self._inline:
            self.scripts.append("".join(self._parts))
            self._inline = False
            self._parts = []


def _html() -> str:
    return SYNTHESIS.read_text(encoding="utf-8")


def _run_controls_javascript(expression: str) -> object:
    node = os.environ.get("NODE_BINARY") or shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    html = _html()
    start = html.index("// SYNTHESIS_CONTROLS_START")
    end = html.index("// SYNTHESIS_CONTROLS_END")
    controls = html[start:end]
    script = controls + f"\nprocess.stdout.write(JSON.stringify({expression}));"
    result = subprocess.run(
        [node, "-e", script],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return json.loads(result.stdout)


def test_synthesis_controls_build_plain_and_segmented_api_payloads() -> None:
    result = _run_controls_javascript(
        "(() => {"
        "const api=globalThis.AnifLiveTTSSynthesisControls;"
        "const generation=api.normalizeGenerationSettings({"
        "seed:'42',top_k:'7',temperature:'0.85',noise_scale:'0.35'});"
        "return {generation,plain:api.buildSpeechRequest({"
        "text:'hello',segments:null,language:'en',model:'roxy',generation}),"
        "segmented:api.buildSpeechRequest({text:'',segments:[{text:'hi'}],"
        "language:'en',model:'roxy',generation})};})()"
    )

    generation = {
        "seed": 42,
        "top_k": 7,
        "temperature": 0.85,
        "noise_scale": 0.35,
        "speed": 1,
    }
    assert result["generation"] == generation
    assert result["plain"] == {
        "text": "hello",
        "language": "en",
        "model": "roxy",
        "generation": generation,
    }
    assert result["segmented"] == {
        "segments": [{"text": "hi"}],
        "language": "en",
        "model": "roxy",
        "generation": generation,
    }


def test_synthesis_sequence_move_and_blind_order_are_deterministic() -> None:
    result = _run_controls_javascript(
        "(() => {"
        "const api=globalThis.AnifLiveTTSSynthesisControls;"
        "return {moved:api.moveSequenceItem(['one','two','three'],0,2),"
        "unchanged:api.moveSequenceItem(['one','two'],-1,1),"
        "even:api.blindVariantOrder(4),odd:api.blindVariantOrder(5)};})()"
    )

    assert result == {
        "moved": ["two", "three", "one"],
        "unchanged": ["one", "two"],
        "even": ["A", "B"],
        "odd": ["B", "A"],
    }


def test_synthesis_session_plan_preserves_text_order_and_segment_controls() -> None:
    result = _run_controls_javascript(
        "(() => {"
        "const api=globalThis.AnifLiveTTSSynthesisControls;"
        "return api.buildSessionPlan({defaultLanguage:'ja',generation:{seed:9},"
        "records:["
        "{key:'a',text:'Hello, ',expression_prompt:null},"
        "{key:'b',text:'world.',expression_prompt:'shy'}],"
        "settings:{a:{language:'en',pause_after_ms:120},"
        "b:{language:'yue',pause_after_ms:0}}});})()"
    )

    assert result["text"] == "Hello, world."
    assert [segment["text"] for segment in result["segments"]] == ["Hello, ", "world."]
    assert [segment["language"] for segment in result["segments"]] == ["en", "yue"]
    assert [segment["pause_after_ms"] for segment in result["segments"]] == [120, 0]
    assert [segment["segment_id"] for segment in result["segments"]] == [
        "seg_01",
        "seg_02",
    ]
    assert result["segments"][1]["expression_prompt"] == "shy"
    assert all(segment["generation"]["speed"] == 1 for segment in result["segments"])


def test_synthesis_inline_javascript_is_valid() -> None:
    node = os.environ.get("NODE_BINARY") or shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    parser = _InlineScriptParser()
    parser.feed(_html())

    assert parser.scripts
    for script in parser.scripts:
        subprocess.run(
            [node, "--check", "-"],
            input=script,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )


@pytest.mark.parametrize(
    "settings",
    [
        {"seed": 1.5, "top_k": 15, "temperature": 1, "noise_scale": 0.5},
        {"seed": 1, "top_k": 0, "temperature": 1, "noise_scale": 0.5},
        {"seed": 1, "top_k": 15, "temperature": 0, "noise_scale": 0.5},
        {"seed": 1, "top_k": 15, "temperature": 1, "noise_scale": 10.1},
    ],
)
def test_synthesis_controls_reject_invalid_generation_values(
    settings: dict[str, float],
) -> None:
    encoded = json.dumps(settings, separators=(",", ":"))
    assert _run_controls_javascript(
        "(() => {try {"
        f"globalThis.AnifLiveTTSSynthesisControls.normalizeGenerationSettings({encoded});"
        "return false;} catch (_) {return true;}})()"
    ) is True


def test_synthesis_waveform_uses_streamed_pcm_and_audio_clock() -> None:
    html = _html()

    assert 'id="pcmWaveform"' in html
    assert "const WAVEFORM_BUCKET_SAMPLES = 256" in html
    assert "view.getInt16(index * 2, true) / 32768" in html
    assert "waveform.peaks.push([waveform.bucketMin, waveform.bucketMax])" in html
    assert "appendPcmWaveform(savedChunk)" in html
    assert "updateWaveformPlayback(pcmElapsed, pcmElapsed >= 0)" in html
    assert "state.playback.firstScheduledContextTime = timing.scheduledStart" in html
    assert "playback.context.currentTime - outputLatency" in html
    assert "Math.random" not in html


def test_synthesis_inspector_exposes_real_controls_and_fixed_speed_contract() -> None:
    html = _html()

    for control_id in (
        "generationSeed",
        "generationTopK",
        "generationTemperature",
        "generationNoiseScale",
    ):
        assert f'id="{control_id}"' in html
    assert 'id="generationTopK" type="number" min="1" max="50"' in html
    assert '<strong>1.0×</strong>' in html
    assert 'data-i18n="generationSpeedFixed"' in html
    assert 'id="generationSpeed"' not in html
    assert "const sessionRequest = await requestSessionStream(" in html
    assert "sessionPlan," in html
    assert '"/api/sessions"' in html
    assert "for (const segment of plan.segments)" in html
    assert html.index("const producer = (async () =>") < html.index(
        "const response = await audioPromise"
    )
    assert "generation: { ...generation, speed: 1.0 }" in html
    assert "generation: { ...generation }," in html
    assert 'data-i18n="historyGeneration"' in html
    assert "@media (max-width: 380px)" in html


def test_synthesis_ab_compare_captures_two_real_pcm_responses_and_replays_them() -> None:
    html = _html()

    for control_id in (
        "abGenerate",
        "abSampleOne",
        "abSampleTwo",
        "abReveal",
        "abSeed",
        "abTopK",
        "abTemperature",
        "abNoiseScale",
    ):
        assert f'id="{control_id}"' in html
    assert 'for (const [variant, generation] of [["A", generationA], ["B", generationB]])' in html
    assert "const capture = await captureSpeechVariant(plan, state.model, controller)" in html
    assert "const sessionRequest = await requestSessionStream(plan, model, controller)" in html
    assert "await sessionRequest.finish()" in html
    assert "chunks.push(saved)" in html
    assert "bytes: concatPcmChunks(chunks, pcmBytes)" in html
    assert "state.ab.order = blindVariantOrder(random[0])" in html
    assert "crypto.getRandomValues(random)" in html
    assert "const sample = state.ab.samples?.[variant]" in html
    assert "const timing = schedulePcm(context, sample.bytes" in html
    assert "sample.chunks" in html


def test_synthesis_segment_order_language_and_pause_are_real_session_controls() -> None:
    html = _html()

    assert 'id="segmentSequence"' in html
    assert "row.draggable = records.length > 1 && !locked" in html
    assert "const reordered = moveSequenceItem(records, fromIndex, toIndex)" in html
    assert "record.annotation.start = start" in html
    assert "record.annotation.end = cursor" in html
    assert "AnifLiveTTSExpressionBoundaries.hasSafeExpressionBoundary" in html
    assert 'el("text").value = text' in html
    assert 'el("text").dispatchEvent(new Event("input", { bubbles: true }))' in html
    assert 'segmentLanguage: "Language"' in html
    assert 'segmentPause: "Pause after · ms"' in html
    assert 'language.value = setting.language || ""' in html
    assert 'pause.min = "0"' in html
    assert 'pause.max = "10000"' in html
    assert "setting.pauseAfterMs = normalizePauseMilliseconds(pause.value)" in html
    assert "state.segmentSettings.set(record.key, setting)" in html
    assert "const plan = buildCurrentSessionPlan(generation)" in html
    assert "await postSessionJson(`${basePath}/segments`, segment, controller.signal)" in html
    assert "await postSessionJson(`${basePath}/flush`, {}, controller.signal)" in html
    assert "activeSessionId" in html
    assert "X-TTS-Neural-State-Continuity" not in html
