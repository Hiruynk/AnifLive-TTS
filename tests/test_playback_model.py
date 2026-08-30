from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from aniflive_tts.playback_model import (
    PlaybackChunk,
    PlaybackTrace,
    analyze_playback_trace,
    simulate_playback,
)


ROOT = Path(__file__).parents[1]


def _trace(*, prebuffer: int = 32) -> PlaybackTrace:
    return PlaybackTrace(
        sample_rate=32000,
        recommended_prebuffer_ms=prebuffer,
        chunks=(
            PlaybackChunk(arrival_seconds=0.050, samples=1600),
            PlaybackChunk(arrival_seconds=0.120, samples=1600),
            PlaybackChunk(arrival_seconds=0.170, samples=1600),
        ),
    )


def test_contractual_prebuffer_distinguishes_network_gap_from_underrun() -> None:
    trace = _trace()

    assert simulate_playback(trace, prebuffer_ms=0)["underrun_count"] == 1
    assert simulate_playback(trace)["underrun_count"] == 0


def test_prebuffer_sweep_reports_minimum_stable_value() -> None:
    result = analyze_playback_trace(_trace())

    assert result["minimum_stable_prebuffer_ms"] == 32
    assert result["zero_prebuffer_gap_stress"]["underrun_samples"] == 640
    assert result["contractual_playback"]["underrun_count"] == 0


def test_final_end_of_stream_is_not_an_underrun() -> None:
    trace = PlaybackTrace(
        sample_rate=32000,
        recommended_prebuffer_ms=0,
        chunks=(PlaybackChunk(arrival_seconds=0.010, samples=320),),
    )

    assert simulate_playback(trace)["underrun_count"] == 0


def test_invalid_trace_is_rejected() -> None:
    with pytest.raises(ValueError, match="monotonic"):
        simulate_playback(
            PlaybackTrace(
                sample_rate=32000,
                recommended_prebuffer_ms=32,
                chunks=(
                    PlaybackChunk(arrival_seconds=0.2, samples=320),
                    PlaybackChunk(arrival_seconds=0.1, samples=320),
                ),
            )
        )


def test_browser_reference_model_matches_python() -> None:
    node = os.environ.get("NODE_BINARY") or shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    trace = _trace()
    expected = simulate_playback(trace)
    js_trace = {
        "sampleRate": trace.sample_rate,
        "recommendedPrebufferMs": trace.recommended_prebuffer_ms,
        "chunks": [
            {"arrivalSeconds": item.arrival_seconds, "samples": item.samples}
            for item in trace.chunks
        ],
    }
    script = (
        "const model=require(process.argv[1]);"
        "const trace=JSON.parse(process.argv[2]);"
        "process.stdout.write(JSON.stringify(model.simulatePlaybackTrace(trace)));"
    )
    completed = subprocess.run(
        [
            node,
            "-e",
            script,
            str(ROOT / "webui" / "playback_model.js"),
            json.dumps(js_trace),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    actual = json.loads(completed.stdout)

    assert actual["underrunCount"] == expected["underrun_count"]
    assert actual["underrunSamples"] == expected["underrun_samples"]
    assert actual["largestUnderrunSamples"] == expected["largest_underrun_samples"]


def test_browser_text_timeline_tracks_unicode_graphemes_without_cumulative_highlight() -> None:
    node = os.environ.get("NODE_BINARY") or shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    script = (
        "const model=require(process.argv[1]);"
        "const timeline=model.buildTextTimeline('即係呢，Hello。','yue');"
        "const first=model.activeTextRange(timeline,0);"
        "const finished=model.activeTextRange(timeline,1);"
        "process.stdout.write(JSON.stringify({timeline,first,finished}));"
    )
    completed = subprocess.run(
        [node, "-e", script, str(ROOT / "webui" / "playback_model.js")],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    result = json.loads(completed.stdout)

    assert result["first"] == {"start": 0, "end": 1}
    assert result["finished"] is None
    assert result["timeline"]["text"] == "即係呢，Hello。"
    assert result["timeline"]["estimatedDurationSeconds"] > 0.6


def test_browser_audio_timeline_pauses_and_uses_acoustic_motion() -> None:
    node = os.environ.get("NODE_BINARY") or shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    script = r"""
      const model=require(process.argv[1]);
      const rate=1000;
      const values=[];
      const add=(count,fn)=>{for(let i=0;i<count;i+=1) values.push(fn(i));};
      add(100,i=>Math.round(5000*Math.sin(i*.25)));
      add(120,_=>0);
      add(100,i=>Math.round(14000*Math.sin(i*1.6)));
      const bytes=new Uint8Array(values.length*2);
      const view=new DataView(bytes.buffer);
      values.forEach((value,index)=>view.setInt16(index*2,value,true));
      const timeline=model.buildTextTimeline('甲乙，丙丁。','yue');
      const detected=model.buildPcmActivityTimeline([bytes],rate,{
        frameMs:10,thresholdDbfs:-45,maximumBridgeMs:30
      });
      const acoustic=model.alignTextTimelineToAudio(timeline,detected);
      const early=model.activeTextRangeAtAudioTime(timeline,acoustic,.03);
      const silence=model.activeTextRangeAtAudioTime(timeline,acoustic,.15);
      const late=model.activeTextRangeAtAudioTime(timeline,acoustic,.25);
      process.stdout.write(JSON.stringify({acoustic,early,silence,late}));
    """
    completed = subprocess.run(
        [node, "-e", script, str(ROOT / "webui" / "playback_model.js")],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    result = json.loads(completed.stdout)

    assert result["early"] is not None
    assert result["silence"] is None
    assert result["late"] is not None
    assert result["late"]["start"] >= 3
    assert result["acoustic"]["totalSpeechMotion"] > 0
    assert len(result["acoustic"]["textAlignment"]) == 4
