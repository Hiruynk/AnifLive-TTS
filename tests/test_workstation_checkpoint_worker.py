from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from aniflive_tts import workstation_checkpoint_worker as worker
from aniflive_tts.workstation_checkpoint_selection import (
    CHECKPOINT_CANDIDATES_SCHEMA,
)


def _failed_speaker_evidence(gpt_epoch: int, sovits_epoch: int) -> dict:
    return {
        "phase": "test",
        "gpt_epoch": gpt_epoch,
        "sovits_epoch": sovits_epoch,
        "seeds": [1234],
        "generation_success_rate": 1.0,
        "content": {
            "median_error": 0.0,
            "p95_error": 0.0,
            "repetition_failures": 0,
            "omission_failures": 0,
        },
        "speaker": {
            "centroid_cosine_median": 0.79,
            "centroid_cosine_p10": 0.78,
        },
        "audio": {
            "invalid": 0,
            "nan": 0,
            "empty": 0,
            "clipped": 0,
            "duration_outliers": 0,
        },
        "stability": 1.0,
    }


def test_failed_selection_returns_machine_readable_diagnostic(
    tmp_path: Path, monkeypatch
) -> None:
    candidates = tmp_path / "candidates"
    bundle = tmp_path / "bundle"
    shared = tmp_path / "shared"
    asr = tmp_path / "asr"
    speaker = tmp_path / "speaker"
    for path in (candidates, bundle / "validation", shared, asr, speaker):
        path.mkdir(parents=True)
    (candidates / "checkpoint-candidates.json").write_text(
        json.dumps(
            {
                "schema": CHECKPOINT_CANDIDATES_SCHEMA,
                "gpt": [{"epoch": 1}],
                "sovits": [{"epoch": 1}],
            }
        ),
        encoding="utf-8",
    )
    (bundle / "validation" / "manifest.json").write_text(
        json.dumps({"split": "validation", "items": [{"source_item_id": "item_1"}]}),
        encoding="utf-8",
    )

    class PairEvaluator:
        provisional_reference = {"item_id": "item_1"}
        references = tuple(
            {"source_item_id": f"reference_{index}"} for index in range(3)
        )
        calls: list[dict] = []

        def __init__(self, **_kwargs) -> None:
            pass

        def evaluate(self, *, gpt_epoch: int, sovits_epoch: int, **_kwargs) -> dict:
            self.calls.append(dict(_kwargs))
            row = _failed_speaker_evidence(gpt_epoch, sovits_epoch)
            reference_index = _kwargs.get("reference_indices", (0,))[0]
            row["reference_probe"] = {
                "count": 1,
                "item_ids": [f"reference_{reference_index}"],
            }
            return row

    monkeypatch.setattr(worker.sys, "platform", "linux")
    monkeypatch.setattr(worker, "_PairEvaluator", PairEvaluator)
    monkeypatch.setattr(
        worker, "validation_records", lambda _bundle: [{"source_item_id": "item_1"}]
    )
    payload, artifacts = worker.run_checkpoint_selection(
        {
            "container_input_paths": {
                "checkpoint_candidates": str(candidates),
                "dataset": str(bundle),
                "shared_dir": str(shared),
                "asr_model": str(asr),
                "speaker_component": str(speaker),
            }
        },
        tmp_path / "output",
    )

    assert payload == {
        "schema": "aniflive-tts-checkpoint-selection-worker-v1",
        "status": "failed",
        "reason": "no checkpoint pair passed the hard gates",
        "test_split_accessed": False,
    }
    assert [path.name for path in artifacts] == [
        "checkpoint-selection-diagnostics.json"
    ]
    diagnostic = json.loads(artifacts[0].read_text(encoding="utf-8"))
    assert diagnostic["status"] == "failed"
    assert diagnostic["test_split_accessed"] is False
    assert diagnostic["passes"]["joint_sweep"][0]["hard_gate"]["failures"] == [
        "speaker-identity"
    ]
    joint_calls = [call for call in PairEvaluator.calls if call["phase"] == "joint-sweep"]
    assert joint_calls
    assert {call["reference_indices"] for call in joint_calls} == {
        (0,),
        (1,),
        (2,),
    }
    pruning_calls = [call for call in PairEvaluator.calls if call["phase"] != "joint-sweep"]
    assert pruning_calls
    assert all("reference_indices" not in call for call in pruning_calls)


def test_reference_audio_uses_stable_source_index(
    tmp_path: Path, monkeypatch
) -> None:
    evaluator = worker._PairEvaluator.__new__(worker._PairEvaluator)
    evaluator.candidate_root = tmp_path
    evaluator.shared_dir = tmp_path
    evaluator.output = tmp_path / "output"
    evaluator.gpt = {1: {"epoch": 1}}
    evaluator.sovits = {1: {"epoch": 1}}
    evaluator.references = [
        {"source_item_id": "reference_1", "transcript": "一", "language": "ja"},
        {"source_item_id": "reference_2", "transcript": "二", "language": "ja"},
        {"source_item_id": "reference_3", "transcript": "三", "language": "ja"},
    ]
    evaluator.reference_paths = [tmp_path / f"reference_{index}.wav" for index in range(3)]
    evaluator.centroid = np.asarray([1.0, 0.0], dtype=np.float32)
    evaluator.speaker = lambda _values, _rate: np.asarray([1.0, 0.0])
    evaluator.asr = object()

    class Inference:
        def __init__(self, *_args) -> None:
            pass

        def infer(self, *_args, **_kwargs):
            return np.ones(24000, dtype=np.float32) * 0.1, 24000

    evaluator.inference_type = Inference
    written: list[Path] = []
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            manual_seed=lambda _seed: None,
            cuda=SimpleNamespace(manual_seed_all=lambda _seed: None),
        ),
    )
    monkeypatch.setattr(worker, "_candidate_path", lambda *_args: tmp_path)
    monkeypatch.setattr(worker, "_write_audio", lambda path, *_args: written.append(path))
    monkeypatch.setattr(worker, "_transcribe", lambda *_args: "測試")
    monkeypatch.setattr(
        worker._PairEvaluator,
        "_source_speaker_score",
        lambda _self, _record: 0.9,
    )

    result = evaluator.evaluate(
        gpt_epoch=1,
        sovits_epoch=1,
        records=[
            {
                "source_item_id": "item_1",
                "transcript": "測試",
                "language": "ja",
                "verification_hashes": {"speaker": "a" * 64},
            }
        ],
        seeds=(1234,),
        phase="joint-sweep",
        reference_indices=(1,),
    )

    assert [path.name for path in written] == ["r2-item_1-seed-1234.wav"]
    assert result["reference_probe"]["item_ids"] == ["reference_2"]
