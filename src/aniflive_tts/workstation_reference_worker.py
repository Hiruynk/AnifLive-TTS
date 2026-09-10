from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile

from .workstation_checkpoint_selection import DEPLOYMENT_CHECKPOINTS_SCHEMA
from .workstation_reference_selection import (
    load_preprocessed_speaker_embeddings,
    rank_reference_candidates,
    write_blind_reference_manifest,
)


_REFERENCE_CASES = {
    "ja": (
        "今日はいい天気ですね。",
        "この方法で本当に大丈夫ですか？",
        "ゆっくり息をして、もう一度始めましょう。",
        "明日の予定を確認したあと、必要な資料を準備します。",
        "長い言葉でも音を省かず、最後まではっきり話してください。",
    ),
    "zh": (
        "今天的天氣很好。",
        "這個方法真的沒問題嗎？",
        "請慢慢呼吸，然後再開始一次。",
        "確認明天的行程後，我們會準備所需資料。",
        "即使句子較長，也要清楚完整地說到最後。",
    ),
    "yue": (
        "今日天氣真係幾好。",
        "呢個方法真係無問題咩？",
        "慢慢呼吸，然後再開始一次。",
        "確認完聽日行程，我哋就準備需要嘅資料。",
        "就算句子比較長，都要清楚完整品講到最後。",
    ),
    "en": (
        "The weather is pleasant today.",
        "Are you sure this method is reliable?",
        "Take a slow breath, and then begin again.",
        "After checking tomorrow's schedule, we will prepare the required material.",
        "Please pronounce every sound clearly, even when the sentence is unusually long.",
    ),
    "ko": (
        "오늘은 날씨가 정말 좋네요.",
        "이 방법이 정말 괜찮은가요?",
        "천천히 숨을 쉬고 다시 시작해 봅시다.",
        "내일 일정을 확인한 뒤 필요한 자료를 준비하겠습니다.",
        "문장이 길어도 모든 소리를 끝까지 분명하게 말해 주세요.",
    ),
}


class ReferenceSelectionWorkerError(RuntimeError):
    pass


def _emit_progress(progress: float, message: str) -> None:
    print(
        "ANIFLIVE_TTS_PROGRESS "
        + json.dumps(
            {"progress": max(0.0, min(1.0, progress)), "message": message},
            ensure_ascii=True,
            separators=(",", ":"),
        ),
        flush=True,
    )


def _object(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ReferenceSelectionWorkerError(f"{label} is missing")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReferenceSelectionWorkerError(f"{label} is unreadable") from error
    if not isinstance(value, dict):
        raise ReferenceSelectionWorkerError(f"{label} is malformed")
    return value


def _regular(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ReferenceSelectionWorkerError(f"{label} is missing")
    return path.resolve(strict=True)


def _write_audio(path: Path, audio: Any, sample_rate: int) -> None:
    values = np.asarray(audio, dtype=np.float32).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ReferenceSelectionWorkerError("reference sweep generated invalid audio")
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, sample_rate, np.clip(values, -1.0, 1.0))


def run_reference_selection(
    manifest: Mapping[str, Any], output: Path
) -> tuple[dict[str, Any], list[Path]]:
    if not sys.platform.startswith("linux"):
        raise ReferenceSelectionWorkerError("reference selection runs only in Linux worker")
    inputs = manifest.get("container_input_paths")
    if not isinstance(inputs, Mapping):
        raise ReferenceSelectionWorkerError("worker manifest has no container inputs")
    for key in ("selected_checkpoints", "dataset", "shared_dir"):
        if not isinstance(inputs.get(key), str):
            raise ReferenceSelectionWorkerError(f"reference selection requires {key}")
    selected_root = Path(str(inputs["selected_checkpoints"])).resolve(strict=True)
    training_bundle = Path(str(inputs["dataset"])).resolve(strict=True)
    shared_dir = Path(str(inputs["shared_dir"])).resolve(strict=True)
    deployment = _object(
        selected_root / "selected" / "deployment-checkpoints.json",
        "deployment checkpoints",
    )
    if deployment.get("schema") != DEPLOYMENT_CHECKPOINTS_SCHEMA:
        raise ReferenceSelectionWorkerError("deployment checkpoints are unsupported")
    train = _object(training_bundle / "train" / "manifest.json", "train manifest")
    records = train.get("items")
    if train.get("split") != "train" or not isinstance(records, list):
        raise ReferenceSelectionWorkerError("train manifest is malformed")
    vectors = load_preprocessed_speaker_embeddings(
        records, selected_root / "selection-assets" / "7-sv_cn"
    )
    report = rank_reference_candidates(records, vectors, limit=5)
    report["dataset_id"] = train.get("dataset_id")
    report["checkpoint_selection_report_sha256"] = deployment["selection"][
        "report_sha256"
    ]
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "reference-selection-report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _emit_progress(0.05, "Representative references ranked")

    source_dir = Path(
        os.environ.get("ANIFLIVE_TTS_SOURCE_DIR", "/app/minimal_inference")
    ).resolve(strict=True)
    for value in (source_dir, source_dir / "GPT_SoVITS"):
        if str(value) not in sys.path:
            sys.path.insert(0, str(value))
    try:
        from run_inference import GPTSoVITSInference
    except ImportError as error:
        raise ReferenceSelectionWorkerError("PyTorch inference source is missing") from error
    gpt = _regular(
        selected_root / "selected" / str(deployment["gpt"]["relative_path"]),
        "selected GPT",
    )
    sovits = _regular(
        selected_root / "selected" / str(deployment["sovits"]["relative_path"]),
        "selected SoVITS",
    )
    engine = GPTSoVITSInference(
        str(gpt),
        str(sovits),
        str(shared_dir / "chinese-hubert-base"),
        str(shared_dir / "chinese-roberta-wwm-ext-large"),
        str(shared_dir / "sv" / "pretrained_eres2netv2w24s4ep4.ckpt"),
    )
    generated: dict[str, list[dict[str, Any]]] = {}
    audio_paths: list[Path] = []
    total_cases = sum(
        len(_REFERENCE_CASES[str(candidate["language"])])
        for candidate in report["top_candidates"]
    )
    completed_cases = 0
    for label, candidate in zip("ABCDE", report["top_candidates"], strict=True):
        language = str(candidate["language"])
        cases = _REFERENCE_CASES[language]
        reference_audio = training_bundle / str(candidate["path"])
        rows: list[dict[str, Any]] = []
        for index, text in enumerate(cases, start=1):
            audio, sample_rate = engine.infer(
                str(reference_audio),
                str(candidate["text"]),
                language,
                text,
                language,
                top_k=15,
                top_p=1.0,
                temperature=1.0,
            )
            path = output / "reference-sweep" / label / f"case-{index}.wav"
            _write_audio(path, audio, int(sample_rate))
            audio_paths.append(path)
            completed_cases += 1
            _emit_progress(
                0.05 + 0.9 * completed_cases / total_cases,
                f"Blind reference audio {completed_cases}/{total_cases}",
            )
            rows.append(
                {
                    "case": index,
                    "text": text,
                    "language": language,
                    "path": path.relative_to(output).as_posix(),
                }
            )
        generated[label] = rows
    blind_path = write_blind_reference_manifest(
        selection_report=report,
        generated_cases=generated,
        output=output / "blind-reference-manifest.json",
    )
    return {
        "schema": "aniflive-tts-reference-selection-worker-v1",
        "status": "blocked-pending-human-evidence",
        "automatic_recommendation": report["automatic_recommendation"],
        "candidate_count": report["candidate_count"],
    }, [report_path, blind_path, *audio_paths]


__all__ = ["ReferenceSelectionWorkerError", "run_reference_selection"]
