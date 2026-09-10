from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _object(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} is missing")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def _safe_file(root: Path, relative: str, label: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise RuntimeError(f"{label} path is unsafe")
    path = (root / candidate).resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise RuntimeError(f"{label} escaped its root") from error
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} is not a regular file")
    return path


def _cosine(left: Any, right: Any) -> float:
    import numpy as np

    first = np.asarray(left, dtype=np.float64).reshape(-1)
    second = np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1e-12:
        raise RuntimeError("speaker cosine denominator is zero")
    return float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit an all-poor reference sweep without reading test data."
    )
    parser.add_argument("--training-bundle", type=Path, required=True)
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--blind-manifest", type=Path, required=True)
    parser.add_argument("--human-evidence", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--speaker-component", type=Path, required=True)
    parser.add_argument("--asr-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    import numpy as np
    import soundfile as sf
    from faster_whisper import WhisperModel

    from aniflive_tts.workstation_evaluation import spoken_content_error_rate
    from aniflive_tts.workstation_reference_diagnostics import (
        build_reference_rejection_diagnostic,
    )
    from aniflive_tts.workstation_reference_selection import (
        load_preprocessed_speaker_embeddings,
        robust_speaker_centroid,
    )
    from aniflive_tts.workstation_speaker import WorkstationSpeakerVerifier

    args = _parse_args()
    bundle = args.training_bundle.resolve(strict=True)
    selection_root = args.selection_root.resolve(strict=True)
    audio_root = args.audio_root.resolve(strict=True)
    report_path = args.reference_report.resolve(strict=True)
    blind_path = args.blind_manifest.resolve(strict=True)
    human_path = args.human_evidence.resolve(strict=True)
    report = _object(report_path, "reference selection report")
    blind = _object(blind_path, "blind reference manifest")
    human = _object(human_path, "all-poor human evidence")
    train = _object(bundle / "train" / "manifest.json", "train manifest")
    if train.get("split") != "train" or not isinstance(train.get("items"), list):
        raise RuntimeError("train manifest is malformed")
    if blind.get("schema") != "aniflive-tts-blind-reference-sweep-v1":
        raise RuntimeError("blind reference manifest is unsupported")
    entries = blind.get("entries")
    candidates = report.get("top_candidates")
    if not isinstance(entries, list) or not isinstance(candidates, list):
        raise RuntimeError("reference sweep evidence is malformed")

    vectors = load_preprocessed_speaker_embeddings(
        train["items"], selection_root / "selection-assets" / "7-sv_cn"
    )
    centroid, _ = robust_speaker_centroid(list(vectors.values()))
    candidate_by_id = {
        str(candidate["item_id"]): candidate
        for candidate in candidates
        if isinstance(candidate, dict)
    }
    source_scores = {
        item_id: _cosine(vectors[item_id], centroid)
        for item_id in candidate_by_id
    }
    speaker = WorkstationSpeakerVerifier(args.speaker_component.resolve(strict=True))
    asr = WhisperModel(
        str(args.asr_model.resolve(strict=True)),
        device="cuda",
        device_index=0,
        compute_type="float16",
        local_files_only=True,
    )
    cases: list[dict[str, Any]] = []
    total = sum(
        len(entry.get("cases", [])) for entry in entries if isinstance(entry, dict)
    )
    completed = 0
    for entry in entries:
        if not isinstance(entry, dict):
            raise RuntimeError("blind reference entry is malformed")
        label = str(entry.get("label") or "")
        item_id = str(entry.get("candidate_item_id") or "")
        rows = entry.get("cases")
        if item_id not in candidate_by_id or not isinstance(rows, list):
            raise RuntimeError(f"blind reference entry {label} has invalid lineage")
        for row in rows:
            if not isinstance(row, dict):
                raise RuntimeError(f"blind reference entry {label} has invalid case")
            path = _safe_file(audio_root, str(row.get("path") or ""), "blind audio")
            samples, sample_rate = sf.read(
                str(path), dtype="float32", always_2d=False
            )
            values = np.asarray(samples, dtype=np.float32)
            if values.ndim == 2:
                values = values.mean(axis=1)
            values = values.reshape(-1)
            finite = bool(values.size and np.isfinite(values).all())
            invalid = not finite
            if invalid:
                embedding = np.zeros_like(centroid)
                hypothesis = ""
                speaker_score = -1.0
                content_error = 1.0
                duration = 0.0
                clipped = False
                silence_ratio = 1.0
                rms_dbfs = float("-inf")
            else:
                embedding = speaker(values, int(sample_rate))
                speaker_score = _cosine(embedding, centroid)
                segments, _ = asr.transcribe(
                    str(path),
                    language="zh" if row.get("language") == "yue" else row.get("language"),
                    beam_size=5,
                    temperature=0.0,
                    vad_filter=False,
                    condition_on_previous_text=False,
                )
                hypothesis = "".join(segment.text for segment in segments).strip()
                content_error = spoken_content_error_rate(
                    str(row.get("text") or ""),
                    hypothesis,
                    str(row.get("language") or ""),
                )
                duration = values.size / int(sample_rate)
                clipped = float(np.mean(np.abs(values) >= 0.999)) > 0.001
                silence_ratio = float(np.mean(np.abs(values) < 10 ** (-50 / 20)))
                rms = float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))
                rms_dbfs = 20.0 * float(np.log10(max(rms, 1e-12)))
            target_length = max(1, len(str(row.get("text") or "").replace(" ", "")))
            cases.append(
                {
                    "label": label,
                    "candidate_item_id": item_id,
                    "case": row.get("case"),
                    "language": row.get("language"),
                    "text": row.get("text"),
                    "hypothesis": hypothesis,
                    "content_error": content_error,
                    "speaker_centroid_cosine": speaker_score,
                    "duration_seconds": duration,
                    "duration_outlier": duration < 0.25
                    or duration > max(8.0, target_length * 0.9),
                    "invalid": invalid,
                    "nan": bool(values.size and not np.isfinite(values).all()),
                    "empty": values.size == 0,
                    "clipped": clipped,
                    "silence_ratio": silence_ratio,
                    "rms_dbfs": rms_dbfs,
                    "audio_sha256": _sha256_file(path),
                    "audio_path": str(path),
                }
            )
            completed += 1
            print(f"Reference diagnostic {completed}/{total}", flush=True)

    payload = build_reference_rejection_diagnostic(
        cases,
        source_speaker_scores=source_scores,
        human_evidence=human,
        reference_selection_report_sha256=_sha256_file(report_path),
        blind_manifest_sha256=_sha256_file(blind_path),
        test_split_accessed=False,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(payload["automatic_summary"], sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
