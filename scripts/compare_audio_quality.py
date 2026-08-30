from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aniflive_tts.backend.audio_quality import compare_audio  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--speaker-model", type=Path)
    args = parser.parse_args()
    report = compare_audio(
        args.reference,
        args.candidate,
        args.output_dir,
        report_path=args.report,
        speaker_model_path=args.speaker_model,
    )
    metrics = report["metrics"]
    print(
        f"log_mel={metrics['log_mel_cosine_similarity']:.6f} "
        f"speaker={metrics['speaker_similarity']['value']} "
        f"duration_reference={report['reference']['duration_seconds']:.6f} "
        f"duration_candidate={report['candidate']['duration_seconds']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
