from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import re
import unicodedata
from typing import Any, Iterable

import numpy as np


_LANGUAGE_ALIASES = {
    "zh": "zh",
    "zh-cn": "zh",
    "zh-tw": "zh",
    "中文": "zh",
    "普通话": "zh",
    "普通話": "zh",
    "mandarin": "zh",
    "yue": "yue",
    "zh-yue": "yue",
    "粤语": "yue",
    "粵語": "yue",
    "广东话": "yue",
    "廣東話": "yue",
    "cantonese": "yue",
    "ja": "ja",
    "jp": "ja",
    "日本語": "ja",
    "japanese": "ja",
    "en": "en",
    "english": "en",
    "英语": "en",
    "英語": "en",
    "ko": "ko",
    "kr": "ko",
    "한국어": "ko",
    "korean": "ko",
}
_CANTONESE_MARKERS = frozenset(
    {
        "係",
        "唔",
        "咗",
        "喺",
        "嘅",
        "冇",
        "啲",
        "佢",
        "哋",
        "咩",
        "噉",
        "嗰",
        "呢",
        "啦",
    }
)
_SPACE = re.compile(r"[\t\v\f\r ]+")


class DatasetQualityError(ValueError):
    pass


@dataclass(frozen=True)
class GptSovitsListEntry:
    audio_path: Path
    speaker: str
    language: str
    text: str
    line_number: int

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["audio_path"] = str(self.audio_path)
        return result


@dataclass(frozen=True)
class AudioQualityReport:
    duration_seconds: float
    rms_dbfs: float
    peak_dbfs: float
    dc_offset: float
    clipped_ratio: float
    silence_ratio: float
    estimated_snr_db: float
    quality_score: float
    method: str = "deterministic-pcm-v1"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def canonical_language(value: str) -> str:
    if not isinstance(value, str):
        raise DatasetQualityError("language must be text")
    normalized = unicodedata.normalize("NFKC", value).strip().casefold()
    try:
        return _LANGUAGE_ALIASES[normalized]
    except KeyError as error:
        raise DatasetQualityError(f"unsupported language: {value}") from error


def normalize_transcript(value: str) -> str:
    if not isinstance(value, str):
        raise DatasetQualityError("transcript must be text")
    normalized = unicodedata.normalize("NFKC", value).replace("\ufeff", "")
    normalized = "\n".join(_SPACE.sub(" ", line).strip() for line in normalized.splitlines())
    normalized = " ".join(part for part in normalized.split("\n") if part)
    if not normalized:
        raise DatasetQualityError("transcript cannot be empty")
    if len(normalized) > 16_000:
        raise DatasetQualityError("transcript is limited to 16000 characters")
    return normalized


def infer_text_language(value: str) -> dict[str, Any]:
    text = normalize_transcript(value)
    hangul = sum("\uac00" <= char <= "\ud7a3" for char in text)
    kana = sum(
        "\u3040" <= char <= "\u30ff" or "\u31f0" <= char <= "\u31ff"
        for char in text
    )
    cjk = sum("\u3400" <= char <= "\u9fff" for char in text)
    latin = sum(char.isascii() and char.isalpha() for char in text)
    visible = max(1, sum(not char.isspace() and not unicodedata.category(char).startswith("P") for char in text))
    if hangul:
        language, evidence = "ko", hangul
    elif kana:
        language, evidence = "ja", kana + cjk
    elif cjk:
        cantonese = sum(char in _CANTONESE_MARKERS for char in text)
        language = "yue" if cantonese >= 2 else "zh"
        evidence = cjk
    elif latin:
        language, evidence = "en", latin
    else:
        language, evidence = "und", 0
    return {
        "language": language,
        "confidence": round(min(1.0, evidence / visible), 6),
        "method": "unicode-script-and-cantonese-lexeme-v1",
        "diagnostic_only": language in {"zh", "yue", "und"},
    }


def parse_gpt_sovits_list(path: Path) -> list[GptSovitsListEntry]:
    source = Path(path).expanduser().resolve(strict=True)
    if not source.is_file() or source.suffix.casefold() != ".list":
        raise DatasetQualityError("GPT-SoVITS metadata must be a .list file")
    try:
        lines = source.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as error:
        raise DatasetQualityError("GPT-SoVITS .list must be readable UTF-8 text") from error
    entries: list[GptSovitsListEntry] = []
    for line_number, raw in enumerate(lines, start=1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        fields = raw.split("|", 3)
        if len(fields) != 4:
            raise DatasetQualityError(
                f"Malformed GPT-SoVITS .list line {line_number}; expected audio|speaker|language|text"
            )
        raw_audio, raw_speaker, raw_language, raw_text = fields
        audio = Path(raw_audio.strip()).expanduser()
        if not audio.is_absolute():
            audio = source.parent / audio
        try:
            audio = audio.resolve(strict=True)
        except OSError as error:
            raise DatasetQualityError(
                f"GPT-SoVITS .list line {line_number} references a missing audio file"
            ) from error
        if not audio.is_file():
            raise DatasetQualityError(
                f"GPT-SoVITS .list line {line_number} does not reference a regular audio file"
            )
        speaker = unicodedata.normalize("NFKC", raw_speaker).strip()
        if not speaker or len(speaker) > 160:
            raise DatasetQualityError(f"GPT-SoVITS .list line {line_number} has an invalid speaker")
        entries.append(
            GptSovitsListEntry(
                audio_path=audio,
                speaker=speaker,
                language=canonical_language(raw_language),
                text=normalize_transcript(raw_text),
                line_number=line_number,
            )
        )
    if not entries:
        raise DatasetQualityError("GPT-SoVITS .list contains no usable records")
    return entries


def analyze_pcm_quality(
    samples: np.ndarray,
    sample_rate: int,
    *,
    silence_threshold_dbfs: float = -50.0,
) -> AudioQualityReport:
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim == 2:
        if values.shape[1] < 1:
            raise DatasetQualityError("audio has no channels")
        mono = values.mean(axis=1)
    elif values.ndim == 1:
        mono = values
    else:
        raise DatasetQualityError("audio samples must be one- or two-dimensional")
    if not isinstance(sample_rate, int) or sample_rate <= 0:
        raise DatasetQualityError("sample_rate must be positive")
    if mono.size == 0 or not np.isfinite(mono).all():
        raise DatasetQualityError("audio must contain finite samples")
    absolute = np.abs(mono)
    rms = math.sqrt(float(np.mean(np.square(mono))))
    peak = float(np.max(absolute))
    rms_dbfs = 20.0 * math.log10(max(rms, 1e-12))
    peak_dbfs = 20.0 * math.log10(max(peak, 1e-12))
    frame = max(1, round(sample_rate * 0.02))
    usable = mono[: (mono.size // frame) * frame]
    if usable.size:
        framed = usable.reshape(-1, frame)
        frame_rms = np.sqrt(np.mean(np.square(framed), axis=1))
        frame_dbfs = 20.0 * np.log10(np.maximum(frame_rms, 1e-12))
        silence_ratio = float(np.mean(frame_dbfs < silence_threshold_dbfs))
        noise_rms = float(np.percentile(frame_rms, 20))
        signal_rms = float(np.percentile(frame_rms, 80))
        estimated_snr = 20.0 * math.log10(max(signal_rms, 1e-12) / max(noise_rms, 1e-12))
    else:
        silence_ratio = float(rms_dbfs < silence_threshold_dbfs)
        estimated_snr = 0.0
    clipped_ratio = float(np.mean(absolute >= 0.999))
    duration = mono.size / sample_rate
    duration_penalty = 0.0
    if duration < 0.5:
        duration_penalty = min(30.0, (0.5 - duration) * 60.0)
    elif duration > 15.0:
        duration_penalty = min(20.0, (duration - 15.0) * 1.5)
    score = 100.0
    score -= min(45.0, clipped_ratio * 4500.0)
    score -= min(25.0, max(0.0, silence_ratio - 0.15) * 35.0)
    score -= min(25.0, max(0.0, 20.0 - estimated_snr) * 1.25)
    score -= min(20.0, abs(float(np.mean(mono))) * 400.0)
    score -= duration_penalty
    return AudioQualityReport(
        duration_seconds=round(duration, 9),
        rms_dbfs=round(rms_dbfs, 6),
        peak_dbfs=round(peak_dbfs, 6),
        dc_offset=round(float(np.mean(mono)), 9),
        clipped_ratio=round(clipped_ratio, 9),
        silence_ratio=round(silence_ratio, 9),
        estimated_snr_db=round(max(-120.0, min(120.0, estimated_snr)), 6),
        quality_score=round(max(0.0, min(100.0, score)), 3),
    )


def aggregate_quality(reports: Iterable[AudioQualityReport]) -> dict[str, float | int]:
    values = list(reports)
    if not values:
        return {"count": 0, "duration_seconds": 0.0, "quality_score_mean": 0.0}
    return {
        "count": len(values),
        "duration_seconds": round(sum(value.duration_seconds for value in values), 9),
        "quality_score_mean": round(
            sum(value.quality_score for value in values) / len(values), 3
        ),
    }


__all__ = [
    "AudioQualityReport",
    "DatasetQualityError",
    "GptSovitsListEntry",
    "aggregate_quality",
    "analyze_pcm_quality",
    "canonical_language",
    "infer_text_language",
    "normalize_transcript",
    "parse_gpt_sovits_list",
]
