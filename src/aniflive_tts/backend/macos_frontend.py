"""Persistent official Python text frontend for the macOS prototype.

Text cleanup stays in the pinned official source. Mandarin G2PW/BERT are
evaluated by the attached native worker; other supported languages use the
official zero-BERT branch. This module does not run neural TTS inference.
"""
from __future__ import annotations

import hashlib
import importlib.util
from itertools import combinations
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


_SUPPORTED_LANGUAGES = {"ja", "en", "yue", "ko"}
# Official GPT-SoVITS TextPreprocessor language modes, exactly as the Windows
# API normalizes them: all_* = monolingual (Latin runs still go to English),
# ja/ko/yue/zh = that language mixed with English, auto/auto_yue = per-run
# language detection (auto_yue reads Chinese runs as Cantonese).
OFFICIAL_LANGUAGES = frozenset({"all_ja", "ja", "en", "all_ko", "ko", "all_yue", "yue",
                                "all_zh", "zh", "auto", "auto_yue"})
# Modes whose official output can contain Mandarin BERT features.
_MANDARIN_MODES = frozenset({"all_zh", "zh", "auto"})
# Base G2P languages each mode may route text to (for dependency metadata).
_MODE_LANGUAGES = {
    "all_ja": {"ja", "en"}, "ja": {"ja", "en"}, "en": {"en"},
    "all_ko": {"ko", "en"}, "ko": {"ko", "en"}, "all_yue": {"yue", "en"}, "yue": {"yue", "en"},
    "all_zh": {"zh", "en"}, "zh": {"zh", "en"},
    "auto": {"ja", "en", "ko", "zh"}, "auto_yue": {"ja", "en", "ko", "yue"},
}
_FRONTEND_VERSION = "v2"
_METADATA_LANGUAGES = ("ja", "en", "yue", "ko", "zh")
_PYTHON_DEPENDENCY_NAMES = (
    "torch", "transformers", "pyopenjtalk", "split-lang", "fast-langdetect",
    "nltk", "g2p-en", "wordsegment", "cn2an", "pypinyin", "tojyutping",
    "g2pk2", "ko-pron", "python-mecab-ko", "jieba-fast",
)


class NativePythonFrontend:
    """Keep one official TextPreprocessor alive across CPU frontend requests."""

    def __init__(self, script: Path, official: Path, *, cache_metadata: bool = False) -> None:
        if type(cache_metadata) is not bool:
            raise TypeError("cache_metadata must be a bool")
        started = time.perf_counter()
        self.metadata_cache_enabled = cache_metadata
        self.metadata_cache_initialization_seconds = 0.0
        self.script = Path(script).expanduser().resolve(strict=True)
        if self.script.is_dir():
            self.script = self.script / "prepare_frontend.py"
        if not self.script.is_file():
            raise FileNotFoundError(f"frontend helper script not found: {self.script}")
        self.scripts_dir = self.script.parent
        self.official = Path(official).expanduser().resolve(strict=True)
        self.gpt_sovits = self.official / "GPT_SoVITS"
        if not self.gpt_sovits.is_dir():
            raise FileNotFoundError(f"official source root has no GPT_SoVITS directory: {self.official}")
        self.data_root = Path(os.environ.get("ANIFLIVE_TTS_MACOS_DATA_ROOT", self.script.parents[2] / "data/macos-v5turbo")).expanduser().resolve()
        self.cache_root = self.data_root / "cache"
        self.cache_root.mkdir(parents=True, exist_ok=True)

        self._request_lock = threading.Lock()
        self._mandarin_bridge: Any | None = None
        self._validate_loaded_frontend_modules()
        if str(self.scripts_dir) not in sys.path:
            sys.path.insert(0, str(self.scripts_dir))
        self.helper = self._load_helper()
        self._validate_loaded_frontend_modules()

        if str(self.gpt_sovits) not in sys.path:
            sys.path.insert(0, str(self.gpt_sovits))
        self.frontend = self.helper._load_official_preprocessor(
            self.official, self.cache_root
        )
        self.fasttext_asset = self.helper._fasttext_asset_record(self.cache_root)
        self._metadata_seen_language_sets: set[frozenset[str]] = set()
        self._metadata_source_snapshots: dict[frozenset[str], dict[str, str]] = {}
        self._metadata_python_dependencies: dict[str, str | None] | None = None
        if self.metadata_cache_enabled:
            metadata_started = time.perf_counter()
            self._metadata_python_dependencies = self._python_dependency_versions()
            self._snapshot_metadata_sources()
            self.metadata_cache_initialization_seconds = time.perf_counter() - metadata_started
        self.initialization_seconds = time.perf_counter() - started

    def enable_mandarin(self, bridge: Any) -> None:
        if self._mandarin_bridge is not None:
            raise RuntimeError("Native Mandarin frontend is already configured")
        bridge.attach(self.frontend["processor"])
        self._mandarin_bridge = bridge

    def _source_paths_for_languages(self, languages: frozenset[str]) -> list[Path]:
        root = self.frontend["root"]
        paths = self.helper._source_files(
            root, set(languages) - {"zh"}, self.frontend["preprocessor_path"],
            self.frontend["segment_source"],
        )
        paths.append(self.frontend["dictionary_source"])
        if "zh" in languages:
            paths.extend(root / name for name in (
                "text/chinese2.py", "text/g2pw/g2pw.py", "text/g2pw/onnx_api.py", "text/g2pw/dataset.py",
            ))
        return sorted(set(paths))

    def _source_hashes_for_languages(self, languages: frozenset[str]) -> dict[str, str]:
        root_parent = self.frontend["root"].parent
        return {
            str(path.relative_to(root_parent)): self.helper._sha256_file(path)
            for path in self._source_paths_for_languages(languages)
        }

    def _snapshot_metadata_sources(self) -> None:
        keys = {frozenset(pair) for pair in combinations(_METADATA_LANGUAGES, 1)}
        keys.update(frozenset(pair) for pair in combinations(_METADATA_LANGUAGES, 2))
        paths_by_key = {key: self._source_paths_for_languages(key) for key in keys}
        all_paths = set().union(*paths_by_key.values())
        root_parent = self.frontend["root"].parent
        hashes = {path: self.helper._sha256_file(path) for path in sorted(all_paths)}
        self._metadata_source_snapshots = {
            key: {
                str(path.relative_to(root_parent)): hashes[path]
                for path in paths_by_key[key]
            }
            for key in keys
        }

    def _python_dependency_versions(self) -> dict[str, str | None]:
        return {name: self.helper._package_version(name) for name in _PYTHON_DEPENDENCY_NAMES}

    def _reporting_metadata(
        self, languages: frozenset[str]
    ) -> tuple[dict[str, str], dict[str, str | None], bool, str]:
        if self.metadata_cache_enabled:
            if self._metadata_python_dependencies is None:
                raise ValueError(f"unsupported metadata language set: {sorted(languages)}")
            if languages not in self._metadata_source_snapshots:
                # Mixed/auto modes touch more than two G2P modules; snapshot
                # that set once, the same way as the precomputed pairs.
                self._metadata_source_snapshots[languages] = self._source_hashes_for_languages(languages)
            hit = languages in self._metadata_seen_language_sets
            self._metadata_seen_language_sets.add(languages)
            return (
                dict(self._metadata_source_snapshots[languages]),
                dict(self._metadata_python_dependencies),
                hit,
                "loaded-process initialization snapshot; refreshed only by frontend/process reload",
            )
        return (
            self._source_hashes_for_languages(languages),
            self._python_dependency_versions(),
            False,
            "current request; no source or dependency metadata reuse",
        )

    def mandarin_reference_source_identity(self) -> dict[str, str]:
        """Hash the small Mandarin text/bridge sources used for registration."""
        if self._mandarin_bridge is None:
            raise RuntimeError("Mandarin requires native G2PW and BERT")
        root = self.frontend["root"]
        paths = self.helper._source_files(
            root, set(), self.frontend["preprocessor_path"], self.frontend["segment_source"]
        )
        paths.extend((
            self.frontend["dictionary_source"],
            root / "text/chinese2.py",
            root / "text/g2pw/g2pw.py",
            root / "text/g2pw/onnx_api.py",
            root / "text/g2pw/dataset.py",
            root / "text/symbols.py",
            root / "text/opencpop-strict.txt",
            Path(__file__).resolve(),
            Path(__file__).with_name("macos_mandarin.py").resolve(),
        ))
        # Mandarin normalization and pronunciation corrections are executable
        # preprocessing inputs too, independent of the native model weights.
        for directory in (root / "text/zh_normalization", root / "text/g2pw"):
            paths.extend(path for path in directory.rglob("*")
                         if path.is_file() and path.suffix in {".py", ".json", ".txt", ".rep"}
                         and "__pycache__" not in path.parts)
        result = {}
        for path in sorted(set(paths)):
            label = ("official/" + path.relative_to(self.official).as_posix()
                     if path.is_relative_to(self.official)
                     else "app/" + path.name)
            result[label] = self.helper._sha256_file(path)
        return result

    def prepare_mandarin_reference(self, *, text: str, output: Path) -> dict[str, Any]:
        """Write actual Mandarin reference phones/BERT through the live bridge."""
        if self._mandarin_bridge is None:
            raise RuntimeError("Mandarin requires native G2PW and BERT")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("reference transcript must be nonempty")
        if not self._request_lock.acquire(blocking=False):
            raise RuntimeError("persistent Python frontend already has a request in progress")
        try:
            g2pw_before = self._mandarin_bridge.g2pw_rpc_count
            bert_before = self._mandarin_bridge.bert_rpc_count
            started = time.perf_counter()
            ids, features, segments, normalized = self._preprocess(text, "zh")
            elapsed = time.perf_counter() - started
            if not ids or features.shape != (1024, len(ids)) or not torch.isfinite(features).all():
                raise ValueError("Mandarin reference phones/BERT are empty, misaligned, or non-finite")
            if torch.count_nonzero(features).item() == 0:
                raise ValueError("Mandarin reference BERT did not produce neural features")
            if self._mandarin_bridge.bert_rpc_count <= bert_before:
                raise RuntimeError("Mandarin reference preprocessing did not call native BERT")

            output_dir = Path(output).expanduser().resolve(strict=True)
            phones_path = output_dir / "reference-phones.npy"
            bert_path = output_dir / "reference-bert.npy"
            report_path = output_dir / "mandarin-reference-frontend.json"
            if phones_path.exists() or bert_path.exists() or report_path.exists():
                raise FileExistsError("DSP-only reference directory already contains Mandarin frontend outputs")
            phones = np.asarray([ids], dtype=np.int64)
            bert = features.unsqueeze(0).cpu().numpy().astype(np.float32, copy=False)
            np.save(phones_path, phones, allow_pickle=False)
            np.save(bert_path, bert, allow_pickle=False)
            report = {
                "schema": "aniflive-tts-macos-v5turbo-mandarin-reference-frontend-v1",
                "kind": "official_mandarin_reference_text_frontend",
                "language": "zh",
                "official_language": "all_zh",
                "text_split_method": "cut0",
                "text_version": _FRONTEND_VERSION,
                "normalized_text": normalized,
                "segments": segments,
                "phone_count": len(ids),
                "phone_shape": list(phones.shape),
                "bert_shape": list(bert.shape),
                "bert_all_zero": False,
                "outputs": {
                    "reference-phones.npy": {
                        "sha256": self.helper._sha256_file(phones_path),
                        "shape": list(phones.shape), "dtype": str(phones.dtype),
                    },
                    "reference-bert.npy": {
                        "sha256": self.helper._sha256_file(bert_path),
                        "shape": list(bert.shape), "dtype": str(bert.dtype),
                    },
                },
                "native_g2pw_rpc_count_delta": self._mandarin_bridge.g2pw_rpc_count - g2pw_before,
                "native_bert_rpc_count_delta": self._mandarin_bridge.bert_rpc_count - bert_before,
                "preprocess_seconds": elapsed,
                "source_files_sha256": self.mandarin_reference_source_identity(),
            }
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                                   encoding="utf-8")
            return report
        finally:
            self._request_lock.release()

    def _preprocess_official(self, text: str, mode: str):
        """Target text in an official language mode (Windows semantics)."""
        if mode not in _MANDARIN_MODES:
            # No Mandarin run is possible: the official zero-BERT branch.
            return self.helper._preprocess(self.frontend["processor"], text, mode, self.frontend["symbols"])
        if mode in {"all_zh", "zh"} and self._mandarin_bridge is None:
            raise ValueError("Mandarin text requires the enabled native G2PW/BERT frontend")
        try:
            results = self.frontend["processor"].preprocess(text, mode, "cut0", _FRONTEND_VERSION)
        except (TypeError, AttributeError) as error:
            if self._mandarin_bridge is None:
                raise ValueError("Text contains Mandarin runs; this package has no native Mandarin frontend") from error
            raise
        if not results:
            raise ValueError(f"Official TextPreprocessor returned no segments for language {mode}")
        phones, parts, segments, normalized = [], [], [], []
        for index, result in enumerate(results):
            ids = [int(value) for value in result["phones"]]
            features = result["bert_features"].detach().to(device="cpu", dtype=torch.float32)
            if not ids or features.shape != (1024, len(ids)) or not torch.isfinite(features).all():
                raise ValueError("Official phones and BERT features do not align")
            if any(token < 0 or token >= len(self.frontend["symbols"]) for token in ids):
                raise ValueError("Official frontend emitted a phone outside symbols2")
            phones.extend(ids)
            parts.append(features)
            normalized.append(str(result["norm_text"]))
            segments.append({"index": index, "normalized_text": str(result["norm_text"]),
                "phone_ids": ids, "phone_symbols": [self.frontend["symbols"][token] for token in ids],
                "phone_count": len(ids), "bert_shape": list(features.shape),
                "bert_all_zero": not bool(torch.count_nonzero(features).item())})
        return phones, torch.cat(parts, dim=1), segments, "".join(normalized)

    def _preprocess(self, text: str, language: str):
        if language != "zh":
            return self.helper._preprocess(self.frontend["processor"], text,
                self.helper.TARGET_LANGUAGES[language], self.frontend["symbols"])
        if self._mandarin_bridge is None:
            raise RuntimeError("Mandarin requires native G2PW and BERT")
        results = self.frontend["processor"].preprocess(text, "all_zh", "cut0", _FRONTEND_VERSION)
        if not results:
            raise ValueError("Mandarin text produced no segments")
        phones, parts, segments, normalized = [], [], [], []
        for index, result in enumerate(results):
            ids = [int(value) for value in result["phones"]]
            features = result["bert_features"].detach().to(device="cpu", dtype=torch.float32)
            if not ids or features.shape != (1024, len(ids)) or not torch.isfinite(features).all():
                raise ValueError("Mandarin phones and native BERT features do not align")
            if any(token < 0 or token >= len(self.frontend["symbols"]) for token in ids):
                raise ValueError("Mandarin frontend emitted a phone outside symbols2")
            phones.extend(ids)
            parts.append(features)
            normalized.append(str(result["norm_text"]))
            segments.append({"index": index, "normalized_text": str(result["norm_text"]),
                "phone_ids": ids, "phone_symbols": [self.frontend["symbols"][token] for token in ids],
                "phone_count": len(ids), "bert_shape": list(features.shape),
                "bert_all_zero": not bool(torch.count_nonzero(features).item())})
        return phones, torch.cat(parts, dim=1), segments, "".join(normalized)

    def prepare(
        self,
        *,
        reference_transcript: Path,
        reference_language: str,
        text: str,
        language: str,
        output: Path,
    ) -> dict[str, Any]:
        """Write reference/target phone arrays and aligned BERT features."""
        # The target language is an official mode (Windows normalization).
        if language not in OFFICIAL_LANGUAGES:
            raise ValueError(f"unsupported target language {language!r}")
        reference_languages = _SUPPORTED_LANGUAGES | ({"zh"} if self._mandarin_bridge is not None else set())
        if reference_language not in reference_languages:
            raise ValueError(f"unsupported reference language {reference_language!r}")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("target text must be a nonempty string")
        if not self._request_lock.acquire(blocking=False):
            raise RuntimeError("persistent Python frontend already has a request in progress")
        english_counter: dict[str, Any] | None = None
        nltk_metadata: dict[str, Any] | None = None
        try:
            involved = _MODE_LANGUAGES[language] | {reference_language}
            if involved & {"en", "ko"}:
                _, _, nltk_metadata = self.helper._ensure_nltk_data(self.data_root)

            if "en" in involved:
                english_counter = self.helper._install_english_fallback_counter()

            reference_path = Path(reference_transcript).expanduser().resolve(strict=True)
            reference_bytes = reference_path.read_bytes()
            reference_text = reference_bytes.decode("utf-8")
            start_total = time.perf_counter()
            start_reference = time.perf_counter()
            reference_ids, reference_bert, reference_segments, reference_normalized = (
                self._preprocess(reference_text, reference_language)
            )
            reference_seconds = time.perf_counter() - start_reference
            if not reference_ids:
                raise ValueError("reference transcript produced no phone IDs")

            start_target = time.perf_counter()
            target_ids, target_bert, target_segments, target_normalized = self._preprocess_official(text, language)
            target_seconds = time.perf_counter() - start_target
            total_seconds = time.perf_counter() - start_total
            if not target_ids:
                raise ValueError("target text produced no phone IDs")

            if reference_language == "zh":
                self._require_mandarin_bert(reference_bert, len(reference_ids), "reference")
            else:
                self._require_zero_bert(reference_bert, len(reference_ids), "reference")
            if language not in _MANDARIN_MODES:
                self._require_zero_bert(target_bert, len(target_ids), "target")
            elif target_bert.shape != (1024, len(target_ids)) or not torch.isfinite(target_bert).all():
                raise ValueError("target phones and BERT features do not align")
            reference_phones = np.asarray([reference_ids], dtype=np.int64)
            target_phones = np.asarray([target_ids], dtype=np.int64)
            phones = np.asarray([reference_ids + target_ids], dtype=np.int64)
            bert_tensor = torch.cat((reference_bert, target_bert), dim=1).unsqueeze(0)
            bert = bert_tensor.cpu().numpy().astype(np.float32, copy=False)
            total_phone_count = len(reference_ids) + len(target_ids)
            if phones.shape != (1, total_phone_count) or bert.shape != (1, 1024, total_phone_count):
                raise ValueError("combined phones/BERT do not align")
            if not np.array_equal(phones[:, : len(reference_ids)], reference_phones):
                raise ValueError("combined phone prefix differs from reference phones")
            if not np.array_equal(phones[:, len(reference_ids) :], target_phones):
                raise ValueError("combined phone suffix differs from target phones")
            if not torch.equal(bert_tensor[:, :, : len(reference_ids)], reference_bert.unsqueeze(0)):
                raise ValueError("combined BERT prefix differs from reference features")
            if not torch.equal(bert_tensor[:, :, len(reference_ids) :], target_bert.unsqueeze(0)):
                raise ValueError("combined BERT suffix differs from target features")

            output_dir = Path(output).expanduser().resolve()
            if output_dir.exists():
                if not output_dir.is_dir() or any(output_dir.iterdir()):
                    raise FileExistsError(f"frontend output must be a fresh empty directory: {output_dir}")
            else:
                output_dir.mkdir(parents=True, exist_ok=True)
            arrays = {
                "reference-phones.npy": reference_phones,
                "target-phones.npy": target_phones,
                "phones.npy": phones,
                "bert.npy": bert,
            }
            outputs: dict[str, dict[str, Any]] = {}
            for name, array in arrays.items():
                path = output_dir / name
                np.save(path, array, allow_pickle=False)
                outputs[name] = {
                    "sha256": self.helper._sha256_file(path),
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                }

            english_calls = None if english_counter is None else int(english_counter["calls"])
            metadata_started = time.perf_counter()
            metadata_languages = frozenset(_MODE_LANGUAGES[language] | {reference_language})
            source_hashes, python_dependencies, metadata_hit, metadata_scope = (
                self._reporting_metadata(metadata_languages)
            )
            metadata_lookup_seconds = time.perf_counter() - metadata_started

            report = {
                "schema": "aniflive-tts-macos-v5turbo-python-frontend-v1",
                "kind": "official_text_frontend_cpu_only",
                "frontend": "official Python TextPreprocessor/cleaner AST-extracted from pinned source",
                "frontend_native": False,
                "native_frontend_claim": False,
                "neural_tts_fallback": False,
                "device": "cpu",
                "language": language,
                "official_language": language,
                "reference_language": reference_language,
                "reference_official_language": "all_zh" if reference_language == "zh"
                    else self.helper.TARGET_LANGUAGES[reference_language],
                "text_split_method": "cut0",
                "text_version": _FRONTEND_VERSION,
                "reference_transcript_path": str(reference_path),
                "reference_transcript_sha256": hashlib.sha256(reference_bytes).hexdigest(),
                "reference_text": reference_text,
                "reference_normalized_text": reference_normalized,
                "reference_segments": reference_segments,
                "reference_phone_count": len(reference_ids),
                "reference_phone_ids": reference_ids,
                "reference_phone_symbols": [self.frontend["symbols"][i] for i in reference_ids],
                "reference_bert_all_zero": not bool(torch.count_nonzero(reference_bert).item()),
                "target_text": text,
                "target_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "target_normalized_text": target_normalized,
                "target_segments": target_segments,
                "target_phone_count": len(target_ids),
                "target_phone_ids": target_ids,
                "target_phone_symbols": [self.frontend["symbols"][i] for i in target_ids],
                "target_bert_all_zero": not bool(torch.count_nonzero(target_bert).item()),
                "native_mandarin_neural": "zh" == reference_language or bool(torch.count_nonzero(target_bert).item()),
                "combined_phone_count": total_phone_count,
                "combined_bert_shape": list(bert.shape),
                "bert_policy": ("native Metal TTS BERT hidden_states[-3], official phone expansion" if "zh" == reference_language or bool(torch.count_nonzero(target_bert).item())
                    else "official TextPreprocessor.get_bert_inf zero-feature branch for non-zh languages"),
                "frontend_preprocess_compute_seconds": {
                    "reference_text_preprocessor": reference_seconds,
                    "target_text_preprocessor": target_seconds,
                    "reference_plus_target_compute": total_seconds,
                    "scope": "excludes per-request asset validation, G2P module initialization, and output file I/O",
                },
                "english_g2p_en": {
                    "used": "en" in involved,
                    "backend": "official text/english.py en_G2p backed by g2p_en",
                    "learned_oov_fallback_calls": english_calls,
                    "cpu_frontend_seconds": reference_seconds + target_seconds
                    if english_calls is not None
                    else None,
                },
                "nltk_asset": nltk_metadata,
                "fasttext_asset": dict(self.fasttext_asset),
                "python_dependencies": python_dependencies,
                "frontend_metadata_cache": {
                    "enabled": self.metadata_cache_enabled,
                    "hit": metadata_hit,
                    "language_key": sorted(metadata_languages),
                    "source_snapshot_scope": metadata_scope,
                    "snapshot_initialization_seconds": self.metadata_cache_initialization_seconds
                    if self.metadata_cache_enabled else None,
                    "lookup_seconds": metadata_lookup_seconds,
                },
                "outputs": outputs,
                "source_files_sha256": source_hashes,
                "qualification_scope": "temporary Python frontend only; native frontend and HTTP performance are not claimed",
            }
            (output_dir / "fixture.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return report
        finally:
            self.helper._restore_english_fallback_counter(english_counter)
            self._request_lock.release()

    @staticmethod
    def _require_zero_bert(features: torch.Tensor, phone_count: int, label: str) -> None:
        if features.shape != (1024, phone_count) or not torch.isfinite(features).all():
            raise ValueError(f"{label} frontend features must be finite [1024,phones]")
        if torch.count_nonzero(features).item() != 0:
            raise ValueError(f"{label} uses a non-zero BERT path; this frontend supports zero-BERT languages only")

    @staticmethod
    def _require_mandarin_bert(features: torch.Tensor, phone_count: int, label: str) -> None:
        if features.shape != (1024, phone_count) or not torch.isfinite(features).all():
            raise ValueError(f"Mandarin {label} BERT features must be finite [1024,phones]")
        if torch.count_nonzero(features).item() == 0:
            raise ValueError(f"Mandarin {label} BERT features are all zero")

    def _validate_loaded_frontend_modules(self) -> None:
        expected_text_root = (self.gpt_sovits / "text").resolve()
        for name, module in tuple(sys.modules.items()):
            if name != "text" and not name.startswith("text."):
                continue
            origins: list[Path] = []
            module_file = getattr(module, "__file__", None)
            if module_file:
                origins.append(Path(module_file).resolve())
            module_path = getattr(module, "__path__", None)
            if module_path:
                origins.extend(Path(item).resolve() for item in module_path)
            if not origins or any(
                origin != expected_text_root and not origin.is_relative_to(expected_text_root)
                for origin in origins
            ):
                raise RuntimeError(
                    f"Refusing to reuse preloaded {name!r} from a different official source root"
                )

    def _load_helper(self):
        helper_name = "_aniflive_prepare_frontend_" + hashlib.sha256(
            str(self.script).encode("utf-8")
        ).hexdigest()[:12]
        loaded = sys.modules.get(helper_name)
        if loaded is not None:
            if Path(loaded.__file__).resolve() != self.script:
                raise RuntimeError("frontend helper module name is already bound to a different file")
            return loaded
        sys.path.insert(0, str(self.scripts_dir))
        extract_module = sys.modules.get("vocoder_fixture")
        if extract_module is not None and Path(extract_module.__file__).resolve() != (
            self.scripts_dir / "vocoder_fixture.py"
        ).resolve():
            raise RuntimeError("vocoder_fixture helper is loaded from a different source directory")
        spec = importlib.util.spec_from_file_location(helper_name, self.script)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load frontend helper {self.script}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[helper_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(helper_name, None)
            raise
        return module
