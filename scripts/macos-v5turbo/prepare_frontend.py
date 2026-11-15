from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from official_source_extract import extract


ROOT = Path(__file__).resolve().parents[2]
FRONTEND_VERSION = "v2"
TARGET_LANGUAGES = {
    "ja": "all_ja",
    "en": "en",
    "ko": "all_ko",
    "yue": "all_yue",
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _nltk_asset_record(data_root: Path) -> dict[str, Any]:
    lock = json.loads((ROOT / "scripts/shared_assets_lock.json").read_text(encoding="utf-8"))
    nltk_lock = lock["nltk"]
    blob_oids = {
        "averaged_perceptron_tagger_eng": "b792e19546a5711218a05f4029ef17c8d62cd839",
        "averaged_perceptron_tagger": "d5bfb6852aa5d288c8f33b3f0ef0fde343afa405",
        "cmudict": "48f1d64a431c474142969a793dfcffe3a2fc877b",
    }
    assets: dict[str, dict[str, Any]] = {}
    for package, package_lock in nltk_lock["packages"].items():
        local_path = data_root / "nltk_data" / package_lock["subdir"] / package
        if not local_path.is_dir():
            raise RuntimeError(f"Pinned NLTK asset is missing from the isolated data tree: {local_path}")
        actual_tree_sha256 = _tree_sha256(local_path)
        if actual_tree_sha256 != package_lock["tree_sha256"]:
            raise RuntimeError(f"Local NLTK asset tree hash mismatch: {package}")
        archive = data_root / "cache/nltk" / f"{package}.zip"
        if not archive.is_file() or _sha256_file(archive) != package_lock["zip_sha256"]:
            raise RuntimeError(f"Local NLTK asset archive is missing or hash-mismatched: {package}")
        nltk_archive = data_root / "nltk_data" / package_lock["subdir"] / f"{package}.zip"
        if not nltk_archive.is_file() or _sha256_file(nltk_archive) != package_lock["zip_sha256"]:
            raise RuntimeError(f"NLTK cannot use its pinned local archive: {package}")
        blob_oid = blob_oids.get(package)
        if blob_oid is None:
            raise RuntimeError(f"No pinned Git blob OID recorded for NLTK asset: {package}")
        assets[package] = {
            "local_path": str(local_path),
            "source_blob_oid": blob_oid,
            "source_blob_api": f"https://api.github.com/repos/nltk/nltk_data/git/blobs/{blob_oid}",
            "zip_sha256": package_lock["zip_sha256"],
            "tree_sha256": actual_tree_sha256,
            "validated": True,
        }
    return {
        "repository": nltk_lock["repository"],
        "revision": nltk_lock["revision"],
        "blocked_live_downloader_url": "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/index.xml",
        "blocked_live_downloader_reason": "NLTK downloader was blocked by URL security resolving to restricted IP 198.18.0.79",
        "assets": assets,
    }


def _ensure_nltk_data(data_root: Path):
    nltk_root = data_root / "nltk_data"
    asset = _nltk_asset_record(data_root)
    os.environ["NLTK_DATA"] = str(nltk_root)
    import nltk

    nltk.data.path[:] = [str(nltk_root)]
    for resource in (
        "taggers/averaged_perceptron_tagger_eng/",
        "taggers/averaged_perceptron_tagger/",
        "taggers/averaged_perceptron_tagger.zip",
        "corpora/cmudict/",
        "corpora/cmudict.zip/cmudict/",
    ):
        try:
            nltk.data.find(resource)
        except LookupError as error:
            raise RuntimeError(f"Validated pinned NLTK resource is not readable: {resource}") from error
    return nltk, nltk_root, asset


def _fasttext_asset_record(cache_root: Path) -> dict[str, Any]:
    lock = json.loads((ROOT / "scripts/shared_assets_lock.json").read_text(encoding="utf-8"))
    expected = lock["fasttext"]["sha256"]
    path = cache_root / "fast_langdetect" / "lid.176.bin"
    if not path.is_file():
        raise RuntimeError("Official LangSegmenter fastText model was not cached locally")
    actual = _sha256_file(path)
    if actual != expected:
        raise RuntimeError("Local fastText language model hash does not match shared_assets_lock.json")
    return {
        "path": str(path),
        "source_url": lock["fasttext"]["url"],
        "sha256": actual,
        "validated": True,
    }


def _load_official_preprocessor(official: Path, cache_root: Path):
    gpt_sovits = official.expanduser().resolve(strict=True) / "GPT_SoVITS"
    if not gpt_sovits.is_dir():
        raise ValueError(f"official source root is missing GPT_SoVITS: {official}")
    sys.path.insert(0, str(gpt_sovits))

    from text import cleaned_text_to_sequence, symbols2
    from text.LangSegmenter import LangSegmenter
    from text.cleaner import clean_text
    from text import japanese as japanese_module
    import pyopenjtalk

    # The pinned source keeps this user dictionary alongside its Japanese G2P.
    # Verify the built dictionary before the official frontend uses it.
    dictionary = Path(japanese_module.USERDIC_BIN_PATH)
    dictionary_source = Path(japanese_module.USERDIC_CSV_PATH)
    expected_md5 = hashlib.md5(dictionary_source.read_bytes()).hexdigest()
    if (
        not dictionary.is_file()
        or Path(japanese_module.USERDIC_HASH_PATH).read_text(encoding="utf-8").strip()
        != expected_md5
    ):
        raise ValueError("Official Japanese user dictionary is missing or does not match its source")
    pyopenjtalk.update_global_jtalk_with_user_dict(str(dictionary))

    # Language detection is lazy. Redirect its model download to the isolated
    # data tree before the first official LangSegmenter call.
    import fast_langdetect.infer as fast_langdetect_infer

    fast_cache = cache_root / "fast_langdetect"
    fast_cache.mkdir(parents=True, exist_ok=True)
    detector = fast_langdetect_infer._default_detector
    detector.config.cache_dir = str(fast_cache)

    segment_source = gpt_sovits / "TTS_infer_pack" / "text_segmentation_method.py"
    spec = importlib.util.spec_from_file_location("official_text_segmentation", segment_source)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the official text segmentation source")
    segmentation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(segmentation)

    namespace = {
        "torch": torch,
        "re": re,
        "threading": threading,
        "LangSegmenter": LangSegmenter,
        "clean_text": clean_text,
        "cleaned_text_to_sequence": cleaned_text_to_sequence,
        "splits": segmentation.splits,
        "get_seg_method": segmentation.get_method,
        "split_big_text": segmentation.split_big_text,
        "tqdm": tqdm,
        "punctuation": set(["!", "?", "…", ",", ".", "-"]),
        "i18n": lambda value: value,
    }
    preprocessor_path = gpt_sovits / "TTS_infer_pack" / "TextPreprocessor.py"
    extract(
        preprocessor_path,
        ["get_first", "merge_short_text_in_array", "TextPreprocessor"],
        namespace,
    )
    processor = namespace["TextPreprocessor"](None, None, torch.device("cpu"))
    return {
        "root": gpt_sovits,
        "processor": processor,
        "clean_text": clean_text,
        "cleaned_text_to_sequence": cleaned_text_to_sequence,
        "symbols": symbols2.symbols,
        "segmentation": segmentation,
        "preprocessor_path": preprocessor_path,
        "segment_source": segment_source,
        "dictionary": dictionary,
        "dictionary_source": dictionary_source,
        "dictionary_md5": expected_md5,
        "fast_cache": fast_cache,
    }


def _install_english_fallback_counter() -> dict[str, Any]:
    english = importlib.import_module("text.english")
    g2p = english._g2p
    original_predict = g2p.predict
    counter = {"calls": 0}

    def counted_predict(*args, **kwargs):
        counter["calls"] += 1
        return original_predict(*args, **kwargs)

    # Instrument only the call count; every OOV prediction delegates to the
    # original official g2p_en implementation with the same arguments/results.
    g2p.predict = counted_predict
    counter["module"] = english
    counter["g2p"] = g2p
    counter["original_predict"] = original_predict
    return counter


def _restore_english_fallback_counter(counter: dict[str, Any] | None) -> None:
    if counter is not None:
        counter["g2p"].predict = counter["original_predict"]


def _preprocess(
    processor,
    text: str,
    language: str,
    symbols: list[str],
) -> tuple[list[int], torch.Tensor, list[dict[str, Any]], str]:
    results = processor.preprocess(text, language, "cut0", FRONTEND_VERSION)
    if not results:
        raise ValueError(f"Official TextPreprocessor returned no segments for language {language}")
    phone_ids: list[int] = []
    bert_parts: list[torch.Tensor] = []
    segments: list[dict[str, Any]] = []
    normalized_text_parts: list[str] = []
    for index, result in enumerate(results):
        ids = [int(value) for value in result["phones"]]
        features = result["bert_features"].detach().to(device="cpu", dtype=torch.float32)
        if features.ndim != 2 or features.shape[0] != 1024 or features.shape[1] != len(ids):
            raise ValueError(
                f"Official segment {index} has misaligned phones/BERT shapes: "
                f"{len(ids)} phones, features {tuple(features.shape)}"
            )
        if not torch.isfinite(features).all():
            raise ValueError(f"Official segment {index} has non-finite BERT features")
        if torch.count_nonzero(features).item() != 0:
            raise ValueError(
                f"Expected the official non-Mandarin zero-BERT branch for {language}"
            )
        if any(token < 0 or token >= len(symbols) for token in ids):
            raise ValueError(f"Official segment {index} emitted a phone ID outside symbols2")
        text_norm = str(result["norm_text"])
        phone_ids.extend(ids)
        bert_parts.append(features)
        normalized_text_parts.append(text_norm)
        segments.append(
            {
                "index": index,
                "normalized_text": text_norm,
                "phone_ids": ids,
                "phone_symbols": [symbols[token] for token in ids],
                "phone_count": len(ids),
                "bert_shape": list(features.shape),
                "bert_all_zero": True,
            }
        )
    bert = torch.cat(bert_parts, dim=1)
    if bert.shape != (1024, len(phone_ids)):
        raise ValueError("Concatenated BERT features do not align with concatenated phones")
    return phone_ids, bert, segments, "".join(normalized_text_parts)


def _save_npy(path: Path, value: np.ndarray) -> dict[str, Any]:
    np.save(path, value, allow_pickle=False)
    return {"sha256": _sha256_file(path), "shape": list(value.shape), "dtype": str(value.dtype)}


def _source_files(root: Path, languages: set[str], preprocessor_path: Path, segment_path: Path):
    paths = [
        preprocessor_path,
        segment_path,
        root / "text/cleaner.py",
        root / "text/__init__.py",
        root / "text/symbols2.py",
        root / "text/chinese.py",
        root / "text/tone_sandhi.py",
        root / "text/LangSegmenter/langsegmenter.py",
    ]
    for language in languages:
        if language == "ja":
            paths.append(root / "text/japanese.py")
        elif language == "en":
            paths.extend([root / "text/english.py", root / "text/en_normalization/expend.py"])
        elif language == "ko":
            paths.append(root / "text/korean.py")
        elif language == "yue":
            paths.extend(
                [
                    root / "text/cantonese.py",
                    root / "text/zh_normalization/text_normlization.py",
                    root / "text/zh_normalization/num.py",
                ]
            )
    return paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare target and reference phones with the official "
            "CPU Python text frontend."
        )
    )
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--reference-transcript", type=Path, required=True)
    parser.add_argument("--reference-language", choices=tuple(TARGET_LANGUAGES), default="ja")
    parser.add_argument("--text", required=True)
    parser.add_argument("--language", choices=tuple(TARGET_LANGUAGES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    data_root = Path(os.environ.get("ANIFLIVE_TTS_MACOS_DATA_ROOT", ROOT / "data/macos-v5turbo")).expanduser().resolve()
    cache_root = data_root / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["XDG_CACHE_HOME"] = str(cache_root)
    _, nltk_root, nltk_asset = _ensure_nltk_data(data_root)
    english_counter = None
    english_frontend_used = "en" in {args.reference_language, args.language}

    try:
        official = _load_official_preprocessor(args.official, cache_root)
        if english_frontend_used:
            english_counter = _install_english_fallback_counter()
        reference_path = args.reference_transcript.expanduser().resolve(strict=True)
        reference_bytes = reference_path.read_bytes()
        reference_text = reference_bytes.decode("utf-8")
        target_text = args.text

        total_start = time.perf_counter()
        reference_start = time.perf_counter()
        reference_ids, reference_bert, reference_segments, reference_normalized = _preprocess(
            official["processor"],
            reference_text,
            TARGET_LANGUAGES[args.reference_language],
            official["symbols"],
        )
        reference_seconds = time.perf_counter() - reference_start
        if not reference_ids:
            raise ValueError("Reference transcript produced no phone IDs")

        target_start = time.perf_counter()
        target_ids, target_bert, target_segments, target_normalized = _preprocess(
            official["processor"],
            target_text,
            TARGET_LANGUAGES[args.language],
            official["symbols"],
        )
        target_seconds = time.perf_counter() - target_start
        total_seconds = time.perf_counter() - total_start
        fasttext_asset = _fasttext_asset_record(cache_root)

        if not target_ids:
            raise ValueError("Target text produced no phone IDs")
        if not torch.equal(reference_bert, torch.zeros_like(reference_bert)):
            raise ValueError("Expected official zero-BERT features for the non-Mandarin reference")
        if not torch.equal(target_bert, torch.zeros_like(target_bert)):
            raise ValueError("Expected zero BERT features for the non-Mandarin target language")

        ref_phone_names = [official["symbols"][token] for token in reference_ids]
        target_phone_names = [official["symbols"][token] for token in target_ids]
        reference_phones_array = np.asarray([reference_ids], dtype=np.int64)
        target_phones_array = np.asarray([target_ids], dtype=np.int64)
        target_bert_array = target_bert.unsqueeze(0).cpu().numpy().astype(np.float32, copy=False)
        combined_phones = np.asarray([reference_ids + target_ids], dtype=np.int64)
        combined_bert_tensor = torch.cat((reference_bert, target_bert), dim=1).unsqueeze(0)
        combined_bert_array = combined_bert_tensor.cpu().numpy().astype(np.float32, copy=False)
        if target_bert_array.shape != (1, 1024, len(target_ids)):
            raise ValueError("Target BERT array shape does not match its phone sequence")
        expected_combined_length = len(reference_ids) + len(target_ids)
        if combined_phones.shape != (1, expected_combined_length):
            raise ValueError("Combined phone array does not contain reference followed by target")
        if combined_bert_array.shape != (1, 1024, expected_combined_length):
            raise ValueError("Combined BERT array does not align with reference and target phones")
        if not np.array_equal(combined_phones[:, : len(reference_ids)], reference_phones_array):
            raise ValueError("Combined phone prefix differs from reference-phones.npy")
        if not np.array_equal(combined_phones[:, len(reference_ids) :], target_phones_array):
            raise ValueError("Combined phone suffix differs from target-phones.npy")
        if not torch.equal(combined_bert_tensor[:, :, : len(reference_ids)], reference_bert.unsqueeze(0)):
            raise ValueError("Combined BERT prefix differs from the reference features")
        if not torch.equal(combined_bert_tensor[:, :, len(reference_ids) :], target_bert.unsqueeze(0)):
            raise ValueError("Combined BERT suffix differs from target features")

        output_dir = args.output.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        arrays = {
            "reference-phones.npy": reference_phones_array,
            "target-phones.npy": target_phones_array,
            "phones.npy": combined_phones,
            "bert.npy": combined_bert_array,
        }
        output_records = {
            name: _save_npy(output_dir / name, value) for name, value in arrays.items()
        }
        source_paths = _source_files(
            official["root"],
            {args.reference_language, args.language},
            official["preprocessor_path"],
            official["segment_source"],
        )
        source_paths.append(official["dictionary_source"])
        source_hashes = {
            str(path.relative_to(official["root"].parent)): _sha256_file(path)
            for path in source_paths
        }
        report = {
            "schema": "aniflive-tts-macos-v5turbo-python-frontend-v1",
            "kind": "official_text_frontend_cpu_only",
            "frontend": "official Python TextPreprocessor/cleaner AST-extracted from pinned source",
            "frontend_native": False,
            "native_frontend_claim": False,
            "device": "cpu",
            "language": args.language,
            "official_language": TARGET_LANGUAGES[args.language],
            "reference_language": args.reference_language,
            "reference_official_language": TARGET_LANGUAGES[args.reference_language],
            "text_split_method": "cut0",
            "text_version": FRONTEND_VERSION,
            "reference_transcript_path": str(reference_path),
            "reference_transcript_sha256": _sha256_bytes(reference_bytes),
            "reference_text": reference_text,
            "reference_normalized_text": reference_normalized,
            "reference_segments": reference_segments,
            "reference_phone_count": len(reference_ids),
            "reference_phone_ids": reference_ids,
            "reference_phone_symbols": ref_phone_names,
            "reference_bert_all_zero": True,
            "target_text": target_text,
            "target_text_sha256": _sha256_bytes(target_text.encode("utf-8")),
            "target_normalized_text": target_normalized,
            "target_segments": target_segments,
            "target_phone_count": len(target_ids),
            "target_phone_ids": target_ids,
            "target_phone_symbols": target_phone_names,
            "target_bert_shape": list(target_bert_array.shape),
            "target_bert_all_zero": bool(np.count_nonzero(target_bert_array) == 0),
            "combined_phone_count": int(combined_phones.shape[1]),
            "combined_bert_shape": list(combined_bert_array.shape),
            "combined_bert_all_zero": bool(np.count_nonzero(combined_bert_array) == 0),
            "bert_policy": "official TextPreprocessor.get_bert_inf returns zero [1024,phones] for all non-zh languages",
            "frontend_cpu_timing_seconds": {
                "reference": reference_seconds,
                "target": target_seconds,
                "total": total_seconds,
            },
            "english_g2p_en": {
                "used": english_frontend_used,
                "backend": "official text/english.py en_G2p backed by g2p_en",
                "learned_oov_fallback_calls": None if english_counter is None else english_counter["calls"],
                "cpu_frontend_seconds": (
                    reference_seconds + target_seconds if english_frontend_used else None
                ),
                "note": "The official English G2P frontend runs on CPU; any learned OOV fallback cost is included in target timing.",
            },
            "nltk_data_directory": str(nltk_root),
            "nltk_asset": {
                **nltk_asset,
                "used": bool({args.reference_language, args.language} & {"en", "ko"}),
            },
            "fast_langdetect_cache_directory": str(official["fast_cache"]),
            "fasttext_asset": fasttext_asset,
            "python_dependencies": {
                name: _package_version(name)
                for name in (
                    "torch",
                    "transformers",
                    "pyopenjtalk",
                    "split-lang",
                    "fast-langdetect",
                    "nltk",
                    "g2p-en",
                    "wordsegment",
                    "cn2an",
                    "pypinyin",
                    "tojyutping",
                    "g2pk2",
                    "ko-pron",
                    "python-mecab-ko",
                    "jieba-fast",
                )
            },
            "outputs": output_records,
            "source_files_sha256": source_hashes,
            "qualification_scope": "text/frontend arrays for one reference and one target; native frontend and HTTP performance are not claimed",
        }
        (output_dir / "fixture.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "output": str(output_dir),
                    "language": args.language,
                    "reference_phones": len(reference_ids),
                    "target_phones": len(target_ids),
                    "frontend_cpu_seconds": report["frontend_cpu_timing_seconds"],
                },
                sort_keys=True,
            )
        )
        return 0
    finally:
        _restore_english_fallback_counter(english_counter)


if __name__ == "__main__":
    raise SystemExit(main())
