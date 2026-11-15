#!/usr/bin/env python3
"""CPU-only public-source preflight for the v1.5 release candidate.

This tool writes evidence only. Pending platform or owner review always keeps
overall release readiness false.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urldefrag, urlsplit


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_RELATIVE = "benchmarks/V1_5_BENCHMARK_SUMMARY.json"

README_PATHS = ("README.md", "README_ZH_HK.md", "README_ZH_CN.md")
BENCHMARK_DOCUMENT_PATH = "docs/BENCHMARKS.md"
PUBLIC_DOCUMENT_PATHS = (
    *README_PATHS,
    BENCHMARK_DOCUMENT_PATH,
    "RELEASE_NOTES_v1.5.0.md",
    "LICENSING.md",
    "THIRD_PARTY_NOTICES.md",
    "docs/V1_5_PLATFORM_MATRIX.md",
    "docs/V1_5_V5_BUILD.md",
    "docs/API_EN.md",
    "docs/API_ZH.md",
    "docs/SDK_EN.md",
    "docs/SDK_ZH.md",
    "docs/DEPLOYMENT_EN.md",
    "docs/DEPLOYMENT_ZH.md",
    "docs/INSTALLATION_EN.md",
    "docs/INSTALLATION_ZH.md",
    "docs/V1_5_VOICE_API.md",
    "assets/THIRD_PARTY_MEDIA.md",
    "webui/media/README.md",
)
LICENSE_PATHS = (
    "LICENSE",
    "LICENSING.md",
    "THIRD_PARTY_NOTICES.md",
    "assets/THIRD_PARTY_MEDIA.md",
    "licenses/FRONTEND-DATA.json",
    "minimal_inference/LICENSE",
    "minimal_inference/GPT_SoVITS/text/g2p_en/LICENSE.txt",
    "licenses/GPT-SoVITS-MIT.txt",
    "licenses/CMUDICT-NOTICE.txt",
    "licenses/IPADIC-NAIST-ICOT.txt",
)

# Every redistributable UI binary has an explicit purpose, evidence file, and
# immutable hash. An unlisted image/audio/video payload is a blocking finding.
UI_MEDIA_ALLOWLIST = {
    "assets/aniflive-tts-studio-demo.gif": {
        "sha256": "554c9580dadd09c0881583c7022ce016a184f3eef7a69d75b3de6ff46ae4df0e",
        "basis": "owner-supplied Studio screen recording; owner authorized public inclusion",
        "evidence": "assets/THIRD_PARTY_MEDIA.md",
    },
    "assets/roxy-v2proplus-cantonese-demo.mp4": {
        "sha256": "0b752658b0075b5e3db0b374be0e61bf77915e4703d45362f0321630d63ea44d",
        "basis": "existing public demo; owner explicitly instructed that this file remain",
        "evidence": "assets/THIRD_PARTY_MEDIA.md",
    },
    "webui/media/voice-workstation-poster.jpg": {
        "sha256": "01e6977d8e325ce9572ebdbee82f6f6c22c1abeeb162dad42f196ee37706f535",
        "basis": "poster derived from the Mixkit-licensed workstation background",
        "evidence": "assets/THIRD_PARTY_MEDIA.md",
    },
    "webui/pwa/studio-icon-192.png": {
        "sha256": "7038ae5a400dea294d254f492e297965230183eb5dfbd612ad9b0661d94b8ba6",
        "basis": "first-party WebUI PWA icon",
        "evidence": "assets/THIRD_PARTY_MEDIA.md",
    },
    "webui/pwa/studio-icon-512.png": {
        "sha256": "8a4a7e8f33d0072e3abf916c67b00a61da69ce66d033943aa104385fd84dc1cd",
        "basis": "first-party WebUI PWA icon",
        "evidence": "assets/THIRD_PARTY_MEDIA.md",
    },
    "webui/media/voice-workstation-background.mp4": {
        "sha256": "26de26f0a6422688505684e7afbab1b13b3d68565b9156d6a9becf4fcd2c3eb4",
        "basis": "Mixkit Stock Video Free License",
        "evidence": "assets/THIRD_PARTY_MEDIA.md",
    },
}
MEDIA_SUFFIXES = {".gif", ".png", ".jpg", ".jpeg", ".webp", ".mp4", ".mov", ".webm"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".pcm", ".raw"}
WEIGHT_SUFFIXES = {
    ".bin", ".ckpt", ".pt", ".pth", ".safetensors", ".gguf", ".engine",
    ".plan", ".trt", ".weights", ".model", ".npy", ".npz", ".pb",
}
WEIGHT_COMPOUND_SUFFIXES = {".onnx.data"}
ONNX_TOPOLOGIES = {
    "scripts/v5_graph/ar-topology.onnx": "74a9f406461c4801615bd88cb426a7887a478d6a2a5ffde27a01a19b9119be1c",
    "scripts/v5_graph/dit-topology.onnx": "312d05fad248cda85b764e7f423ccc8043e75ec1a8bcbc7a4586849a13c37d51",
    "scripts/v5_graph/encoder-topology.onnx": "43f3ad50e725b915e5d568e9b4dc57ec53357dbf4827dd62574b5e9fcf40c129",
    "scripts/v5_graph/vocoder-topology.onnx": "0d686f635432f5e4ea2a507c113c64c52f206e2279d412178aa432922c227284",
}
DIT_PADDING_SENTINEL_SHA256 = "0d1abbe3b9da7a48d463edb0a844f3a102dcf7fdea35f9c771d885027b31b322"
LICENSED_FRONTEND_DATA = {
    "minimal_inference/GPT_SoVITS/text/g2p_en/checkpoint20.npz": {
        "sha256": "b8af35e4596d8dd5836dfd3fe9b2ba4f97b9c311efe8879544cbcfcbd566d8c6",
        "purpose": "generic English grapheme-to-phoneme frontend model data",
        "license": "Apache-2.0",
        "license_file": "minimal_inference/GPT_SoVITS/text/g2p_en/LICENSE.txt",
        "notice_file": "THIRD_PARTY_NOTICES.md",
    },
}
UPSTREAM_FRONTEND_DICTIONARIES = {
    "minimal_inference/GPT_SoVITS/text/engdict_cache.pickle": {
        "sha256": "9bff9393f4b192d873a11335efc8f124771087b6dc847d34fd240c2846889d2b",
        "upstream_blob": "935da5b95f403f02968c192ef3885ed3298feb74",
        "license_context": "g2p-en Apache-2.0 and CMU Pronouncing Dictionary notice",
    },
    "minimal_inference/GPT_SoVITS/text/namedict_cache.pickle": {
        "sha256": "559552094c4a6e995213e3fa586330e078ef8cb3a7a95a3109e945111cd2bfc1",
        "upstream_blob": "a9a9dc3986ad88d93685c63643248e6234d5c011",
        "license_context": "g2p-en Apache-2.0 and CMU Pronouncing Dictionary notice",
    },
    "minimal_inference/GPT_SoVITS/text/g2pw/polyphonic.pickle": {
        "sha256": "f425246160a32c578557cd3151cd0bb97f5f44c3aaf65e718dd2c3213c04fb4b",
        "upstream_blob": "fbe46ce1dcc498e4cc508f5a68e77bee3428a393",
        "license_context": "pinned upstream multilingual frontend dictionary data",
    },
}
JAPANESE_COMPILED_DICTIONARY = {
    "path": "minimal_inference/GPT_SoVITS/text/ja_userdic/user.dict",
    "sha256": "b44817ce96e24be7bcfdd009d834b5237fe044dc9ed5f2f9709f71da9d506fed",
    "source_csv": "minimal_inference/GPT_SoVITS/text/ja_userdic/userdict.csv",
    "source_csv_sha256": "d857e443ee48d9641096816a98996669602895411e4330d7d91d1dbe1103389f",
    "source_blob": "2be580b461185ce18c9000d9ec6c8032623e17fd",
    "license_context": "Japanese dictionary covered by the IPADIC NAIST/ICOT notice",
}
FRONTEND_DATA_MANIFEST = "licenses/FRONTEND-DATA.json"
FRONTEND_DATA_REVISION = "6916eb2e9c227aeb28468ce6fa86ad5b17ead532"
PRIVATE_PUBLIC_TERMS = {
    "first_party_tts_framing": re.compile(r"\bfirst[- ]party TTS\b|第一方 TTS", re.IGNORECASE),
    "cantonese_first_class": re.compile(
        r"first[- ]class Cantonese|(?:粵語|廣東話|粤语|广东话)\s*(?:優先|优先)",
        re.IGNORECASE,
    ),
}
_private_terms_file = os.environ.get("ANIFLIVE_TTS_PRIVATE_TERMS_FILE")
if _private_terms_file:
    _private_terms = json.loads(Path(_private_terms_file).read_text(encoding="utf-8"))
    if not isinstance(_private_terms, list) or not all(
        isinstance(term, str) and term for term in _private_terms
    ):
        raise ValueError("Private term configuration must be a list of non-empty strings")
    PRIVATE_PUBLIC_TERMS.update(
        {f"private_term_{index}": re.compile(re.escape(term), re.IGNORECASE)
         for index, term in enumerate(_private_terms)}
    )
MARKDOWN_LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
HTML_MEDIA_RE = re.compile(
    r"""<(?:img|source|video)\b[^>]*\bsrc\s*=\s*["']([^"']+)["']""",
    re.IGNORECASE,
)
SECRET_PATTERNS = {
    "private_key_block": re.compile(
        r"-----BEGIN (?:RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----"
        r"[\s\S]{40,}?-----END (?:RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----",
        re.IGNORECASE,
    ),
    "aws_access_key": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "github_token": re.compile(
        r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b"
    ),
    "service_secret": re.compile(r"\bsk-(?:live|proj|ant)-[A-Za-z0-9_-]{20,}\b"),
    "long_secret_assignment": re.compile(
        r"""(?im)\b(?:api[_-]?key|access[_-]?token|client[_-]?secret|private[_-]?key)"""
        r"""\s*[:=]\s*["']?(?!\$\{|<|your_|example|changeme)[A-Za-z0-9_+/=-]{24,}"""
    ),
}
HOST_PATH_PATTERNS = (
    re.compile(
        r"(?i)\b[A-Z]:[\\/](?:Users|Work|Win|Documents and Settings)"
        r"[\\/][^\s\"'<>|]+"
    ),
    re.compile(r"(?i)(?<![\w.])/(?:Users|home)/[A-Za-z0-9_.-]+(?:/[^\s\"'<>|]*)?"),
)


def _finding(code: str, path: str | None = None, **details: object) -> dict[str, object]:
    result: dict[str, object] = {"code": code, "severity": "error"}
    if path is not None:
        result["path"] = path
    result.update(details)
    return result


def scan_text_content(relative_path: str, text: str) -> list[dict[str, object]]:
    """Find secrets/host paths without copying matched values into evidence."""
    findings: list[dict[str, object]] = []
    if _private_terms_file:
        for index, term in enumerate(_private_terms):
            if term.casefold() in text.casefold():
                findings.append(_finding(f"private_term_{index}", relative_path, redacted=True))
    for code, pattern in SECRET_PATTERNS.items():
        line_numbers = sorted(
            {text.count("\n", 0, match.start()) + 1 for match in pattern.finditer(text)}
        )
        if line_numbers:
            findings.append(
                _finding(code, relative_path, redacted=True, line_numbers=line_numbers)
            )
    for pattern in HOST_PATH_PATTERNS:
        line_numbers = sorted(
            {text.count("\n", 0, match.start()) + 1 for match in pattern.finditer(text)}
        )
        if line_numbers:
            findings.append(
                _finding(
                    "machine_specific_absolute_path",
                    relative_path,
                    redacted=True,
                    line_numbers=line_numbers,
                )
            )
            break
    return findings


def _text(data: bytes) -> str | None:
    if b"\x00" in data[:8192]:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def inspect_payload(relative_path: str, data: bytes) -> dict[str, object] | None:
    """Classify private audio/weights/engines and unreviewed binary payloads."""
    relative_path = relative_path.replace("\\", "/")
    lowered = relative_path.casefold()
    suffix = PurePosixPath(relative_path).suffix.casefold()
    if any(lowered.endswith(item) for item in WEIGHT_COMPOUND_SUFFIXES):
        return _finding("model_weight_payload", relative_path)
    if suffix in AUDIO_SUFFIXES:
        return _finding("audio_payload", relative_path)
    if suffix in WEIGHT_SUFFIXES:
        return _finding("model_or_engine_payload", relative_path)
    if suffix == ".onnx" and relative_path not in ONNX_TOPOLOGIES:
        return _finding("unallowlisted_onnx_payload", relative_path)
    if suffix in MEDIA_SUFFIXES and relative_path not in UI_MEDIA_ALLOWLIST:
        return _finding("unallowlisted_ui_media", relative_path)
    if data.startswith(b"RIFF"):
        if data[8:12] == b"WAVE":
            return _finding("audio_signature", relative_path)
        if data[8:12] == b"WEBP" and relative_path not in UI_MEDIA_ALLOWLIST:
            return _finding("unallowlisted_ui_media_signature", relative_path)
    if data.startswith((b"fLaC", b"OggS", b"ID3", b"\x1a\x45\xdf\xa3")):
        return _finding("audio_or_video_signature", relative_path)
    if len(data) >= 12 and data[4:8] == b"ftyp" and relative_path not in UI_MEDIA_ALLOWLIST:
        return _finding("unallowlisted_video_signature", relative_path)
    if b"\x00" in data[:8192] and suffix != ".onnx":
        return _finding("unreviewed_binary_payload", relative_path)
    return None


def _git_source_paths(root: Path) -> tuple[list[str], list[dict[str, object]]]:
    try:
        result = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={root}",
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "-z",
            ],
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError:
        return [], [_finding("git_inventory_unavailable")]
    if result.returncode:
        return [], [_finding("git_inventory_failed")]
    paths = {
        item.decode("utf-8", errors="surrogateescape")
        for item in result.stdout.split(b"\x00")
        if item
    }
    safe: list[str] = []
    findings: list[dict[str, object]] = []
    for item in sorted(paths):
        relative = PurePosixPath(item)
        if relative.is_absolute() or ".." in relative.parts:
            findings.append(_finding("unsafe_release_path"))
            continue
        full = root.joinpath(*relative.parts)
        if full.is_file():
            safe.append(relative.as_posix())
    return safe, findings


def _media_evidence_ok(root: Path, relative_path: str) -> bool:
    entry = UI_MEDIA_ALLOWLIST.get(relative_path)
    if entry is None:
        return False
    evidence = root / str(entry["evidence"])
    if not evidence.is_file():
        return False
    notice = evidence.read_text(encoding="utf-8", errors="replace").casefold()
    if relative_path.casefold() not in notice:
        return False
    if relative_path == "assets/aniflive-tts-studio-demo.gif":
        return "owner-supplied" in notice and "authorized its public inclusion" in notice
    if relative_path == "assets/roxy-v2proplus-cantonese-demo.mp4":
        return "owner explicitly instructed" in notice and "this demo remain" in notice
    if relative_path == "webui/media/voice-workstation-poster.jpg":
        media_readme = root / "webui/media/README.md"
        return (
            "mixkit stock video free license" in notice
            and media_readme.is_file()
            and "derived poster sha-256" in media_readme.read_text(
                encoding="utf-8", errors="replace"
            ).casefold()
        )
    if relative_path.startswith("webui/pwa/"):
        return "first-party application icons" in notice
    if relative_path == "webui/media/voice-workstation-background.mp4":
        return "mixkit stock video free license" in notice
    return False


def _licensed_frontend_data_ok(root: Path, relative_path: str, actual_hash: str) -> bool:
    entry = LICENSED_FRONTEND_DATA.get(relative_path)
    if entry is None or actual_hash != entry["sha256"]:
        return False
    manifest = _read_frontend_data_manifest(root)
    model_entry = manifest.get("english_g2p_parameters") if manifest else None
    if not isinstance(model_entry, dict) or not (
        model_entry.get("file") == relative_path
        and model_entry.get("sha256") == actual_hash
        and model_entry.get("license") == entry["license"]
        and model_entry.get("license_file") == entry["license_file"]
    ):
        return False
    license_path = root / str(entry["license_file"])
    notice_path = root / str(entry["notice_file"])
    if not license_path.is_file() or not notice_path.is_file():
        return False
    license_text = license_path.read_text(encoding="utf-8", errors="replace")
    notice_text = notice_path.read_text(encoding="utf-8", errors="replace")
    return (
        "Apache License" in license_text
        and "Version 2.0" in license_text
        and "g2p-en 2.1.0" in notice_text
        and "Apache-2.0" in notice_text
    )


def _read_frontend_data_manifest(root: Path) -> dict[str, object] | None:
    path = root / FRONTEND_DATA_MANIFEST
    if not path.is_file():
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != "aniflive-frontend-data-provenance-v1"
        or manifest.get("upstream_revision") != "6916eb2e9c227aeb28468ce6fa86ad5b17ead532"
    ):
        return None
    return manifest


def _frontend_dictionary_evidence(
    root: Path,
    relative_path: str,
    actual_hash: str,
    manifest: dict[str, object] | None,
) -> dict[str, object] | None:
    if manifest is None:
        return None
    notices_path = root / "THIRD_PARTY_NOTICES.md"
    if not notices_path.is_file():
        return None
    notices = notices_path.read_text(encoding="utf-8", errors="replace")
    upstream_files = manifest.get("upstream_exact_files", [])
    upstream_row = next(
        (
            row for row in upstream_files
            if isinstance(row, dict) and row.get("file") == relative_path
        ),
        None,
    ) if isinstance(upstream_files, list) else None
    entry = UPSTREAM_FRONTEND_DICTIONARIES.get(relative_path)
    if entry is not None:
        if not upstream_row or not (
            actual_hash == entry["sha256"]
            and upstream_row.get("sha256") == actual_hash
            and upstream_row.get("upstream_blob") == entry["upstream_blob"]
            and manifest.get("upstream_license", "").startswith("Apache-2.0")
        ):
            return None
        if relative_path.endswith(("engdict_cache.pickle", "namedict_cache.pickle")):
            license_path = root / "minimal_inference/GPT_SoVITS/text/g2p_en/LICENSE.txt"
            cmu_notice = root / "licenses/CMUDICT-NOTICE.txt"
            if not (
                license_path.is_file()
                and "Apache License" in license_path.read_text(encoding="utf-8", errors="replace")
                and cmu_notice.is_file()
                and "CMU Pronouncing Dictionary" in notices
            ):
                return None
        elif not (
            (root / "minimal_inference/LICENSE").is_file()
            and "GPT-SoVITS | MIT" in notices
            and "GPT-SoVITS minimal inference | Apache-2.0" in notices
        ):
            return None
        return {
            "path": relative_path,
            "sha256": actual_hash,
            "classification": "upstream_frontend_dictionary_cache",
            "upstream_revision": manifest["upstream_revision"],
            "upstream_blob": upstream_row["upstream_blob"],
            "license_context": entry["license_context"],
            "evidence": FRONTEND_DATA_MANIFEST,
        }
    if relative_path == JAPANESE_COMPILED_DICTIONARY["path"]:
        entry = JAPANESE_COMPILED_DICTIONARY
        japanese = manifest.get("japanese_compiled_dictionary")
        source_row = next(
            (
                row for row in upstream_files
                if isinstance(row, dict) and row.get("file") == entry["source_csv"]
            ),
            None,
        ) if isinstance(upstream_files, list) else None
        source_path = root / entry["source_csv"]
        if not (
            isinstance(japanese, dict)
            and actual_hash == entry["sha256"]
            and japanese.get("sha256") == actual_hash
            and japanese.get("rebuild_byte_equal") is True
            and japanese.get("source_sha256") == entry["source_csv_sha256"]
            and source_path.is_file()
            and hashlib.sha256(source_path.read_bytes()).hexdigest() == entry["source_csv_sha256"]
            and source_row
            and source_row.get("sha256") == entry["source_csv_sha256"]
            and source_row.get("upstream_blob") == entry["source_blob"]
            and (root / "licenses/IPADIC-NAIST-ICOT.txt").is_file()
            and "NAIST / ICOT terms" in notices
        ):
            return None
        return {
            "path": relative_path,
            "sha256": actual_hash,
            "classification": "compiled_frontend_dictionary",
            "source_csv": entry["source_csv"],
            "source_csv_sha256": entry["source_csv_sha256"],
            "upstream_blob": entry["source_blob"],
            "byte_equal_rebuild_verified": True,
            "license_context": entry["license_context"],
            "evidence": FRONTEND_DATA_MANIFEST,
        }
    return None


def _manifest_source_paths(
    root: Path, manifest_path: Path
) -> tuple[list[str], list[dict[str, object]], dict[str, object]]:
    if not manifest_path.is_file():
        return [], [_finding("source_manifest_missing")], {}
    try:
        raw = manifest_path.read_bytes()
        manifest = json.loads(raw.decode("utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return [], [_finding("source_manifest_invalid")], {}
    paths_value = manifest.get("paths") if isinstance(manifest, dict) else None
    if not isinstance(paths_value, list) or any(not isinstance(item, str) for item in paths_value):
        return [], [_finding("source_manifest_invalid")], {}
    paths: list[str] = []
    findings: list[dict[str, object]] = []
    for item in paths_value:
        relative = PurePosixPath(item.replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts:
            findings.append(_finding("unsafe_release_path"))
            continue
        if not root.joinpath(*relative.parts).is_file():
            findings.append(_finding("manifest_source_file_missing", relative.as_posix()))
            continue
        paths.append(relative.as_posix())
    details = {
        "source_commit": str(manifest.get("source_commit", "")),
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "file_count": len(paths),
    }
    return sorted(set(paths)), findings, details


def scan_source_tree(root: Path, source_manifest: Path | None = None) -> dict[str, object]:
    if source_manifest is None:
        paths, findings = _git_source_paths(root)
        manifest_details: dict[str, object] = {}
    else:
        paths, findings, manifest_details = _manifest_source_paths(root, source_manifest)
    allowed_media: list[dict[str, object]] = []
    licensed_frontend_data: list[dict[str, object]] = []
    frontend_dictionary_data: list[dict[str, object]] = []
    frontend_manifest = _read_frontend_data_manifest(root)
    if frontend_manifest is None:
        findings.append(_finding("frontend_data_manifest_missing_or_invalid", FRONTEND_DATA_MANIFEST))
    discovered_topologies: set[str] = set()
    for relative_path in paths:
        full = root.joinpath(*PurePosixPath(relative_path).parts)
        try:
            size = full.stat().st_size
            if size > 128 * 1024 * 1024:
                findings.append(_finding("oversized_unreviewed_file", relative_path, bytes=size))
                continue
            data = full.read_bytes()
        except OSError:
            findings.append(_finding("source_file_unreadable", relative_path))
            continue
        suffix = PurePosixPath(relative_path).suffix.casefold()
        if suffix == ".onnx" and relative_path in ONNX_TOPOLOGIES:
            discovered_topologies.add(relative_path)
        if suffix in MEDIA_SUFFIXES:
            entry = UI_MEDIA_ALLOWLIST.get(relative_path)
            if entry is None:
                findings.append(_finding("unallowlisted_ui_media", relative_path))
            else:
                actual_hash = hashlib.sha256(data).hexdigest()
                if actual_hash != entry["sha256"]:
                    findings.append(_finding("allowlisted_media_hash_mismatch", relative_path))
                elif not _media_evidence_ok(root, relative_path):
                    findings.append(_finding("media_license_evidence_missing", relative_path))
                else:
                    allowed_media.append(
                        {
                            "path": relative_path,
                            "sha256": actual_hash,
                            "basis": entry["basis"],
                            "evidence": entry["evidence"],
                        }
                    )
        licensed_data_ok = False
        licensed_entry = LICENSED_FRONTEND_DATA.get(relative_path)
        if licensed_entry is not None:
            actual_hash = hashlib.sha256(data).hexdigest()
            licensed_data_ok = _licensed_frontend_data_ok(root, relative_path, actual_hash)
            if licensed_data_ok:
                licensed_frontend_data.append(
                    {
                        "path": relative_path,
                        "sha256": actual_hash,
                        "purpose": licensed_entry["purpose"],
                        "license": licensed_entry["license"],
                        "license_file": licensed_entry["license_file"],
                        "notice_file": licensed_entry["notice_file"],
                    }
                )
            else:
                findings.append(
                    _finding("licensed_frontend_data_hash_or_license_mismatch", relative_path)
                )
        dictionary_evidence = None
        if (
            relative_path in UPSTREAM_FRONTEND_DICTIONARIES
            or relative_path == JAPANESE_COMPILED_DICTIONARY["path"]
        ):
            actual_hash = hashlib.sha256(data).hexdigest()
            dictionary_evidence = _frontend_dictionary_evidence(
                root, relative_path, actual_hash, frontend_manifest
            )
            if dictionary_evidence is None:
                findings.append(
                    _finding("frontend_dictionary_provenance_missing_or_mismatched", relative_path)
                )
            else:
                frontend_dictionary_data.append(dictionary_evidence)
        payload_finding = inspect_payload(relative_path, data)
        if payload_finding:
            is_allowlisted_media = suffix in MEDIA_SUFFIXES and relative_path in UI_MEDIA_ALLOWLIST
            is_allowlisted_topology_binary = suffix == ".onnx" and relative_path in ONNX_TOPOLOGIES
            is_licensed_frontend_data = licensed_data_ok and relative_path in LICENSED_FRONTEND_DATA
            is_verified_frontend_dictionary = dictionary_evidence is not None
            duplicate_media_finding = (
                payload_finding["code"] == "unallowlisted_ui_media"
                and suffix in MEDIA_SUFFIXES
            )
            if not (
                is_allowlisted_media
                or is_allowlisted_topology_binary
                or is_licensed_frontend_data
                or is_verified_frontend_dictionary
                or duplicate_media_finding
            ):
                payload_finding["bytes"] = size
                payload_finding["sha256"] = hashlib.sha256(data).hexdigest()
                findings.append(payload_finding)
        decoded = _text(data)
        if decoded is not None:
            findings.extend(scan_text_content(relative_path, decoded))
    for relative_path in sorted(set(ONNX_TOPOLOGIES) - discovered_topologies):
        findings.append(_finding("required_weightless_topology_missing", relative_path))
    return {
        "status": "failed" if findings else "passed",
        "source_file_count": len(paths),
        "allowed_ui_media": allowed_media,
        "licensed_frontend_model_data": licensed_frontend_data,
        "licensed_frontend_dictionary_data": frontend_dictionary_data,
        "findings": findings,
        "topology_paths_found": sorted(discovered_topologies),
        "source_manifest": manifest_details,
    }


def check_markdown_links(root: Path, relative_docs: tuple[str, ...] | list[str]) -> dict[str, object]:
    findings: list[dict[str, object]] = []
    checked = 0
    root_resolved = root.resolve()
    for relative_doc in relative_docs:
        source = root / relative_doc
        if not source.is_file():
            findings.append(_finding("current_public_document_missing", relative_doc))
            continue
        text = source.read_text(encoding="utf-8", errors="replace")
        targets = [match.group(1) for match in MARKDOWN_LINK_RE.finditer(text)]
        targets.extend(match.group(1) for match in HTML_MEDIA_RE.finditer(text))
        for raw_target in targets:
            target = raw_target.strip().split(None, 1)[0].strip("<>")
            if not target or target.startswith("#"):
                continue
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc:
                continue
            local_path = unquote(urldefrag(parsed.path)[0])
            destination = source if not local_path else (source.parent / local_path).resolve()
            try:
                destination.relative_to(root_resolved)
            except ValueError:
                findings.append(_finding("local_link_escapes_release_tree", relative_doc))
                continue
            checked += 1
            if not destination.exists():
                findings.append(
                    _finding("local_link_target_missing", relative_doc, target=local_path)
                )
    return {
        "status": "failed" if findings else "passed",
        "local_links_checked": checked,
        "findings": findings,
    }


def check_current_public_terms(root: Path) -> dict[str, object]:
    findings: list[dict[str, object]] = []
    for relative_path in PUBLIC_DOCUMENT_PATHS:
        path = root / relative_path
        if not path.is_file():
            findings.append(_finding("current_public_document_missing", relative_path))
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for code, pattern in PRIVATE_PUBLIC_TERMS.items():
            if pattern.search(text):
                findings.append(_finding(code, relative_path, redacted=True))
    return {"status": "failed" if findings else "passed", "findings": findings}


def check_license_files(root: Path) -> dict[str, object]:
    missing = [relative for relative in LICENSE_PATHS if not (root / relative).is_file()]
    findings = [_finding("license_or_notice_missing", relative) for relative in missing]
    license_path = root / "LICENSE"
    if license_path.is_file():
        text = license_path.read_text(encoding="utf-8", errors="replace")
        if "PolyForm Noncommercial License 1.0.0" not in text:
            findings.append(_finding("unexpected_primary_license", "LICENSE"))
    if _read_frontend_data_manifest(root) is None:
        findings.append(_finding("frontend_data_manifest_invalid", FRONTEND_DATA_MANIFEST))
    return {
        "status": "failed" if findings else "passed",
        "license_files_present": len(LICENSE_PATHS) - len(missing),
        "findings": findings,
    }


def _round_decimal(value: object, quantum: str) -> Decimal:
    return Decimal(str(value)).quantize(Decimal(quantum), rounding=ROUND_HALF_UP)


def _cell_numbers(cell: str) -> list[Decimal]:
    cleaned = cell.replace("**", "").replace(",", "")
    matches = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", cleaned)
    return [Decimal(value) for value in matches]


def _platform_key(value: object) -> str | None:
    normalized = re.sub(r"[^a-z0-9]+", "", str(value).casefold())
    if normalized.startswith("macos"):
        return "macos"
    if normalized.startswith("windows"):
        return "windows"
    return None


def _has_per_configuration_count(text: str, measured_total: int) -> bool:
    normalized = re.sub(r"\s+", " ", text.casefold().replace(",", "").replace("，", ""))
    count = re.escape(str(measured_total))
    markers = (
        rf"(?<!\d){count}(?!\d)\s+total\s+per\s+configuration\b",
        rf"每個配置\s*合共\s*{count}\s*次",
        rf"每个配置\s*共\s*{count}\s*次",
    )
    return any(re.search(marker, normalized) for marker in markers)


def _benchmark_rows(readme: str) -> dict[tuple[str, str], list[list[str]]]:
    rows: dict[tuple[str, str], list[list[str]]] = {}
    for line in readme.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        model = re.sub(r"[^a-z0-9]+", "", cells[0].casefold()) if cells else ""
        if len(cells) < 7 or "v5turbo" not in model:
            continue
        platform = "macos" if any(token in model for token in ("macos", "m5max", "macstudio")) else "windows"
        condition = cells[1].casefold()
        if "keep-alive" in condition or "keepalive" in condition or "持續連線" in condition or "持久连接" in condition:
            rows.setdefault((platform, "keepalive_pcm"), []).append(cells)
        elif "new connection" in condition or "新連線" in condition or "新连接" in condition:
            rows.setdefault((platform, "new_connection_pcm"), []).append(cells)
    return rows


def _comparison_benchmark_rows(
    readme: str,
) -> dict[tuple[str, str], dict[str, list[list[str]]]]:
    rows: dict[tuple[str, str], dict[str, list[list[str]]]] = {}
    summary_name = Path(SUMMARY_RELATIVE).name.casefold()
    section: str | None = None
    metric: str | None = None
    for line in readme.splitlines():
        heading = line.strip()
        if heading.startswith("## ") and not heading.startswith("### "):
            section = None
            metric = None
            continue
        if heading.startswith("### ") and not heading.startswith("#### "):
            lowered = heading.casefold()
            if "gpt-sovits" in lowered:
                section = "gpt"
            elif ("other" in lowered and "tts" in lowered) or (
                "其他" in lowered
                and ("tts" in lowered or "开源" in lowered or "開源" in lowered)
            ):
                section = "open_tts"
            else:
                section = None
            metric = None
            continue
        if heading.startswith("#### ") and section is not None:
            metric = "rtf" if re.search(r"\brtf\b", heading, re.IGNORECASE) else "first_output"
            continue
        if not line.startswith("|"):
            continue
        if section is None or metric is None:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if not any(summary_name in cell.casefold() for cell in cells):
            continue
        row_text = " ".join(cells).casefold()
        model = re.sub(r"[^a-z0-9]+", "", row_text)
        if "gsvcpp" in model:
            key = "gsv_cpp"
        elif "v2proplus" in model:
            key = "windows_v2proplus"
        elif "v5turbo" in model:
            machine = row_text
            key = "macos_v5turbo" if any(
                marker in machine
                for marker in ("macos", "mac studio", "m5 max", "apple silicon", "蘋果", "苹果")
            ) else "windows_v5turbo"
        else:
            continue
        rows.setdefault((section, metric), {}).setdefault(key, []).append(cells)
    return rows


def check_readme_benchmark_values(
    root: Path,
    summary: dict[str, object],
    readme_paths: tuple[str, ...] | list[str] = README_PATHS,
) -> dict[str, object]:
    method = summary.get("method", {})
    if not isinstance(method, dict):
        return {"status": "failed", "findings": [_finding("benchmark_method_missing")]}
    expected_samples = method.get("measured_requests_per_condition")
    measured_total = method.get("measured_total")
    results = summary.get("results", [])
    if not isinstance(results, list):
        return {"status": "failed", "findings": [_finding("benchmark_results_missing")]}
    findings: list[dict[str, object]] = []
    expected_by_platform_workload: dict[tuple[str, str], dict[str, object]] = {}
    verified_platforms: set[str] = set()
    for item in results:
        if (
            not isinstance(item, dict)
            or item.get("model_family") != "gsv-v5turbo"
            or not str(item.get("status", "")).startswith("verified")
        ):
            continue
        platform = _platform_key(item.get("platform", ""))
        if platform is None:
            findings.append(_finding("benchmark_platform_missing", workload=item.get("workload")))
            continue
        verified_platforms.add(platform)
        workload = item.get("workload")
        if workload in {"keepalive_pcm", "new_connection_pcm"}:
            key = (platform, str(workload))
            if key in expected_by_platform_workload:
                findings.append(_finding("benchmark_workload_duplicated", platform=platform, workload=workload))
            expected_by_platform_workload[key] = item
    detailed_rows_checked = 0
    workloads = ("keepalive_pcm", "new_connection_pcm")
    if not verified_platforms:
        findings.extend(_finding("benchmark_workload_missing", workload) for workload in workloads)
    else:
        for platform in sorted(verified_platforms):
            for workload in workloads:
                if (platform, workload) not in expected_by_platform_workload:
                    findings.append(_finding("benchmark_workload_missing", platform=platform, workload=workload))
    benchmark_path = root / BENCHMARK_DOCUMENT_PATH
    detailed_rows: dict[tuple[str, str], list[list[str]]] = {}
    if not benchmark_path.is_file():
        findings.append(_finding("benchmark_document_missing", BENCHMARK_DOCUMENT_PATH))
    else:
        benchmark_text = benchmark_path.read_text(encoding="utf-8", errors="replace")
        detailed_rows = _benchmark_rows(benchmark_text)
        if measured_total is not None and not _has_per_configuration_count(
            benchmark_text, int(measured_total)
        ):
            findings.append(
                _finding("benchmark_total_count_mismatch", BENCHMARK_DOCUMENT_PATH)
            )
    for platform in sorted(verified_platforms):
        for workload in workloads:
            key_for_row = (platform, workload)
            if key_for_row not in expected_by_platform_workload:
                continue
            matches = detailed_rows.get(key_for_row, [])
            if len(matches) != 1:
                findings.append(
                    _finding(
                        "readme_benchmark_row_missing_or_duplicated",
                        BENCHMARK_DOCUMENT_PATH,
                        platform=platform,
                        workload=workload,
                    )
                )
                continue
            cells = matches[0]
            if len(cells) < 7:
                findings.append(
                    _finding(
                        "readme_benchmark_row_malformed",
                        BENCHMARK_DOCUMENT_PATH,
                        platform=platform,
                        workload=workload,
                    )
                )
                continue
            item = expected_by_platform_workload[key_for_row]
            for metric, column, pair_index, quantum in (
                ("audible_ttfa_p50_ms", 3, 0, "0.01"),
                ("audible_ttfa_p95_ms", 3, 1, "0.01"),
                ("rtf_p50", 5, 0, "0.00001"),
                ("rtf_p95", 5, 1, "0.00001"),
            ):
                values = _cell_numbers(cells[column])
                actual = values[pair_index] if len(values) > pair_index else None
                expected = _round_decimal(item[metric], quantum)
                if actual != expected:
                    findings.append(
                        _finding(
                            "readme_benchmark_value_mismatch",
                            BENCHMARK_DOCUMENT_PATH,
                            platform=platform,
                            workload=workload,
                            metric=metric,
                        )
                    )
            sample_cell = cells[6].replace(",", "")
            if expected_samples is not None and str(expected_samples) not in sample_cell:
                findings.append(
                    _finding(
                        "readme_benchmark_sample_count_mismatch",
                        BENCHMARK_DOCUMENT_PATH,
                        platform=platform,
                        workload=workload,
                    )
                )
            detailed_rows_checked += 1

    comparison_specs = {
        ("gpt", "first_output"): (
            "windows_v2proplus",
            "windows_v5turbo",
            "macos_v5turbo",
            "gsv_cpp",
        ),
        ("gpt", "rtf"): (
            "windows_v2proplus",
            "windows_v5turbo",
            "macos_v5turbo",
            "gsv_cpp",
        ),
        ("open_tts", "first_output"): (
            "windows_v2proplus",
            "windows_v5turbo",
        ),
        ("open_tts", "rtf"): (
            "windows_v2proplus",
            "windows_v5turbo",
        ),
    }
    expected_keys = {key for table_keys in comparison_specs.values() for key in table_keys}
    comparison_expected: dict[str, dict[str, object]] = {}
    for item in results:
        if not isinstance(item, dict):
            continue
        is_gsv_cpp = item.get("system") == "gsv.cpp"
        mode = item.get("condition") if is_gsv_cpp else item.get("workload")
        if mode != "keepalive_pcm":
            continue
        status = str(item.get("status", ""))
        if is_gsv_cpp:
            compact_key = "gsv_cpp" if status == "complete_multi_session_exploratory" else None
        elif item.get("system") == "AnifLive-TTS v1.5.0" and status.startswith("verified"):
            platform = _platform_key(item.get("platform", ""))
            compact_key = (
                f"{platform}_{'v2proplus' if item.get('model_family') == 'gsv-v2proplus' else 'v5turbo'}"
                if platform in {"windows", "macos"}
                and item.get("model_family") in {"gsv-v2proplus", "gsv-v5turbo"}
                else None
            )
        else:
            compact_key = None
        if compact_key not in expected_keys:
            continue
        if compact_key in comparison_expected:
            findings.append(_finding("benchmark_workload_duplicated", compact_key=compact_key))
        comparison_expected[compact_key] = item

    comparison_rows_checked = 0
    for relative_path in readme_paths:
        readme_path = root / relative_path
        if not readme_path.is_file():
            findings.append(_finding("readme_missing", relative_path))
            continue
        comparison_rows = _comparison_benchmark_rows(
            readme_path.read_text(encoding="utf-8", errors="replace")
        )
        for table_key, table_models in comparison_specs.items():
            table_rows = comparison_rows.get(table_key, {})
            table_label = f"{table_key[0]}_{table_key[1]}"
            for unexpected_key in set(table_rows) - set(table_models):
                findings.append(
                    _finding(
                        "readme_benchmark_row_unexpected",
                        relative_path,
                        table=table_label,
                        workload=unexpected_key,
                    )
                )
            expected_cell_count = 6 if table_key == ("open_tts", "first_output") else 5
            value_column = 2 if table_key[1] == "first_output" else 1
            metric = "audible_ttfa_p50_ms" if table_key[1] == "first_output" else "rtf_p50"
            quantum = "0.01" if metric == "audible_ttfa_p50_ms" else "0.00001"
            for comparison_key in table_models:
                expected = comparison_expected.get(comparison_key)
                if expected is None:
                    findings.append(
                        _finding(
                            "benchmark_workload_missing",
                            platform=("macos" if comparison_key == "macos_v5turbo" else "windows"),
                            workload=comparison_key,
                        )
                    )
                    continue
                matches = table_rows.get(comparison_key, [])
                if len(matches) != 1:
                    findings.append(
                        _finding(
                            "readme_benchmark_row_missing_or_duplicated",
                            relative_path,
                            table=table_label,
                            workload=comparison_key,
                        )
                    )
                    continue
                cells = matches[0]
                if len(cells) != expected_cell_count:
                    findings.append(
                        _finding(
                            "readme_benchmark_row_malformed",
                            relative_path,
                            table=table_label,
                            workload=comparison_key,
                        )
                    )
                    continue
                if table_key[1] == "first_output":
                    metric_label = cells[1].casefold()
                    has_p50 = (
                        "p50" in metric_label
                        if table_key[0] == "gpt"
                        else "p50" in cells[3].casefold()
                    )
                    if "ttfa" not in metric_label or not has_p50:
                        findings.append(
                            _finding(
                                "readme_benchmark_metric_mismatch",
                                relative_path,
                                table=table_label,
                                workload=comparison_key,
                            )
                        )
                values = _cell_numbers(cells[value_column])
                actual = values[0] if len(values) == 1 else None
                expected_value = expected.get(metric)
                if expected_value is None or actual != _round_decimal(expected_value, quantum):
                    findings.append(
                        _finding(
                            "readme_benchmark_value_mismatch",
                            relative_path,
                            table=table_label,
                            workload=comparison_key,
                            metric=metric,
                        )
                    )
                comparison_rows_checked += 1
    return {
        "status": "failed" if findings else "passed",
        "rows_checked": detailed_rows_checked + comparison_rows_checked,
        "detailed_rows_checked": detailed_rows_checked,
        "comparison_rows_checked": comparison_rows_checked,
        "per_condition_samples": expected_samples,
        "measured_total": measured_total,
        "findings": findings,
    }

def _all_graphs(graph: object, label: str = "main"):
    yield graph, label
    for node_index, node in enumerate(graph.node):
        for attribute in node.attribute:
            if attribute.type == 5:
                yield from _all_graphs(attribute.g, f"{label}/node-{node_index}:{node.op_type}")
            elif attribute.type == 10:
                for sub_index, subgraph in enumerate(attribute.graphs):
                    yield from _all_graphs(
                        subgraph, f"{label}/node-{node_index}:{node.op_type}/sub-{sub_index}"
                    )


def _json_number(value: object) -> int | float | str | None:
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if math.isnan(converted):
        return "NaN"
    if math.isinf(converted):
        return "-Infinity" if converted < 0 else "Infinity"
    if converted.is_integer():
        return int(converted)
    return converted


def _array_metrics(array: object) -> dict[str, object]:
    import numpy as np

    flat = np.asarray(array).reshape(-1)
    nan_count = int(np.isnan(flat).sum()) if np.issubdtype(flat.dtype, np.number) else 0
    posinf_count = int(np.isposinf(flat).sum()) if np.issubdtype(flat.dtype, np.number) else 0
    neginf_count = int(np.isneginf(flat).sum()) if np.issubdtype(flat.dtype, np.number) else 0
    finite = flat[np.isfinite(flat)] if np.issubdtype(flat.dtype, np.number) else flat
    if finite.size:
        minimum = _json_number(finite.min())
        maximum = _json_number(finite.max())
    elif neginf_count and not (nan_count or posinf_count):
        minimum = maximum = "-Infinity"
    elif posinf_count and not (nan_count or neginf_count):
        minimum = maximum = "Infinity"
    elif nan_count:
        minimum = maximum = "NaN"
    else:
        minimum = maximum = None
    return {
        "element_count": int(flat.size),
        "minimum": minimum,
        "maximum": maximum,
        "nan_count": nan_count,
        "positive_infinity_count": posinf_count,
        "negative_infinity_count": neginf_count,
        "values_if_small": [_json_number(x) for x in flat.tolist()] if flat.size <= 16 else None,
        "sha256": hashlib.sha256(np.asarray(array).tobytes(order="C")).hexdigest(),
    }


def _constant_payload(node: object, attribute: object, onnx_module: object) -> tuple[str, tuple[int, ...], object] | None:
    import numpy as np
    from onnx import numpy_helper

    if attribute.type == onnx_module.AttributeProto.TENSOR:
        tensor = attribute.t
        return (
            onnx_module.TensorProto.DataType.Name(tensor.data_type),
            tuple(int(x) for x in tensor.dims),
            np.asarray(numpy_helper.to_array(tensor)),
        )
    if attribute.name == "value_float":
        return "FLOAT", (), np.asarray(attribute.f, dtype=np.float32)
    if attribute.name == "value_floats":
        return "FLOAT", (len(attribute.floats),), np.asarray(attribute.floats, dtype=np.float32)
    if attribute.name == "value_int":
        return "INT64", (), np.asarray(attribute.i, dtype=np.int64)
    if attribute.name == "value_ints":
        return "INT64", (len(attribute.ints),), np.asarray(attribute.ints, dtype=np.int64)
    if attribute.name == "value_string":
        return "STRING", (), np.asarray(attribute.s)
    return None


def _fixed_float_pattern(
    family: str,
    op: str,
    dtype: str,
    shape: tuple[int, ...],
    array: object,
    downstream_ops: set[str],
    native_padding_sentinel: bool = False,
) -> tuple[str | None, str]:
    import numpy as np

    values = np.asarray(array)
    flat = values.reshape(-1)
    if family == "dit":
        if (
            dtype == "FLOAT16"
            and shape == (1024,)
            and downstream_ops == {"LayerNormalization"}
            and np.all((flat == 0) | (flat == 1))
        ):
            return ("dit_non_affine_layernorm_zero" if np.all(flat == 0) else "dit_non_affine_layernorm_one", "passed")
        if dtype == "FLOAT16" and shape == () and np.all(flat == 0) and downstream_ops == {"Where"}:
            return "dit_f16_zero_mask_where", "passed"
        if dtype == "FLOAT" and shape == (1,) and np.all(flat == 0) and downstream_ops == {"Where"}:
            return "dit_f32_zero_mask_where", "passed"
        if dtype == "FLOAT16" and shape == () and np.all(flat == 1) and downstream_ops == {"Add"}:
            return "dit_f16_one_add", "passed"
        if dtype == "FLOAT16" and shape == () and np.all(flat == 1) and downstream_ops == {"Mul"}:
            return "dit_f16_one_mul", "passed"
        if dtype == "FLOAT" and shape == (1,) and np.all(flat == 1) and downstream_ops == {"Div"}:
            return "dit_f32_one_div", "passed"
        if dtype == "FLOAT16" and shape == () and np.all(flat == 4) and downstream_ops == {"Mul"}:
            return "dit_f16_four_scale_mul", "passed"
        if dtype == "FLOAT" and shape == (1,) and np.isneginf(flat).all() and downstream_ops == {"Where"}:
            return "dit_f32_negative_infinity_where", "passed"
        if native_padding_sentinel:
            return "recognized_native_padding_sentinel", "passed"
    elif family == "encoder":
        if dtype == "FLOAT" and shape == (3,) and np.array_equal(flat, np.asarray([1, 1, 2], dtype=np.float32)) and downstream_ops == {"Resize"}:
            return "encoder_resize_ratio_1_1_2", "passed"
        if dtype == "FLOAT16" and shape == () and np.all(flat == np.float16(math.sqrt(96))) and downstream_ops == {"Div"}:
            return "encoder_sqrt96_divisor", "passed"
        if dtype == "FLOAT16" and shape == () and np.all(flat == np.float16(math.sqrt(128))) and downstream_ops == {"Div"}:
            return "encoder_sqrt128_divisor", "passed"
        if dtype == "FLOAT16" and shape == () and np.all(flat == np.float16(-10000)) and downstream_ops == {"Where"}:
            return "encoder_attention_mask_negative_10000", "passed"
        if dtype == "FLOAT16" and shape in {(), (1,)} and np.all(flat == 0) and downstream_ops in ({"Add"}, {"Equal"}):
            return f"encoder_zero_{next(iter(downstream_ops)).casefold()}", "passed"
    elif family == "vocoder":
        if dtype == "FLOAT16" and shape == () and np.all(flat == 3) and downstream_ops == {"Div"}:
            return "vocoder_residual_average_divisor", "passed"
    return None, "failed"


EXPECTED_CONSTANT_COUNTS = {
    "ar": {},
    "dit": {
        "dit_non_affine_layernorm_zero": 45,
        "dit_non_affine_layernorm_one": 45,
        "dit_f16_zero_mask_where": 24,
        "dit_f32_zero_mask_where": 44,
        "dit_f16_one_add": 45,
        "dit_f16_one_mul": 88,
        "dit_f32_one_div": 22,
        "dit_f32_negative_infinity_where": 22,
        "dit_f16_four_scale_mul": 1,
        "recognized_native_padding_sentinel": 1,
    },
    "encoder": {
        "encoder_resize_ratio_1_1_2": 2,
        "encoder_sqrt96_divisor": 12,
        "encoder_sqrt128_divisor": 1,
        "encoder_attention_mask_negative_10000": 13,
        "encoder_zero_equal": 3,
        "encoder_zero_add": 2,
    },
    "vocoder": {"vocoder_residual_average_divisor": 5},
}


def _where_context(node: object, output_name: str, graph: object, producers: dict[str, object], consumers: dict[str, list[object]]) -> list[dict[str, object]]:
    result = []
    graph_outputs = {item.name for item in graph.output}
    for consumer in consumers.get(output_name, []):
        if consumer.op_type != "Where":
            continue
        roles = []
        for index, role in enumerate(("condition", "true_value", "false_value")):
            if index >= len(consumer.input):
                continue
            value_name = consumer.input[index]
            producer = producers.get(value_name)
            roles.append(
                {
                    "role": role,
                    "value": value_name,
                    "producer_name": producer.name if producer else None,
                    "producer_op": producer.op_type if producer else ("graph_input" if value_name in {x.name for x in graph.input} else None),
                }
            )
        result.append(
            {
                "consumer_name": consumer.name,
                "consumer_op": consumer.op_type,
                "input_index": list(consumer.input).index(output_name),
                "where_inputs": roles,
                "where_outputs": list(consumer.output),
                "graph_outputs": [name for name in consumer.output if name in graph_outputs],
            }
        )
    return result


def _is_native_padding_sentinel(
    relative_path: str,
    file_sha256: str,
    node: object,
    attribute: object,
    dtype: str,
    shape: tuple[int, ...],
    array: object,
    where_contexts: list[dict[str, object]],
    graph: object,
    producers: dict[str, object],
) -> bool:
    import numpy as np

    if (
        relative_path != "scripts/v5_graph/dit-topology.onnx"
        or file_sha256 != ONNX_TOPOLOGIES[relative_path]
        or node.name != "/ConstantOfShape"
        or node.op_type != "ConstantOfShape"
        or attribute.name != "value"
        or dtype != "FLOAT16"
        or shape != (1,)
        or hashlib.sha256(np.asarray(array).tobytes(order="C")).hexdigest()
        != DIT_PADDING_SENTINEL_SHA256
        or not np.isnan(np.asarray(array)).all()
        or len(where_contexts) != 1
    ):
        return False
    shape_node = producers.get(node.input[0]) if node.input else None
    if (
        shape_node is None
        or shape_node.name != "/Shape_24"
        or shape_node.op_type != "Shape"
        or list(shape_node.input) != ["/proj_out/Add_output_0"]
    ):
        return False
    context = where_contexts[0]
    roles = context.get("where_inputs", [])
    return (
        context.get("consumer_name") == "/Where"
        and context.get("consumer_op") == "Where"
        and context.get("input_index") == 2
        and context.get("where_outputs") == ["velocity"]
        and context.get("graph_outputs") == ["velocity"]
        and len(roles) == 3
        and roles[0].get("role") == "condition"
        and roles[0].get("value") == "/Reshape_output_0"
        and roles[0].get("producer_name") == "/Reshape"
        and roles[0].get("producer_op") == "Reshape"
        and roles[1].get("role") == "true_value"
        and roles[1].get("value") == "/proj_out/Add_output_0"
        and roles[2].get("role") == "false_value"
        and roles[2].get("value") == "/ConstantOfShape_output_0"
    )


def audit_onnx_topologies(root: Path) -> dict[str, object]:
    try:
        import onnx
    except ImportError:
        return {"status": "failed", "topologies": [], "findings": [_finding("onnx_verifier_unavailable")]}
    topologies: list[dict[str, object]] = []
    findings: list[dict[str, object]] = []
    review_pending = False
    for relative_path, expected_hash in ONNX_TOPOLOGIES.items():
        path = root / relative_path
        if not path.is_file():
            findings.append(_finding("required_weightless_topology_missing", relative_path))
            continue
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        hash_match = actual_hash == expected_hash
        if not hash_match:
            findings.append(_finding("onnx_topology_inventory_hash_mismatch", relative_path))
        try:
            model = onnx.load(str(path), load_external_data=False)
        except Exception:
            findings.append(_finding("onnx_topology_unreadable", relative_path))
            continue
        family = PurePosixPath(relative_path).stem.split("-", 1)[0]
        entry: dict[str, object] = {
            "path": relative_path,
            "sha256": actual_hash,
            "expected_sha256": expected_hash,
            "hash_match": hash_match,
            "dense_initializers": 0,
            "sparse_initializers": 0,
            "integer_shape_constant_count": 0,
            "max_integer_constant_elements": 0,
            "float_constant_pattern_counts": {},
            "float_constant_evidence": [],
            "learned_constant_nodes": [],
            "unrecognized_constant_nodes": [],
        }
        category_counts: dict[str, int] = {}
        for graph, graph_label in _all_graphs(model.graph):
            entry["dense_initializers"] = int(entry["dense_initializers"]) + len(graph.initializer)
            entry["sparse_initializers"] = int(entry["sparse_initializers"]) + len(graph.sparse_initializer)
            producers = {output: node for node in graph.node for output in node.output}
            consumers: dict[str, list[object]] = {}
            for node in graph.node:
                for name in node.input:
                    consumers.setdefault(name, []).append(node)
            for node_index, node in enumerate(graph.node):
                if node.op_type.casefold() == "learnedconstant":
                    entry["learned_constant_nodes"].append(
                        {"graph": graph_label, "node": node.name or f"{node.op_type}@{node_index}"}
                    )
                    continue
                if node.op_type not in {"Constant", "ConstantOfShape"}:
                    continue
                for attribute in node.attribute:
                    payload = _constant_payload(node, attribute, onnx)
                    if payload is None:
                        entry["unrecognized_constant_nodes"].append(
                            {"graph": graph_label, "node": node.name or f"{node.op_type}@{node_index}", "attribute": attribute.name}
                        )
                        continue
                    dtype, shape, array = payload
                    import numpy as np

                    flat = np.asarray(array).reshape(-1)
                    if dtype in {"INT32", "INT64", "BOOL"}:
                        count = int(flat.size)
                        entry["integer_shape_constant_count"] = int(entry["integer_shape_constant_count"]) + 1
                        entry["max_integer_constant_elements"] = max(
                            int(entry["max_integer_constant_elements"]), count
                        )
                        if count > 64:
                            entry["unrecognized_constant_nodes"].append(
                                {"graph": graph_label, "node": node.name or f"{node.op_type}@{node_index}", "reason": "integer_shape_constant_too_large", "elements": count}
                            )
                        continue
                    if not dtype.startswith(("FLOAT", "DOUBLE", "BFLOAT")):
                        entry["unrecognized_constant_nodes"].append(
                            {"graph": graph_label, "node": node.name or f"{node.op_type}@{node_index}", "reason": "unsupported_constant_dtype", "dtype": dtype}
                        )
                        continue
                    downstream_ops = {
                        consumer.op_type
                        for output in node.output
                        for consumer in consumers.get(output, [])
                    }
                    where_contexts = [
                        context
                        for output in node.output
                        for context in _where_context(node, output, graph, producers, consumers)
                    ]
                    native_padding_sentinel = _is_native_padding_sentinel(
                        relative_path,
                        actual_hash,
                        node,
                        attribute,
                        dtype,
                        shape,
                        array,
                        where_contexts,
                        graph,
                        producers,
                    )
                    category, classification = _fixed_float_pattern(
                        family,
                        node.op_type,
                        dtype,
                        shape,
                        array,
                        downstream_ops,
                        native_padding_sentinel=native_padding_sentinel,
                    )
                    details = _array_metrics(array)
                    evidence = {
                        "graph": graph_label,
                        "node_index": node_index,
                        "node_name": node.name or f"{node.op_type}@{node_index}",
                        "op": node.op_type,
                        "attribute": attribute.name,
                        "dtype": dtype,
                        "shape": list(shape),
                        **details,
                        "upstream_ops": sorted(
                            {producers[name].op_type for name in node.input if name in producers}
                        ),
                        "downstream_ops": sorted(downstream_ops),
                        "classification": category,
                        "classification_status": classification,
                    }
                    if where_contexts:
                        evidence["where_contexts"] = where_contexts
                    entry["float_constant_evidence"].append(evidence)
                    if classification == "review_pending":
                        review_pending = True
                    elif classification == "failed" or category is None:
                        entry["unrecognized_constant_nodes"].append(
                            {
                                "graph": graph_label,
                                "node": node.name or f"{node.op_type}@{node_index}",
                                "dtype": dtype,
                                "shape": list(shape),
                                "classification": "not_in_fixed_formula_inventory",
                                "sha256": details["sha256"],
                            }
                        )
                    else:
                        category_counts[category] = category_counts.get(category, 0) + 1
        entry["float_constant_pattern_counts"] = category_counts
        expected_counts = EXPECTED_CONSTANT_COUNTS[family]
        if category_counts != expected_counts:
            entry["unrecognized_constant_nodes"].append(
                {
                    "reason": "fixed_formula_pattern_count_mismatch",
                    "expected": expected_counts,
                    "actual": category_counts,
                }
            )
        if entry["dense_initializers"] or entry["sparse_initializers"]:
            findings.append(
                _finding(
                    "onnx_initializers_present",
                    relative_path,
                    dense=int(entry["dense_initializers"]),
                    sparse=int(entry["sparse_initializers"]),
                )
            )
        if entry["learned_constant_nodes"]:
            findings.append(_finding("onnx_learned_constant_nodes_present", relative_path))
        if entry["unrecognized_constant_nodes"]:
            findings.append(_finding("onnx_constant_outside_fixed_inventory", relative_path))
        if entry["unrecognized_constant_nodes"] or entry["dense_initializers"] or entry["sparse_initializers"] or entry["learned_constant_nodes"] or not hash_match:
            entry["status"] = "failed"
        elif any(item["classification_status"] == "review_pending" for item in entry["float_constant_evidence"]):
            entry["status"] = "review_pending"
        else:
            entry["status"] = "passed"
        topologies.append(entry)
    if findings:
        status = "failed"
    elif review_pending:
        status = "review_pending"
    else:
        status = "passed"
    return {
        "status": status,
        "topology_count": len(topologies),
        "topologies": topologies,
        "findings": findings,
    }


def derive_release_gates(summary: dict[str, object]) -> dict[str, str]:
    results = summary.get("results", [])
    if not isinstance(results, list):
        results = []
    macos_performance_verified = any(
        isinstance(item, dict)
        and _platform_key(item.get("platform", "")) == "macos"
        and item.get("model_family") == "gsv-v5turbo"
        and str(item.get("status", "")).casefold().startswith("verified_macos")
        for item in results
    )
    v2 = next(
        (item for item in results if isinstance(item, dict) and item.get("model_family") == "gsv-v2proplus"),
        None,
    )
    gsv = next(
        (item for item in results if isinstance(item, dict) and item.get("system") == "gsv.cpp"),
        None,
    )
    remaining = summary.get("remaining", [])
    remaining_text = " ".join(str(item).casefold() for item in remaining) if isinstance(remaining, list) else ""
    integration = summary.get("quality_and_integration", {})
    switching = integration.get("switching_checks", {}) if isinstance(integration, dict) else {}
    switch_evidence = switching.get("cross_family_evidence", {}) if isinstance(switching, dict) else {}
    switch_verified = (
        isinstance(switch_evidence, dict)
        and switch_evidence.get("status") == "verified_real_http_roundtrip"
        and switch_evidence.get("sequence") == ["gsv-v5turbo", "gsv-v2proplus", "gsv-v5turbo"]
        and switch_evidence.get("synthesis_after_each_switch") is True
        and switch_evidence.get("roundtrip_pcm_exact") is True
        and bool(switch_evidence.get("source_record"))
    )
    return {
        "macos_handoff": "passed" if macos_performance_verified else "pending",
        "fresh_v2_benchmark": "passed" if isinstance(v2, dict) and str(v2.get("status", "")).startswith("verified") else "pending",
        "local_gsv_cpp_benchmark": "passed" if isinstance(gsv, dict) and (str(gsv.get("status", "")).startswith("verified") or (gsv.get("status") == "complete_multi_session_exploratory" and gsv.get("formal_10round") is False)) else "pending",
        "main_comparison": "passed" if summary.get("main_comparison_complete") is True else "pending",
        "integrated_release_source_validation": (
            "pending" if "integratedrelease-source validation" in remaining_text else "passed"
        ),
        "cross_family_model_switch": "passed" if switch_verified else "pending",
        "functional_release_candidate": "pending" if summary.get("release_ready") is False else "passed",
    }


def is_release_ready(
    summary: dict[str, object],
    check_statuses: dict[str, str],
    release_gates: dict[str, str] | None = None,
) -> bool:
    gates = release_gates if release_gates is not None else derive_release_gates(summary)
    return (
        summary.get("main_comparison_complete") is True
        and all(status == "passed" for status in check_statuses.values())
        and all(status == "passed" for status in gates.values())
    )


def _read_summary(path: Path) -> tuple[dict[str, object] | None, dict[str, object]]:
    if not path.is_file():
        return None, {"status": "failed", "findings": [_finding("public_benchmark_summary_missing")]}
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, {"status": "failed", "findings": [_finding("public_benchmark_summary_invalid")]}
    if not isinstance(summary, dict):
        return None, {"status": "failed", "findings": [_finding("public_benchmark_summary_invalid")]}
    return summary, {"status": "passed", "findings": []}


def run_preflight(
    root: Path,
    summary_path: Path | None = None,
    source_manifest: Path | None = None,
) -> dict[str, object]:
    root = root.resolve()
    summary_file = summary_path.resolve() if summary_path else root / SUMMARY_RELATIVE
    summary, summary_check = _read_summary(summary_file)
    if summary is None:
        summary = {}
    source_scan = scan_source_tree(root, source_manifest)
    checks = {
        "public_benchmark_summary": summary_check,
        "markdown_links": check_markdown_links(root, list(PUBLIC_DOCUMENT_PATHS)),
        "readme_benchmark_values": check_readme_benchmark_values(root, summary),
        "public_document_terms": check_current_public_terms(root),
        "source_hygiene": source_scan,
        "license_files": check_license_files(root),
        "onnx_topologies": audit_onnx_topologies(root),
    }
    gates = derive_release_gates(summary)
    statuses = {name: str(result.get("status", "failed")) for name, result in checks.items()}
    release_ready = is_release_ready(summary, statuses, gates)
    core_names = {
        "public_benchmark_summary",
        "markdown_links",
        "readme_benchmark_values",
        "public_document_terms",
        "source_hygiene",
        "license_files",
    }
    core_failures = sorted(name for name in core_names if statuses.get(name) != "passed")
    status = "ready" if release_ready else "partial_nonmac_release_preparation"
    return {
        "schema": 1,
        "release": str(summary.get("release", "1.5.0")),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "overall_release_ready": release_ready,
        "non_mac_core_status": "failed" if core_failures else "passed",
        "non_mac_core_failures": core_failures,
        "checks": checks,
        "release_gates": gates,
        "main_comparison_complete": summary.get("main_comparison_complete") is True,
        "remaining_release_evidence": summary.get("remaining", []),
        "notes": [
            "A pending Mac handoff always keeps overall_release_ready false.",
            "Unrecognized ONNX constants stay review-pending or failed; they are never silently allowed.",
            "Secrets and matched source content are never copied into this report.",
            "Licensed generic frontend model data and upstream dictionary caches are disclosed separately from voice/model weights.",
        ],
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--summary", type=Path)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        help="Git-generated JSON inventory of tracked and non-ignored untracked source paths.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run_preflight(args.root, args.summary, args.source_manifest)
    onnx_check = report["checks"]["onnx_topologies"]
    onnx_report_path = args.output.parent / "RELEASE_PREFLIGHT_ONNX_CONSTANTS.json"
    _write_json(onnx_report_path, onnx_check)
    report["onnx_constant_evidence_file"] = onnx_report_path.name
    _write_json(args.output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "overall_release_ready": report["overall_release_ready"],
                "non_mac_core_status": report["non_mac_core_status"],
                "report_file": args.output.name,
                "onnx_evidence_file": onnx_report_path.name,
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["overall_release_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
