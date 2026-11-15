#!/usr/bin/env python3
"""Backport the LightGlue trust fix to the qualified Transformers wheel.

Adapted from Hugging Face Transformers commit
676559d5022b74aaa0cee1cee0842b7f27c5320e (Apache-2.0):
https://github.com/huggingface/transformers/commit/676559d5022b74aaa0cee1cee0842b7f27c5320e

Copyright 2025 The HuggingFace Team. All rights reserved.
Licensed under the Apache License, Version 2.0; see
https://www.apache.org/licenses/LICENSE-2.0 for the license text.
"""

from __future__ import annotations

import hashlib
from importlib import metadata
import json
from pathlib import Path


VERSION = "4.57.6"
UPSTREAM = "676559d5022b74aaa0cee1cee0842b7f27c5320e"
PRE_HASHES = {
    "configuration_lightglue.py": "8e28c9b208e564467e312640450d64294896ff5554e41f855651c6b71597e1b7",
    "modeling_lightglue.py": "2aa47ee7ce82ef28845495275c53366d9d57d4e2676b4b8be7074a5966586843",
    "modular_lightglue.py": "ed6e0f5e178a65cd45506d8e9a4d9ce79593b0139d4d1b563fbc8944610ac65c",
}
POST_HASHES = {
    "configuration_lightglue.py": "4e3018597482dd77c76e8c7738311fee4ded1fd1b06c46183cc16e6ddc34ba12",
    "modeling_lightglue.py": "5ae143f426bb26d893d3f3592882ddc484d809b14c9169a5edee19960c10328a",
    "modular_lightglue.py": "305fa1a2bde50beb02ac89f8b11a9cca289aa301c93261103df55e060c763239",
}

_DOC = (
    '        trust_remote_code (`bool`, *optional*, defaults to `False`):\n'
    '            Whether to trust remote code when using other models than SuperPoint as keypoint detector.\n'
)
_PARAM = '        trust_remote_code: bool = False,\n'
_ASSIGNMENT = (
    '        # LightGlue can be used with other models than SuperPoint as keypoint detector\n'
    '        # We provide the trust_remote_code argument to allow the use of other models\n'
    '        # that are not registered in the CONFIG_MAPPING dictionary (for example DISK)\n'
    '        self.trust_remote_code = trust_remote_code\n\n'
)
_OLD_CONFIG = (
    '            if keypoint_detector_config["model_type"] not in CONFIG_MAPPING:\n'
    '                keypoint_detector_config = AutoConfig.from_pretrained(\n'
    '                    keypoint_detector_config["_name_or_path"], trust_remote_code=self.trust_remote_code\n'
    '                )\n'
    '            else:\n'
    '                keypoint_detector_config = CONFIG_MAPPING[keypoint_detector_config["model_type"]](\n'
    '                    **keypoint_detector_config, attn_implementation="eager"\n'
    '                )\n'
)
_NEW_CONFIG = (
    '            keypoint_detector_config = CONFIG_MAPPING[keypoint_detector_config["model_type"]](\n'
    '                **keypoint_detector_config, attn_implementation="eager"\n'
    '            )\n'
)
_OLD_MODEL = (
    '        self.keypoint_detector = AutoModelForKeypointDetection.from_config(\n'
    '            config.keypoint_detector_config, trust_remote_code=config.trust_remote_code\n'
    '        )\n'
)
_NEW_MODEL = '        self.keypoint_detector = AutoModelForKeypointDetection.from_config(config.keypoint_detector_config)\n'


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError("LightGlue source does not match the expected upstream patch context")
    return text.replace(old, new, 1)


def patched_bytes(name: str, source: bytes) -> bytes:
    text = source.decode("utf-8")
    if name in ("configuration_lightglue.py", "modular_lightglue.py"):
        for old, new in ((_DOC, ""), (_PARAM, ""), (_ASSIGNMENT, ""), (_OLD_CONFIG, _NEW_CONFIG)):
            text = _replace_once(text, old, new)
    if name in ("modeling_lightglue.py", "modular_lightglue.py"):
        text = _replace_once(text, _OLD_MODEL, _NEW_MODEL)
    return text.encode("utf-8")


def patch_package(package_root: Path, version: str) -> dict:
    if version != VERSION:
        raise RuntimeError(f"Expected transformers {VERSION}; found {version}")
    lightglue = package_root / "models" / "lightglue"
    originals = {name: (lightglue / name).read_bytes() for name in PRE_HASHES}
    actual = {name: _sha(data) for name, data in originals.items()}
    if actual == POST_HASHES:
        return {"status": "already_patched", "version": version, "upstream": UPSTREAM, "sha256": actual}
    if actual != PRE_HASHES:
        raise RuntimeError(f"Unexpected Transformers LightGlue source hashes: {actual}")
    updated = {name: patched_bytes(name, data) for name, data in originals.items()}
    after = {name: _sha(data) for name, data in updated.items()}
    if after != POST_HASHES:
        raise RuntimeError(f"Backport output failed exact SHA-256 verification: {after}")
    for name, data in updated.items():
        (lightglue / name).write_bytes(data)
    verified = {name: _sha((lightglue / name).read_bytes()) for name in PRE_HASHES}
    if verified != POST_HASHES:
        raise RuntimeError(f"Written Transformers LightGlue source failed verification: {verified}")
    return {"status": "patched", "version": version, "upstream": UPSTREAM,
            "before_sha256": actual, "sha256": verified}


def main() -> None:
    dist = metadata.distribution("transformers")
    package_root = Path(dist.locate_file("transformers"))
    print(json.dumps(patch_package(package_root, dist.version), sort_keys=True))


if __name__ == "__main__":
    main()
