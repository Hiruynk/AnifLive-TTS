#!/usr/bin/env python3
"""Backport upstream security fixes to the qualified Transformers wheel.

Adapted from Hugging Face Transformers commit
676559d5022b74aaa0cee1cee0842b7f27c5320e (Apache-2.0):
https://github.com/huggingface/transformers/commit/676559d5022b74aaa0cee1cee0842b7f27c5320e
https://github.com/huggingface/transformers/commit/a7f8e7ff37d87d1a1a0c8cf607971c607741452f
https://github.com/huggingface/transformers/commit/eaaaf8494dd5386634ae37d1d122212fdc315be5

Copyright 2025 The HuggingFace Team. All rights reserved.
Licensed under the Apache License, Version 2.0; see
https://www.apache.org/licenses/LICENSE-2.0 for the license text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from importlib import metadata
from pathlib import Path

VERSION = "4.57.6"
UPSTREAM = "676559d5022b74aaa0cee1cee0842b7f27c5320e"
UPSTREAM_FIXES = {
    "CVE-2026-5241": UPSTREAM,
    "CVE-2026-4372": "a7f8e7ff37d87d1a1a0c8cf607971c607741452f",
    "CVE-2026-9856": "eaaaf8494dd5386634ae37d1d122212fdc315be5",
}
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
PACKAGE_FILES = {name: f"models/lightglue/{name}" for name in PRE_HASHES}
PACKAGE_FILES.update({
    "configuration_utils.py": "configuration_utils.py",
    "hub_kernels.py": "integrations/hub_kernels.py",
    "processing_utils.py": "processing_utils.py",
    "tokenization_utils_base.py": "tokenization_utils_base.py",
})
PRE_HASHES.update({
    "configuration_utils.py": "08a3889c1b8a73d340a37c93f72346ee0e385e339b7a72a03b5c026d4943baf7",
    "hub_kernels.py": "845f33ef26be915f62efa82aaf3695bf26e6e165eb5e49a2542ae4fc69ea154b",
    "processing_utils.py": "6c29de5c3fc674cf21ba6377a39cbafb2ba3e4bc2d1202d4b944f8cf840e6f33",
    "tokenization_utils_base.py": "a2d62f7e5d58f908b3a2d0b24ce6329e1e52c815082cedde29a5c007a23e7e83",
})
POST_HASHES.update({
    "configuration_utils.py": "541dba62b7200857f19ca828c32e06eb794ab3dacfa783f6c007128275ac0c75",
    "hub_kernels.py": "123c1eba64bf176d3cf2e163fb7c3c6e2f3c36a8752ec4799a88f8e0f071d864",
    "processing_utils.py": "7bac9b4cb09728dac905e7616680581b275510690d88a021f52d23e7da2f0620",
    "tokenization_utils_base.py": "831e539d76180226d214c8abc5b9be96df2351bb0e4c27fc7569a18fd2e11d45",
})

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
        raise RuntimeError("Transformers source does not match the expected upstream patch context")
    return text.replace(old, new, 1)


def patched_bytes(name: str, source: bytes) -> bytes:
    text = source.decode("utf-8")
    if name in ("configuration_lightglue.py", "modular_lightglue.py"):
        for old, new in ((_DOC, ""), (_PARAM, ""), (_ASSIGNMENT, ""), (_OLD_CONFIG, _NEW_CONFIG)):
            text = _replace_once(text, old, new)
    if name in ("modeling_lightglue.py", "modular_lightglue.py"):
        text = _replace_once(text, _OLD_MODEL, _NEW_MODEL)
    if name == "configuration_utils.py":
        old = (
            '            try:\n'
            '                setattr(self, key, value)\n'
            '            except AttributeError as err:\n'
            '                logger.error(f"Can\'t set {key} with value {value} for {self}")\n'
            '                raise err\n'
        )
        new = (
            '            # Check this to avoid deserializing problematic fields from hub configs - they should use the public field\n'
            '            if key not in ("_attn_implementation_internal", "_experts_implementation_internal"):\n'
            + "".join("    " + line for line in old.splitlines(keepends=True))
        )
        text = _replace_once(text, old, new)
    if name == "hub_kernels.py":
        # 4.57.6 has no module logger; the upstream fix uses warning_once.
        text = _replace_once(text, 'from ..modeling_flash_attention_utils import lazy_import_flash_attention\n',
                             'from ..modeling_flash_attention_utils import lazy_import_flash_attention\nfrom ..utils import logging\n')
        text = _replace_once(text, '\n\ntry:\n', '\n\nlogger = logging.get_logger(__name__)\n\n\ntry:\n')
        text = _replace_once(text, (
            '    return (\n'
            '        attn_implementation is not None\n'
            '        and re.search(r"^[^/:]+/[^/:]+(?:@[^/:]+)?(?::[^/:]+)?$", attn_implementation) is not None\n'
            '    )\n'
        ), (
            '    if attn_implementation is None:\n'
            '        return False\n'
            '    match_object = re.search(r"^(?:paged\\|)?([^/:]+)/[^/:]+(?:@[^/:]+)?(?::[^/:]+)?$", attn_implementation)\n'
            '    if match_object is not None:\n'
            '        # Due to security reason, we only allow kernels from the `kernels-community` repo, as otherwise loading\n'
            '        # random kernels will lead to arbitrary code execution\n'
            '        repo_id = match_object.group(1)\n'
            '        if repo_id != "kernels-community":\n'
            '            logger.warning_once(\n'
            '                "For security reasons, we currently only accept kernels from the `kernels-community` repository. We "\n'
            '                f"will skip kernel from `{repo_id}`"\n'
            '            )\n'
            '            return False\n'
            '        return True\n'
            '    return False\n'
        ))
    if name in ("processing_utils.py", "tokenization_utils_base.py"):
        indent = " " * (24 if name == "processing_utils.py" else 20)
        old = indent + 'template_filepath = os.path.join(chat_template_dir, f"{template_name}.jinja")\n'
        text = _replace_once(text, old, old + (
            indent + '# template_name is an untrusted dict key; reject path traversal (CWE-22)\n'
            + indent + 'if Path(template_filepath).resolve().parent != Path(chat_template_dir).resolve():\n'
            + indent + '    raise ValueError(f"Invalid chat template name: {template_name!r}")\n'
        ))
    return text.encode("utf-8")


def patch_package(package_root: Path, version: str) -> dict:
    if version != VERSION:
        raise RuntimeError(f"Expected transformers {VERSION}; found {version}")
    originals = {name: (package_root / path).read_bytes() for name, path in PACKAGE_FILES.items()}
    actual = {name: _sha(data) for name, data in originals.items()}
    if actual == POST_HASHES:
        return {"status": "already_patched", "version": version, "upstream": UPSTREAM,
                "upstream_fixes": UPSTREAM_FIXES, "sha256": actual}
    if actual != PRE_HASHES:
        raise RuntimeError(f"Unexpected Transformers source hashes: {actual}")
    updated = {name: patched_bytes(name, data) for name, data in originals.items()}
    after = {name: _sha(data) for name, data in updated.items()}
    if after != POST_HASHES:
        raise RuntimeError(f"Backport output failed exact SHA-256 verification: {after}")
    for name, data in updated.items():
        (package_root / PACKAGE_FILES[name]).write_bytes(data)
    verified = {name: _sha((package_root / path).read_bytes()) for name, path in PACKAGE_FILES.items()}
    if verified != POST_HASHES:
        raise RuntimeError(f"Written Transformers source failed verification: {verified}")
    return {"status": "patched", "version": version, "upstream": UPSTREAM,
            "upstream_fixes": UPSTREAM_FIXES, "before_sha256": actual, "sha256": verified}


def verify_package(package_root: Path, version: str) -> dict:
    """Verify installed bytes without importing or changing Transformers."""
    if version != VERSION:
        raise RuntimeError(f"Expected transformers {VERSION}; found {version}")
    actual = {name: _sha((package_root / path).read_bytes()) for name, path in PACKAGE_FILES.items()}
    if actual != POST_HASHES:
        raise RuntimeError(f"Installed Transformers security source verification failed: {actual}")
    return {"status": "verified", "version": version, "upstream_fixes": UPSTREAM_FIXES, "sha256": actual}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Verify installed fixed bytes without modifying files")
    args = parser.parse_args()
    dist = metadata.distribution("transformers")
    package_root = Path(dist.locate_file("transformers"))
    action = verify_package if args.verify_only else patch_package
    print(json.dumps(action(package_root, dist.version), sort_keys=True))


if __name__ == "__main__":
    main()
