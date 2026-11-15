#!/usr/bin/env python3
"""Assemble an FP32 functional candidate package beside the frozen FP32 package.

Usage: fp32_functional_candidate.py BASE_PACKAGE WORKER_PAYLOAD DEST MODEL_ID {package,dynamic}

BASE_PACKAGE is the delivered frozen FP32 package; it is only read. Weights,
MLX runtime libraries, reference inputs and voice profiles are hard-linked
from it (no large copies); the manifest is written fresh, and
fp16_package_refresh.py then unlinks and replaces the worker binary,
engine manifest and checksums, so nothing is written through a link into the
frozen package. "dynamic" produces the shared-base form (no fixed voice).
"""
import json
import os
import subprocess
import sys
from pathlib import Path

base, payload, dest = (Path(p).resolve() for p in sys.argv[1:4])
model_id, mode = sys.argv[4], sys.argv[5]
if mode not in ("package", "dynamic"):
    raise SystemExit("mode must be package or dynamic")
manifest = json.loads((base / "manifest.json").read_text())
if manifest.get("precision") != "FP32" or manifest.get("reference_mode") != "package":
    raise SystemExit("base must be the frozen FP32 package-mode package")
if dest.exists():
    raise SystemExit(f"destination exists: {dest}")
worker_manifest = json.loads((payload / "build-manifest.json").read_text())

skip = {"manifest.json", "checksums.json", "native/engine-manifest.json", "native/bin/aniflive-macos-v5turbo"}
if mode == "dynamic":
    skip_prefixes = ("voices/", "native/reference-inputs/")
else:
    skip_prefixes = ()
for src in sorted(base.rglob("*")):
    rel = src.relative_to(base).as_posix()
    if not src.is_file() or rel in skip or rel.startswith(skip_prefixes):
        continue
    target = dest / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    os.link(src, target)
(dest / "native/bin").mkdir(parents=True, exist_ok=True)
(dest / "native/bin/aniflive-macos-v5turbo").write_bytes(b"")  # placeholder; refresh unlinks and copies

candidate = dict(manifest.get("qualification_candidate", {}))
candidate["baseline_source_commit"] = candidate.pop("source_commit", None)
candidate.update({
    "frozen_private_release_candidate": False,
    "functional_candidate": "fp32-f1",
    "worker_source_commit": worker_manifest["source_commit"],
    "worker_payload": "fp16-candidate-p1 native-worker (precision selected by this manifest)",
    "numeric_equivalence": "p1 worker in FP32 bit-identical to the frozen FP32 worker on 10 complete/stream cases",
    "frozen_baseline_package_preserved": True,
})
manifest.update({"model_id": model_id, "qualification_candidate": candidate})
if mode == "dynamic":
    manifest.update({"reference_mode": "dynamic", "voice_profiles": [], "default_voice_profile": None})
    manifest["native_runtime"].pop("reference_inputs", None)
(dest / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

refresh = Path(__file__).with_name("fp16_package_refresh.py")
subprocess.run([sys.executable, str(refresh), str(dest), str(payload)], check=True)
