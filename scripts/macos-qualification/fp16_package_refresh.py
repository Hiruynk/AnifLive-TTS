#!/usr/bin/env python3
"""Install a built worker payload into an FP16 or FP32 model package.

The precision comes from the package manifest. Unlinks the old binary and
metadata first (never writes through a hard link), then rewrites
the engine fingerprint (sha256 over sorted relpath:sha256 of native/bin, lib,
models) and the complete checksums.json inventory.
"""
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
package, payload = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()
precision = json.loads((package / 'manifest.json').read_text())['precision']
if precision not in ('FP16', 'FP32'):
    raise SystemExit(f'unsupported package precision: {precision}')
commit = json.loads((payload / 'build-manifest.json').read_text())['source_commit']
native = payload / 'native' if (payload / 'native/bin').is_dir() else payload
assets = [Path('bin/aniflive-macos-v5turbo'), *(Path('lib') / name for name in
          ('libmlx.dylib', 'libjaccl.dylib', 'mlx.metallib'))]
licenses = native / 'licenses'
if not licenses.is_dir():
    raise SystemExit('Worker payload is missing licenses')
assets.extend(p.relative_to(native) for p in sorted(licenses.rglob('*')) if p.is_file())
for relative in assets:
    source, destination = native / relative, package / 'native' / relative
    if not source.is_file():
        raise SystemExit(f'Worker payload is missing {relative}')
    if source.resolve() == destination.resolve():
        raise SystemExit('Worker payload must be outside the destination model package')
for relative in assets:
    source, destination = native / relative, package / 'native' / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    shutil.copy2(source, destination)
binary = package / 'native/bin/aniflive-macos-v5turbo'
os.chmod(binary, 0o755)
ident = sorted((p.relative_to(package).as_posix(), sha(p)) for d in ('native/bin', 'native/lib', 'native/models') for p in (package / d).rglob('*') if p.is_file())
fp = hashlib.sha256('\n'.join(f'{a}:{b}' for a, b in ident).encode()).hexdigest()
em = {"kind": "aniflive-tts-gsv-v5turbo-mlx-metal", "fingerprint": fp, "backend": "mlx-metal", "qualified": False, "precision": precision,
      "fingerprint_scope": "sha256 over sorted relpath:sha256 of native/bin, native/lib, native/models", "source_commit": commit,
      "worker_sha256": sha(binary)}
(package / 'native/engine-manifest.json').unlink(missing_ok=True)
(package / 'native/engine-manifest.json').write_text(json.dumps(em, indent=2) + '\n')
files = {p.relative_to(package).as_posix(): sha(p) for p in sorted(package.rglob('*')) if p.is_file() and p.name != 'checksums.json'}
(package / 'checksums.json').unlink(missing_ok=True)
(package / 'checksums.json').write_text(json.dumps(files, indent=2, sort_keys=True) + '\n')
print(json.dumps({"fingerprint": fp, "worker_sha256": em["worker_sha256"], "source_commit": commit, "files": len(files)}))
