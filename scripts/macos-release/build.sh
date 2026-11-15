#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/macos-release/common.sh
source "$script_dir/common.sh"

usage() {
    cat <<'USAGE'
Usage: build.sh [--output DIR] [--offline]

Builds a relocatable native worker payload (binary + MLX runtime libraries).
It does not download, convert, or package model weights or reference audio.
USAGE
}

output="${ANIFLIVE_TTS_NATIVE_OUTPUT:-}"
offline="${ANIFLIVE_TTS_CARGO_OFFLINE:-0}"
while (($#)); do
    case "$1" in
        --output)
            (($# >= 2)) || { usage >&2; exit 2; }
            output="$2"
            shift 2
            ;;
        --offline)
            offline=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            macos_release_error "unknown option: $1"
            usage >&2
            exit 2
            ;;
    esac
done

macos_release_check_platform
macos_release_require_python
command -v cargo >/dev/null 2>&1 || { macos_release_error "Cargo is required"; exit 1; }
command -v rustc >/dev/null 2>&1 || { macos_release_error "rustc is required"; exit 1; }

mlx_root="$("$ANIFLIVE_TTS_PYTHON" - <<'PY'
import importlib.metadata
import importlib.util
from pathlib import Path

spec = importlib.util.find_spec("mlx")
if spec is None:
    raise SystemExit("The configured Python 3.12 environment has no MLX package")
if importlib.metadata.version("mlx") != "0.32.2":
    raise SystemExit("The release build requires mlx==0.32.2")
if importlib.metadata.version("mlx-metal") != "0.32.2":
    raise SystemExit("The release build requires mlx-metal==0.32.2")
locations = list(spec.submodule_search_locations or ())
if not locations:
    raise SystemExit("The configured MLX package has no package directory")
print(Path(locations[0]).resolve())
PY
)"
mlx_lib="$mlx_root/lib"
for file in libmlx.dylib libjaccl.dylib mlx.metallib; do
    [[ -f "$mlx_lib/$file" ]] || { macos_release_error "missing pinned MLX runtime file: $mlx_lib/$file"; exit 1; }
done
[[ -f "$mlx_root/share/cmake/MLX/MLXConfigVersion.cmake" ]] || {
    macos_release_error "configured MLX wheel lacks CMake metadata required by mlx-sys: $mlx_root"
    exit 1
}

source_commit="$(git -C "$MACOS_RELEASE_REPO_ROOT" rev-parse HEAD 2>/dev/null || printf 'unknown')"
if [[ -z "$output" ]]; then
    output="$ANIFLIVE_TTS_MACOS_RUNTIME_ROOT/build/native-worker-${source_commit:0:8}-$(date +%Y%m%d%H%M%S)"
fi
mkdir -p "$(dirname "$output")"
output="$(cd "$(dirname "$output")" && pwd)/$(basename "$output")"
if [[ -e "$output" ]]; then
    macos_release_error "output already exists; choose a fresh path: $output"
    exit 1
fi
temporary="$output.tmp.$$"
if [[ -e "$temporary" ]]; then
    macos_release_error "temporary output already exists: $temporary"
    exit 1
fi
mkdir -p "$(dirname "$output")"
mkdir -p "$temporary/native/bin" "$temporary/native/lib" "$temporary/native/licenses"
cleanup() { [[ ! -d "$temporary" ]] || rm -rf "$temporary"; }
trap cleanup EXIT

cargo_target="${ANIFLIVE_TTS_CARGO_TARGET_DIR:-$ANIFLIVE_TTS_MACOS_RUNTIME_ROOT/build/cargo-target-worker}"
mkdir -p "$cargo_target"
export MLX_SYS_PREBUILT="$mlx_root"
export CARGO_TARGET_DIR="$cargo_target"
# Rust worker compatibility follows the repository target; bundled MLX dylibs
# still set the effective package OS floor (documented separately).
export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-11.0}"
# The installed MLX dylibs are staged beside the executable in native/lib.
export RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }-C link-arg=-Wl,-rpath,@executable_path/../lib"
export PATH="$(dirname "$ANIFLIVE_TTS_PYTHON"):$PATH"
args=(build --release --locked --manifest-path "$MACOS_RELEASE_REPO_ROOT/rust/macos-v5turbo/Cargo.toml" --bin aniflive-macos-v5turbo)
case "$offline" in
    0) ;;
    1) args+=(--offline) ;;
    *) macos_release_error "ANIFLIVE_TTS_CARGO_OFFLINE must be 0 or 1"; exit 2 ;;
esac
(
    cd "$MACOS_RELEASE_REPO_ROOT"
    cargo "${args[@]}"
)

binary="$cargo_target/release/aniflive-macos-v5turbo"
[[ -x "$binary" ]] || { macos_release_error "Cargo did not produce the native worker: $binary"; exit 1; }
install -m 755 "$binary" "$temporary/native/bin/aniflive-macos-v5turbo"
for file in libmlx.dylib libjaccl.dylib mlx.metallib; do
    install -m 644 "$mlx_lib/$file" "$temporary/native/lib/$file"
done
install -m 644 "$MACOS_RELEASE_REPO_ROOT/LICENSE" "$temporary/LICENSE-AnifLive-TTS.txt"

"$ANIFLIVE_TTS_PYTHON" - "$temporary" "$mlx_root" "$source_commit" "$MACOS_RELEASE_REPO_ROOT" "$MACOSX_DEPLOYMENT_TARGET" <<'PY'
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys

root = Path(sys.argv[1])
mlx_root = Path(sys.argv[2])
source_commit = sys.argv[3]
repo_root = Path(sys.argv[4])
deployment_target = sys.argv[5]

def copy_license(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise SystemExit(f"Missing required third-party license file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)

for package, output_name in (("mlx", "mlx-Python-MIT.txt"),
                             ("mlx-metal", "mlx-metal-MIT.txt")):
    distribution = importlib.metadata.distribution(package)
    licenses = [f for f in (distribution.files or ())
                if "licenses" in str(f).replace("\\", "/").split("/")
                and Path(str(f)).name.upper() in {"LICENSE", "LICENSE.TXT", "LICENSE.MD"}]
    if not licenses:
        raise SystemExit(f"The installed {package} distribution does not expose its license file")
    copy_license(Path(distribution.locate_file(licenses[0])), root / "native/licenses" / output_name)

metadata = json.loads(subprocess.run(
    ["cargo", "metadata", "--format-version", "1", "--locked", "--manifest-path",
     str(repo_root / "rust/macos-v5turbo/Cargo.toml")],
    cwd=repo_root, capture_output=True, text=True, check=True,
).stdout)
mlx_package = next((p for p in metadata["packages"] if p["name"] == "mlx-rs" and p["version"] == "0.34.0"), None)
if mlx_package is None:
    raise SystemExit("Pinned mlx-rs 0.34.0 package was not resolved by Cargo")
mlx_workspace = Path(mlx_package["manifest_path"]).resolve().parent.parent
copy_license(mlx_workspace / "LICENSE-MIT", root / "native/licenses" / "mlx-rs-MIT.txt")
copy_license(mlx_workspace / "LICENSE-APACHE", root / "native/licenses" / "mlx-rs-APACHE.txt")
copy_license(mlx_workspace / "mlx-sys/src/mlx-c/LICENSE", root / "native/licenses" / "mlx-c-MIT.txt")

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

payload = sorted(p for p in root.rglob("*") if p.is_file()
                 and p.name not in {"checksums.json", "build-manifest.json"})
checksums = {p.relative_to(root).as_posix(): sha(p) for p in payload}
(root / "checksums.json").write_text(json.dumps(checksums, indent=2, sort_keys=True) + "\n")
dirty = bool(subprocess.run(["git", "-C", str(repo_root), "status", "--porcelain"],
                            capture_output=True, text=True, check=False).stdout.strip())
manifest = {
    "kind": "aniflive-macos-native-worker-payload",
    "model_assets_included": False,
    "reference_audio_included": False,
    "source_commit": source_commit,
    "source_tree_dirty_at_build": dirty,
    "rust_toolchain_observed": subprocess.run(["rustc", "--version"], capture_output=True,
                                               text=True, check=True).stdout.strip(),
    "mlx_version": "0.32.2",
    "mlx_prebuilt_source": "configured Python 3.12 environment's mlx wheel",
    "macos_deployment_target": deployment_target,
    "files_sha256": checksums,
}
(root / "build-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print(json.dumps({"worker_sha256": checksums["native/bin/aniflive-macos-v5turbo"],
                  "payload_file_count": len(checksums)}, sort_keys=True))
PY

mv "$temporary" "$output"
trap - EXIT
printf 'Native worker payload: %s\n' "$output"
