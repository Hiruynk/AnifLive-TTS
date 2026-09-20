# ADR: v1.5 Rust SDK packaging

Status: selected design for implementation; qualification pending.
Date: 2026-09-20. Baseline: `2bb0b48d2d4821eec6f0d19208c27538812be2fc`.
PR00 changes no build backend, dependency metadata, native code or Docker build.

## Existing evidence

`pyproject.toml` uses setuptools.build_meta with setuptools 84.0.0 and wheel
0.48.0, a src package layout, Python >=3.10,<3.13, the aniflive-tts distribution
and cli:main entry point, and workstation_assets_lock.json package data.
Its five runtime requirements are fastapi>=0.115,<1, httpx>=0.28,<1,
numpy>=1.26,<3, pydantic>=2.10,<3 and uvicorn[standard]>=0.34,<1
(uvicorn includes its standard extras).
The CLI lazily imports command implementations. Current root import only
sets version. requirements/base.txt and CUDA/TensorRT requirement files
separately pin deployment dependencies.

CI installs requirements/dev.txt then `pip install -e .` and runs existing
pytest/server/security checks. It has no native wheel matrix. The ci.yml
security gate still specifies 1.4.0.dev0 while baseline metadata is 1.4.0;
version work must reconcile this explicitly, not hide it in SDK test results.

## Decision and alternatives

Retain setuptools.build_meta and add setuptools-rust to build one private
PyO3 cdylib at aniflive_tts._native. Preserve distribution/import name,
src layout, CLI, Python modules and package data. Cargo workspace separates the
client core from bindings. No pure-Python HTTP fallback and no same-version
server-only wheel missing the extension.

| Option | Assessment |
| --- | --- |
| setuptools + setuptools-rust | Selected: adds native build integration while retaining current metadata, discovery and entry points |
| maturin mixed project | Maintained and viable; migration would require requalifying package-data/source-layout/build configuration without removing Docker native-build work |
| separate overlapping Python distribution | Rejected for Beta: introduces namespace/file ownership and version-skew installation hazards |
| custom build commands or Python HTTP fallback | Rejected: unnecessary maintenance or violates single Rust protocol ownership |

The [setuptools-rust project](https://pypi.org/project/setuptools-rust/1.13.0/)
documents pyproject extension targets, src-style discovery, editable install and
explicit Rust source inclusion. This establishes feasibility, not proof that
this repository's complete build passes. The
[maturin mixed-project guide](https://www.maturin.rs/project_layout.html)
supports private nested extensions and a configured Python source directory;
maturin is not rejected as incapable.

## Lightweight client and deliberate runtime migration

The intended bare wheel has no Python runtime dependencies beyond the standard
library; Rust transport is linked into the extension. Move the existing five
requirement entries unchanged into a `runtime` extra. A full source/developer
install becomes `pip install -e '.[runtime]'`; bare editable installation
remains valid for SDK development and requires the Rust build toolchain.
Runtime/server documentation and CI must migrate in the same implementation PR.
The runtime extra is not a replacement for locked inference/worker requirements.

This is an intentional change to what bare installation brings in, requiring
migration notes before release. Keep CLI commands and lazy imports; a client-only
user invoking a runtime command must receive an actionable missing-runtime
dependency message. Existing runtime deployments install their explicit locked
requirements and the identical wheel with --no-deps. Do not resolve broader
extras over the qualified pinned runtime environment. No CUDA, TensorRT,
PyTorch, NumPy C API, models or Docker dependency enters the native client.

## Upstream evidence and candidate versions

Observed from primary upstream sources on 2026-09-20; these are researched
candidates, not an existing lockfile or a claim of joint build qualification.

| Component | Candidate | Declared license | Evidence |
| --- | --- | --- | --- |
| PyO3 | 0.29.2 | MIT OR Apache-2.0 | [release manifest](https://github.com/PyO3/pyo3/blob/v0.29.2/Cargo.toml) |
| pyo3-async-runtimes | 0.29.0 | Apache-2.0 | [upstream manifest](https://github.com/PyO3/pyo3-async-runtimes/blob/main/Cargo.toml), [released crate](https://docs.rs/crate/pyo3-async-runtimes/0.29.0) |
| setuptools-rust | 1.13.0 | MIT | [release metadata](https://pypi.org/project/setuptools-rust/1.13.0/) |
| maturin (alternative) | 1.15.0 | MIT OR Apache-2.0 | [release metadata](https://pypi.org/project/maturin/1.15.0/) |
| reqwest | 0.13.5 | MIT OR Apache-2.0 | [released docs](https://docs.rs/reqwest/0.13.5/reqwest/), [manifest](https://github.com/seanmonstar/reqwest/blob/master/Cargo.toml) |

The async bridge declares PyO3 0.29 and a Tokio feature. Use that matched line,
not the older pyo3-asyncio crate or experimental PyO3 async features.
Select minimal reqwest JSON/stream/TLS features, explicitly disabling unwanted
defaults; verify exact 0.13 feature names in implementation. Prefer rustls to
avoid a system OpenSSL client requirement. Inspect crypto-provider and all
transitive licenses/build requirements before freezing Cargo.lock. Lock exact
resolved crates and a tested Rust toolchain; upstream minimum versions alone
are not a tested toolchain. Preserve repository licensing and add bundled
dependency notices/SBOM; these permissive declarations do not relicense the
original project or constitute a completed license audit.

Choose Rust 1.98.1 as the initial pinned builder/CI toolchain, from the
[official release announcement](https://blog.rust-lang.org/releases/latest/)
(2026-09-03). Set the initial Beta MSRV to 1.98, subject to the first actual
build; do not promise older toolchains based on a single dependency minimum.
Use a manylinux_2_28 builder for distributable Linux wheels. A Python 3.12 QA
container with this toolchain can qualify the initial local skeleton, but its
locally built wheel is not automatically a repaired manylinux release wheel.

## Wheels, ABI and sources

Prefer PyO3 abi3-py310 for the existing CPython 3.10-3.12 range, with setuptools
limited-API wheel tagging. [PyO3 distribution guidance](https://pyo3.rs/v0.29.2/building-and-distribution.html)
explains stable ABI limitations. The extension feature alone does not prove
the wheel tag or async compatibility. Test the same wheel on all three
interpreters; if ABI3 fails, document evidence and use per-interpreter wheels
without claiming unsupported ABI compatibility. Do not widen requires-python
merely because upstream PyO3 supports newer Python.

Qualification targets: Windows x86_64, Linux x86_64 (proposed
manylinux_2_28 baseline), macOS arm64. No platform is supported until wheel
build/install and SDK contracts pass there. No musllinux, PyPy,
free-threaded Python or cross-interpreter claim in Beta. CI uses appropriate
native runners/build containers and wheel repair/inspection tools; no local
Linux run can qualify Windows/macOS.

Ordinary wheel users need no Rust compiler. Source/sdist and editable builds
require a documented Rust toolchain; rebuilding Rust is necessary after native
source edits. Include workspace/crate manifests, Cargo.lock, .rs sources and
required build files explicitly in sdist; keep local execution instructions and
scratch artifacts excluded. Include py.typed, _native.pyi, existing JSON data,
licenses and the entry point in wheels. Check archive contents and rebuild a
wheel from unpacked sdist. Never label native wheels py3-none-any.

## Docker migration impact

| Path at baseline | Required later change |
| --- | --- |
| Dockerfile copies metadata/src then pip install --no-deps /app | Build identical wheel in a builder stage with Rust/manifests/sources; copy/install wheel into current runtime dependency environment |
| Dockerfile.dev installs copied source with --no-deps --no-build-isolation | Replace install with exact revision wheel from builder; adding build requirements alone cannot fix this no-isolation path |
| Dockerfile.workstation-worker has the same source/no-isolation install | Same wheel flow after existing worker dependency setup; preserve worker assets and runtime qualification |
| Dockerfile.studio overlays Python source into Python 3.12 site-packages of WORKER_IMAGE | Install matching whole wheel; do not overlay a new facade over an older inherited native extension |
| docker-compose.yml and docker-compose.cu126.yml | Preserve existing CUDA base selection, mounts, command and healthcheck; pass builder inputs if required and test both variants |

Builder stages must match wheel target architecture and compatible libc, use
locked inputs, and never copy Rust toolchains/caches to final images.
Each final image must show matching Python/native/distribution versions and
retain CLI/package-data behavior. Runtime images contain the same complete
package, regardless of whether their entry point uses the SDK. This ADR
specifies migrations; it does not report these Docker builds as already passing.

A pre-existing clean-source issue also blocks the production build:
Dockerfile copies assets/everynight_dance.gif, absent at the audited HEAD.
The existing WebUI route already returns a transparent GIF when this optional
asset is missing (tests/test_webui_optional_assets.py). Remove the stale COPY
in the Docker implementation PR and retain that tested fallback. Do not restore
the historical media: assets/THIRD_PARTY_MEDIA.md records no documented
redistribution permission. This is separate from native packaging and no
baseline Docker success is inferred.

## Qualification and rollback

Before packaging completion: build/editable import; native import without heavy
dependencies; CLI --help plus representative existing runtime tests; sdist
rebuild; clean wheel installation on the matrix; stub/package-data inspection;
all existing tests plus SDK contracts; all four Docker builds and both CUDA
Compose configurations; final-image import/CLI/version smoke tests without Rust.
Keep fmt/clippy/Rust unit tests and existing Python/security checks. Confirm
Cargo source dependencies are complete and scans/licenses include native code.

PR00 review is static repository/upstream evidence only. No native build,
ABI spike, wheel matrix, real-server/GPU test or Docker image build was run
for this design. Later implementation must record actual commands/results.
Revert packaging/native/facade/Docker changes together if qualification fails;
do not deploy a mismatched facade/native pair or substitute Python transport.
