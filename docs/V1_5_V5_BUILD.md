# v5Turbo Model Packages

The CUDA／TensorRT（Docker） backend for v5Turbo uses the shared upstream model and stored
Voice/Reference audio. It produces 48 kHz mono PCM16 with four CFM steps and
CFG 0. v2ProPlus remains a separate 32 kHz backend. See the
[platform matrix](V1_5_PLATFORM_MATRIX.md) and [Voice API](V1_5_VOICE_API.md).

## Runtime Image

The default CUDA 12.8 image includes both v2ProPlus and v5Turbo dependencies:

```sh
docker build -t aniflive-tts:1.5.0-cu128 .
```

To reuse a v1.5 CUDA 12.8 base built from the same source revision and current
SDK, build the v5Turbo overlay instead:

```sh
docker build -f Dockerfile.v5 -t aniflive-tts:1.5.0-v5-cu128 .
```

The overlay shares the base CUDA/PyTorch layers. Both recipes explicitly apply
the upstream LightGlue configuration security fix to the pinned Transformers
profile; the patch verifies its input and output source hashes.

The v5Turbo profile pins PyTorch 2.10.0+cu128, TensorRT 11.2.1.2 and Transformers
4.57.6. It needs Linux containers, an NVIDIA GPU and NVIDIA Container Toolkit
(Docker Desktop with WSL2 on Windows). Use the [macOS installation guide](MACOS_INSTALLATION_EN.md) for Apple Silicon（macOS／MLX／Metal）.

## Inputs And Local Engine Build

Obtain [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) source at commit
`04a44eb977586645f33594b72af13a035a2056c6` and the upstream v5Turbo model assets.
Keep the upstream license. The assembler checks the pinned AR, acoustic and
vocoder checkpoint hashes, source files, and prepared frontend resources.
The source checkout is separate from the output package.

Run the following inside the v5Turbo CUDA 12.8 container, with local input and output
directories mounted. Paths below are container paths supplied by the operator.
Use a new output directory for each build.

```sh
python /app/scripts/v5_fetch_headers.py --output /trt-sdk
python /app/scripts/v5_build_native.py --checkpoint /models/s1v3.ckpt \
  --tensorrt-headers /trt-sdk --output /build/native
python /app/scripts/v5_build_ar.py --checkpoint /models/s1v3.ckpt \
  --native-build /build/native --output /build/ar
python /app/scripts/v5_build_acoustic.py --component encoder \
  --checkpoint /models/gsv-v5-pretrained/s2Gv5turbo.pth --output /build/encoder
python /app/scripts/v5_build_acoustic.py --component dit --source /upstream \
  --checkpoint /models/gsv-v5-pretrained/s2Gv5turbo.pth --output /build/dit
python /app/scripts/v5_build_acoustic.py --component vocoder \
  --checkpoint /models/gsv-v5-pretrained/vocoder.pth --output /build/vocoder
```

`/trt-sdk` must contain the matching TensorRT SDK headers. A working C/C++
compiler and CUDA headers are needed for the native helpers. The ONNX topology
templates contain no learned initializers; the builders recover and verify
them from the operator-supplied checkpoints. `--export-only` is available for
the AR and acoustic exporters; it does not build or validate an engine.

TensorRT tactics can differ after rebuilding. Rebuilt plans start as `unqualified`; validate their audio and streaming behavior on the target GPU before deployment, keeping the previous package available for rollback.

## Assemble The Package

The assembler takes the pinned source, model assets, prepared Japanese/G2PW
frontend caches, native helpers and engine build directories. It includes the
vendored Apache-licensed English G2P implementation and its provenance, without
the unused Distance dependency.

Prepare the derived dictionaries in the operator's pinned source checkout,
then capture the actual build target's fingerprint (inside the GPU container):

```sh
PYTHONPATH=/upstream/GPT_SoVITS python -c 'import text.japanese; import text.g2pw.g2pw'
python -c 'import json; from aniflive_tts.model_package import runtime_fingerprint; print(json.dumps(runtime_fingerprint(), indent=2))' > /build/target-runtime.json
```

Use the resulting `/upstream/GPT_SoVITS/text/ja_userdic` and
`/upstream/GPT_SoVITS/text/g2pw` as the two cache inputs below, or copy them to
the illustrated `/frontend` directories.

```sh
python /app/scripts/v5_assemble_package.py --model-id v5turbo-shared-base \
  --source /upstream --models /models --g2pw-model /models/G2PWModel \
  --ja-cache /frontend/ja_userdic --g2pw-cache /frontend/g2pw \
  --native /build/native --ar /build/ar --encoder /build/encoder \
  --dit /build/dit --vocoder /build/vocoder \
  --runtime-fingerprint /build/target-runtime.json \
  --output /packages/v5turbo-shared-base
```

`target-runtime.json` records the actual target's TensorRT/CUDA versions, GPU
compute capability, SM count, memory, operating system and architecture. The
runtime validates the fingerprint before loading the package. See
[model-package validation](V1_5_VOICE_API.md).

The output contains `manifest.json`, `source-lock.json`, checksums, source and
model assets, and `engines/<fingerprint>/engine-manifest.json`. It contains no
user Reference audio or preselected voice. `--link-models` can hardlink immutable
weights on the same filesystem; it fails if hardlinking is unavailable. Do not
modify hardlinked inputs or outputs.

Select the package through the existing model registry and create/prepare a
Reference using the [Voice API](V1_5_VOICE_API.md). Keep Voice data persistent
across container replacement. See [deployment](DEPLOYMENT_EN.md) and
[rollback](ROLLBACK.md) for the existing volume and upgrade workflow.
