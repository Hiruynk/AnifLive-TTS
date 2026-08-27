# AnifLive-TTS v1.2 Mamba-2 TensorRT 11 Feasibility Lab

This directory is an isolated build and validation environment. It does not
modify or depend on the AnifLive-TTS production Dockerfile.

## Scope

- CUDA 12.8 development image with `nvcc`
- TensorRT 11.2.1.2 public headers and runtime
- Standalone C++ `IPluginV3` Mamba-2 recurrent update
- Configurable Mamba-2 state dimensions up to 128, including the
  GPT-SoVITS V2ProPlus contract (`dim=512`, `nheads=16`, `dstate=32`)
- FP16 output and recurrent-state parity against a NumPy FP32 reference with
  FP16 state quantization
- Two persistent TensorRT execution contexts with ping-pong recurrent buffers
- 4,096-step stability test
- CUDA-event and wall-clock microbenchmark
- Offline runtime validation via `docker run --network none --pull never`

The experiment does not include a PyTorch neural runtime fallback, Mamba
training, a hybrid GPT checkpoint, or production API integration.

## Run

```powershell
powershell -ExecutionPolicy Bypass -File .\build_and_validate.ps1
```

The script uses Docker CLI only. It refuses to start when the Docker daemon is
unavailable and never launches Docker Desktop.

## Feasibility Result

The standalone plugin gate was completed on the target RTX 5070 Ti. The
experiment proved all of the following:

1. TensorRT engine build and deserialization succeed.
2. `enqueueV3` executes the compiled plugin.
3. One-step and 64-step FP16 reference parity pass.
4. The recurrent state remains finite for 4,096 steps.
5. The report contains measured GPU and wall-clock latency.

This result establishes standalone TensorRT 11 plugin feasibility only. The
subsequent 1:1 Transformer/Mamba-2 autoregressive student did not pass the
production gate: across 42 validation records its mean teacher/student token
similarity was `0.160671`, while mean student time was `3.566 s` versus
`1.121 s` for the Transformer teacher. Mamba-2 is therefore not an AnifLive-TTS
v1.2 runtime backend.
