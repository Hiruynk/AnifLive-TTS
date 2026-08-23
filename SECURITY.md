# Security Policy

## Supported Version

Security fixes are provided for the latest AnifLive-TTS v1 release. Older
snapshots, locally modified builds, model packages, and third-party deployment
overlays are outside this policy.

## Reporting a Vulnerability

Use [GitHub private vulnerability reporting](https://github.com/Hiruynk/AnifLive-TTS/security/advisories/new).
Do not disclose a suspected vulnerability in a public issue, discussion, or
pull request. Include the affected version, a minimal reproduction, expected
impact, and any proposed mitigation, but do not include credentials, private
model assets, reference audio, or personal data.

AnifLive-TTS does not currently accept external contributions or pull requests.
A private security report is a disclosure channel, not a contribution request.
Please coordinate public disclosure through the private advisory.

## Model And Checkpoint Trust Boundary

Treat every checkpoint and model package as executable-risk input. The default
converter path uses `torch.load(weights_only=True)`, but this reduces one
class of pickle risk and is **not a security sandbox**. Only process checkpoints
obtained from a source you trust.

The `--allow-unsafe-pickle` option can execute pickle reconstruction behavior.
Use it only for a trusted local checkpoint, preferably in an isolated,
network-disabled converter container with read-only checkpoint mounts. Never
enable it in the API request path.

The `cu128` image uses PyTorch 2.10 or newer, which contains the upstream fixes
for `GHSA-53q9-r3pm-6pq6` and `GHSA-63cw-57p8-fm3p`. PyTorch does not publish a
2.10 CUDA 12.1 wheel, so the compatibility-focused `cu121` image retains an
older PyTorch runtime. In that profile, checkpoint conversion is an explicit
trusted-administrator operation and not a remote API capability. Do not mount
untrusted checkpoints into the `cu121` container.

The vendored GPT-SoVITS frontend uses a patched Transformers release and is
also constrained to local, checksum-pinned assets with
`local_files_only=True`, `trust_remote_code=False`, `HF_HUB_OFFLINE=1`, and
`TRANSFORMERS_OFFLINE=1`.

TensorRT engines are specific to their build environment and must be rebuilt on
the target GPU from the package's portable ONNX files. Verify release digests,
source revision, SBOM, and provenance before deployment.

## Deployment Boundary

The API binds to loopback by default and does not provide public-edge
authentication. Public deployments require an authenticated reverse proxy,
request limits, transport security, and a private model storage policy.
