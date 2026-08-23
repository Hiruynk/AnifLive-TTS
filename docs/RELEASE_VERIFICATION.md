# Release Image Verification

AnifLive-TTS publishes separate `cu121` and `cu128` images. A tag is a
convenient name, not an immutable identity. Production deployments should pin
the digest recorded by the release workflow.

For each profile, download the matching
`RELEASE-METADATA-AnifLive-TTS-v1.0.0-<profile>.json` and
`SBOM-AnifLive-TTS-v1.0.0-<profile>.spdx.json` and
`TRIVY-AnifLive-TTS-v1.0.0-<profile>.json` release assets, then verify:

1. `source_commit` equals the v1.0.0 release commit.
2. `image_digest` matches the digest reported by GHCR.
3. The OCI `org.opencontainers.image.revision` label and index annotation
   equal `source_commit`.
4. The image-derived SPDX document passes `spdx-tools==0.8.3` validation and
   names the same image.
5. The Trivy report passes `scripts/check_trivy_report.py` for its profile.
6. BuildKit provenance is attached to the immutable digest.

Inspect an image without changing its contents:

```bash
docker buildx imagetools inspect \
  ghcr.io/hiruynk/aniflive-tts@sha256:<digest>
```

Pull and run by digest:

```bash
docker pull ghcr.io/hiruynk/aniflive-tts@sha256:<digest>
```

Digest and provenance fields remain pending until the image is rebuilt by the
release workflow from the final v1.0.0 commit. Evidence from an older workflow
run must not be presented as evidence for a newer source revision.
