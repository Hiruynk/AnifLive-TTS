# Repository Security Settings

This checklist records the GitHub settings required before changing the
AnifLive-TTS repository or its GHCR packages from private to public. Repository
and package visibility are separate controls.

## Required Before Public Release

- Keep the repository and both GHCR packages private until the release commit,
  image digests, SBOMs, and provenance have been reviewed together.
- Enable GitHub private vulnerability reporting.
- Enable the dependency graph, Dependabot alerts, security updates, secret
  scanning, and push protection where the account plan provides them.
- Set the default Actions token to read-only. Grant `packages: write` only to
  the image build job.
- Restrict Actions to trusted publishers or the full-SHA-pinned actions in this
  repository. Require approval for workflows from forks.
- Protect `main`: require the `static-quality` check, block force pushes and
  branch deletion, and limit bypass rights.
- Keep Issues, Discussions, and pull requests disabled while external
  contributions are not accepted.
- Review collaborators, deploy keys, Actions secrets, environments, webhooks,
  installed GitHub Apps, and cached artifacts.
- Run secret scanning across the complete Git history. Rotate any credential
  found in current or historical objects before publication.

## Release Evidence

The `container` workflow builds each image from `${{ github.sha }}`, writes
that commit to the OCI `org.opencontainers.image.revision` label and index
annotation, enables BuildKit SBOM and maximum provenance attestations, and
uploads these files:

- `SBOM-AnifLive-TTS-v1.1.0-cu126.spdx.json`
- `SBOM-AnifLive-TTS-v1.1.0-cu128.spdx.json`
- `RELEASE-METADATA-AnifLive-TTS-v1.1.0-cu126.json`
- `RELEASE-METADATA-AnifLive-TTS-v1.1.0-cu128.json`
- `TRIVY-AnifLive-TTS-v1.1.0-cu126.json`
- `TRIVY-AnifLive-TTS-v1.1.0-cu128.json`

The metadata file records the immutable image digest, source commit, workflow
run URL, SBOM filename, and attestation policy. Attach the reviewed files to the
matching GitHub Release. Do not reuse evidence produced for an earlier commit.

## Visibility Change

1. Verify the release commit has passed CI and the container security gate.
2. Rebuild both image tags from that exact commit.
3. Compare the OCI revision and metadata source commit with the release commit.
4. Verify each digest and its attached SBOM/provenance before changing
   visibility.
5. Change repository visibility only after the history scan is clean.
6. Change each GHCR package visibility separately, then verify anonymous pulls
   by immutable digest.
7. Confirm the release assets contain no model, voice, reference-audio,
   credential, cache, report, or private deployment data.

If any identity, digest, or provenance value differs, keep the release private
and rebuild from the intended commit.
