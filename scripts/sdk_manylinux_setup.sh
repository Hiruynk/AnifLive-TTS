#!/usr/bin/env bash
set -euo pipefail
# Official rustup 1.28.2 archive; checksum published by static.rust-lang.org.
sdk_bootstrap_dir="$(mktemp -d)"
trap 'rm -rf "$sdk_bootstrap_dir"' EXIT
curl --proto '=https' --tlsv1.2 --fail --silent --show-error --location \
  https://static.rust-lang.org/rustup/archive/1.28.2/x86_64-unknown-linux-gnu/rustup-init \
  --output "$sdk_bootstrap_dir/rustup-init"
printf '%s  %s\n' \
  20a06e644b0d9bd2fbdbfd52d42540bdde820ea7df86e92e533c073da0cdd43c \
  "$sdk_bootstrap_dir/rustup-init" | sha256sum --check
chmod +x "$sdk_bootstrap_dir/rustup-init"
"$sdk_bootstrap_dir/rustup-init" -y --profile minimal --default-toolchain 1.98.1 --no-modify-path
