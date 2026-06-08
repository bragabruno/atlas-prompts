#!/usr/bin/env bash
# docker.sh — container build verification.
# This repo is a prompt-registry + eval library, NOT a deployed service: it ships
# no Dockerfile and no deploy/ chart by design, so there is no image to build.
# This stage skips cleanly (keeps `make ci` uniform across repos).
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
# shellcheck source=scripts/lib/colors.sh
source scripts/lib/colors.sh
# shellcheck source=scripts/lib/common.sh
source scripts/lib/common.sh
trap 'on_err "$LINENO" "$?"' ERR

image="${ATLAS_IMAGE:-atlas-prompts:dev}"
if [[ ! -f Dockerfile ]]; then
  skip "docker build" "no Dockerfile (prompt-registry/eval library — not a deployed service)"
  exit 0
fi
require_cmd docker "Docker Desktop / a running daemon"
if ! docker info >/dev/null 2>&1; then
  skip "docker build" "docker daemon not running"
  exit 0
fi
run "docker build ${image}" docker build -t "$image" .
log_ok "image built: ${image}"
