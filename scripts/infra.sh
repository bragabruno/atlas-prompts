#!/usr/bin/env bash
# infra.sh — deploy-manifest validation: lint + render an in-repo Helm chart.
# This repo is a prompt-registry + eval library, NOT a deployed service: it ships
# no deploy/ chart by design, so this stage skips cleanly.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
# shellcheck source=scripts/lib/colors.sh
source scripts/lib/colors.sh
# shellcheck source=scripts/lib/common.sh
source scripts/lib/common.sh
trap 'on_err "$LINENO" "$?"' ERR

if [[ ! -d deploy ]]; then
  skip "infra" "no deploy/ chart (prompt-registry/eval library — not a deployed service)"
  exit 0
fi
if ! has_cmd helm; then
  skip "helm lint" "helm not installed"
  exit 0
fi
run "helm lint" helm lint deploy
log_ok "deploy manifests valid"
