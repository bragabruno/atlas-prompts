#!/usr/bin/env bash
# build.sh — build verification: the package imports cleanly. This repo is a
# prompt-registry + eval library (not a deployed service), so there is no OpenAPI
# contract to export — the smoke test is the importable module. Publishes nothing.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
# shellcheck source=scripts/lib/colors.sh
source scripts/lib/colors.sh
# shellcheck source=scripts/lib/common.sh
source scripts/lib/common.sh
trap 'on_err "$LINENO" "$?"' ERR

require_cmd python "pip install -e .[dev]"
run "import smoke (atlas_prompts)" python -c "import atlas_prompts"
log_ok "build verification passed"
