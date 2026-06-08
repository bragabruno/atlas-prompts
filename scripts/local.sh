#!/usr/bin/env bash
# local.sh — local dev inner loop. This repo is a prompt-registry + eval library
# (no long-running service), so "run it locally" means validating the authored
# prompt/agent schemas — the fast feedback loop while editing prompts/ + agents/.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
# shellcheck source=scripts/lib/colors.sh
source scripts/lib/colors.sh
# shellcheck source=scripts/lib/common.sh
source scripts/lib/common.sh
trap 'on_err "$LINENO" "$?"' ERR

require_cmd python "pip install -e .[dev]"
log_info "atlas-prompts — validating prompt/agent schemas (INF-1)"
exec python scripts/lint_schemas.py
