#!/usr/bin/env bash
# eval.sh — Gate 2 (eval quality gate, REG-11). Compares a candidate eval run to
# the production baseline via gate.py and fails on any blocking regression.
#
# This gate only runs when there is a candidate run to compare. The candidate and
# baseline run IDs are supplied via env (mirroring the Bitbucket Gate-2 step):
#   CANDIDATE_RUN_ID       — run ID produced by the eval runner (REG-8).
#   BASELINE_EVAL_RUN_ID   — production baseline run ID.
# When CANDIDATE_RUN_ID is unset there is nothing to evaluate (e.g. a local `make
# ci` or a PR that doesn't touch prompts/agents/evals), so this stage skips
# cleanly. Set GATE_NO_BASELINE=1 for the first-ever run of a prompt with no
# production baseline (gate always passes).
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
# shellcheck source=scripts/lib/colors.sh
source scripts/lib/colors.sh
# shellcheck source=scripts/lib/common.sh
source scripts/lib/common.sh
trap 'on_err "$LINENO" "$?"' ERR

if [[ -z "${CANDIDATE_RUN_ID:-}" ]]; then
  skip "eval quality gate (Gate 2)" "CANDIDATE_RUN_ID unset (no eval run to compare)"
  exit 0
fi

require_cmd python "pip install -e .[dev]"

if [[ "${GATE_NO_BASELINE:-0}" == "1" ]]; then
  run "eval quality gate (Gate 2, first run)" \
    python -m atlas_prompts.gate "${CANDIDATE_RUN_ID}" --no-baseline
else
  if [[ -z "${BASELINE_EVAL_RUN_ID:-}" ]]; then
    log_error "BASELINE_EVAL_RUN_ID unset — set it, or pass GATE_NO_BASELINE=1 for first-run mode"
    exit 2
  fi
  run "eval quality gate (Gate 2)" \
    python -m atlas_prompts.gate "${CANDIDATE_RUN_ID}" "${BASELINE_EVAL_RUN_ID}"
fi
log_ok "eval quality gate passed"
