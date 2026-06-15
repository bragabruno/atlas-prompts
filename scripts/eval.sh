#!/usr/bin/env bash
# eval.sh — Gate 2 (eval quality gate, REG-11 + REG-12). Compares a candidate
# eval run to the production baseline via gate.py, fails on any blocking
# regression (red required check), and posts the metric-diff as a PR comment.
#
# This gate only runs when there is a candidate run to compare. The candidate and
# baseline run IDs are supplied via env (mirroring the Bitbucket Gate-2 step):
#   CANDIDATE_RUN_ID       — run ID produced by the eval runner (REG-8).
#   BASELINE_EVAL_RUN_ID   — production baseline run ID.
# When CANDIDATE_RUN_ID is unset there is nothing to evaluate (e.g. a local `make
# ci` or a PR that doesn't touch prompts/agents/evals), so this stage skips
# cleanly. Set GATE_NO_BASELINE=1 for the first-ever run of a prompt with no
# production baseline (gate always passes).
#
# REG-12: the gate writes a machine-readable comparison via --json; that file is
# fed to the PR-comment poster, which renders the metric diff and posts it to the
# pull request. The comment is posted on BOTH pass and regression. The gate's
# exit code is captured and re-raised afterwards so a regression still turns the
# required check red. Outside a PR (no BITBUCKET_PR_ID etc.) the poster is a clean
# no-op, so local `make ci` is unaffected.
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

# Machine-readable comparison the poster consumes (cleaned up on exit).
GATE_DIFF_JSON="$(mktemp -t atlas-eval-diff.XXXXXX.json)"
trap 'rm -f "${GATE_DIFF_JSON}"' EXIT

# Run the gate, capturing its exit code WITHOUT tripping `set -e`: a blocking
# regression exits 1 and we must still post the comment before re-exiting.
gate_rc=0
if [[ "${GATE_NO_BASELINE:-0}" == "1" ]]; then
  run "eval quality gate (Gate 2, first run)" \
    python -m atlas_prompts.gate "${CANDIDATE_RUN_ID}" --no-baseline \
      --json "${GATE_DIFF_JSON}" || gate_rc=$?
else
  if [[ -z "${BASELINE_EVAL_RUN_ID:-}" ]]; then
    log_error "BASELINE_EVAL_RUN_ID unset — set it, or pass GATE_NO_BASELINE=1 for first-run mode"
    exit 2
  fi
  run "eval quality gate (Gate 2)" \
    python -m atlas_prompts.gate "${CANDIDATE_RUN_ID}" "${BASELINE_EVAL_RUN_ID}" \
      --json "${GATE_DIFF_JSON}" || gate_rc=$?
fi

# A gate usage error (rc 2) means there is no comparison to report — surface it.
if [[ "${gate_rc}" -ge 2 ]]; then
  exit "${gate_rc}"
fi

# Post the metric-diff comment on BOTH pass and regression. The poster no-ops
# cleanly when there is no PR context (local runs) and fails fast on a real
# misconfiguration (PR context set but token missing).
post_rc=0
run "eval-gate PR comment (REG-12)" \
  python -m atlas_prompts.evals.gate.pr_comment "${GATE_DIFF_JSON}" || post_rc=$?

# Preserve the gate's verdict as the stage's exit code so a regression keeps the
# required check red. When the gate passed, a poster failure (misconfig) is a
# real problem and surfaces as the stage's exit code.
if [[ "${gate_rc}" -ne 0 ]]; then
  log_error "eval quality gate failed (rc=${gate_rc}) — comment posted, failing the check"
  exit "${gate_rc}"
fi
if [[ "${post_rc}" -ne 0 ]]; then
  exit "${post_rc}"
fi
log_ok "eval quality gate passed"
