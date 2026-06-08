# atlas-prompts — build scripts

Single source of truth for build & validation. Developers and CI run the **same**
scripts; `make ci` (or `./scripts/ci.sh`) runs the full gate locally, and the
Bitbucket pipeline calls the same per-stage scripts. Cross-repo guide:
[atlas-docs/07-build-system.md](../../atlas-docs/07-build-system.md).

| Script | Make target | What it does |
|---|---|---|
| `lint.sh` | `make lint` | dep age audit (XCUT-4; also audits `.trunk/trunk.yaml` pins) + Trunk(ruff lint+format) + pyright (strict) + prompt/agent schema lint (INF-1) |
| `test.sh` | `make test` | offline unit tests (pytest; FakeGatewayClient, zero API spend) |
| `eval.sh` | `make eval` | Gate 2 eval quality gate (`gate.py`; runs only with `CANDIDATE_RUN_ID` set) |
| `coverage.sh` | `make coverage` | pytest-cov gate (recommended; `ATLAS_COV_MIN`) |
| `build.sh` | `make build` | import smoke (`import atlas_prompts`) |
| `docker.sh` | `make docker` | container build — skips (prompt-registry/eval library, not a deployed service) |
| `infra.sh` | `make infra` | `helm lint`/`template` the `deploy/` chart — skips (no chart) |
| `security.sh` | `make security` | secret / CVE / fs scans (advisory; `ATLAS_SECURITY_STRICT=1`) |
| `ci.sh` | `make ci` | runs all of the above, in order |
| `local.sh` | `make local` | validate prompt/agent schemas (local dev inner loop) |

`lib/common.sh` + `lib/colors.sh` hold the shared helpers (logging, timing,
command checks, error trap). All scripts are bash with `set -Eeuo pipefail`,
shellcheck-clean, idempotent, and run on Linux + macOS. Stages that are N/A for
this repo, or whose tools/inputs are absent, print `↷ skip` and exit 0 — so the
same command works on a laptop and in CI.

## Eval quality gate (Gate 2)

`eval.sh` is the regression gate (REG-11). It runs **only** when an eval run is
available to compare, signalled via env (mirroring the Bitbucket Gate-2 step):

| Env var | Meaning |
|---|---|
| `CANDIDATE_RUN_ID` | Candidate eval run ID (from the eval runner, REG-8). Unset → stage skips. |
| `BASELINE_EVAL_RUN_ID` | Production baseline run ID to compare against. |
| `GATE_NO_BASELINE=1` | First-ever run of a prompt (no baseline); gate always passes. |

With no IDs (e.g. a local `make ci`, or a PR not touching `prompts/agents/evals`)
the stage skips cleanly. Otherwise it runs
`python gate.py "$CANDIDATE_RUN_ID" "$BASELINE_EVAL_RUN_ID"` and fails the build
on any blocking metric regression.

## Other validators (not in the `ci.sh` chain)

- `validate_diagrams.sh` (XCUT-6) — Mermaid + PlantUML diagram validation; run by
  its own pipeline step (`./scripts/validate_diagrams.sh docs/diagrams`).
- `dep_audit.py` (XCUT-4) — supply-chain pin + age audit; invoked by `lint.sh`.
- `lint_schemas.py` (INF-1) — prompt/agent schema linter; invoked by `lint.sh`.
