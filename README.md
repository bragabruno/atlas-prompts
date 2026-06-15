# atlas-prompts

Prompts as code — git-tracked prompt templates, agent definitions, and the eval-gate pipeline (Gate 2) that guards every promotion to production.

## Purpose

This repo owns the full prompt lifecycle for Atlas: authoring, versioning, evaluation, and promotion. No prompt reaches production without passing the eval gate. The gateway's registry is the authoritative runtime record; this repo is the authoritative source of truth for what was evaluated and why it was promoted.

## Layout

The Python code follows the PyPA `src/` layout (single installable package `atlas_prompts`); the prompt/agent/eval **data** are the repo's deliverable and live at the root.

```
atlas-prompts/
├── prompts/
│   └── <name>/
│       └── <semver>/               # e.g. 1.0.0, 1.1.0-candidate
│           ├── template.jinja      # Jinja2 prompt template
│           └── meta.yaml           # name, version, status, tags, description
├── agents/
│   └── <name>.yaml                 # agent definition (model, prompt refs, tools, params)
├── datasets/                       # golden-set manifests, <name>/<version>.jsonl (data lives in Azure Blob)
├── rubrics/                        # versioned LLM-as-judge rubrics (judge_v<N>.yaml)
├── src/
│   └── atlas_prompts/              # the installable package
│       ├── dataset_loader.py       # loads golden-set manifests from datasets/
│       ├── schemas.py / lint.py    # prompt/agent schema models + `lint-schemas` CLI
│       ├── gate.py                 # `gate` CLI: candidate-vs-baseline diff; exits non-zero on regression
│       ├── evals/
│       │   ├── runner/             # eval runner: calls gateway, computes metrics
│       │   └── gate/               # baseline comparator + gate config
│       └── eval_runs/
│           ├── db/                 # SQLAlchemy models for the eval_runs/eval_results tables (this repo owns the schema)
│           └── eval_results/       # per-run result output dir
├── alembic/                        # migrations for the eval_runs/eval_results tables
├── bitbucket-pipelines.yml         # Gate 2 pipeline definition
└── docs/
    └── diagrams/
        ├── eval-gate-flow.md
        ├── prompt-lifecycle-state.md
        └── eval-runner-sequence.md
```

## Prompt versioning

Each prompt version lives at `prompts/<name>/<semver>/`. Status is encoded in `meta.yaml` and mirrored server-side by the gateway registry. Valid statuses:

| Status | Meaning |
|---|---|
| `draft` | Work in progress; not evaluated |
| `candidate` | Manually promoted; triggers eval gate on the next PR |
| `production` | Passed the gate; active in the gateway registry |
| `retired` | Superseded or withdrawn |

Rollback is a pointer flip in the gateway registry — no template files are deleted.

## Eval-gate flow (Gate 2)

Triggered on any PR that touches `prompts/**` or `agents/**`.

1. The Bitbucket pipeline runs `src/atlas_prompts/evals/runner/` against the golden set (manifests in `datasets/`, data in Azure Blob).
2. The runner calls the gateway via the generated Python client (source: gateway OpenAPI spec).
3. Metrics computed per golden case — metric implementations come from **DeepEval**; the gate/promotion logic stays custom (see [ADR-017](../atlas-docs/02-tech-stack-and-adrs.md)):
   - Exact match
   - Semantic match (embedding cosine similarity)
   - Citation validity %
   - Cost and latency delta vs. baseline
   - Advisory LLM-as-judge (pinned judge model, temperature 0)
4. Results are written to `eval_runs/eval_results` and logged to MLflow.
5. The `gate` CLI (`python -m atlas_prompts.gate`) compares the candidate's metrics to the production baseline. Any regression exits non-zero, blocking the PR and posting a metric-diff comment.
6. A passing gate allows the PR to merge and the status pointer on the gateway registry to advance to `production`.

See [`docs/diagrams/eval-gate-flow.md`](docs/diagrams/eval-gate-flow.md) for the full flow diagram.

## Gateway client

The Python client under `src/atlas_prompts/evals/runner/` is generated from the gateway's OpenAPI spec. The spec is the contract source of truth — regenerate the client after any gateway API change. Do not hand-edit generated files.

## MLflow + eval tables ownership

This repo owns:
- The `eval_runs/eval_results` schema and migrations.
- The MLflow experiment structure and run metadata written by the eval runner.

The gateway and other Atlas repos read results via MLflow or the eval_results tables; they do not own those schemas.

## Development

**Python 3.12** with pinned dependencies (see `requirements.lock`). The `src/` layout
means the package must be installed (editable) to be importable — there is no
`pythonpath` shim.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"          # installs atlas_prompts + the `gate` / `lint-schemas` CLIs

# Lint (Trunk runs ruff, pinned in .trunk/trunk.yaml)
trunk check

# Typecheck
pyright

# Tests
pytest
```

Generating Alembic migrations requires `trunk` on your `PATH`: `alembic.ini`'s
`[post_write_hooks]` runs `trunk check --fix` on each newly generated migration.

No secrets in this repo. All credentials (Azure Blob SAS, gateway API key, MLflow tracking URI) are injected via environment variables or CI secrets. Use `.env` locally (`.env` is gitignored).

## Related

- `atlas-gateway` — the gateway registry and OpenAPI spec (client source of truth).
- `atlas-docs` — architecture documentation and ADRs.
- `datasets/` manifests reference data stored in the shared Azure Blob container.
