# atlas-prompts

Prompts as code — git-tracked prompt templates, agent definitions, and the eval-gate pipeline (Gate 2) that guards every promotion to production.

## Purpose

This repo owns the full prompt lifecycle for Atlas: authoring, versioning, evaluation, and promotion. No prompt reaches production without passing the eval gate. The gateway's registry is the authoritative runtime record; this repo is the authoritative source of truth for what was evaluated and why it was promoted.

## Layout

```
atlas-prompts/
├── prompts/
│   └── <name>/
│       └── <semver>/           # e.g. 1.0.0, 1.1.0-candidate
│           ├── template.jinja  # Jinja2 prompt template
│           └── meta.yaml       # name, version, status, tags, description
├── agents/
│   └── <name>.yaml             # agent definition (model, prompt refs, tools, params)
├── evals/
│   ├── runner/                 # eval runner: calls gateway, computes metrics via DeepEval (ADR-017)
│   └── datasets/               # golden-set manifests (data lives in Azure Blob)
├── eval_runs/
│   └── eval_results/           # SQLite/Postgres migrations + result tables (this repo owns the schema)
├── gate.py                     # compares candidate metrics to production baseline; exits non-zero on regression
├── bitbucket-pipelines.yml     # Gate 2 pipeline definition
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

1. The Bitbucket pipeline runs `evals/runner/` against the golden set (manifests in `evals/datasets/`, data in Azure Blob).
2. The runner calls the gateway via the generated Python client (source: gateway OpenAPI spec).
3. Metrics computed per golden case — metric implementations come from **DeepEval**; the gate/promotion logic stays custom (see [ADR-017](../atlas-docs/02-tech-stack-and-adrs.md)):
   - Exact match
   - Semantic match (embedding cosine similarity)
   - Citation validity %
   - Cost and latency delta vs. baseline
   - Advisory LLM-as-judge (pinned judge model, temperature 0)
4. Results are written to `eval_runs/eval_results` and logged to MLflow.
5. `gate.py` compares the candidate's metrics to the production baseline. Any regression exits non-zero, blocking the PR and posting a metric-diff comment.
6. A passing gate allows the PR to merge and the status pointer on the gateway registry to advance to `production`.

See [`docs/diagrams/eval-gate-flow.md`](docs/diagrams/eval-gate-flow.md) for the full flow diagram.

## Gateway client

The Python client under `evals/runner/` is generated from the gateway's OpenAPI spec. The spec is the contract source of truth — regenerate the client after any gateway API change. Do not hand-edit generated files.

## MLflow + eval tables ownership

This repo owns:
- The `eval_runs/eval_results` schema and migrations.
- The MLflow experiment structure and run metadata written by the eval runner.

The gateway and other Atlas repos read results via MLflow or the eval_results tables; they do not own those schemas.

## Development

**Python 3.12** with pinned dependencies (see `requirements.txt` and `requirements-dev.txt`).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt

# Lint
ruff check .

# Typecheck
pyright

# Tests
pytest
```

No secrets in this repo. All credentials (Azure Blob SAS, gateway API key, MLflow tracking URI) are injected via environment variables or CI secrets. Use `.env` locally (`.env` is gitignored).

## Related

- `atlas-gateway` — the gateway registry and OpenAPI spec (client source of truth).
- `atlas-docs` — architecture documentation and ADRs.
- `evals/datasets/` manifests reference data stored in the shared Azure Blob container.
