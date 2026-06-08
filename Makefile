# atlas-prompts — build system (single source of truth).
# Developers and CI run the same targets. Logic lives in scripts/, not here and
# not in the pipeline YAML. See atlas-docs/07-build-system.md.
.DEFAULT_GOAL := help
SHELL := bash

.PHONY: help lint test eval coverage build docker infra security ci local

help: ## Show this help
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-9s\033[0m %s\n",$$1,$$2}'

lint: ## Dep age audit (+trunk pins) + Trunk(ruff) + pyright + schema lint
	@./scripts/lint.sh

test: ## Offline unit tests (pytest; FakeGatewayClient, zero API spend)
	@./scripts/test.sh

eval: ## Eval quality gate (Gate 2; runs only with CANDIDATE_RUN_ID set)
	@./scripts/eval.sh

coverage: ## Coverage gate (pytest-cov; recommended, ATLAS_COV_MIN)
	@./scripts/coverage.sh

build: ## Build verification (import smoke: atlas_prompts)
	@./scripts/build.sh

docker: ## Build the container image (skips — not a deployed service)
	@./scripts/docker.sh

infra: ## Validate the deploy/ Helm chart (skips — no chart)
	@./scripts/infra.sh

security: ## Security scans (secret/CVE/fs; advisory, ATLAS_SECURITY_STRICT=1)
	@./scripts/security.sh

ci: ## Run the full gate — what CI runs
	@./scripts/ci.sh

local: ## Validate prompt/agent schemas (local dev inner loop)
	@./scripts/local.sh
