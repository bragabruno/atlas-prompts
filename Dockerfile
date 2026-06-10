# syntax=docker/dockerfile:1
# atlas-prompts drift-eval job image (POL-3/POL-5) — multi-stage, non-root,
# pinned base. This builds the image the K8s nightly CronJob runs
# (bragabruno/atlas-prompts). It is NOT a long-running service: it consumes
# shadow traffic from Kafka, logs drift to MLflow, optionally fires a webhook
# alert, then exits (0 = no drift, 1 = alert fired). In the local compose loop
# it lives behind the `jobs` profile so `docker compose up` never starts it.

FROM python:3.12.13-slim-bookworm AS build
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
COPY . .
# Install the package + its pinned runtime deps into an isolated venv.
RUN python -m venv /venv \
 && /venv/bin/pip install --no-cache-dir .

FROM python:3.12.13-slim-bookworm AS runtime
# Non-root runtime user.
RUN groupadd --system app \
 && useradd --system --gid app --home-dir /app --shell /usr/sbin/nologin app
WORKDIR /app
COPY --from=build /venv /venv
# The entrypoint imports evals.* and lives in scripts/ — neither is part of the
# installed wheel ([tool.setuptools.packages.find] ships only atlas_prompts*),
# so copy the full source tree. nightly_drift_eval.py inserts /app on sys.path.
COPY --from=build /app /app
ENV PATH="/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
USER app
# Batch job — reads ATLAS_KAFKA_BOOTSTRAP / MLFLOW_TRACKING_URI / ATLAS_DRIFT_*
# from env, runs once, exits. For a Kafka-free dry run override the command:
#   docker compose --profile jobs run --rm drift-eval \
#     python scripts/nightly_drift_eval.py --source shadow.jsonl
CMD ["python", "scripts/nightly_drift_eval.py"]
