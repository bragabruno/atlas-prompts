#!/usr/bin/env python3
"""POL-3 — Nightly shadow/drift eval CLI.

Reads from the ``atlas.shadow.v1`` Kafka topic (or a local JSONL file for dry
runs), computes drift metrics, logs to MLflow, and exits non-zero when an alert
fires.

Usage
-----
Production (Kafka):
    python scripts/nightly_drift_eval.py

Dry run (local JSONL):
    python scripts/nightly_drift_eval.py --source shadow.jsonl

Environment variables
---------------------
    ATLAS_KAFKA_BOOTSTRAP     Kafka broker addresses (required for live mode)
    MLFLOW_TRACKING_URI       MLflow server URI
    ATLAS_DRIFT_EXPERIMENT    MLflow experiment name (default: atlas-drift)
    ATLAS_SHADOW_LIMIT        Max records to consume (default: 1000)
    ATLAS_DRIFT_ALERT_PCT     Alert threshold % regression (default: 10.0)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("drift_eval")

sys.path.insert(0, str(Path(__file__).parent.parent))


def _build_source(args: argparse.Namespace):  # type: ignore[return]
    if args.source:
        from evals.drift.shadow_source import JsonlShadowSource
        path = Path(args.source)
        if not path.is_file():
            log.error("source file not found: %s", path)
            sys.exit(1)
        log.info("dry_run mode: reading from %s", path)
        return JsonlShadowSource(path)

    bootstrap = os.environ.get("ATLAS_KAFKA_BOOTSTRAP")
    if not bootstrap:
        log.error("ATLAS_KAFKA_BOOTSTRAP env var is required in live mode")
        sys.exit(1)

    from evals.drift.kafka_shadow_source import KafkaShadowSource
    return KafkaShadowSource(bootstrap_servers=bootstrap)


def main() -> int:
    parser = argparse.ArgumentParser(description="Nightly shadow/drift eval")
    parser.add_argument(
        "--source",
        metavar="FILE",
        help="Path to a local JSONL shadow file (dry run — skips Kafka)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=int(os.environ.get("ATLAS_SHADOW_LIMIT", "1000")),
        help="Max shadow records to consume (default 1000)",
    )
    parser.add_argument(
        "--alert-pct",
        type=float,
        default=float(os.environ.get("ATLAS_DRIFT_ALERT_PCT", "10.0")),
        help="Alert threshold %% regression (default 10.0)",
    )
    args = parser.parse_args()

    source = _build_source(args)

    from evals.drift.drift_job import run_drift_eval

    report = run_drift_eval(
        source=source,
        mlflow_tracking_uri=os.environ.get("MLFLOW_TRACKING_URI"),
        mlflow_experiment=os.environ.get("ATLAS_DRIFT_EXPERIMENT", "atlas-drift"),
        alert_threshold_pct=args.alert_pct,
        sample_limit=args.limit,
    )

    print(report.format_summary())

    if report.has_alerts:
        from evals.drift.alerter import WebhookAlerter

        alerter = WebhookAlerter(os.environ.get("ATLAS_DRIFT_WEBHOOK_URL"))
        alerter.fire(
            summary=report.format_summary(),
            evaluated_at=report.evaluated_at,
            versions=[
                {
                    "prompt_version": v.prompt_version,
                    "n_samples": v.n_samples,
                    "alerted": v.alerted,
                    "run_id": v.run_id,
                }
                for v in report.versions
            ],
        )
        log.warning("drift_alerts_fired — check MLflow dashboard")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
