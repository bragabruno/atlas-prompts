#!/usr/bin/env python3
"""REG-14 — Demo: blocked-PR + rollback.

Demonstrates the full eval-gate lifecycle end-to-end:

  1. A *candidate* prompt version (regdoc-qa/1.1.0) regresses on the golden
     dataset → the gate blocks it and prints a structured report.
  2. A *fix* prompt version (regdoc-qa/1.1.1) recovers the metrics → the gate
     promotes it (would trigger the GitHub-status update in CI).
  3. An operator rolls back to the last-known-good version (regdoc-qa/1.0.0)
     by re-running the gate in bypass mode — simulating an emergency rollback.

All calls use ``FakeGatewayClient`` so no live gateway is required.

Usage
-----
    python scripts/demo_blocked_pr_rollback.py

Expected output
---------------
    [STEP 1] Running eval gate for candidate version regdoc-qa/1.1.0 ...
    GATE BLOCKED — regdoc-qa/1.1.0 failed 3 of 8 metrics
    Regressions:
      exact_match:  0.250 (baseline 0.875, -71.4%)
      semantic_match: 0.583 (baseline 0.875, -33.3%)
      citation_validity: 0.750 (baseline 1.000, -25.0%)

    [STEP 2] Running eval gate for fixed version regdoc-qa/1.1.1 ...
    GATE PASSED  — regdoc-qa/1.1.1 passed all 8 metrics
    Would set GitHub commit status: success

    [STEP 3] Emergency rollback to regdoc-qa/1.0.0 ...
    ROLLBACK OK  — regdoc-qa/1.0.0 gate passed (pinned baseline; no regression check)
    Deployment: atlas-gateway rolls back to image tagged regdoc-qa-1.0.0
"""

from __future__ import annotations

import sys
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Minimal stubs so the demo runs without a live gateway or database.
# ---------------------------------------------------------------------------

@dataclass
class _FakeResult:
    metric: str
    value: float
    baseline_value: float | None
    passed: bool
    run_id: str = "demo-run"
    prompt_version: str = ""
    dataset_name: str = "regdoc-golden"
    dataset_version: str = "1.0.0"
    triggered_by: str = "demo"
    created_at: str = "2026-06-10T00:00:00+00:00"
    case_index: int = 0
    input_snippet: str = "demo"


def _make_results(prompt_version: str, scenario: str) -> list[_FakeResult]:
    """Return fake metric results for each demo scenario."""
    baseline = {
        "exact_match": 0.875,
        "semantic_match": 0.875,
        "citation_validity": 1.000,
        "latency_ms": 200.0,
    }
    if scenario == "regressed":
        values = {
            "exact_match": 0.250,   # big regression
            "semantic_match": 0.583,
            "citation_validity": 0.750,
            "latency_ms": 210.0,
        }
    else:  # "passing"
        values = {
            "exact_match": 0.900,
            "semantic_match": 0.900,
            "citation_validity": 1.000,
            "latency_ms": 195.0,
        }

    # latency: pass if ≤ threshold (lower is better)
    thresholds = {
        "exact_match": 0.80,
        "semantic_match": 0.80,
        "citation_validity": 0.95,
        "latency_ms": 250.0,  # pass when value ≤ this
    }

    results = []
    for metric, value in values.items():
        # latency: lower is better (pass when ≤ threshold)
        passed = value <= thresholds[metric] if metric == "latency_ms" else value >= thresholds[metric]
        results.append(_FakeResult(
            metric=metric,
            value=value,
            baseline_value=baseline.get(metric),
            passed=passed,
            prompt_version=prompt_version,
        ))
    return results


def _run_gate(prompt_version: str, scenario: str) -> bool:
    """Simulate running the eval gate.  Returns True if all metrics passed."""
    results = _make_results(prompt_version, scenario)
    failed = [r for r in results if not r.passed]

    if failed:
        print(f"GATE BLOCKED — {prompt_version} failed {len(failed)} of {len(results)} metrics")
        print("Regressions:")
        for r in failed:
            if r.baseline_value is not None and r.baseline_value > 0:
                pct = (r.value - r.baseline_value) / r.baseline_value * 100
                print(
                    f"  {r.metric:<22} {r.value:.3f} "
                    f"(baseline {r.baseline_value:.3f}, {pct:+.1f}%)"
                )
            else:
                print(f"  {r.metric:<22} {r.value:.3f} (no baseline)")
        return False
    else:
        print(f"GATE PASSED  — {prompt_version} passed all {len(results)} metrics")
        return True


def main() -> int:
    print("=" * 65)
    print("Atlas eval gate demo — blocked-PR + rollback")
    print("=" * 65)
    print()

    # ---- Step 1: Regressed candidate is blocked ---------------------------
    print("[STEP 1] Running eval gate for candidate version regdoc-qa/1.1.0 ...")
    passed = _run_gate("regdoc-qa/1.1.0", "regressed")
    if passed:
        print("ERROR: expected gate to block this version in the demo")
        return 1
    print("  → PR status set to: failure  (merge blocked)")
    print()

    # ---- Step 2: Fixed version passes -------------------------------------
    print("[STEP 2] Running eval gate for fixed version regdoc-qa/1.1.1 ...")
    passed = _run_gate("regdoc-qa/1.1.1", "passing")
    if not passed:
        print("ERROR: expected gate to pass the fixed version in the demo")
        return 1
    print("  → PR status set to: success  (merge unblocked)")
    print()

    # ---- Step 3: Emergency rollback ---------------------------------------
    print("[STEP 3] Emergency rollback to regdoc-qa/1.0.0 ...")
    passed = _run_gate("regdoc-qa/1.0.0", "passing")
    if not passed:
        print("ERROR: baseline version should always pass in the demo")
        return 1
    print("  → Deployment: atlas-gateway image rolled back to regdoc-qa-1.0.0")
    print()

    print("Demo complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
