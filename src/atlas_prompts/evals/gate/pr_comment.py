"""PR-comment posting for the eval quality gate (REG-12, Gate 2).

The regression gate (``gate.py`` / :mod:`atlas_prompts.evals.gate.comparator`)
already produces a red required check on regression.  This module adds the
*second* half of REG-12's done-when: posting the metric-diff as a comment on the
pull request, on **both** pass and regression.

Three pieces, each independently testable:

``format_metric_diff_comment``
    Pure function: render a deterministic markdown comment from a
    :class:`~atlas_prompts.evals.gate.comparator.GateResult`.  No I/O — this is
    the unit-tested core.

``BitbucketCommentPoster``
    The provider behind a small seam (:class:`CommentPoster`).  Posts to the
    Bitbucket Cloud PR-comments REST endpoint.  Reads all configuration from the
    environment; constructed only when full PR context is present.

``post_comment`` / ``main``
    Env-gated orchestration.  Outside a PR (no PR-context env) this is a clean
    no-op — it must never break a local ``make ci``.  When PR context *is*
    present but credentials are missing it fails fast with an explicit error,
    rather than silently skipping a real misconfiguration.

The HTTP call uses the standard library (:mod:`urllib.request`) so REG-12 adds
no new dependency.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from atlas_prompts.evals.gate.comparator import GateResult

# ---------------------------------------------------------------------------
# Pure formatter — the unit-tested core
# ---------------------------------------------------------------------------

_COMMENT_MARKER = "<!-- atlas-eval-gate -->"
"""Hidden marker so a future update can find and edit its own comment.

Always emitted as the first line; harmless in rendered markdown.
"""


def _fmt_metric(value: float | None) -> str:
    """Render a metric mean, or an em dash when there is no baseline."""
    if value is None:
        return "—"
    return f"{value:.4f}"


def format_metric_diff_comment(comparison: GateResult) -> str:
    """Render a deterministic markdown PR comment from a gate comparison.

    The comment has three parts:

    1. A verdict header — ``PASS ✅`` or ``REGRESSION ❌`` (first-run with no
       baseline still passes, so it renders as PASS).
    2. A per-metric table: metric · baseline · candidate · Δ · verdict.
    3. The candidate / baseline run ids for traceability.

    Pure function — no environment access, no I/O.  Given the same
    ``comparison`` it always returns byte-identical output.
    """
    verdict = "PASS ✅" if comparison.passed else "REGRESSION ❌"

    lines: list[str] = [
        _COMMENT_MARKER,
        f"## Eval quality gate — {verdict}",
        "",
        "| Metric | Baseline | Candidate | Δ | Verdict |",
        "| --- | ---: | ---: | ---: | :---: |",
    ]

    for d in comparison.diffs:
        delta = "—" if d.baseline_mean is None else f"{d.delta:+.4f}"
        lines.append(
            f"| `{d.metric}` | {_fmt_metric(d.baseline_mean)} "
            f"| {_fmt_metric(d.candidate_mean)} | {delta} | {d.status} |"
        )

    baseline_id = comparison.baseline_run_id or "none (first run)"
    lines += [
        "",
        f"Candidate run: `{comparison.candidate_run_id or 'unknown'}`",
        f"Baseline run: `{baseline_id}`",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Poster seam + Bitbucket provider
# ---------------------------------------------------------------------------


class CommentPoster(Protocol):
    """Seam for posting a rendered comment to a PR provider."""

    def post(self, body: str) -> None:
        """Post ``body`` as a comment on the pull request.

        Raises on transport / auth failure — callers must not swallow it.
        """
        ...


@dataclass(frozen=True)
class BitbucketCommentPoster:
    """Posts a PR comment via the Bitbucket Cloud REST API (2.0).

    Endpoint:
        ``POST /2.0/repositories/{workspace}/{repo}/pullrequests/{pr_id}/comments``

    Constructed only when full PR context is present (see :func:`_poster_from_env`);
    every field is required so there is no half-configured state at this layer.
    """

    workspace: str
    repo_slug: str
    pr_id: str
    token: str
    base_url: str = "https://api.bitbucket.org"

    @property
    def _url(self) -> str:
        return (
            f"{self.base_url}/2.0/repositories/{self.workspace}/{self.repo_slug}"
            f"/pullrequests/{self.pr_id}/comments"
        )

    def post(self, body: str) -> None:
        payload = json.dumps({"content": {"raw": body}}).encode("utf-8")
        request = urllib.request.Request(  # noqa: S310 — fixed https Bitbucket host
            self._url,
            data=payload,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
        )
        try:
            # urlopen raises HTTPError for any non-2xx, so a normal return is success.
            with urllib.request.urlopen(request, timeout=30):  # noqa: S310
                pass
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Bitbucket comment POST failed: HTTP {exc.code} {exc.reason} — {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Bitbucket comment POST failed: {exc.reason}") from exc


# ---------------------------------------------------------------------------
# Env-gated orchestration
# ---------------------------------------------------------------------------

_PR_CONTEXT_KEYS = ("BITBUCKET_WORKSPACE", "BITBUCKET_REPO_SLUG", "BITBUCKET_PR_ID")
_TOKEN_KEYS = ("BITBUCKET_TOKEN", "BITBUCKET_ACCESS_TOKEN")


def _env(name: str) -> str | None:
    """Return a non-empty env var value, or None when unset/blank."""
    value = os.environ.get(name)
    return value if value else None


def _poster_from_env() -> BitbucketCommentPoster | None:
    """Build a poster from the environment, or None when there is no PR context.

    No PR context (none of the PR-context keys set) → return None so the caller
    no-ops cleanly (local ``make ci``, non-PR pipeline).

    Partial PR context (some but not all keys, or PR context without a token) →
    raise ``RuntimeError``.  Half-configured CI is a real misconfiguration and
    must fail loudly rather than silently skip the comment.
    """
    present = [k for k in _PR_CONTEXT_KEYS if _env(k)]
    if not present:
        return None

    missing = [k for k in _PR_CONTEXT_KEYS if not _env(k)]
    if missing:
        raise RuntimeError(
            "Partial Bitbucket PR context: set all of "
            f"{', '.join(_PR_CONTEXT_KEYS)} (missing: {', '.join(missing)})."
        )

    token = next((_env(k) for k in _TOKEN_KEYS if _env(k)), None)
    if not token:
        raise RuntimeError(
            f"Bitbucket PR context is set but no token found: provide {' or '.join(_TOKEN_KEYS)}."
        )

    # All keys validated non-None above; assert for the type checker.
    workspace = _env("BITBUCKET_WORKSPACE")
    repo_slug = _env("BITBUCKET_REPO_SLUG")
    pr_id = _env("BITBUCKET_PR_ID")
    assert workspace and repo_slug and pr_id  # noqa: S101 — narrowing after the checks above
    return BitbucketCommentPoster(
        workspace=workspace,
        repo_slug=repo_slug,
        pr_id=pr_id,
        token=token,
    )


def post_comment(comparison: GateResult, poster: CommentPoster | None = None) -> bool:
    """Format ``comparison`` and post it as a PR comment.

    Returns True when a comment was posted, False when skipped (no PR context).

    Parameters
    ----------
    comparison:
        The gate result to render and post.
    poster:
        Override the provider (used by tests).  When None the provider is built
        from the environment; absent PR context → clean no-op.
    """
    resolved = poster if poster is not None else _poster_from_env()
    if resolved is None:
        print("eval-gate PR comment: no PR context, skipping comment.", file=sys.stderr)
        return False

    body = format_metric_diff_comment(comparison)
    resolved.post(body)
    print("eval-gate PR comment: posted metric-diff comment.", file=sys.stderr)
    return True


def _load_comparison(json_path: Path) -> GateResult:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return GateResult.from_dict(data)


def main(argv: list[str] | None = None) -> int:
    """CLI: read the gate's --json comparison, format it, post the PR comment.

    Usage::

        python -m atlas_prompts.evals.gate.pr_comment <comparison.json>

    Exit codes::

        0   comment posted, or cleanly skipped (no PR context)
        2   usage error, or PR context present but misconfigured / post failed
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print(
            "usage: python -m atlas_prompts.evals.gate.pr_comment <comparison.json>",
            file=sys.stderr,
        )
        return 2

    json_path = Path(args[0])
    if not json_path.is_file():
        print(f"pr_comment: comparison file not found: {json_path}", file=sys.stderr)
        return 2

    try:
        comparison = _load_comparison(json_path)
        post_comment(comparison)
    except (RuntimeError, KeyError, ValueError) as exc:
        print(f"pr_comment: {exc}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
