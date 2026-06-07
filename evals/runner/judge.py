# pyright: reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownMemberType=false
"""LLM-as-judge metric (REG-9).

Design
------
``LlmJudge`` wraps the existing ``GatewayClient`` Protocol (REG-8) and applies
a versioned rubric (stored in ``evals/rubrics/``) to score a single
candidate–question–reference triple.

Key properties
--------------
- **Pinned judge model**: callers pass the model alias; the same alias is
  recorded in every ``JudgeScoreResult`` for traceability.
- **Temperature 0**: every call is made with ``temperature=0`` regardless of
  the caller's completion model.
- **Versioned rubric**: the rubric YAML lives in ``evals/rubrics/`` and is
  referenced by filename (``judge_v1.yaml``).  Breaking rubric changes MUST
  bump the version so historical scores stay comparable.
- **Multi-sample margin**: ``score_sample()`` calls the judge ``n_samples``
  times and returns the mean.  ``spread_within_margin`` is True when
  ``max − min ≤ margin`` — callers can log a variance warning when False.
- **Advisory only**: ``JudgeScoreResult.passed`` is a convenience flag;
  the metric is never wired as ``blocking=True`` in the gate.

Offline testing
---------------
Pass a ``FakeGatewayClient`` with a ``completion_map`` or
``default_completion`` that returns a plain integer string (e.g. ``"4"``).
The judge parses the integer from the response, so any valid integer within the
rubric's ``[score_min, score_max]`` range is accepted.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from evals.runner.gateway_client import GatewayClient

from evals.runner.gateway_client import ChatMessage, CompletionRequest
from evals.runner.metrics import JudgeScoreResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rubric loading
# ---------------------------------------------------------------------------

_RUBRICS_ROOT = Path(__file__).parent.parent / "rubrics"


def _rubric_path(version: str) -> Path:
    """Return the absolute path for a rubric version, e.g. ``"1"`` → ``judge_v1.yaml``."""
    return _RUBRICS_ROOT / f"judge_v{version}.yaml"


def load_rubric(version: str) -> dict[str, object]:
    """Load and return the parsed rubric YAML for *version*.

    Parameters
    ----------
    version:
        Rubric version string, e.g. ``"1"``.

    Raises
    ------
    FileNotFoundError
        If the rubric file does not exist.
    ValueError
        If the YAML is missing required keys.
    """
    path = _rubric_path(version)
    if not path.is_file():
        raise FileNotFoundError(
            f"Rubric version {version!r} not found at {path}. "
            f"Available rubrics: {[p.name for p in _RUBRICS_ROOT.glob('judge_v*.yaml')]}"
        )
    raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Rubric file {path} must be a YAML mapping; got {type(raw)}")
    required_keys = {
        "version",
        "score_min",
        "score_max",
        "criteria",
        "system_prompt_template",
        "user_prompt_template",
        "scale_anchors",
    }
    missing = required_keys - set(raw.keys())
    if missing:
        raise ValueError(f"Rubric {path} is missing required keys: {missing}")
    return raw  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _build_scale_anchors_text(anchors: dict[int, str], score_min: int, score_max: int) -> str:
    lines = []
    for score in range(score_min, score_max + 1):
        label = anchors.get(score, "")
        lines.append(f"  {score}: {label}")
    return "\n".join(lines)


def _build_criteria_text(criteria: list[dict[str, str]]) -> str:
    lines = []
    for i, crit in enumerate(criteria, start=1):
        lines.append(f"  {i}. {crit['id']}: {crit['description'].strip()}")
    return "\n".join(lines)


def _build_prompts(
    rubric: dict[str, object],
    question: str,
    candidate_answer: str,
    reference_answer: str,
) -> tuple[str, str]:
    """Return ``(system_prompt, user_prompt)`` for one judge call."""
    score_min = int(rubric["score_min"])  # type: ignore[arg-type]
    score_max = int(rubric["score_max"])  # type: ignore[arg-type]
    anchors: dict[int, str] = {
        int(k): str(v)
        for k, v in (rubric["scale_anchors"] or {}).items()  # type: ignore[union-attr]
    }
    criteria: list[dict[str, str]] = list(rubric["criteria"])  # type: ignore[arg-type]

    scale_anchors_text = _build_scale_anchors_text(anchors, score_min, score_max)
    criteria_text = _build_criteria_text(criteria)

    system_tpl: str = str(rubric["system_prompt_template"])
    user_tpl: str = str(rubric["user_prompt_template"])

    system_prompt = system_tpl.format(
        score_min=score_min,
        score_max=score_max,
        scale_anchors_text=scale_anchors_text,
        criteria_text=criteria_text,
    )
    user_prompt = user_tpl.format(
        question=question,
        candidate_answer=candidate_answer,
        reference_answer=reference_answer,
        score_min=score_min,
        score_max=score_max,
    )
    return system_prompt, user_prompt


# ---------------------------------------------------------------------------
# Score parsing
# ---------------------------------------------------------------------------


def parse_score(response_text: str, score_min: int, score_max: int) -> float:
    """Extract the integer score from the judge response.

    Accepts a bare integer or the first integer found in the response.
    Clamps to ``[score_min, score_max]`` and raises ``ValueError`` when no
    integer is found at all.
    """
    stripped = response_text.strip()
    match = re.search(r"\b(\d+)\b", stripped)
    if match is None:
        raise ValueError(f"Judge response contains no integer score: {response_text!r}")
    value = int(match.group(1))
    if value < score_min or value > score_max:
        logger.warning(
            "judge_score_out_of_range value=%d range=[%d, %d] — clamping",
            value,
            score_min,
            score_max,
        )
        value = max(score_min, min(score_max, value))
    return float(value)


# ---------------------------------------------------------------------------
# LlmJudge
# ---------------------------------------------------------------------------


class LlmJudge:
    """LLM-as-judge evaluator (REG-9).

    Parameters
    ----------
    client:
        Any object satisfying ``GatewayClient`` (live or fake).
    judge_model:
        Model alias sent to the gateway for judge calls.
    rubric_version:
        Version of the rubric to load, e.g. ``"1"``.
    n_samples:
        Number of independent judge calls per scored triple.
        Higher values reduce per-sample flakiness at the cost of more API calls.
        Default 3.
    margin:
        Allowed spread ``max(raw_scores) − min(raw_scores)`` across samples.
        Scores outside this spread trigger a variance log warning but are not
        rejected.  Default 1.0 (one rubric point).
    pass_threshold:
        Minimum normalised score (0–1) for ``JudgeScoreResult.passed`` to be
        True.  Default 0.6 (3 out of 5 on a 1–5 scale).
    """

    def __init__(
        self,
        client: GatewayClient,
        judge_model: str,
        rubric_version: str = "1",
        n_samples: int = 3,
        margin: float = 1.0,
        pass_threshold: float = 0.6,
    ) -> None:
        self._client = client
        self._judge_model = judge_model
        self._rubric_version = rubric_version
        self._n_samples = n_samples
        self._margin = margin
        self._pass_threshold = pass_threshold
        self._rubric = load_rubric(rubric_version)

    @property
    def rubric_version(self) -> str:
        return self._rubric_version

    @property
    def judge_model(self) -> str:
        return self._judge_model

    def score_sample(
        self,
        question: str,
        candidate_answer: str,
        reference_answer: str,
    ) -> JudgeScoreResult:
        """Score one (question, candidate, reference) triple.

        Calls the judge ``n_samples`` times at ``temperature=0`` and
        returns a :class:`~evals.runner.metrics.JudgeScoreResult` with the
        normalised mean score and per-sample raw scores.

        Parameters
        ----------
        question:
            The original user question from the golden record.
        candidate_answer:
            The model's actual response text.
        reference_answer:
            The golden expected answer (used as the reference for the judge).

        Returns
        -------
        JudgeScoreResult
            Advisory score; never directly blocking in the gate.
        """
        score_min = int(self._rubric["score_min"])  # type: ignore[arg-type]
        score_max = int(self._rubric["score_max"])  # type: ignore[arg-type]

        system_prompt, user_prompt = _build_prompts(
            self._rubric,
            question=question,
            candidate_answer=candidate_answer,
            reference_answer=reference_answer,
        )

        raw_scores: list[float] = []
        for sample_idx in range(self._n_samples):
            response = self._client.chat_completion(
                CompletionRequest(
                    model=self._judge_model,
                    messages=[
                        ChatMessage(role="system", content=system_prompt),
                        ChatMessage(role="user", content=user_prompt),
                    ],
                    temperature=0.0,
                )
            )
            try:
                raw = parse_score(response.content, score_min, score_max)
            except ValueError:
                logger.warning(
                    "judge_parse_error sample=%d response=%r — using score_min",
                    sample_idx,
                    response.content,
                )
                raw = float(score_min)
            raw_scores.append(raw)

        mean_raw = sum(raw_scores) / len(raw_scores)
        # Normalise to [0, 1]
        score_range = float(score_max - score_min)
        normalised = (mean_raw - score_min) / score_range if score_range > 0.0 else 0.0
        normalised = round(normalised, 6)

        spread = max(raw_scores) - min(raw_scores)
        spread_within_margin = spread <= self._margin

        if not spread_within_margin:
            logger.warning(
                "judge_variance_exceeded spread=%.1f margin=%.1f scores=%s question=%r",
                spread,
                self._margin,
                raw_scores,
                question[:80],
            )

        passed = normalised >= self._pass_threshold

        logger.info(
            "judge_score question=%r score=%.4f raw=%s passed=%s",
            question[:80],
            normalised,
            raw_scores,
            passed,
        )

        return JudgeScoreResult(
            score=normalised,
            raw_scores=raw_scores,
            rubric_version=self._rubric_version,
            judge_model=self._judge_model,
            n_samples=self._n_samples,
            margin=self._margin,
            spread_within_margin=spread_within_margin,
            passed=passed,
            pass_threshold=self._pass_threshold,
        )
