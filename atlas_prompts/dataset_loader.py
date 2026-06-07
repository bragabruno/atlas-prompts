"""Versioned golden-set loader (REG-7).

Usage
-----
    from atlas_prompts.dataset_loader import load_dataset

    records = load_dataset("regdoc-qa", "1.0.0")

Dataset layout (local)
----------------------
    evals/datasets/<name>/<version>.jsonl

Each line is a JSON object that must validate against
:class:`atlas_prompts.datasets.GoldenRecord`.

Versions are treated as **immutable**: once a version file exists on disk it is
never modified.  The loader enforces this by reading files in a read-only manner
and never writing to them.

Blob backend (deferred)
-----------------------
Azure Blob Storage is the eventual storage backend (ADR-016).  The seam is
:class:`DatasetSource` — swap in an :class:`AzureBlobSource` implementation that
downloads the JSONL bytes and feeds them through the same ``_parse_jsonl`` helper.
No other code needs to change.

    # Future usage (not implemented here):
    #
    #   from atlas_prompts.dataset_loader import load_dataset, AzureBlobSource
    #   records = load_dataset("regdoc-qa", "1.0.0", source=AzureBlobSource(...))
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import ValidationError

from atlas_prompts.datasets import GoldenRecord

# ---------------------------------------------------------------------------
# Source abstraction — seam for Blob backend
# ---------------------------------------------------------------------------

_DATASETS_ROOT = Path(__file__).parent.parent / "evals" / "datasets"


class DatasetSource(ABC):
    """Abstract source of raw JSONL bytes for a named, versioned dataset.

    Implement this interface and pass an instance to :func:`load_dataset` to
    swap the storage backend without changing any call-site code.
    """

    @abstractmethod
    def read_jsonl(self, name: str, version: str) -> str:
        """Return the full JSONL text for ``<name>/<version>``.

        Raises
        ------
        FileNotFoundError
            If the dataset/version does not exist in this source.
        """


class LocalFileSource(DatasetSource):
    """Reads JSONL from ``evals/datasets/<name>/<version>.jsonl`` on the local filesystem.

    Parameters
    ----------
    root:
        Directory that contains the per-dataset sub-directories.
        Defaults to ``<repo_root>/evals/datasets``.
    """

    def __init__(self, root: Path | None = None) -> None:
        self._root = root if root is not None else _DATASETS_ROOT

    def read_jsonl(self, name: str, version: str) -> str:
        path = self._root / name / f"{version}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Dataset '{name}' version '{version}' not found at {path}")
        return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Parsing helper
# ---------------------------------------------------------------------------


def _parse_jsonl(text: str, name: str, version: str) -> list[GoldenRecord]:
    """Parse and validate each non-empty line as a :class:`GoldenRecord`.

    Raises
    ------
    ValueError
        If any line fails JSON parsing or schema validation.  The error message
        includes the dataset name, version, and 1-based line number so failures
        are easy to locate.
    """
    records: list[GoldenRecord] = []
    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue  # skip blank lines between records
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Dataset '{name}' v{version} line {lineno}: invalid JSON — {exc}"
            ) from exc
        try:
            records.append(GoldenRecord.model_validate(obj))
        except ValidationError as exc:
            raise ValueError(
                f"Dataset '{name}' v{version} line {lineno}: schema error — {exc}"
            ) from exc
    return records


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_DEFAULT_SOURCE: DatasetSource = LocalFileSource()


def load_dataset(
    name: str,
    version: str,
    *,
    source: DatasetSource | None = None,
) -> list[GoldenRecord]:
    """Load and validate a golden dataset by name and version.

    Parameters
    ----------
    name:
        Dataset name (directory name under ``evals/datasets/``).
    version:
        Semver string, e.g. ``"1.0.0"``.
    source:
        Storage backend.  Defaults to :class:`LocalFileSource`.
        Pass an alternative implementation to use Azure Blob or any other backend.

    Returns
    -------
    list[GoldenRecord]
        Validated, immutable-by-convention list of golden records.

    Raises
    ------
    FileNotFoundError
        If the dataset/version is not found in the source.
    ValueError
        If any record fails schema validation.
    """
    effective_source = source if source is not None else _DEFAULT_SOURCE
    text = effective_source.read_jsonl(name, version)
    return _parse_jsonl(text, name, version)
