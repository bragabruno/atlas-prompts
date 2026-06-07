"""Tests for atlas_prompts.datasets and atlas_prompts.dataset_loader (REG-7)."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
from pydantic import ValidationError

from atlas_prompts.dataset_loader import LocalFileSource, load_dataset
from atlas_prompts.datasets import CitationRequirement, GoldenRecord

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
SEED_NAME = "regdoc-qa"
SEED_VERSION = "1.0.0"


def _write_jsonl(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(dict(rec)) + "\n")


# ---------------------------------------------------------------------------
# GoldenRecord schema tests
# ---------------------------------------------------------------------------


class TestGoldenRecordSchema:
    def test_valid_exact_only(self) -> None:
        rec = GoldenRecord(input="What is the lead limit?", expected_exact="0.05%")
        assert rec.expected_exact == "0.05%"
        assert rec.expected_semantic is None
        assert rec.required_citations == []

    def test_valid_semantic_only(self) -> None:
        rec = GoldenRecord(
            input="Describe CLP labelling obligations.",
            expected_semantic="Suppliers must affix hazard labels.",
        )
        assert rec.expected_semantic is not None
        assert rec.expected_exact is None

    def test_valid_both_expected_fields(self) -> None:
        rec = GoldenRecord(
            input="Is DINP in Annex XIV?",
            expected_exact="No",
            expected_semantic="DINP is not listed in Annex XIV.",
            required_citations=[CitationRequirement(source_id="EUR-REACH-Annex-XIV")],
        )
        assert len(rec.required_citations) == 1
        assert rec.required_citations[0].source_id == "EUR-REACH-Annex-XIV"

    def test_citation_min_count_default(self) -> None:
        cit = CitationRequirement(source_id="EUR-REACH-Annex-XVII-Entry-27")
        assert cit.min_count == 1

    def test_citation_min_count_custom(self) -> None:
        cit = CitationRequirement(source_id="doc-abc", min_count=3)
        assert cit.min_count == 3

    def test_missing_both_expected_raises(self) -> None:
        with pytest.raises(ValidationError, match="at least one"):
            GoldenRecord(input="What is the limit?")

    def test_empty_input_raises(self) -> None:
        with pytest.raises(ValidationError):
            GoldenRecord(input="", expected_exact="some answer")

    def test_citation_min_count_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            CitationRequirement(source_id="doc-x", min_count=0)

    def test_tags_default_empty(self) -> None:
        rec = GoldenRecord(input="Q?", expected_exact="A")
        assert rec.tags == []

    def test_tags_populated(self) -> None:
        rec = GoldenRecord(input="Q?", expected_exact="A", tags=["reach", "annex-xiv"])
        assert rec.tags == ["reach", "annex-xiv"]


# ---------------------------------------------------------------------------
# Seed dataset happy-path test
# ---------------------------------------------------------------------------


class TestSeedDataset:
    def test_seed_loads_all_records(self) -> None:
        """The committed seed file must load and validate cleanly."""
        records = load_dataset(SEED_NAME, SEED_VERSION)
        assert len(records) >= 1, "Expected at least one record in the seed dataset"
        for rec in records:
            assert isinstance(rec, GoldenRecord)

    def test_seed_records_have_expected_field(self) -> None:
        records = load_dataset(SEED_NAME, SEED_VERSION)
        for rec in records:
            has_expected = rec.expected_exact is not None or rec.expected_semantic is not None
            assert has_expected, f"Record missing both expected fields: {rec.input!r}"

    def test_seed_citations_have_source_ids(self) -> None:
        records = load_dataset(SEED_NAME, SEED_VERSION)
        for rec in records:
            for cit in rec.required_citations:
                assert cit.source_id, "Citation source_id must be non-empty"


# ---------------------------------------------------------------------------
# Versioned loader tests
# ---------------------------------------------------------------------------


class TestLoadDataset:
    def test_load_by_version(self, tmp_path: Path) -> None:
        """load_dataset reads the correct version file."""
        v1_records = [{"input": "Q1?", "expected_exact": "A1"}]
        v2_records = [
            {"input": "Q1?", "expected_exact": "A1"},
            {"input": "Q2?", "expected_semantic": "Answer two."},
        ]
        _write_jsonl(tmp_path / "myset" / "1.0.0.jsonl", v1_records)
        _write_jsonl(tmp_path / "myset" / "2.0.0.jsonl", v2_records)

        source = LocalFileSource(root=tmp_path)
        r1 = load_dataset("myset", "1.0.0", source=source)
        r2 = load_dataset("myset", "2.0.0", source=source)

        assert len(r1) == 1
        assert len(r2) == 2

    def test_version_not_found_raises(self, tmp_path: Path) -> None:
        source = LocalFileSource(root=tmp_path)
        with pytest.raises(FileNotFoundError, match="not found"):
            load_dataset("myset", "9.9.9", source=source)

    def test_unknown_dataset_raises(self, tmp_path: Path) -> None:
        source = LocalFileSource(root=tmp_path)
        with pytest.raises(FileNotFoundError):
            load_dataset("does-not-exist", "1.0.0", source=source)

    def test_blank_lines_skipped(self, tmp_path: Path) -> None:
        """Blank lines in JSONL are silently skipped."""
        path = tmp_path / "ds" / "1.0.0.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text(
            '\n{"input": "Q?", "expected_exact": "A"}\n\n',
            encoding="utf-8",
        )
        source = LocalFileSource(root=tmp_path)
        records = load_dataset("ds", "1.0.0", source=source)
        assert len(records) == 1

    def test_invalid_json_line_raises(self, tmp_path: Path) -> None:
        """A line that is not valid JSON raises ValueError."""
        path = tmp_path / "bad" / "1.0.0.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text("not json at all\n", encoding="utf-8")
        source = LocalFileSource(root=tmp_path)
        with pytest.raises(ValueError, match="invalid JSON"):
            load_dataset("bad", "1.0.0", source=source)

    def test_schema_validation_rejects_malformed_record(self, tmp_path: Path) -> None:
        """A record missing both expected fields must raise ValueError."""
        bad_record = {"input": "Q?"}  # no expected_exact or expected_semantic
        _write_jsonl(tmp_path / "malformed" / "1.0.0.jsonl", [bad_record])
        source = LocalFileSource(root=tmp_path)
        with pytest.raises(ValueError, match="schema error"):
            load_dataset("malformed", "1.0.0", source=source)

    def test_schema_validation_rejects_empty_input(self, tmp_path: Path) -> None:
        """A record with an empty input string must raise ValueError."""
        bad_record = {"input": "", "expected_exact": "A"}
        _write_jsonl(tmp_path / "emptyinput" / "1.0.0.jsonl", [bad_record])
        source = LocalFileSource(root=tmp_path)
        with pytest.raises(ValueError, match="schema error"):
            load_dataset("emptyinput", "1.0.0", source=source)


# ---------------------------------------------------------------------------
# Immutability test
# ---------------------------------------------------------------------------


class TestImmutability:
    def test_load_does_not_mutate_source_file(self, tmp_path: Path) -> None:
        """Loading a dataset must not alter the JSONL file on disk."""
        records_data = [{"input": "Q?", "expected_exact": "A"}]
        path = tmp_path / "stable" / "1.0.0.jsonl"
        _write_jsonl(path, records_data)

        mtime_before = path.stat().st_mtime
        size_before = path.stat().st_size

        source = LocalFileSource(root=tmp_path)
        load_dataset("stable", "1.0.0", source=source)

        assert path.stat().st_mtime == mtime_before, "load_dataset must not touch the file mtime"
        assert path.stat().st_size == size_before, "load_dataset must not change the file size"

    def test_mutating_returned_list_does_not_affect_reload(self, tmp_path: Path) -> None:
        """Mutating the returned list does not affect a subsequent load call."""
        records_data = [
            {"input": "Q1?", "expected_exact": "A1"},
            {"input": "Q2?", "expected_exact": "A2"},
        ]
        _write_jsonl(tmp_path / "stable2" / "1.0.0.jsonl", records_data)
        source = LocalFileSource(root=tmp_path)

        first_load = load_dataset("stable2", "1.0.0", source=source)
        first_load.clear()  # mutate returned list

        second_load = load_dataset("stable2", "1.0.0", source=source)
        assert len(second_load) == 2, "Mutating the returned list must not affect the source file"
