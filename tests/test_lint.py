"""Tests for atlas_prompts.lint schema validation."""

from __future__ import annotations

from pathlib import Path

import yaml

from atlas_prompts.lint import lint_agent, lint_all, lint_meta

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent


def _write(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        yaml.safe_dump(data, fh)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


# ---------------------------------------------------------------------------
# Seed-file happy-path tests
# ---------------------------------------------------------------------------


def test_seed_meta_valid() -> None:
    """The committed meta.yaml for regdoc-qa must pass lint."""
    meta_path = REPO_ROOT / "prompts" / "regdoc-qa" / "1.0.0" / "meta.yaml"
    assert meta_path.is_file(), f"Seed meta.yaml missing at {meta_path}"
    errors = lint_meta(meta_path)
    assert errors == [], f"Seed meta.yaml has lint errors: {errors}"


def test_seed_agent_valid() -> None:
    """The committed regdoc-qa.yaml agent spec must pass lint."""
    agent_path = REPO_ROOT / "agents" / "regdoc-qa.yaml"
    assert agent_path.is_file(), f"Seed agent YAML missing at {agent_path}"
    errors = lint_agent(agent_path, REPO_ROOT)
    assert errors == [], f"Seed agent YAML has lint errors: {errors}"


def test_seed_template_exists() -> None:
    """The seed template.jinja file must exist."""
    template_path = REPO_ROOT / "prompts" / "regdoc-qa" / "1.0.0" / "template.jinja"
    assert template_path.is_file(), f"Seed template.jinja missing at {template_path}"


def test_lint_all_clean() -> None:
    """lint_all on the real repo root must return no errors."""
    errors = lint_all(REPO_ROOT)
    assert errors == [], f"lint_all found errors: {errors}"


# ---------------------------------------------------------------------------
# Invalid-fixture tests (use tmp_path — nothing broken is committed)
# ---------------------------------------------------------------------------


def test_meta_missing_required_field(tmp_path: Path) -> None:
    """meta.yaml without 'status' must produce a lint error."""
    meta_dir = tmp_path / "prompts" / "bad-prompt" / "1.0.0"
    template_path = meta_dir / "template.jinja"
    _write_text(template_path, "hello {{ question }}")

    bad_meta: dict[str, object] = {
        "name": "bad-prompt",
        "version": "1.0.0",
        # status intentionally omitted
        "model_alias": "claude-sonnet-4-6",
        "description": "A bad meta without status",
    }
    meta_path = meta_dir / "meta.yaml"
    _write(meta_path, bad_meta)

    errors = lint_meta(meta_path)
    assert any("status" in e for e in errors), f"Expected 'status' field error, got: {errors}"


def test_meta_invalid_status(tmp_path: Path) -> None:
    """meta.yaml with an unknown status value must produce a lint error."""
    meta_dir = tmp_path / "prompts" / "bad-status" / "1.0.0"
    _write_text(meta_dir / "template.jinja", "{{ question }}")

    bad_meta = {
        "name": "bad-status",
        "version": "1.0.0",
        "status": "published",  # not a valid status
        "model_alias": "claude-sonnet-4-6",
        "description": "Wrong status",
    }
    _write(meta_dir / "meta.yaml", bad_meta)

    errors = lint_meta(meta_dir / "meta.yaml")
    assert errors, "Expected lint errors for invalid status, got none"


def test_meta_missing_template(tmp_path: Path) -> None:
    """meta.yaml with no companion template.jinja must produce a lint error."""
    meta_dir = tmp_path / "prompts" / "no-template" / "1.0.0"
    meta_dir.mkdir(parents=True)

    good_meta = {
        "name": "no-template",
        "version": "1.0.0",
        "status": "draft",
        "model_alias": "claude-sonnet-4-6",
        "description": "Meta without a template",
    }
    meta_path = meta_dir / "meta.yaml"
    _write(meta_path, good_meta)
    # deliberately do NOT create template.jinja

    errors = lint_meta(meta_path)
    assert any("template" in e.lower() for e in errors), (
        f"Expected missing-template error, got: {errors}"
    )


def test_agent_missing_required_field(tmp_path: Path) -> None:
    """Agent YAML without 'max_iterations' must produce a lint error."""
    agent_path = tmp_path / "agents" / "bad-agent.yaml"

    bad_agent: dict[str, object] = {
        "agent_name": "bad-agent",
        "agent_version": "1.0.0",
        "system_prompt_ref": "prompts/regdoc-qa/1.0.0/template.jinja",
        "model_alias": "claude-sonnet-4-6",
        # max_iterations intentionally omitted
        "token_budget": 4096,
        "timeout_s": 60,
    }
    _write(agent_path, bad_agent)

    errors = lint_agent(agent_path, REPO_ROOT)
    assert any("max_iterations" in e for e in errors), (
        f"Expected 'max_iterations' error, got: {errors}"
    )


def test_agent_broken_prompt_ref(tmp_path: Path) -> None:
    """Agent YAML referencing a non-existent template must produce a lint error."""
    agent_path = tmp_path / "agents" / "bad-ref.yaml"

    bad_agent = {
        "agent_name": "bad-ref",
        "agent_version": "1.0.0",
        "system_prompt_ref": "prompts/does-not-exist/1.0.0/template.jinja",
        "model_alias": "claude-sonnet-4-6",
        "max_iterations": 5,
        "token_budget": 2048,
        "timeout_s": 30,
    }
    _write(agent_path, bad_agent)

    errors = lint_agent(agent_path, REPO_ROOT)
    assert any("system_prompt_ref" in e for e in errors), (
        f"Expected prompt-ref error, got: {errors}"
    )


def test_lint_all_catches_bad_meta(tmp_path: Path) -> None:
    """lint_all on a tmp tree with a bad meta.yaml returns errors."""
    # Build minimal valid structure
    prompts_dir = tmp_path / "prompts" / "myp" / "1.0.0"
    _write_text(prompts_dir / "template.jinja", "{{ question }}")
    bad_meta = {
        "name": "myp",
        "version": "1.0.0",
        # missing status
        "model_alias": "claude-sonnet-4-6",
        "description": "missing status",
    }
    _write(prompts_dir / "meta.yaml", bad_meta)

    errors = lint_all(tmp_path)
    assert errors, "Expected lint_all to surface missing-status error"
