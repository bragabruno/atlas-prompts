"""Schema lint: validates meta.yaml, agent YAMLs, and referenced template files.

Exits non-zero when any validation error is found.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml
from pydantic import ValidationError

from atlas_prompts.schemas import AgentSpec, PromptMeta


def _load_yaml(path: Path) -> dict[str, object]:
    with path.open() as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a YAML mapping, got {type(raw).__name__}")
    return raw  # type: ignore[return-value]


def lint_meta(path: Path) -> list[str]:
    """Validate a single meta.yaml; return a list of error strings (empty = OK)."""
    errors: list[str] = []
    try:
        data = _load_yaml(path)
    except Exception as exc:
        return [f"{path}: failed to load YAML: {exc}"]

    try:
        PromptMeta.model_validate(data)
    except ValidationError as exc:
        for e in exc.errors():
            field = " -> ".join(str(loc) for loc in e["loc"])
            errors.append(f"{path}: [{field}] {e['msg']}")

    # Ensure the companion template.jinja exists
    template = path.parent / "template.jinja"
    if not template.is_file():
        errors.append(f"{path}: companion template not found: {template}")

    return errors


def lint_agent(path: Path, repo_root: Path) -> list[str]:
    """Validate a single agent YAML; return a list of error strings (empty = OK)."""
    errors: list[str] = []
    try:
        data = _load_yaml(path)
    except Exception as exc:
        return [f"{path}: failed to load YAML: {exc}"]

    try:
        spec = AgentSpec.model_validate(data)
    except ValidationError as exc:
        for e in exc.errors():
            field = " -> ".join(str(loc) for loc in e["loc"])
            errors.append(f"{path}: [{field}] {e['msg']}")
        return errors  # skip ref check if schema itself is broken

    # Validate that system_prompt_ref points to an existing file
    ref_path = repo_root / spec.system_prompt_ref
    if not ref_path.is_file():
        errors.append(
            f"{path}: system_prompt_ref '{spec.system_prompt_ref}' does not exist at {ref_path}"
        )

    return errors


def lint_all(repo_root: Path) -> list[str]:
    """Run all lints; return accumulated error strings."""
    all_errors: list[str] = []

    # Lint every meta.yaml under prompts/
    prompts_dir = repo_root / "prompts"
    if prompts_dir.is_dir():
        for meta_path in sorted(prompts_dir.rglob("meta.yaml")):
            all_errors.extend(lint_meta(meta_path))

    # Lint every yaml under agents/
    agents_dir = repo_root / "agents"
    if agents_dir.is_dir():
        for agent_path in sorted(agents_dir.glob("*.yaml")):
            all_errors.extend(lint_agent(agent_path, repo_root))

    return all_errors


def main() -> None:
    repo_root = Path(__file__).parent.parent
    errors = lint_all(repo_root)
    if errors:
        for msg in errors:
            print(f"ERROR: {msg}", file=sys.stderr)
        sys.exit(1)
    print("lint: all schema checks passed.")


if __name__ == "__main__":
    main()
