"""Pydantic v2 models for meta.yaml and agent YAML validation."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

PromptStatus = Literal["draft", "candidate", "production", "retired"]

VALID_STATUSES: tuple[str, ...] = ("draft", "candidate", "production", "retired")


class PromptMeta(BaseModel):
    """Schema for prompts/<name>/<semver>/meta.yaml."""

    name: Annotated[str, Field(min_length=1)]
    version: Annotated[str, Field(min_length=1)]
    status: PromptStatus
    model_alias: Annotated[str, Field(min_length=1)]
    description: Annotated[str, Field(min_length=1)]
    tags: list[str] = Field(default_factory=list)
    params_schema: dict[str, object] = Field(default_factory=dict)


class AgentSpec(BaseModel):
    """Schema for agents/<name>.yaml."""

    agent_name: Annotated[str, Field(min_length=1)]
    agent_version: Annotated[str, Field(min_length=1)]
    system_prompt_ref: Annotated[str, Field(min_length=1)]
    model_alias: Annotated[str, Field(min_length=1)]
    tool_whitelist: list[str] = Field(default_factory=list)
    max_iterations: Annotated[int, Field(gt=0)]
    token_budget: Annotated[int, Field(gt=0)]
    timeout_s: Annotated[int, Field(gt=0)]
