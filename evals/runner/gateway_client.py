"""Gateway client abstraction (REG-8).

Defines a typed ``GatewayClient`` Protocol so the eval runner is decoupled from
both live HTTP calls and test fakes.

Usage (production)
------------------
    from evals.runner.gateway_client import HttpGatewayClient
    client = HttpGatewayClient(base_url="http://gateway:8000", api_key="...")

Usage (offline / testing)
--------------------------
    from evals.runner.gateway_client import FakeGatewayClient
    client = FakeGatewayClient(
        completion_map={"What is X?": "X is Y."},
        embedding_fn=lambda text: [1.0, 0.0],
    )

Wire-format
-----------
The client follows the OpenAI-compatible shape exposed by the Atlas gateway:

POST /v1/chat/completions
    body: {"model": str, "messages": [...], "temperature": float, ...}
    response: {"choices": [{"message": {"content": str}}], "usage": {...}}

POST /v1/embeddings
    body: {"model": str, "input": str}
    response: {"data": [{"embedding": [float, ...]}], "usage": {...}}
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Wire-format models
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    """One message in a chat completion request."""

    role: str
    content: str


class CompletionRequest(BaseModel):
    """Request body for POST /v1/chat/completions."""

    model: str
    messages: list[ChatMessage]
    temperature: float = 0.0
    max_tokens: int | None = None


class UsageInfo(BaseModel):
    """Token usage returned by the gateway."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class CompletionResponse(BaseModel):
    """Parsed response from POST /v1/chat/completions."""

    content: str
    usage: UsageInfo = Field(default_factory=UsageInfo)
    latency_ms: float = 0.0
    """Wall-clock latency in milliseconds (injected by the client, not the API)."""

    cost_usd: float = 0.0
    """Estimated cost in USD (injected by the client when pricing info is available)."""


class EmbeddingRequest(BaseModel):
    """Request body for POST /v1/embeddings."""

    model: str
    input: str


class EmbeddingResponse(BaseModel):
    """Parsed response from POST /v1/embeddings."""

    embedding: list[float]
    usage: UsageInfo = Field(default_factory=UsageInfo)
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Protocol — the seam
# ---------------------------------------------------------------------------


@runtime_checkable
class GatewayClient(Protocol):
    """Minimal typed interface for calling the Atlas gateway.

    Both ``HttpGatewayClient`` (live) and ``FakeGatewayClient`` (offline/tests)
    satisfy this protocol.  The eval runner accepts ``GatewayClient`` — it never
    imports a concrete class.
    """

    def chat_completion(self, request: CompletionRequest) -> CompletionResponse:
        """Call POST /v1/chat/completions and return a parsed response."""
        ...

    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Call POST /v1/embeddings and return a parsed response."""
        ...


# ---------------------------------------------------------------------------
# Live HTTP client
# ---------------------------------------------------------------------------


class HttpGatewayClient:
    """Production client that calls the Atlas gateway over HTTP.

    Parameters
    ----------
    base_url:
        Gateway base URL, e.g. ``"http://gateway:8000"``.  No trailing slash.
    api_key:
        Value sent in the ``Authorization: Bearer <key>`` header.
    timeout_s:
        HTTP timeout in seconds (default 60).
    """

    def __init__(self, base_url: str, api_key: str, timeout_s: int = 60) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout_s = timeout_s

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        payload = json.dumps(body).encode()
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:  # noqa: S310
                return json.loads(resp.read().decode())  # type: ignore[no-any-return]
        except urllib.error.HTTPError as exc:
            body_text = exc.read().decode(errors="replace")
            raise RuntimeError(f"Gateway {path} returned HTTP {exc.code}: {body_text}") from exc

    def chat_completion(self, request: CompletionRequest) -> CompletionResponse:
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [m.model_dump() for m in request.messages],
            "temperature": request.temperature,
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens

        t0 = time.perf_counter()
        raw = self._post("/v1/chat/completions", payload)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        content: str = raw["choices"][0]["message"]["content"]
        usage_raw: dict[str, int] = raw.get("usage", {})
        return CompletionResponse(
            content=content,
            usage=UsageInfo(
                prompt_tokens=usage_raw.get("prompt_tokens", 0),
                completion_tokens=usage_raw.get("completion_tokens", 0),
                total_tokens=usage_raw.get("total_tokens", 0),
            ),
            latency_ms=round(latency_ms, 2),
        )

    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        payload: dict[str, Any] = {"model": request.model, "input": request.input}

        t0 = time.perf_counter()
        raw = self._post("/v1/embeddings", payload)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        embedding: list[float] = raw["data"][0]["embedding"]
        usage_raw: dict[str, int] = raw.get("usage", {})
        return EmbeddingResponse(
            embedding=embedding,
            usage=UsageInfo(
                prompt_tokens=usage_raw.get("prompt_tokens", 0),
                completion_tokens=usage_raw.get("completion_tokens", 0),
                total_tokens=usage_raw.get("total_tokens", 0),
            ),
            latency_ms=round(latency_ms, 2),
        )


# ---------------------------------------------------------------------------
# Fake client — deterministic offline / test stub
# ---------------------------------------------------------------------------


class FakeGatewayClient:
    """Offline stub that returns deterministic responses from in-memory maps.

    Designed for unit tests: no network, no live gateway required.

    Parameters
    ----------
    completion_map:
        Maps the *content* of the last ``user`` message to a response string.
        If a message is not found, ``default_completion`` is returned.
    default_completion:
        Fallback response content when no entry in ``completion_map`` matches.
    embedding_fn:
        Callable that receives a string and returns a ``list[float]`` embedding.
        Defaults to a simple deterministic hash-based vector.
    completion_latency_ms:
        Simulated latency reported in the response (does not actually sleep).
    embedding_latency_ms:
        Simulated latency reported in the embedding response.
    prompt_tokens_per_char:
        Rough approximation used to fill ``usage.prompt_tokens``.
    completion_tokens_per_char:
        Rough approximation used to fill ``usage.completion_tokens``.
    """

    def __init__(
        self,
        completion_map: dict[str, str] | None = None,
        default_completion: str = "Default fake response.",
        embedding_fn: Any | None = None,
        completion_latency_ms: float = 50.0,
        embedding_latency_ms: float = 20.0,
        prompt_tokens_per_char: float = 0.25,
        completion_tokens_per_char: float = 0.25,
    ) -> None:
        self._completion_map: dict[str, str] = completion_map or {}
        self._default_completion = default_completion
        self._embedding_fn = embedding_fn or default_embedding_fn
        self._completion_latency_ms = completion_latency_ms
        self._embedding_latency_ms = embedding_latency_ms
        self._prompt_tokens_per_char = prompt_tokens_per_char
        self._completion_tokens_per_char = completion_tokens_per_char

    def chat_completion(self, request: CompletionRequest) -> CompletionResponse:
        user_content = ""
        for msg in reversed(request.messages):
            if msg.role == "user":
                user_content = msg.content
                break

        content = self._completion_map.get(user_content, self._default_completion)
        prompt_chars = sum(len(m.content) for m in request.messages)
        prompt_tokens = max(1, int(prompt_chars * self._prompt_tokens_per_char))
        completion_tokens = max(1, int(len(content) * self._completion_tokens_per_char))
        return CompletionResponse(
            content=content,
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            latency_ms=self._completion_latency_ms,
        )

    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        embedding: list[float] = self._embedding_fn(request.input)
        input_tokens = max(1, int(len(request.input) * 0.25))
        return EmbeddingResponse(
            embedding=embedding,
            usage=UsageInfo(
                prompt_tokens=input_tokens,
                completion_tokens=0,
                total_tokens=input_tokens,
            ),
            latency_ms=self._embedding_latency_ms,
        )


def default_embedding_fn(text: str) -> list[float]:
    """Return a deterministic unit-norm 8-dimensional fake embedding.

    Not meaningful as a semantic representation — only useful for testing that
    the runner correctly computes cosine similarity between two vectors.
    """
    dim = 8
    vec = [0.0] * dim
    for i, ch in enumerate(text):
        vec[i % dim] += ord(ch)
    norm = sum(v * v for v in vec) ** 0.5
    if norm == 0.0:
        vec[0] = 1.0
        return vec
    return [v / norm for v in vec]
