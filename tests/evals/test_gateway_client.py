"""Tests for the GatewayClient Protocol and FakeGatewayClient (REG-8)."""

# pyright: reportUnknownMemberType=false

from __future__ import annotations

import math

import pytest

from evals.runner.gateway_client import (
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    FakeGatewayClient,
    GatewayClient,
    default_embedding_fn,
)

# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestGatewayClientProtocol:
    def test_fake_client_satisfies_protocol(self) -> None:
        client = FakeGatewayClient()
        assert isinstance(client, GatewayClient)

    def test_protocol_is_runtime_checkable(self) -> None:
        # GatewayClient must be runtime_checkable so runners can guard on it
        assert hasattr(GatewayClient, "__protocol_attrs__")


# ---------------------------------------------------------------------------
# FakeGatewayClient — chat_completion
# ---------------------------------------------------------------------------


class TestFakeClientChatCompletion:
    def _make_request(self, content: str) -> CompletionRequest:
        return CompletionRequest(
            model="test-model",
            messages=[
                ChatMessage(role="system", content="You are helpful."),
                ChatMessage(role="user", content=content),
            ],
        )

    def test_returns_mapped_response(self) -> None:
        client = FakeGatewayClient(completion_map={"What is X?": "X is Y."})
        resp = client.chat_completion(self._make_request("What is X?"))
        assert isinstance(resp, CompletionResponse)
        assert resp.content == "X is Y."

    def test_returns_default_for_unknown_input(self) -> None:
        client = FakeGatewayClient(default_completion="FALLBACK")
        resp = client.chat_completion(self._make_request("Unknown question"))
        assert resp.content == "FALLBACK"

    def test_usage_populated(self) -> None:
        client = FakeGatewayClient(completion_map={"Q": "A"})
        resp = client.chat_completion(self._make_request("Q"))
        assert resp.usage.prompt_tokens >= 1
        assert resp.usage.completion_tokens >= 1
        assert resp.usage.total_tokens == resp.usage.prompt_tokens + resp.usage.completion_tokens

    def test_simulated_latency_reported(self) -> None:
        client = FakeGatewayClient(completion_latency_ms=123.0)
        resp = client.chat_completion(self._make_request("any"))
        assert resp.latency_ms == pytest.approx(123.0)

    def test_empty_completion_map_uses_default(self) -> None:
        client = FakeGatewayClient()
        resp = client.chat_completion(self._make_request("hello"))
        assert resp.content == "Default fake response."

    def test_last_user_message_used_for_lookup(self) -> None:
        """The lookup key is the last user-role message, not the first."""
        client = FakeGatewayClient(completion_map={"second": "got-second"})
        req = CompletionRequest(
            model="m",
            messages=[
                ChatMessage(role="user", content="first"),
                ChatMessage(role="assistant", content="intermediate"),
                ChatMessage(role="user", content="second"),
            ],
        )
        resp = client.chat_completion(req)
        assert resp.content == "got-second"


# ---------------------------------------------------------------------------
# FakeGatewayClient — embed
# ---------------------------------------------------------------------------


class TestFakeClientEmbed:
    def test_returns_embedding_response(self) -> None:
        client = FakeGatewayClient()
        resp = client.embed(EmbeddingRequest(model="emb-model", input="hello world"))
        assert isinstance(resp, EmbeddingResponse)
        assert len(resp.embedding) > 0
        assert all(isinstance(v, float) for v in resp.embedding)

    def test_custom_embedding_fn(self) -> None:
        def fixed_fn(text: str) -> list[float]:
            return [1.0, 0.0, 0.0]

        client = FakeGatewayClient(embedding_fn=fixed_fn)
        resp = client.embed(EmbeddingRequest(model="m", input="any"))
        assert resp.embedding == [1.0, 0.0, 0.0]

    def test_simulated_latency_reported(self) -> None:
        client = FakeGatewayClient(embedding_latency_ms=7.5)
        resp = client.embed(EmbeddingRequest(model="m", input="text"))
        assert resp.latency_ms == pytest.approx(7.5)

    def test_usage_populated(self) -> None:
        client = FakeGatewayClient()
        resp = client.embed(EmbeddingRequest(model="m", input="hello"))
        assert resp.usage.prompt_tokens >= 1
        assert resp.usage.completion_tokens == 0


# ---------------------------------------------------------------------------
# Default embedding function
# ---------------------------------------------------------------------------


class TestDefaultEmbeddingFn:
    def test_returns_8_dimensions(self) -> None:
        vec = default_embedding_fn("hello")
        assert len(vec) == 8

    def test_is_unit_norm(self) -> None:
        vec = default_embedding_fn("some text here")
        norm = math.sqrt(sum(v * v for v in vec))
        assert norm == pytest.approx(1.0, abs=1e-6)

    def test_deterministic(self) -> None:
        assert default_embedding_fn("abc") == default_embedding_fn("abc")

    def test_different_texts_differ(self) -> None:
        assert default_embedding_fn("foo") != default_embedding_fn("bar")

    def test_empty_string_returns_valid_vector(self) -> None:
        vec = default_embedding_fn("")
        assert len(vec) == 8
        assert vec[0] == 1.0  # sentinel for zero-input
