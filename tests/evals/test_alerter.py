"""POL-5 — Tests for WebhookAlerter."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
from urllib.request import Request

import pytest

from atlas_prompts.evals.drift.alerter import WebhookAlerter

_VERSIONS = [{"prompt_version": "v1.2", "n_samples": 100, "alerted": True, "run_id": "abc"}]
_SUMMARY = "1/1 versions alerted"
_EVALUATED_AT = "2026-06-10T02:00:00+00:00"


def test_fire_logs_when_no_url_configured(caplog: pytest.LogCaptureFixture) -> None:
    alerter = WebhookAlerter(None)
    import logging

    with caplog.at_level(logging.WARNING):
        alerter.fire(_SUMMARY, _EVALUATED_AT, _VERSIONS)
    assert "no webhook configured" in caplog.text


def test_fire_posts_json_to_webhook() -> None:
    captured: list[bytes] = []

    mock_resp = MagicMock()
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.status = 200

    def fake_urlopen(req: Request, timeout: float = 10) -> MagicMock:
        assert isinstance(req.data, bytes)
        captured.append(req.data)
        return mock_resp

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        alerter = WebhookAlerter("https://hooks.example.com/atlas")
        alerter.fire(_SUMMARY, _EVALUATED_AT, _VERSIONS)

    assert len(captured) == 1
    payload = json.loads(captured[0])
    assert payload["evaluated_at"] == _EVALUATED_AT
    assert payload["versions"][0]["prompt_version"] == "v1.2"
    assert payload["versions"][0]["alerted"] is True


def test_fire_swallows_http_error(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
        alerter = WebhookAlerter("https://hooks.example.com/atlas")
        with caplog.at_level(logging.ERROR):
            alerter.fire(_SUMMARY, _EVALUATED_AT, _VERSIONS)

    assert "webhook dispatch failed" in caplog.text


def test_fire_includes_all_versions() -> None:
    versions = [
        {"prompt_version": f"v{i}", "n_samples": 50, "alerted": i % 2 == 0, "run_id": str(i)}
        for i in range(3)
    ]
    captured: list[bytes] = []

    mock_resp = MagicMock()
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.status = 200

    def fake_urlopen(req: Request, timeout: float = 10) -> MagicMock:
        assert isinstance(req.data, bytes)
        captured.append(req.data)
        return mock_resp

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        WebhookAlerter("https://hooks.example.com/atlas").fire(_SUMMARY, _EVALUATED_AT, versions)

    payload = json.loads(captured[0])
    assert len(payload["versions"]) == 3
