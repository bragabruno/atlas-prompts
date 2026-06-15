"""POL-5 — Drift alert dispatcher.

Sends a JSON webhook POST when the nightly drift eval fires an alert.
The dispatcher is injected into the CLI via the ``ATLAS_DRIFT_WEBHOOK_URL``
env var; when unset no HTTP call is made (safe offline default).

The payload shape is deliberately flat so it round-trips cleanly into
Slack/Teams inbound webhooks and simple alert aggregators:

    {
      "text": "Atlas drift alert ...",
      "summary": "...",
      "evaluated_at": "...",
      "versions": [{"prompt_version": ..., "alerted": ..., ...}, ...]
    }
"""

from __future__ import annotations

import json
import logging
import urllib.request
from collections.abc import Mapping, Sequence

log = logging.getLogger(__name__)


class WebhookAlerter:
    """POST a drift-alert JSON payload to a configured webhook URL.

    Parameters
    ----------
    url:
        Full webhook URL (Slack inbound, Teams, PagerDuty, or any HTTP sink).
        When ``None`` the alert is logged but not dispatched — safe default.
    timeout_s:
        HTTP timeout in seconds (default 10).
    """

    def __init__(self, url: str | None, *, timeout_s: int = 10) -> None:
        self._url = url
        self._timeout = timeout_s

    def fire(
        self, summary: str, evaluated_at: str, versions: Sequence[Mapping[str, object]]
    ) -> None:
        payload = {
            "text": f"Atlas drift alert — {evaluated_at}",
            "summary": summary,
            "evaluated_at": evaluated_at,
            "versions": versions,
        }

        if self._url is None:
            log.warning("drift_alert (no webhook configured): %s", summary)
            return

        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            self._url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                log.info("drift_alert dispatched to webhook: HTTP %s", resp.status)
        except Exception as exc:
            log.error("drift_alert webhook dispatch failed: %s", exc)
