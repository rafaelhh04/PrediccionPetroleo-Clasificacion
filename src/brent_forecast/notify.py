"""Notifications of scheduled jobs (retraining, drift) to a chat webhook.

One JSON ``POST`` to ``notify.webhook_url`` with the message under both
``text`` (Slack, Mattermost, Google Chat, Teams workflows) and ``content``
(Discord), so the same URL setting works with the usual incoming webhooks.
In the cloud the URL comes from Secret Manager (``BRENT_NOTIFY__WEBHOOK_URL``).

A notification never fails the job that sends it: errors are logged and
:func:`notify` returns ``False``.
"""

import json
import logging
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence

logger = logging.getLogger(__name__)

Post = Callable[[str, bytes, float], int]
"""``(url, body, timeout) -> HTTP status``."""


def http_post(url: str, body: bytes, timeout: float) -> int:
    """POST a JSON body (the default transport); return the status code."""
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": "brent-forecast"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return int(response.status)


def format_message(title: str, lines: Sequence[str]) -> str:
    """Title in bold followed by one bullet per line (Markdown understood by most chats)."""
    return "\n".join([f"*{title}*", *(f"• {line}" for line in lines)])


def notify(
    webhook_url: str | None,
    title: str,
    lines: Sequence[str] = (),
    *,
    timeout: float = 10.0,
    post: Post = http_post,
) -> bool:
    """Send ``title`` and ``lines`` to the webhook; return whether it was delivered.

    Without a URL the message is only logged (local runs, tests).
    """
    message = format_message(title, lines)
    if not webhook_url:
        logger.info("No notification webhook configured:\n%s", message)
        return False
    body = json.dumps({"text": message, "content": message}).encode()
    try:
        status = post(webhook_url, body, timeout)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        logger.warning("Notification not delivered: %s", exc)
        return False
    if not 200 <= status < 300:
        logger.warning("Notification rejected by the webhook (HTTP %d)", status)
        return False
    return True
