"""Outgoing email through a Google Apps Script web app (MAIL_WEBHOOK_URL + MAIL_WEBHOOK_SECRET).

The script sends the message with MailApp from the Google account that deployed it.
It always answers HTTP 200; the real result is the "status" field of the JSON it returns.
"""
import logging
import re

import requests
from jinja2 import Environment, FileSystemLoader, select_autoescape
from langsmith import traceable

from . import config

log = logging.getLogger(__name__)

_env = Environment(
    loader=FileSystemLoader(config.BASE_DIR / "app" / "templates" / "mail"),
    autoescape=select_autoescape(["html"]),
)


def render(template: str, **ctx) -> str:
    return _env.get_template(template).render(base_url=config.BASE_URL, **ctx)


def _html_to_text(html: str) -> str:
    text = re.sub(r"<(br|/p|/div|/tr|/h\d|/li)[^>]*>", "\n", html, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


# The body and the secret are left out of the LangSmith trace on purpose.
@traceable(run_type="tool", name="send_email",
           process_inputs=lambda d: {k: d.get(k) for k in ("to", "subject")})
def send_email(to: str, subject: str, html: str) -> bool:
    """Returns True only when the Apps Script confirmed the email was sent."""
    if not config.MAIL_ENABLED:
        log.warning("Email to %s not sent: MAIL_WEBHOOK_URL / MAIL_WEBHOOK_SECRET are not set", to)
        return False
    payload = {
        "secret": config.MAIL_WEBHOOK_SECRET,
        "to": to,
        "subject": subject,
        "htmlBody": html,
        "textBody": _html_to_text(html),
    }
    try:
        # Apps Script answers with a redirect; requests follows it as a GET, which is what Google expects.
        r = requests.post(config.MAIL_WEBHOOK_URL, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
    except (requests.RequestException, ValueError) as e:
        log.error("Email to %s failed: %s", to, e)
        return False
    if data.get("status") != 200:
        log.error("Email to %s failed: %s %s", to, data.get("status"), data.get("message"))
        return False
    log.info("Email sent to %s (%s)", to, subject)
    return True
