# Tool: send an email from your Gmail account over SMTP.
#
# Part of 6-Agents/08-notify-agent (book Chapter 8, "An agent that acts").
# Not run directly; agent.py imports it.
#
# Setup (once): turn on 2-Step Verification for your Google account, create an
# App Password (Google Account -> Security -> App passwords), and put
# GMAIL_ADDRESS and GMAIL_APP_PASSWORD in the repo-root .env. Never use your
# normal Gmail password here.
#
# Safety: the tool only sends to addresses listed in ALLOWED_EMAIL_RECIPIENTS
# (comma-separated). If that is not set, the only allowed recipient is
# GMAIL_ADDRESS itself — the agent can email you and nobody else.

import os
import smtplib
from email.message import EmailMessage

from langchain_core.tools import tool


def _allowed_recipients() -> set[str]:
    configured = os.getenv("ALLOWED_EMAIL_RECIPIENTS") or os.getenv("GMAIL_ADDRESS", "")
    return {address.strip().lower() for address in configured.split(",") if address.strip()}


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email from the user's Gmail account.

    Use this when the user asks to email something. `to` must be one of the
    user's approved recipients; keep the subject short and the body plain text.
    """
    if to.strip().lower() not in _allowed_recipients():
        # The code, not the model, decides who may receive email.
        return f"Refused: {to} is not an approved recipient."

    if os.getenv("NOTIFY_DRY_RUN") == "1":
        print(f"[DRY RUN] Email -> to={to!r} subject={subject!r}\n{body}")
        return "Dry run: the email was not really sent."

    sender, password = os.getenv("GMAIL_ADDRESS"), os.getenv("GMAIL_APP_PASSWORD")
    if not sender or not password:
        return "Gmail is not configured (GMAIL_ADDRESS / GMAIL_APP_PASSWORD missing)."

    message = EmailMessage()
    message["From"], message["To"], message["Subject"] = sender, to, subject
    message.set_content(body)
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=20) as smtp:
            smtp.login(sender, password)
            smtp.send_message(message)
    except (smtplib.SMTPException, OSError) as error:
        return f"Email failed: {error}"
    return f"Email sent to {to}."
