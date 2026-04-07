"""Email notification utilities for emergency alerts."""

from __future__ import annotations

import json
import os
import smtplib
from email.message import EmailMessage
from typing import Any, Dict, Optional


def _build_email_body(raw_text: str, result: Dict[str, Any]) -> str:
    """Build a structured plain-text email payload."""
    lines = [
        "Emergency Information Extractor Alert",
        "=" * 40,
        "",
        f"Incident: {result.get('incident', 'Unknown')}",
        f"Severity: {result.get('severity', 'Unknown')}",
        f"Priority Score: {result.get('priority_score', 'N/A')}",
        f"Location: {result.get('location', 'Not detected')}",
        f"Latitude: {result.get('latitude', 'N/A')}",
        f"Longitude: {result.get('longitude', 'N/A')}",
        f"Victims: {result.get('victims', 'N/A')}",
        f"Language: {result.get('language', 'N/A')}",
        f"Duplicate: {result.get('duplicate', 'N/A')}",
        f"Confidence: {result.get('confidence', 'N/A')}",
        "",
        "Original Report Text:",
        raw_text,
        "",
        "Structured JSON:",
        json.dumps(result, indent=2),
    ]
    return "\n".join(lines)


def send_emergency_email(
    result: Dict[str, Any],
    recipient_email: str,
    raw_text: str,
    *,
    smtp_host: Optional[str] = None,
    smtp_port: Optional[int] = None,
    sender_email: Optional[str] = None,
    sender_password: Optional[str] = None,
    use_tls: Optional[bool] = None,
) -> None:
    """Send a structured emergency alert email.

    Config values default to these environment variables:
    - SMTP_HOST (default: smtp.gmail.com)
    - SMTP_PORT (default: 587)
    - SMTP_SENDER_EMAIL (required if sender_email not provided)
    - SMTP_SENDER_PASSWORD (required if sender_password not provided)
    - SMTP_USE_TLS (default: true)
    """
    smtp_host = smtp_host or os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port = smtp_port if smtp_port is not None else int(os.getenv("SMTP_PORT", "587"))
    sender_email = sender_email or os.getenv("SMTP_SENDER_EMAIL")
    sender_password = sender_password or os.getenv("SMTP_SENDER_PASSWORD")
    if use_tls is None:
        use_tls = os.getenv("SMTP_USE_TLS", "true").strip().lower() in {"1", "true", "yes"}

    if not sender_email or not sender_password:
        raise ValueError(
            "Missing SMTP credentials. Set SMTP_SENDER_EMAIL and SMTP_SENDER_PASSWORD environment variables."
        )

    incident = str(result.get("incident", "Emergency"))
    severity = str(result.get("severity", "Unknown")).upper()
    subject = f"[EMERGENCY ALERT] {incident} | Severity: {severity}"

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = sender_email
    message["To"] = recipient_email
    message.set_content(_build_email_body(raw_text=raw_text, result=result))

    with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as smtp:
        if use_tls:
            smtp.starttls()
        smtp.login(sender_email, sender_password)
        smtp.send_message(message)
