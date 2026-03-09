"""Severity estimation module."""

from typing import Dict, Any


def estimate_severity(text: str) -> str:
    """Estimate emergency severity based on keywords."""

    text = text.lower()

    high_keywords = [
        "explosion",
        "collapse",
        "earthquake",
        "major accident",
        "huge fire",
        "bridge collapse"
    ]

    medium_keywords = [
        "fire",
        "gas leak",
        "flood",
        "accident"
    ]

    for word in high_keywords:
        if word in text:
            return "high"

    for word in medium_keywords:
        if word in text:
            return "medium"

    return "low"