"""Text preprocessing utilities for the emergency information pipeline."""

import re
from typing import List


def clean_text(text: str) -> str:
    """
    Normalize emergency text messages.

    Steps:
    - lowercase
    - remove URLs
    - remove punctuation
    - remove extra whitespace
    """

    if not isinstance(text, str):
        return ""

    # lowercase
    text = text.lower()

    # remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # remove punctuation
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

    # remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def tokenize(text: str) -> List[str]:
    """
    Simple whitespace tokenizer for emergency text.
    """

    if not isinstance(text, str):
        return []

    tokens = text.split()

    return tokens