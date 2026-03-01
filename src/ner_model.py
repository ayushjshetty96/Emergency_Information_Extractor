"""Named Entity Recognition (NER) model for emergency-related entities."""

from typing import List, Dict, Any


def extract_entities(text: str) -> List[Dict[str, Any]]:
    """Placeholder for NER model inference.

    Expected output: list of entities with fields like:
    {
        "text": "Mumbai",
        "label": "LOCATION",
        "start": 10,
        "end": 16,
    }
    """
    raise NotImplementedError("Implement NER model inference.")

