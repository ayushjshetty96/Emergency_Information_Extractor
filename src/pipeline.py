"""End-to-end emergency information extraction pipeline."""

from typing import Dict, Any

from .speech_to_text import transcribe_audio
from .preprocessing import clean_text
from .language_detection import detect_language
from .ner_model import extract_entities
from .classification import classify_emergency
from .severity_model import estimate_severity


def run_pipeline_from_audio(audio_path: str) -> Dict[str, Any]:
    """Run the full pipeline starting from an audio file."""
    # NOTE: All called functions are currently placeholders.
    text = transcribe_audio(audio_path)
    text_clean = clean_text(text)
    language = detect_language(text_clean)
    entities = extract_entities(text_clean)
    emergency = classify_emergency(text_clean)
    severity = estimate_severity(text_clean)

    return {
        "raw_text": text,
        "clean_text": text_clean,
        "language": language,
        "entities": entities,
        "emergency": emergency,
        "severity": severity,
    }

