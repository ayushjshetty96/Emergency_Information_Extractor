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


if __name__ == "__main__":

    # Example test sentences
    test_sentences = [
        "Whitefield road par bada accident hua",
        "Huge fire broke out in a residential building near MG Road Bangalore",
        "Major road accident involving two buses near Silk Board junction",
        "Gas leak detected inside a restaurant kitchen in Koramangala",
        "Flood water entering houses after heavy rain in Whitefield",
        "Earthquake tremors felt across Delhi NCR region",
        "A gang robbery reported outside a jewelry shop near Indiranagar metro",
        "Bridge collapse reported near railway station causing traffic jam",
        "Gas leak reported near Indiranagar metro station Bangalore",
        # Kannada sentence written in English (transliteration)
        "Bengaluru Majestic bus stand nalli accident aagide"
    ]

    for text in test_sentences:

        text_clean = clean_text(text)
        language = detect_language(text_clean)
        entities = extract_entities(text_clean)
        emergency = classify_emergency(text_clean)
        severity = estimate_severity(text_clean)

        locations = list(set(e["text"] for e in entities))
        location_text = ", ".join(locations)

        print("\n==============================")
        print("Emergency Message Analysis")
        print("==============================")

        print("\nInput:")
        print(text)

        print("\nIncident:", emergency["label"], "(confidence:", round(emergency["score"], 3), ")")
        print("Location:", location_text if location_text else "Unknown")
        print("Severity:", severity)
        print("Language:", language)