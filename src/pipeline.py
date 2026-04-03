"""End-to-end emergency information extraction pipeline."""

from typing import Dict, Any

from src.geocoding import get_coordinates
from src.priority_score import calculate_priority
from src.duplicate_detection import is_duplicate
from src.victim_extraction import extract_victims

from .speech_to_text import transcribe_audio
from .preprocessing import clean_text
from .language_detection import detect_language
from .ner_model import extract_entities
from .classification import classify_emergency
from .severity_model import estimate_severity


def run_pipeline_from_audio(audio_path: str) -> Dict[str, Any]:
    """Run the full pipeline starting from an audio file."""

    # Convert speech to text
    text = transcribe_audio(audio_path)

    return run_pipeline_from_text(text)


def run_pipeline_from_text(text: str) -> Dict[str, Any]:
    """Run the full pipeline starting from text."""

    # Preprocessing
    text_clean = clean_text(text)

    # Language detection
    language = detect_language(text_clean)

    # Named entity extraction
    entities = extract_entities(text_clean)

    # Emergency classification
    emergency = classify_emergency(text_clean)

    # Severity estimation
    severity = estimate_severity(text_clean)

    # Extract location from entities
    locations = list(set(e["text"] for e in entities))
    location_text = ", ".join(locations) if locations else None

    # Geolocation coordinates
    latitude, longitude = get_coordinates(location_text)

    # Priority score
    priority_score = calculate_priority(text_clean, severity)

    # Duplicate detection
    duplicate_report = is_duplicate(text_clean)

    # Victim extraction
    victims = extract_victims(text_clean)

    # Final structured output
    result = {
        "raw_text": text,
        "clean_text": text_clean,
        "language": language,
        "entities": entities,
        "incident": emergency["label"],
        "confidence": emergency["score"],
        "location": location_text,
        "latitude": latitude,
        "longitude": longitude,
        "severity": severity,
        "priority_score": priority_score,
        "victims": victims,
        "duplicate_report": duplicate_report,
    }

    return result


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
        "Bengaluru Majestic bus stand nalli accident aagide"
    ]

    for text in test_sentences:

        result = run_pipeline_from_text(text)

        print("\n==============================")
        print("Emergency Message Analysis")
        print("==============================")

        print("\nInput:")
        print(result["raw_text"])

        print("\nIncident:", result["incident"], "(confidence:", round(result["confidence"], 3), ")")
        print("Location:", result["location"] if result["location"] else "Unknown")
        print("Latitude:", result["latitude"])
        print("Longitude:", result["longitude"])
        print("Severity:", result["severity"])
        print("Priority Score:", result["priority_score"])
        print("Victims:", result["victims"])
        print("Language:", result["language"])
        print("Duplicate Report:", result["duplicate_report"])