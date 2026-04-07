"""End-to-end emergency information extraction pipeline."""

from typing import Dict, Any

from src.geocoding import get_coordinates
from src.priority_score import calculate_priority
from src.duplicate_detection import is_duplicate
from src.victim_extraction import extract_victims

from .preprocessing import clean_text
from .language_detection import detect_language
from .ner_model import extract_entities
from .classification import classify_emergency
from .severity_model import estimate_severity


def run_pipeline_from_text(text: str) -> Dict[str, Any]:
    """Run the full pipeline starting from text."""

    # Preprocessing
    text_clean = clean_text(text)

    # Language detection
    language = detect_language(text_clean)

    # Named entity extraction
    entities = extract_entities(text)

    # Emergency classification
    emergency = classify_emergency(text_clean)

    # Severity estimation
    severity = estimate_severity(text_clean)

    # Extract location from entities
    locations = list(set(e["text"] for e in entities))
    
    # Also check for common city names in the text that NER might miss
    common_cities = ["bangalore", "bengaluru", "delhi", "mumbai", "pune", "gurgaon", 
                    "noida", "chennai", "hyderabad", "kolkata", "ahmedabad", "jaipur"]
    text_lower = text_clean.lower()
    for city in common_cities:
        if city in text_lower and not any(city in loc for loc in locations):
            locations.append(city)
    
    location_text = ", ".join(locations) if locations else None

    # Geolocation coordinates
    latitude, longitude = get_coordinates(location_text)

    # Priority score
    priority_score = calculate_priority(text_clean, severity)

    # Duplicate detection
    duplicate_report = is_duplicate(text_clean)

    # Victim extraction
    victims = extract_victims(text_clean)

    # Final structured output - Clean API-ready format
    result = {
        "incident": emergency["label"],
        "location": location_text,
        "latitude": latitude,
        "longitude": longitude,
        "severity": severity,
        "priority_score": priority_score,
        "victims": victims,
        "language": language,
        "duplicate": duplicate_report,
        "confidence": emergency["score"]
    }

    return result


if __name__ == "__main__":

    # Example test sentences
    test_sentences = [
        # "Whitefield road par bada accident hua",
        # "Major road accident involving two buses near Silk Board junction",
        # "Gas leak detected inside a restaurant kitchen in Koramangala",
        # "Flood water entering houses after heavy rain in Whitefield",
        # "Earthquake tremors felt across Delhi NCR region",
        # "A gang robbery reported outside a jewelry shop near Indiranagar metro",
        # "Bridge collapse reported near railway station causing traffic jam",
        # "Bengaluru Majestic bus stand nalli accident aagide",
        "Fire near mg road gurugram haryana, multiple vehicles burnt, people injured",
        "Colaba Causeway, Mumbai par fire hogaya, logon ko bachao"
        
    ]

    for text in test_sentences:

        result = run_pipeline_from_text(text)

        # Display as clean JSON-like response
        print("\n==============================")
        print("Emergency Alert Response")
        print("==============================\n")
        
        import json
        print(json.dumps(result, indent=2))