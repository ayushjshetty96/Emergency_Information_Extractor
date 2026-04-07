from transformers import pipeline
import re

LABELS = [
    "accident",
    "fire",
    "flood",
    "gas leak",
    "earthquake",
    "crime",
    "building collapse"
]

# Keywords mapping for efficient classification
KEYWORDS = {
    "accident": ["accident", "crash", "collision", "injured", "ambulance", "wreck"],
    "fire": ["fire", "burning", "ablaze", "flames", "burnt", "smoke"],
    "flood": ["flood", "water", "inundated", "submerged", "drowning"],
    "gas leak": ["gas leak", "leak", "fumes", "gas"],
    "earthquake": ["earthquake", "tremor", "seismic", "quake"],
    "crime": ["robbery", "theft", "robbery", "murder", "assault", "kidnap"],
    "building collapse": ["collapse", "collapsed", "crumbled"]
}

def classify_emergency(text: str):
    """Classify emergency type using keyword matching (fast, memory-efficient)."""
    text_lower = text.lower()
    
    # Count keyword matches for each label
    scores = {}
    for label, keywords in KEYWORDS.items():
        match_count = sum(1 for kw in keywords if kw in text_lower)
        scores[label] = match_count
    
    # Get the label with most matches, default to "accident" if none found
    best_label = max(scores, key=scores.get) if max(scores.values()) > 0 else "accident"
    confidence = min(0.95, 0.5 + (max(scores.values()) * 0.15))
    
    return {
        "label": best_label,
        "score": confidence
    }