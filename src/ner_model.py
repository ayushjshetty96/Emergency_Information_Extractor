"""Named Entity Recognition (NER) model for emergency-related entities."""

from transformers import pipeline
from typing import List, Dict, Any

# Load pretrained NER model
ner_pipeline = pipeline(
    "ner",
    model="dslim/bert-base-NER",
    aggregation_strategy="simple"
)


def extract_entities(text: str) -> List[Dict[str, Any]]:
    """Extract location entities using NER + simple rules."""

    entities = []

    # -----------------------------
    # Step 1 — NER model detection
    # -----------------------------
    results = ner_pipeline(text)

    for ent in results:
        if ent["entity_group"] in ["LOC", "ORG"]:
            word = ent["word"].replace("##", "")
            if len(word) > 3:
                entities.append(word.lower())

    # -------------------------------------
    # Step 2 — Detect capitalized place names
    # (helps with Hinglish like "Whitefield road par accident hua")
    # -------------------------------------
    tokens = text.split()

    # words that are NOT locations even if capitalized
    non_location_caps = {
        "major", "huge", "gas", "flood", "earthquake",
        "bridge", "accident", "fire"
    }

    for token in tokens:
        word = token.strip(",.")
        if (
            word
            and word[0].isupper()
            and len(word) > 3
            and word.lower() not in non_location_caps
        ):
            entities.append(word.lower())

    # -------------------------------------
    # Step 2b — "X road" style pattern
    # Works even when the whole text is lowercased, e.g.
    # "whitefield road par bada accident hua" → "whitefield"
    # and keeps "mg road" as a single location.
    # -------------------------------------
    for i, token in enumerate(tokens):
        word = token.strip(",.").lower()
        if word == "road" and i > 0:
            prev = tokens[i - 1].strip(",.")
            if prev and prev.isalpha():
                # For "mg road" keep full phrase, for others keep just area name
                if prev.lower() == "mg":
                    entities.append("mg road")
                elif len(prev) > 3:
                    entities.append(prev.lower())

    # -------------------------------------
    # Step 2c — "[Area] board" pattern
    # e.g. "Major, Silk Board" → "silk board"
    # -------------------------------------
    for i, token in enumerate(tokens):
        word = token.strip(",.").lower()
        if word == "board" and i > 0:
            prev = tokens[i - 1].strip(",.")
            if prev and prev.isalpha():
                entities.append(f"{prev.lower()} board")

    # -------------------------------------
    # Step 3 — Phrase extraction
    # near / in / at / outside / inside
    # -------------------------------------
    cues = ["near", "in", "at", "outside", "inside"]
    words = text.split()

    for i, w in enumerate(words):
        if w.lower() in cues and i + 1 < len(words):

            phrase_words = []

            for j in range(i + 1, min(i + 6, len(words))):
                token = words[j].strip(",.")
                phrase_words.append(token)

                # stop if next word is lowercase
                if j > i + 1 and token[0].islower():
                    break

            phrase = " ".join(phrase_words).lower()
            entities.append(phrase)

    # -------------------------------------
    # Step 3b — Kannada markers (nalli / hatra / hattira)
    # Use the word(s) BEFORE these markers as location.
    # e.g. "Bengaluru Majestic bus stand nalli accident aagide" → "majestic"
    # -------------------------------------
    kannada_markers = {"nalli", "hatra", "hattira"}
    generic_place_words = {"bus", "stand", "station", "road", "area", "city"}

    for i, w in enumerate(words):
        word_l = w.strip(",.").lower()
        if word_l in kannada_markers and i > 0:
            # walk backwards to find the most specific place token
            for j in range(i - 1, -1, -1):
                token = words[j].strip(",.")
                token_l = token.lower()

                if not token or not token.isalpha():
                    continue

                if token_l in generic_place_words:
                    continue

                if len(token_l) <= 2:
                    continue

                entities.append(token_l)
                break

    # -------------------------------------
    # Step 4 — Cleaning rules
    # -------------------------------------
    stop_words = {
        "a", "an", "the",
        "restaurant", "shop", "building", "houses",
        "kitchen", "traffic", "reported", "causing",
        "road",   # important for cases like "Whitefield road"
        "major", "huge", "gas", "flood", "earthquake",
        "bridge", "accident", "fire"
    }

    cleaned = []

    for e in entities:

        words = e.split()

        # remove phrases starting with articles
        if words[0] in {"a", "an", "the"}:
            continue

        # trim generic words like "road"
        if words[-1] in stop_words and len(words) > 1:
            # special-case "mg road" → keep full phrase
            if not (words[-1] == "road" and words[-2].lower() == "mg"):
                words = words[:-1]
                e = " ".join(words)

        # drop single tokens that are generic non-location words
        if len(words) == 1 and words[0] in stop_words:
            continue

        # remove broken tokens
        if "##" in e or e.startswith("dir"):
            continue

        cleaned.append(e)

    # remove duplicates
    cleaned = list(set(cleaned))

    return [{"text": e, "label": "LOCATION"} for e in cleaned]