from transformers import pipeline

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

LABELS = [
    "accident",
    "fire",
    "flood",
    "gas leak",
    "earthquake",
    "crime",
    "building collapse"
]

def classify_emergency(text: str):

    result = classifier(text, LABELS)

    return {
        "label": result["labels"][0],
        "score": result["scores"][0]
    }