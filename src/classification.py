"""Emergency type classification module."""

from typing import Dict, Any
from transformers import pipeline
import json

# load trained classifier
classifier = pipeline(
    "text-classification",
    model="models/classifier",
    tokenizer="xlm-roberta-base"
)

# load label mapping (the correct one you generated during training)
with open("models/classifier/label_mapping.json") as f:
    label_map = json.load(f)


def classify_emergency(text: str) -> Dict[str, Any]:

    result = classifier(text)[0]

    raw_label = result["label"]
    score = result["score"]

    # convert LABEL_X → actual label
    if raw_label.startswith("LABEL_"):
        label_id = raw_label.split("_")[1]
        label = label_map.get(label_id, raw_label)
    else:
        label = raw_label

    return {
        "label": label,
        "score": score
    }