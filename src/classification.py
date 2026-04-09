from pathlib import Path
from typing import Dict, Any

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "classifier"
_classifier = None


def _load_classifier():
    global _classifier
    if _classifier is not None:
        return _classifier

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    _classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        return_all_scores=False,
        device=-1,
    )
    return _classifier


def get_classifier_info() -> Dict[str, Any]:
    """Return information about the loaded classifier model."""
    classifier = _load_classifier()
    model = classifier.model
    return {
        "model_type": getattr(model.config, "model_type", "unknown"),
        "model_path": str(MODEL_DIR),
        "num_labels": getattr(model.config, "num_labels", None),
        "label_mapping": getattr(model.config, "id2label", {}),
    }


def classify_emergency(text: str) -> Dict[str, Any]:
    """Classify emergency type using the XLM-RoBERTa transformer model."""
    classifier = _load_classifier()
    result = classifier(text, truncation=True, max_length=128)
    result = result[0] if isinstance(result, list) else result

    return {
        "label": result["label"],
        "score": float(result["score"]),
    }
