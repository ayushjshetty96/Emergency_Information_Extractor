import json
from transformers import pipeline

# Load label mapping
with open("models/classifier/label_mapping.json") as f:
    label_map = json.load(f)

classifier = pipeline(
    "text-classification",
    model="models/classifier",
    tokenizer="xlm-roberta-base"
)

texts = [
    "Huge fire broke out in a residential building near MG Road Bangalore",
    "Major road accident involving two buses near Silk Board junction",
    "Gas leak detected inside a restaurant kitchen in Koramangala",
    "Flood water entering houses after heavy rain in Whitefield",
    "Earthquake tremors felt across Delhi NCR region",
    "A gang robbery reported outside a jewelry shop near Indiranagar metro",
    "Bridge collapse reported near railway station causing traffic jam"
]

for text in texts:
    result = classifier(text)[0]

    label_id = result["label"].replace("LABEL_", "")
    label = label_map[label_id]

    print("\nINPUT:", text)
    print("PREDICTED INCIDENT:", label)
    print("CONFIDENCE:", round(result["score"],3))