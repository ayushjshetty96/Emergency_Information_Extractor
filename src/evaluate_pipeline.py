import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

from .classification import classify_emergency, get_classifier_info

# Load dataset - using the newly generated emergency classification dataset
df = pd.read_csv("data/raw/emergency_classification_dataset.csv")

classifier_info = get_classifier_info()
print("Using classifier model:", classifier_info["model_type"])
print("Model path:", classifier_info["model_path"])
print("Label mapping:", classifier_info["label_mapping"])

true_labels = []
pred_labels = []

print("\nRunning Evaluation on Emergency Classification Dataset...\n")

for i, row in df.iterrows():

    text = row["text"]
    true_label = row["label"]

    result = classify_emergency(text)
    predicted_label = result["label"]

    print(f"{text} → {predicted_label} (true: {true_label})")

    # Error analysis
    if predicted_label != true_label:
        print("❌ ERROR")

    true_labels.append(true_label)
    pred_labels.append(predicted_label)


# =============================
# METRICS
# =============================

print("\n==============================")
print("EVALUATION RESULTS")
print("==============================")

accuracy = accuracy_score(true_labels, pred_labels)
print("Accuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(true_labels, pred_labels))

# =============================
# CONFUSION MATRIX (LABELED)
# =============================

labels = sorted(list(set(true_labels)))

cm = confusion_matrix(true_labels, pred_labels, labels=labels)

fig, ax = plt.subplots()

im = ax.imshow(cm)

# Labels
ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))

ax.set_xticklabels(labels, rotation=45)
ax.set_yticklabels(labels)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")

# Numbers inside cells
for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j, i, cm[i, j], ha="center", va="center")

plt.colorbar(im)
plt.tight_layout()
plt.show()

# =============================
# SUMMARY (IMPORTANT FOR REPORT)
# =============================

print("\n==============================")
print("SUMMARY")
print("==============================")

if accuracy == 1.0:
    print("The model achieved perfect accuracy on the evaluation dataset.")
    print("This indicates strong pattern learning but may reflect similarity between training and test data.")
else:
    print("The model shows good performance with some misclassifications.")