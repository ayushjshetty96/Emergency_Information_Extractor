import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

# Load datasets
train_df = pd.read_csv("data/processed/train.csv")
val_df = pd.read_csv("data/processed/validation.csv")

# Convert to HuggingFace dataset
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")


def tokenize_function(example):
    return tokenizer(
        example["clean_text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )


train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Encode labels
labels = train_df["label"].unique().tolist()

label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}

import json
with open("models/classifier/label_mapping.json", "w") as f:
    json.dump(id2label, f)

train_dataset = train_dataset.map(lambda x: {"label": label2id[x["label"]]})
val_dataset = val_dataset.map(lambda x: {"label": label2id[x["label"]]})

# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    "xlm-roberta-base",
    num_labels=len(labels)
)

training_args = TrainingArguments(
    output_dir="models/classifier",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=2,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="logs"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()

trainer.save_model("models/classifier")