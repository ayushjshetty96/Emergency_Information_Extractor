import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing import clean_text

# Load dataset
df = pd.read_csv("data/processed/emergency_dataset_final.csv")

print("Original dataset shape:", df.shape)


# Keep only rows that contain text
df = df.dropna(subset=["text"])


# Clean text
df["clean_text"] = df["text"].apply(clean_text)


# Remove empty cleaned text
df = df[df["clean_text"].str.len() > 3]


print("Dataset after cleaning:", df.shape)


# If incident column exists use it as label
if "incident" in df.columns:
    df["label"] = df["incident"].fillna("unknown")
else:
    df["label"] = "unknown"


# Keep only required columns
df = df[["clean_text", "label"]]


# Train / Validation / Test split
train_df, temp_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=42
)


print("Train size:", train_df.shape)
print("Validation size:", val_df.shape)
print("Test size:", test_df.shape)


# Save datasets
train_df.to_csv("data/processed/train.csv", index=False)
val_df.to_csv("data/processed/validation.csv", index=False)
test_df.to_csv("data/processed/test.csv", index=False)


print("Training datasets saved successfully.")