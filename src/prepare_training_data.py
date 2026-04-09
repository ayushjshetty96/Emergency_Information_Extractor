import pandas as pd
from sklearn.model_selection import train_test_split
from src.preprocessing import clean_text

# Load dataset
df = pd.read_csv("data/processed/emergency_dataset_final.csv")
print(df["incident"].value_counts())
print("Original dataset shape:", df.shape)
print("Final dataset size before split:", len(df))
# Remove rows without text
df = df.dropna(subset=["text"])

# Clean text
df["clean_text"] = df["text"].apply(clean_text)

# Remove very short texts
df = df[df["clean_text"].str.len() > 3]

print("Dataset after text cleaning:", df.shape)

# Ensure incident column exists
if "incident" not in df.columns:
    raise ValueError("Dataset must contain an 'incident' column.")

# Remove rows with missing incident values
df = df.dropna(subset=["incident"])

# Remove unknown class
df = df[df["incident"] != "unknown"]

# Create label column
df["label"] = df["incident"].astype(str)

# Keep only required columns
df = df[["clean_text", "label"]]

print("Dataset after label cleaning:", df.shape)

# Split dataset
train_df, temp_df = train_test_split(
    df,
    test_size=0.3,
    random_state=42,
    stratify=df["label"]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=42,
    stratify=temp_df["label"]
)

print("Train size:", train_df.shape)
print("Validation size:", val_df.shape)
print("Test size:", test_df.shape)

# Save datasets
train_df.to_csv("data/processed/train.csv", index=False)
val_df.to_csv("data/processed/validation.csv", index=False)
test_df.to_csv("data/processed/test.csv", index=False)

print("Training datasets saved successfully.")