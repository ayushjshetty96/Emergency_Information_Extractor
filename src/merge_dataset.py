import pandas as pd

# Load datasets
synthetic = pd.read_csv(r"data/raw/generated_emergency_dataset.csv")

kaggle = pd.read_csv(r"data/raw/kaggle_disaster_tweets.csv")

crisis = pd.read_csv(r"data/raw/crisisnlp.csv")


# Combine them
combined = pd.concat([
    synthetic,
    kaggle,
    crisis
])

# Shuffle dataset
combined = combined.sample(frac=1, random_state=42)

# Save final dataset
combined.to_csv(
    "data/processed/emergency_dataset_final.csv",
    index=False
)

print("Final dataset created successfully.")