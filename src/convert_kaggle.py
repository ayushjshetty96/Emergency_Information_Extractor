import pandas as pd

df = pd.read_csv(r"C:\Users\Ayush\OneDrive\Desktop\Emergency_Information_Extractor\data\raw\kaggle_disaster_tweets.csv")

df = df[["text","location","target"]]

df["incident"] = df["target"].apply(
    lambda x: "disaster" if x == 1 else "non_emergency"
)

df["severity"] = None
df["source"] = "kaggle"

df = df[["text","incident","location","severity","source"]]

df.to_csv(r"C:\Users\Ayush\OneDrive\Desktop\Emergency_Information_Extractor\data\raw\kaggle_processed.csv",index=False)

print("Kaggle dataset processed") 