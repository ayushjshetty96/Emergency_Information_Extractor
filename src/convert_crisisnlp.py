import pandas as pd
import glob

files = glob.glob("data/raw/crisisnlp/*.tsv")

rows = []

for file in files:

    df = pd.read_csv(file, sep="\t")

    # detect correct column
    if "tweet_text" in df.columns:
        text_column = "tweet_text"

    elif "text" in df.columns:
        text_column = "text"

    else:
        print("Unknown format:", file)
        continue

    for text in df[text_column]:

        rows.append({
            "text": text,
            "incident": "disaster",
            "location": None,
            "severity": None
        })

combined = pd.DataFrame(rows)

combined.to_csv(
    "data/raw/crisisnlp_processed.csv",
    index=False
)

print("CrisisNLP dataset processed")