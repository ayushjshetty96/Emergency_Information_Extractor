from datasets import load_dataset
import pandas as pd

dataset = load_dataset("wikiann","en")

rows = []

for item in dataset["train"]:
    
    sentence = " ".join(item["tokens"])
    
    rows.append({
        "text": sentence
    })

df = pd.DataFrame(rows)

df.to_csv(r"C:\Users\Ayush\OneDrive\Desktop\Emergency_Information_Extractor\data\ner\wikiann.csv",index=False)

print("Export complete")