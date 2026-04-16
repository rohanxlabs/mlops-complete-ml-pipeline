import pandas as pd

df = pd.read_csv("data/spam.csv")

df = df[["v1", "v2"]]
df.columns = ["label", "message"]

df.to_csv("data/raw.csv", index=False)
print("Data ingestion completed successfully.")