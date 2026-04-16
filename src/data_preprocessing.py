import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("data/raw.csv")

le = LabelEncoder()
df["label"] = le.fit_transform(df["label"])

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=3000
)

X = vectorizer.fit_transform(df["message"])
y = df["label"]

joblib.dump(vectorizer, "models/vectorizer.pkl")

processed_df = pd.DataFrame(X.toarray())
processed_df["label"] = y.values
processed_df.to_csv("data/processed.csv", index=False)
print("Data preprocessing completed successfully.")