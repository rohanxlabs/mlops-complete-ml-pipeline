import pandas as pd 
import joblib
from sklearn.metrics import classification_report

import mlflow.sklearn

df = pd.read_csv("data/processed.csv")
X = df.drop("label", axis=1)
y = df["label"]

model = joblib.load("models/model.pkl")
preds = model.predict(X)
print(classification_report(y, preds))