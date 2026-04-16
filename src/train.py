import pandas as pd 
import joblib
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

df = pd.read_csv("data/processed.csv")

X = df.drop("label", axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.start_run()

model = MultinomialNB()
model.fit(X_train, y_train)

preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

mlflow.log_metric("accuracy", acc)

joblib.dump(model, "models/model.pkl")
mlflow.sklearn.log_model(model, "model")

mlflow.end_run()

print(f"Training complete | Model accuracy: {acc}")