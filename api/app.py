from fastapi import FastAPI
import joblib

app = FastAPI()

model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")


@app.post("/predict/")
def predict(text: str):
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]
    return {"spam": bool(prediction)}