# mlops-complete-ml-pipeline

An end-to-end MLOps pipeline for **SMS spam detection**. It takes a raw SMS
dataset, transforms it into TF-IDF features, trains a Naive Bayes classifier,
tracks experiments with MLflow, versions the pipeline with DVC, and serves
predictions through a FastAPI endpoint.

## Overview

The project demonstrates a complete, reproducible ML workflow:

- **Data ingestion** – load and clean the raw SMS dataset.
- **Preprocessing** – label-encode targets and vectorize text with TF-IDF.
- **Training** – fit a `MultinomialNB` model and log metrics to MLflow.
- **Evaluation** – produce a classification report on the processed data.
- **Serving** – expose a `/predict/` REST endpoint via FastAPI.
- **Orchestration & tracking** – DVC for pipeline stages, MLflow for experiments.

## Tech stack

| Concern            | Tool                                   |
| ------------------ | -------------------------------------- |
| Language           | Python                                 |
| ML                 | scikit-learn (TF-IDF + MultinomialNB)  |
| Data handling      | pandas, numpy                          |
| Experiment tracking| MLflow (backed by `mlflow.db`)         |
| Pipeline versioning| DVC                                    |
| Serving            | FastAPI + Uvicorn                      |
| Persistence        | joblib (`.pkl` artifacts)              |

## Project structure

```
mlops-complete-ml-pipeline/
├── api/
│   └── app.py                  # FastAPI app exposing /predict/
├── data/
│   ├── spam.csv                # Raw source dataset
│   ├── raw.csv                 # Cleaned dataset (label, message)
│   └── processed.csv           # TF-IDF feature matrix + label
├── models/
│   ├── vectorizer.pkl          # Fitted TfidfVectorizer
│   └── model.pkl               # Trained MultinomialNB model
├── src/
│   ├── data_ingestion.py       # spam.csv -> raw.csv
│   ├── data_preprocessing.py   # raw.csv -> processed.csv + vectorizer.pkl
│   ├── train.py                # processed.csv -> model.pkl (+ MLflow logging)
│   └── evaluate.py             # classification report
├── mlruns/                     # MLflow run artifacts
├── mlflow.db                   # MLflow tracking backend (SQLite)
├── dvc.yaml                    # DVC pipeline definition
├── requirements.txt
├── LICENSE
└── README.md
```

## Getting started

### 1. Prerequisites

- Python 3.10+
- `git` and (optionally) `dvc`

### 2. Set up the environment

```bash
# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows (PowerShell/CMD)
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

## Running the pipeline

### Option A: run each stage manually

```bash
python src/data_ingestion.py       # -> data/raw.csv
python src/data_preprocessing.py   # -> data/processed.csv, models/vectorizer.pkl
python src/train.py                # -> models/model.pkl (logs to MLflow)
python src/evaluate.py             # prints classification report
```

### Option B: reproduce with DVC

```bash
dvc repro
```

> Note: the stage definitions in `dvc.yaml` may need to be aligned with the
> actual script paths in `src/` (e.g. `train.py`) and data locations before
> `dvc repro` runs cleanly.

## Experiment tracking with MLflow

Training runs are logged (accuracy metric and the serialized model). Launch the
MLflow UI against the local SQLite backend:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open http://127.0.0.1:5000 in your browser.

## Serving predictions

Start the API with Uvicorn:

```bash
uvicorn api.app:app --reload
```

The service loads `models/model.pkl` and `models/vectorizer.pkl` and exposes a
prediction endpoint. Interactive docs are available at
http://127.0.0.1:8000/docs.

### Example request

```bash
curl -X POST "http://127.0.0.1:8000/predict/?text=Congratulations!%20You%20won%20a%20free%20prize"
```

### Example response

```json
{ "spam": true }
```

A non-spam message returns `{ "spam": false }`.

## Deployment

The API is production-ready and can be served with Gunicorn using Uvicorn
workers (the same command used for the Render deployment):

```bash
gunicorn api.app:app -k uvicorn.workers.UvicornWorker
```

A live instance is available at
https://mlops-complete-ml-pipeline.onrender.com (predictions via
`POST /predict/`; the root path `/` intentionally returns 404).

> Tip: retrain the model with the same scikit-learn version used in production
> to avoid `InconsistentVersionWarning` when the pickled artifacts are loaded.

## How it works

1. **`data_ingestion.py`** reads the raw SMS CSV, keeps the label/message
   columns, and writes a clean `raw.csv`.
2. **`data_preprocessing.py`** label-encodes the target, fits a
   `TfidfVectorizer` (English stop words, top 3000 features), and saves both the
   feature matrix (`processed.csv`) and the fitted vectorizer.
3. **`train.py`** splits the data, trains a `MultinomialNB` classifier, logs the
   accuracy to MLflow, and persists the model. On the bundled dataset it reaches
   roughly **98% accuracy**.
4. **`evaluate.py`** loads the trained model and prints a full classification
   report.
5. **`api/app.py`** vectorizes incoming text with the saved vectorizer and
   returns whether the message is spam.

## License

See the [LICENSE](LICENSE) file for details.
