# MLOps Complete ML Pipeline

Reproducible end-to-end ML pipeline for SMS spam detection — from raw data to a deployed REST API.

---

## Overview

This project walks through the full ML engineering lifecycle: data ingestion, validation, preprocessing, model training, evaluation, and API serving — all wired together with DVC for pipeline reproducibility and MLflow for experiment tracking. The trained model is served via FastAPI and deployed on Render.

---

## Architecture

```
┌─────────────────┐
│   data/spam.csv │  UCI SMS Spam dataset (5 572 messages)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Data Ingestion  │  select columns, rename, write raw.csv
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Data Validation │  schema, nulls, label values, class balance
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Preprocessing  │  TF-IDF vectorisation → processed.csv + vectorizer.pkl
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Training     │  MultinomialNB → model.pkl  (logged to MLflow)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Evaluation    │  held-out test set → metrics.json
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Model Artifact │  models/model.pkl  +  models/vectorizer.pkl
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    FastAPI      │  POST /predict  ·  GET /health
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Render (live)  │  https://mlops-complete-ml-pipeline.onrender.com
└─────────────────┘
```

---

## Tech Stack

| Concern              | Tool                                       |
|----------------------|--------------------------------------------|
| Language             | Python 3.11                                |
| ML                   | scikit-learn (TF-IDF + MultinomialNB)      |
| Data handling        | pandas, numpy                              |
| Experiment tracking  | MLflow (SQLite backend — `mlflow.db`)      |
| Pipeline versioning  | DVC                                        |
| API framework        | FastAPI + Uvicorn                          |
| Model persistence    | joblib (`.pkl`)                            |
| Testing              | pytest + httpx                             |
| Linting              | ruff                                       |
| CI                   | GitHub Actions                             |
| Deployment           | Render                                     |
| Container            | Docker                                     |

---

## Project Structure

```
mlops-complete-ml-pipeline/
├── .github/
│   └── workflows/
│       └── ci.yml              # CI: install → lint → pipeline → tests
├── api/
│   └── app.py                  # FastAPI app — /health + /predict
├── data/
│   ├── spam.csv                # Raw UCI SMS Spam dataset
│   ├── raw.csv                 # Cleaned dataset (label, message)
│   └── processed.csv           # TF-IDF feature matrix + label
├── models/
│   ├── vectorizer.pkl          # Fitted TfidfVectorizer
│   └── model.pkl               # Trained MultinomialNB
├── src/
│   ├── config.py               # Central path + hyperparameter config
│   ├── data_ingestion.py       # spam.csv → raw.csv
│   ├── data_validation.py      # schema + quality checks on raw.csv
│   ├── data_preprocessing.py   # raw.csv → processed.csv + vectorizer.pkl
│   ├── train.py                # processed.csv → model.pkl (MLflow logging)
│   └── evaluate.py             # held-out test evaluation → metrics.json
├── tests/
│   ├── test_pipeline.py        # ingestion / preprocessing / model tests
│   └── test_api.py             # health / predict / validation tests
├── mlruns/                     # MLflow run artifacts
├── mlflow.db                   # MLflow tracking backend (SQLite)
├── metrics.json                # Latest evaluation metrics (DVC tracked)
├── dvc.yaml                    # DVC pipeline definition
├── Dockerfile                  # API container
├── requirements.txt            # Full dev dependencies
├── requirements-prod.txt       # Minimal API-only dependencies
├── .env.example                # Environment variable template
└── README.md
```

---

## Dataset

**UCI SMS Spam Collection** — 5 572 English SMS messages labelled `ham` or `spam`.

| Class | Count | Share  |
|-------|-------|--------|
| ham   | 4 825 | 86.6 % |
| spam  |   747 | 13.4 % |

---

## Pipeline

### 1. Data Ingestion — `src/data_ingestion.py`
Reads `data/spam.csv`, selects the `v1`/`v2` columns (label, message), drops nulls, and writes `data/raw.csv`.

### 2. Data Validation — `src/data_validation.py`
Validates `raw.csv` before preprocessing:
- required columns present
- no null values in label/message
- label values exactly `{ham, spam}`
- minimum row count
- warns on class imbalance

### 3. Preprocessing — `src/data_preprocessing.py`
- Label-encodes target (`ham=0`, `spam=1`)
- Fits a `TfidfVectorizer` (English stop words, top 3 000 features)
- Writes `data/processed.csv` with named feature columns (eliminates sklearn feature-name warnings)
- Saves `models/vectorizer.pkl`

### 4. Training — `src/train.py`
- Splits data 80/20 with `random_state=42`
- Fits `MultinomialNB`
- Logs params + accuracy to MLflow (`mlflow.db`)
- Saves `models/model.pkl`

### 5. Evaluation — `src/evaluate.py`
- Reproduces the same 80/20 split (same `random_state`)
- Evaluates **only** on the held-out 20 % test set
- Prints full classification report
- Writes `metrics.json` for DVC metrics tracking

### 6. API — `api/app.py`
FastAPI application; loads both artifacts at startup and exposes two endpoints.

---

## Model Results

Evaluated on a held-out 20 % test set (1 115 messages, never seen during training).

| Metric           | Value  |
|------------------|--------|
| Accuracy         | 0.9794 |
| F1 (weighted)    | 0.9786 |
| Precision (spam) | 1.00   |
| Recall (spam)    | 0.85   |
| F1 (spam)        | 0.92   |

Full classification report:

```
              precision    recall  f1-score   support

         ham       0.98      1.00      0.99       965
        spam       1.00      0.85      0.92       150

    accuracy                           0.98      1115
   macro avg       0.99      0.92      0.95      1115
weighted avg       0.98      0.98      0.98      1115
```

---

## API

### Endpoints

| Method | Path       | Description                         |
|--------|------------|-------------------------------------|
| GET    | `/health`  | Liveness check                      |
| POST   | `/predict` | Classify a message as spam or ham   |

### Request

```json
POST /predict
Content-Type: application/json

{
  "text": "Congratulations! You've won a FREE prize. Call now!"
}
```

### Response

```json
{
  "spam": true,
  "label": "spam"
}
```

### Validation

- `text` is required
- `text` must not be blank or whitespace-only
- `text` max length: 10 000 characters
- Returns `HTTP 422` on invalid input

### Interactive docs

When running locally: http://127.0.0.1:8000/docs

---

## Local Setup

```bash
# 1. Clone
git clone https://github.com/rohanxlabs/mlops-complete-ml-pipeline.git
cd mlops-complete-ml-pipeline

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Running the Pipeline

### Option A — run each stage manually

```bash
python src/data_ingestion.py
python src/data_validation.py
python src/data_preprocessing.py
python src/train.py
python src/evaluate.py
```

### Option B — reproduce with DVC

```bash
# Initialise DVC (first time only)
dvc init

dvc repro
```

---

## Starting the API

```bash
uvicorn api.app:app --reload
```

The API will be available at http://127.0.0.1:8000.

### Example curl request

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Congratulations! You won a free prize. Call now!"}'
```

---

## Experiment Tracking (MLflow)

Training runs are logged to a local SQLite database (`mlflow.db`). Launch the UI with:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open http://127.0.0.1:5000.

---

## Docker

Build and run the API in a container:

```bash
# Build
docker build -t spam-api .

# Run
docker run -p 8000:8000 spam-api
```

The container uses `requirements-prod.txt` (minimal runtime dependencies).

---

## CI/CD

GitHub Actions runs on every push and pull request to `main`:

1. Install dependencies
2. Lint with `ruff`
3. Run the full pipeline (ingestion → validation → preprocessing → training → evaluation)
4. Run the test suite (`pytest`)

See `.github/workflows/ci.yml`.

---

## Deployment

The API is deployed on **Render** and available at:

**https://mlops-complete-ml-pipeline.onrender.com**

```bash
# Test the live deployment
curl -X POST "https://mlops-complete-ml-pipeline.onrender.com/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Free prize winner! Call now to claim."}'
```

Render runs the API with:

```bash
gunicorn api.app:app -k uvicorn.workers.UvicornWorker
```

> Render free-tier instances spin down after inactivity. The first request may take ~30 s to wake the service.

---

## Running Tests

```bash
pytest tests/ -v
```

26 tests covering:
- pipeline stages (ingestion, preprocessing, model accuracy)
- API health, valid predictions, and input validation edge cases

---

## Limitations

- The model is a simple bag-of-words + Naive Bayes classifier. It works well on this dataset but will not generalise to adversarial or multilingual spam.
- No data versioning remote is configured (DVC is set up locally only).
- MLflow tracking is local (SQLite). A remote tracking server is not configured.
- The Render free tier spins down on inactivity — expect a cold-start delay.

---

## Future Improvements

- Add a DVC remote (S3/GCS) for proper data versioning
- Configure a remote MLflow tracking server
- Add model comparison across MLflow runs before promoting a new model
- Experiment with transformer-based text classifiers for better recall on spam

---

## License

See [LICENSE](LICENSE).
