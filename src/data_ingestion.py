"""
data_ingestion.py
-----------------
Reads the raw SMS spam CSV, selects the two relevant columns,
renames them, and writes a clean raw.csv.

Input:  data/spam.csv
Output: data/raw.csv
"""

import logging
import sys

import pandas as pd

from config import RAW_DATA_PATH, CLEAN_DATA_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def ingest() -> None:
    logger.info("Starting data ingestion")
    logger.info("Source: %s", RAW_DATA_PATH)

    if not RAW_DATA_PATH.exists():
        logger.error("Source file not found: %s", RAW_DATA_PATH)
        sys.exit(1)

    try:
        df = pd.read_csv(RAW_DATA_PATH, encoding="latin-1")
    except Exception as exc:
        logger.error("Failed to read source CSV: %s", exc)
        sys.exit(1)

    # The UCI SMS Spam dataset has columns v1, v2 plus 3 unnamed extras
    required_cols = {"v1", "v2"}
    if not required_cols.issubset(df.columns):
        logger.error(
            "Expected columns %s not found. Got: %s",
            required_cols,
            df.columns.tolist(),
        )
        sys.exit(1)

    df = df[["v1", "v2"]].copy()
    df.columns = ["label", "message"]

    # Drop rows where either column is null
    before = len(df)
    df.dropna(subset=["label", "message"], inplace=True)
    dropped = before - len(df)
    if dropped:
        logger.warning("Dropped %d rows with null label/message", dropped)

    CLEAN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_DATA_PATH, index=False)

    logger.info(
        "Ingestion complete — %d rows written to %s  (ham=%d, spam=%d)",
        len(df),
        CLEAN_DATA_PATH,
        (df["label"] == "ham").sum(),
        (df["label"] == "spam").sum(),
    )


if __name__ == "__main__":
    ingest()
