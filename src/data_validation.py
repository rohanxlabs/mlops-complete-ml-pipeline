"""
data_validation.py
------------------
Validates raw.csv before it enters the preprocessing stage.

Checks:
  - Required columns present
  - No null values in label / message
  - Label values are exactly {"ham", "spam"}
  - Message column is string type
  - Minimum row count
  - Class balance (warns if extreme imbalance)

Input:  data/raw.csv
Output: raises SystemExit on hard failures; warnings for soft issues
"""

import logging
import sys

import pandas as pd

from config import CLEAN_DATA_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

REQUIRED_COLUMNS  = {"label", "message"}
VALID_LABELS      = {"ham", "spam"}
MIN_ROWS          = 100
IMBALANCE_WARNING = 0.10   # warn if minority class < 10 %


def validate() -> None:
    logger.info("Starting data validation")
    logger.info("Target file: %s", CLEAN_DATA_PATH)

    if not CLEAN_DATA_PATH.exists():
        logger.error("raw.csv not found — run data_ingestion.py first")
        sys.exit(1)

    try:
        df = pd.read_csv(CLEAN_DATA_PATH)
    except Exception as exc:
        logger.error("Failed to read raw.csv: %s", exc)
        sys.exit(1)

    errors: list[str] = []

    # 1. Schema check
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")

    if errors:
        for e in errors:
            logger.error("VALIDATION FAILED — %s", e)
        sys.exit(1)

    # 2. Null check
    null_counts = df[list(REQUIRED_COLUMNS)].isnull().sum()
    for col, count in null_counts.items():
        if count > 0:
            errors.append(f"Column '{col}' has {count} null values")

    # 3. Label values
    unexpected_labels = set(df["label"].unique()) - VALID_LABELS
    if unexpected_labels:
        errors.append(f"Unexpected label values: {unexpected_labels}")

    # 4. Row count
    if len(df) < MIN_ROWS:
        errors.append(f"Too few rows: {len(df)} (minimum {MIN_ROWS})")

    # 5. Message dtype — should be string
    non_string = df["message"].apply(lambda x: not isinstance(x, str)).sum()
    if non_string:
        errors.append(f"{non_string} messages are not strings")

    if errors:
        for e in errors:
            logger.error("VALIDATION FAILED — %s", e)
        sys.exit(1)

    # 6. Class balance (soft warning only)
    counts = df["label"].value_counts(normalize=True)
    for label, pct in counts.items():
        if pct < IMBALANCE_WARNING:
            logger.warning(
                "Class '%s' represents only %.1f%% of data (imbalanced dataset)",
                label,
                pct * 100,
            )

    logger.info(
        "Validation passed — %d rows | ham=%.1f%%  spam=%.1f%%",
        len(df),
        counts.get("ham", 0) * 100,
        counts.get("spam", 0) * 100,
    )


if __name__ == "__main__":
    validate()
