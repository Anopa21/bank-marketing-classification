"""Shared helpers for the modelling scripts: loading, feature sets, splits and metrics."""

from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "bank-additional-clean.csv"
IMAGES_DIR = PROJECT_ROOT / "images"

RANDOM_STATE = 42
TARGET = "subscribed"

# Text columns (they need encoding before most models can use them)
CATEGORICAL_COLS = [
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "poutcome",
]

# Groups of columns known BEFORE a call is made
CLIENT_COLS = ["age", "job", "marital", "education", "default", "housing", "loan"]
CONTACT_COLS = ["contact", "month", "day_of_week"]
HISTORY_COLS = ["previous", "poutcome", "pdays_days", "pdays_recorded", "previously_contacted"]
ECON_COLS = ["emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]

PRE_CALL = CLIENT_COLS + CONTACT_COLS + HISTORY_COLS + ECON_COLS

# Never used as features: y and subscribed (the target), original_row (a clock),
# and the raw pdays (replaced by pdays_days plus the two flags).
FEATURE_SETS = {
    "R: pre-call": PRE_CALL,
    "E: pre-call, no economic columns": CLIENT_COLS + CONTACT_COLS + HISTORY_COLS,
    "D: pre-call + campaign": PRE_CALL + ["campaign"],
    "F: everything incl. duration": PRE_CALL + ["campaign", "duration"],
}


def load_data():
    """Load the cleaned data in date order and add a numeric pdays column."""
    df = pd.read_csv(DATA_PATH).sort_values("original_row").reset_index(drop=True)
    # pdays uses 999 as a code. pdays_days holds real days only (0 when not recorded);
    # the pdays_recorded flag tells the model which rows are real.
    df["pdays_days"] = df["pdays"].where(df["pdays"] != 999, 0)
    return df


def random_split(df, feature_cols, test_size=0.2):
    """Stratified random split: both sides keep the same share of subscribers."""
    return train_test_split(
        df[feature_cols], df[TARGET],
        test_size=test_size, random_state=RANDOM_STATE, stratify=df[TARGET],
    )


def chronological_split(df, feature_cols, test_size=0.2):
    """Earlier calls for training, the latest calls for testing."""
    ordered = df.sort_values("original_row").reset_index(drop=True)
    cut = int(len(ordered) * (1 - test_size))
    train, test = ordered.iloc[:cut], ordered.iloc[cut:]
    return train[feature_cols], test[feature_cols], train[TARGET], test[TARGET]


def classification_metrics(y_true, y_pred, scores):
    """Threshold-based metrics (from y_pred) and threshold-free metrics (from scores)."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, scores),
        "pr_auc": average_precision_score(y_true, scores),
    }