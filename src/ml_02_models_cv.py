"""Modelling step 2: first models, scored by cross-validation on the TRAINING part only.
The test sides of both splits are not used here."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_common import (
    CATEGORICAL_COLS,
    FEATURE_SETS,
    RANDOM_STATE,
    chronological_split,
    classification_metrics,
    load_data,
    random_split,
)

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)

FEATURES = FEATURE_SETS["R: pre-call"]
THRESHOLD = 0.5  # predict "yes" when the model's probability is at least this


def build_pipeline(model, feature_cols):
    """One-hot encode text columns, scale numeric columns, then fit the model."""
    categorical = [c for c in feature_cols if c in CATEGORICAL_COLS]
    numeric = [c for c in feature_cols if c not in CATEGORICAL_COLS]
    prep = ColumnTransformer([
        ("categories", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ("numbers", StandardScaler(), numeric),
    ])
    return Pipeline([("prep", prep), ("model", model)])


def make_models():
    """Fresh, untrained models (a new set is needed for every fold)."""
    return {
        "Logistic regression": LogisticRegression(max_iter=2000),
        "Logistic regression (balanced)": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Gradient boosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
    }


splits = {
    "random (stratified)": (random_split, StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)),
    "chronological": (chronological_split, TimeSeriesSplit(n_splits=5)),
}

df = load_data()
records = []
for split_name, (splitter, cv) in splits.items():
    # Only the training side is used. The test side is deliberately ignored.
    X_train, _, y_train, _ = splitter(df, FEATURES)

    for fold, (tr, va) in enumerate(cv.split(X_train, y_train), start=1):
        X_tr, X_va = X_train.iloc[tr], X_train.iloc[va]
        y_tr, y_va = y_train.iloc[tr], y_train.iloc[va]

        for model_name, model in make_models().items():
            pipe = build_pipeline(model, FEATURES)
            pipe.fit(X_tr, y_tr)
            scores = pipe.predict_proba(X_va)[:, 1]
            pred = (scores >= THRESHOLD).astype(int)
            metrics = classification_metrics(y_va, pred, scores)
            records.append({
                "split": split_name, "model": model_name, "fold": fold,
                "train_rows": len(X_tr), "base_rate": y_va.mean(), **metrics,
            })

results = pd.DataFrame(records)

summary = results.groupby(["split", "model"], sort=False).agg(
    roc_auc=("roc_auc", "mean"),
    roc_auc_std=("roc_auc", "std"),
    pr_auc=("pr_auc", "mean"),
    pr_auc_std=("pr_auc", "std"),
    base_rate=("base_rate", "mean"),
    precision=("precision", "mean"),
    recall=("recall", "mean"),
    f1=("f1", "mean"),
)
print("=== CROSS-VALIDATION ON THE TRAINING PART (5 folds, realistic feature set R) ===")
print(summary.round(3).to_string())
print()
print(f"precision, recall and f1 use the threshold {THRESHOLD}.")
print("base_rate = share of subscribers in the validation folds; it is the PR-AUC of a useless model.")
print("_std = spread between folds.")
print()

# The chronological folds one by one, for the gradient boosting model
print("=== CHRONOLOGICAL FOLDS, GRADIENT BOOSTING ===")
one = results[(results["split"] == "chronological") & (results["model"] == "Gradient boosting")]
print(one[["fold", "train_rows", "base_rate", "roc_auc", "pr_auc", "precision", "recall"]].round(3).to_string(index=False))