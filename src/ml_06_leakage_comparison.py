"""Modelling step 6: how much do 'campaign' and 'duration' inflate the scores?
Cross-validation on the TRAINING side of each split only; the test sides are not used."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_common import (
    CATEGORICAL_COLS,
    FEATURE_SETS,
    RANDOM_STATE,
    chronological_split,
    load_data,
    random_split,
)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)


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
        "LR": LogisticRegression(max_iter=2000),
        "GB": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
    }


df = load_data()

sets = {
    "R": FEATURE_SETS["R: pre-call"],
    "D": FEATURE_SETS["D: pre-call + campaign"],
    "F": FEATURE_SETS["F: everything incl. duration"],
}
all_cols = sets["F"]  # the largest set contains every column we need

splits = {
    "random (stratified)": (random_split, StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)),
    "chronological": (chronological_split, TimeSeriesSplit(n_splits=5)),
}

records = []
for split_name, (splitter, cv) in splits.items():
    # Only the training side is used. The test side is deliberately ignored.
    X_train, _, y_train, _ = splitter(df, all_cols)

    for fold, (tr, va) in enumerate(cv.split(X_train, y_train), start=1):
        y_va = y_train.iloc[va]
        for set_name, cols in sets.items():
            for model_name, model in make_models().items():
                pipe = build_pipeline(model, cols)
                pipe.fit(X_train.iloc[tr][cols], y_train.iloc[tr])
                scores = pipe.predict_proba(X_train.iloc[va][cols])[:, 1]
                records.append({
                    "split": split_name, "features": set_name, "model": model_name,
                    "fold": fold, "base_rate": y_va.mean(),
                    "roc_auc": roc_auc_score(y_va, scores),
                    "pr_auc": average_precision_score(y_va, scores),
                })

results = pd.DataFrame(records)
summary = results.groupby(["split", "features", "model"], sort=False).agg(
    base_rate=("base_rate", "mean"),
    roc_auc=("roc_auc", "mean"),
    roc_sd=("roc_auc", "std"),
    pr_auc=("pr_auc", "mean"),
    pr_sd=("pr_auc", "std"),
).reset_index()
summary["pr_lift"] = summary["pr_auc"] / summary["base_rate"]

print("=== CROSS-VALIDATION ON THE TRAINING PART: THE EFFECT OF campaign AND duration ===")
print(summary.round(3).to_string(index=False))
print()
print("R = 20 pre-call features; D = R + campaign; F = R + campaign + duration")
print("LR = logistic regression; GB = gradient boosting")
print("pr_lift = pr_auc / base_rate (1.0 means no better than a useless model); _sd = spread between folds")
print("Only R is a realistic feature set: campaign and duration are not known before a call.")