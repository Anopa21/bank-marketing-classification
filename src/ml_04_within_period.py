"""Modelling step 4: ranking quality WITHIN each period.
Out-of-fold predictions on the TRAINING side of the random split; the test sides are not used."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_common import (
    CATEGORICAL_COLS,
    ECON_COLS,
    FEATURE_SETS,
    RANDOM_STATE,
    TARGET,
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


df = load_data()
df["period"] = pd.qcut(df["original_row"], 10, labels=False) + 1  # 1 = earliest tenth

R = FEATURE_SETS["R: pre-call"]
E = FEATURE_SETS["E: pre-call, no economic columns"]
N = [c for c in E if c != "month"]
T = ["month"] + ECON_COLS  # time markers only: knows WHEN a call happened, nothing about the client
feature_sets = {"R": R, "E": E, "N": N, "T": T}

models = {
    "LR": lambda: LogisticRegression(max_iter=2000),
    "GB": lambda: HistGradientBoostingClassifier(random_state=RANDOM_STATE),
}

# Training side of the random split only; the test side is never used here
X_train, _, _, _ = random_split(df, R)
train = df.loc[X_train.index].copy()
y = train[TARGET]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

summary_rows = []
gb_roc_by_period = {}
for set_name, cols in feature_sets.items():
    for model_name, make_model in models.items():
        pipe = build_pipeline(make_model(), cols)
        # Every client is scored by a model that did not see that client
        oof = pd.Series(
            cross_val_predict(pipe, train[cols], y, cv=cv, method="predict_proba")[:, 1],
            index=train.index,
        )
        within_roc, within_lift = [], []
        for period in range(1, 11):
            mask = train["period"] == period
            roc = roc_auc_score(y[mask], oof[mask])
            within_roc.append(roc)
            within_lift.append(average_precision_score(y[mask], oof[mask]) / y[mask].mean())
            if model_name == "GB":
                gb_roc_by_period[(set_name, period)] = roc
        summary_rows.append({
            "features": set_name,
            "model": model_name,
            "pooled_roc": roc_auc_score(y, oof),
            "mean_within_roc": float(np.mean(within_roc)),
            "pooled_pr_lift": average_precision_score(y, oof) / y.mean(),
            "mean_within_pr_lift": float(np.mean(within_lift)),
        })

print("=== POOLED vs WITHIN-PERIOD RANKING (5-fold out-of-fold predictions) ===")
print(pd.DataFrame(summary_rows).round(3).to_string(index=False))
print()
print("pooled_* = scored over all training clients together")
print("mean_within_* = average over the ten periods, each scored on its own")
print("R = all 20 pre-call features; E = no economic columns; N = no economic columns, no month")
print("T = month and economic columns only (knows when the call happened, nothing about the client)")
print()

table = train.groupby("period").agg(
    clients=(TARGET, "count"),
    subscribers=(TARGET, "sum"),
    base_rate_pct=(TARGET, lambda s: s.mean() * 100),
)
for set_name in feature_sets:
    table[f"roc_{set_name}"] = [gb_roc_by_period[(set_name, p)] for p in table.index]
print("=== PERIOD BY PERIOD, GRADIENT BOOSTING (ROC-AUC within the period) ===")
print(table.round(3))
print()
print("Periods 1-3 have few subscribers (about 100 each), so their scores are noisy.")