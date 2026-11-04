"""Modelling step 7: final one-time evaluation on the held-out TEST sides.
Choices fixed BEFORE looking at these numbers: realistic feature set R, default settings (no tuning),
logistic regression and gradient boosting, plus the clock-only reference T and the unrealistic F.
Run once; do not tune on these results."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_common import (
    CATEGORICAL_COLS,
    ECON_COLS,
    FEATURE_SETS,
    RANDOM_STATE,
    chronological_split,
    load_data,
    random_split,
)

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)

N_BOOT = 1000
rng = np.random.default_rng(42)  # fixed seed so the intervals are reproducible


def build_pipeline(model, feature_cols):
    """One-hot encode text columns, scale numeric columns, then fit the model."""
    categorical = [c for c in feature_cols if c in CATEGORICAL_COLS]
    numeric = [c for c in feature_cols if c not in CATEGORICAL_COLS]
    prep = ColumnTransformer([
        ("categories", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ("numbers", StandardScaler(), numeric),
    ])
    return Pipeline([("prep", prep), ("model", model)])


def capture(y_values, scores, fraction):
    """Share (%) of all subscribers found among the top-ranked `fraction` of clients."""
    y_values = np.asarray(y_values)
    order = np.argsort(-np.asarray(scores), kind="stable")
    k = int(np.ceil(fraction * len(order)))
    return 100 * y_values[order][:k].sum() / y_values.sum()


def bootstrap_ci(y_true, scores):
    """95% bootstrap intervals for ROC-AUC and PR-AUC, resampling the test clients."""
    n = len(y_true)
    roc_values, pr_values = [], []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        roc_values.append(roc_auc_score(y_true[idx], scores[idx]))
        pr_values.append(average_precision_score(y_true[idx], scores[idx]))
    return np.percentile(roc_values, [2.5, 97.5]), np.percentile(pr_values, [2.5, 97.5])


df = load_data()
R = FEATURE_SETS["R: pre-call"]
T = ["month"] + ECON_COLS
F = FEATURE_SETS["F: everything incl. duration"]

specs = {
    "LR_R": (lambda: LogisticRegression(max_iter=2000), R),
    "GB_R": (lambda: HistGradientBoostingClassifier(random_state=RANDOM_STATE), R),
    "GB_T": (lambda: HistGradientBoostingClassifier(random_state=RANDOM_STATE), T),
    "GB_F": (lambda: HistGradientBoostingClassifier(random_state=RANDOM_STATE), F),
}
splits = {"random (stratified)": random_split, "chronological": chronological_split}

rows = []
store = {}
for split_name, splitter in splits.items():
    X_train, X_test, y_train, y_test = splitter(df, F)  # F contains every column we need
    y_true = y_test.to_numpy()
    base = y_true.mean()
    for key, (make_model, cols) in specs.items():
        pipe = build_pipeline(make_model(), cols)
        pipe.fit(X_train[cols], y_train)
        scores = pipe.predict_proba(X_test[cols])[:, 1]
        store[(split_name, key)] = scores
        roc = roc_auc_score(y_true, scores)
        pr = average_precision_score(y_true, scores)
        (roc_lo, roc_hi), (pr_lo, pr_hi) = bootstrap_ci(y_true, scores)
        rows.append({
            "split": split_name,
            "model": key,
            "test_clients": len(y_true),
            "test_rate": base,
            "roc_auc": roc,
            "roc_low": roc_lo,
            "roc_high": roc_hi,
            "pr_auc": pr,
            "pr_low": pr_lo,
            "pr_high": pr_hi,
            "pr_lift": pr / base,
            "top20_reach_%": capture(y_true, scores, 0.20),
        })

print("=== FINAL TEST-SIDE RESULTS (each test side used once) ===")
print(pd.DataFrame(rows).round(3).to_string(index=False))
print()
print("LR = logistic regression, GB = gradient boosting")
print("R = 20 pre-call features (realistic); T = month and economic columns only (clock reference)")
print("F = R + campaign + duration (NOT realistic: duration is not known before a call)")
print("roc_low/high and pr_low/high = 95% bootstrap intervals over the test clients")
print("test_rate = share of subscribers in the test side; it is the PR-AUC of a useless model")
print("top20_reach_% = share of subscribers found by calling the top-ranked 20% (random selection: 20%)")
print()

# Inside each era of the RANDOM test side: do the client columns beat the clock-only reference?
period = pd.qcut(df["original_row"], 10, labels=False) + 1
era = pd.cut(period, bins=[0, 3, 6, 10], labels=["early", "middle", "late"])
X_train, X_test, y_train, y_test = random_split(df, F)
test_era = era.loc[X_test.index].to_numpy()
y_all = y_test.to_numpy()

rows2 = []
for era_name in ["early", "middle", "late"]:
    mask = test_era == era_name
    y_e = y_all[mask]
    row = {
        "era": era_name,
        "test_clients": int(mask.sum()),
        "subscribers": int(y_e.sum()),
        "base_rate_%": y_e.mean() * 100,
    }
    for key in ["LR_R", "GB_R", "GB_T"]:
        row[f"roc_{key}"] = roc_auc_score(y_e, store[("random (stratified)", key)][mask])
    rows2.append(row)

print("=== RANDOM TEST SIDE: ROC-AUC INSIDE EACH ERA ===")
print(pd.DataFrame(rows2).round(3).to_string(index=False))
print("Single run per era, no intervals; the early and middle eras have few subscribers, so they are noisy.")