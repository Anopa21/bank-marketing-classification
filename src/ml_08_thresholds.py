"""Modelling step 8: choosing a probability threshold.
Thresholds are chosen on the TRAINING side only (out-of-fold predictions) and then applied once to
the test side. The 'value of a subscriber' figures are ASSUMPTIONS for illustration, not data:
the dataset contains no costs or revenues."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
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

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)

R = FEATURE_SETS["R: pre-call"]
THRESHOLDS = [0.0] + [round(0.02 * k, 2) for k in range(1, 31)]  # 0.0 means "call everyone"
SHOW_THRESHOLDS = [0.05, 0.10, 0.20, 0.30, 0.50]
VALUES = [5, 10, 20]  # assumed value of one subscriber, in units of the cost of one call


def build_pipeline(model, feature_cols):
    """One-hot encode text columns, scale numeric columns, then fit the model."""
    categorical = [c for c in feature_cols if c in CATEGORICAL_COLS]
    numeric = [c for c in feature_cols if c not in CATEGORICAL_COLS]
    prep = ColumnTransformer([
        ("categories", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ("numbers", StandardScaler(), numeric),
    ])
    return Pipeline([("prep", prep), ("model", model)])


def outcomes(y_true, scores, threshold):
    """Who would be called at this threshold, and how many subscribers that reaches."""
    called = scores >= threshold
    tp = int((called & (y_true == 1)).sum())
    fp = int((called & (y_true == 0)).sum())
    return {
        "called_%": 100 * called.mean(),
        "precision": tp / (tp + fp) if (tp + fp) > 0 else float("nan"),
        "recall": tp / int(y_true.sum()),
        "tp": tp,
        "fp": fp,
    }


def net_per_1000(y_true, scores, threshold, value):
    """Net gain per 1,000 clients if each call costs 1 unit and each subscriber is worth `value` units."""
    o = outcomes(y_true, scores, threshold)
    return 1000 * (value * o["tp"] - (o["tp"] + o["fp"])) / len(y_true)


def f1_at(y_true, scores, threshold):
    """F1 score at a threshold (0 when nobody is called)."""
    o = outcomes(y_true, scores, threshold)
    p, r = o["precision"], o["recall"]
    if np.isnan(p) or (p + r) == 0:
        return 0.0
    return 2 * p * r / (p + r)


# Each rule: (assumed value used to report net gain, function to maximise on the TRAINING side)
rules = {
    "max F1": (10, lambda y, s, t: f1_at(y, s, t)),
    "max net, V=5": (5, lambda y, s, t: net_per_1000(y, s, t, 5)),
    "max net, V=10": (10, lambda y, s, t: net_per_1000(y, s, t, 10)),
    "max net, V=20": (20, lambda y, s, t: net_per_1000(y, s, t, 20)),
}

df = load_data()
splits = {"random (stratified)": random_split, "chronological": chronological_split}

for split_name, splitter in splits.items():
    X_train, X_test, y_train, y_test = splitter(df, R)
    y_tr, y_te = y_train.to_numpy(), y_test.to_numpy()

    pipe = build_pipeline(LogisticRegression(max_iter=2000), R)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof = cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    pipe.fit(X_train, y_train)
    test_scores = pipe.predict_proba(X_test)[:, 1]

    print("=" * 78)
    print(f"{split_name.upper()}  (logistic regression, realistic features)")
    print("=" * 78)
    print(f"Training side: actual subscription rate {y_tr.mean():.3f}, mean predicted probability {oof.mean():.3f} (out-of-fold)")
    print(f"Test side:     actual subscription rate {y_te.mean():.3f}, mean predicted probability {test_scores.mean():.3f}")
    print()

    rows = []
    for t in SHOW_THRESHOLDS:
        o = outcomes(y_tr, oof, t)
        rows.append({
            "threshold": t,
            "called_%": o["called_%"],
            "precision": o["precision"],
            "recall": o["recall"],
            "f1": f1_at(y_tr, oof, t),
        })
    print("--- Training side (out-of-fold): precision and recall at fixed thresholds ---")
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print()

    rows = []
    for rule_name, (value, objective) in rules.items():
        best = max(THRESHOLDS, key=lambda t: objective(y_tr, oof, t))
        tr = outcomes(y_tr, oof, best)
        te = outcomes(y_te, test_scores, best)
        rows.append({
            "rule": rule_name,
            "threshold": best,
            "train_called_%": tr["called_%"],
            "train_precision": tr["precision"],
            "train_recall": tr["recall"],
            "test_called_%": te["called_%"],
            "test_precision": te["precision"],
            "test_recall": te["recall"],
            "test_net_per_1000": net_per_1000(y_te, test_scores, best, value),
        })
    print("--- Thresholds chosen on the training side, then applied once to the test side ---")
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print("(test_net_per_1000 uses V = 10 for 'max F1' and the rule's own V otherwise.)")
    print()

    rate = y_te.mean()
    reference = pd.DataFrame({
        "value_of_a_subscriber_V": VALUES,
        "call_everyone_net_per_1000": [1000 * (v * rate - 1) for v in VALUES],
    })
    print("--- Reference on the test side: calling EVERYONE (calling no one gives 0) ---")
    print(reference.round(0).to_string(index=False))
    print(f"Break-even: calling everyone only pays if a subscriber is worth more than {1 / rate:.1f} calls.")
    print()

print("Each call costs 1 unit; V is the ASSUMED value of one subscriber in those units. Neither is in the data.")