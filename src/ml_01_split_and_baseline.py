"""Modelling step 1: define the task, build two kinds of split, and score an 'always no' baseline."""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from ml_common import (
    FEATURE_SETS,
    chronological_split,
    classification_metrics,
    load_data,
    random_split,
)

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)

df = load_data()

# 1. Leakage guard: these columns must never be features in the realistic set
forbidden = {"duration", "campaign", "y", "subscribed", "original_row", "pdays"}
realistic = FEATURE_SETS["R: pre-call"]
assert not forbidden & set(realistic), "a forbidden column is in the realistic feature set"
print("Leakage guard passed: no forbidden column in the realistic feature set.")
print()

# 2. The feature sets
print("=== FEATURE SETS ===")
for name, cols in FEATURE_SETS.items():
    print(f"{name}: {len(cols)} features")
print()
print("Realistic set (R):", realistic)
print()

# 3. Two kinds of split, and an 'always no' baseline on each
splits = {"random (stratified)": random_split, "chronological": chronological_split}
size_rows, metric_rows = [], []
for split_name, splitter in splits.items():
    X_train, X_test, y_train, y_test = splitter(df, realistic)
    size_rows.append({
        "split": split_name,
        "train_clients": len(X_train),
        "test_clients": len(X_test),
        "train_rate_%": y_train.mean() * 100,
        "test_rate_%": y_test.mean() * 100,
        "test_subscribers": int(y_test.sum()),
    })

    # The baseline ignores every feature, so we pass it a dummy column of zeros
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(np.zeros((len(y_train), 1)), y_train)
    x_dummy = np.zeros((len(y_test), 1))
    y_pred = baseline.predict(x_dummy)
    scores = baseline.predict_proba(x_dummy)[:, 1]
    metric_rows.append({"split": split_name, **classification_metrics(y_test, y_pred, scores)})

print("=== SPLIT SIZES AND SUBSCRIPTION RATES ===")
print(pd.DataFrame(size_rows).round(1).to_string(index=False))
print()
print("=== BASELINE: ALWAYS PREDICT 'NO' (scored on the test side of each split) ===")
print(pd.DataFrame(metric_rows).round(3).to_string(index=False))
print()
print("accuracy = share of clients classified correctly")
print("precision = of the clients predicted 'yes', how many truly subscribed")
print("recall = of the clients who truly subscribed, how many were found")
print("roc_auc = 0.5 means no better than guessing; pr_auc for a useless model equals the test subscription rate")