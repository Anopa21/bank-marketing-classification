"""Modelling step 3: is the skill in the random-split scores real, or is the model recognising the period?
Uses only the TRAINING side of the random split. The test sides are not touched."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_common import CATEGORICAL_COLS, FEATURE_SETS, RANDOM_STATE, TARGET, load_data, random_split

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)


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
df["period"] = pd.qcut(df["original_row"], 10, labels=False) + 1
df["era"] = pd.cut(df["period"], bins=[0, 3, 6, 10], labels=["early", "middle", "late"])

R = FEATURE_SETS["R: pre-call"]
E = FEATURE_SETS["E: pre-call, no economic columns"]
N = [c for c in E if c != "month"]
feature_sets = {"R": R, "E": E, "N": N}

# Training side of the random split only; the test side is never used here
X_train, _, _, _ = random_split(df, R)
train = df.loc[X_train.index]

scopes = {"all": train}
for era in ["early", "middle", "late"]:
    scopes[era] = train[train["era"] == era]

print("=== SCOPES (training side of the random split) ===")
for name, data in scopes.items():
    print(f"{name}: {len(data)} clients, {int(data[TARGET].sum())} subscribers ({data[TARGET].mean() * 100:.1f}%)")
print()

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
records = []
for scope_name, data in scopes.items():
    y = data[TARGET]
    for set_name, cols in feature_sets.items():
        X = data[cols]
        for fold, (tr, va) in enumerate(cv.split(X, y), start=1):
            for model_name, model in make_models().items():
                pipe = build_pipeline(model, cols)
                pipe.fit(X.iloc[tr], y.iloc[tr])
                scores = pipe.predict_proba(X.iloc[va])[:, 1]
                records.append({
                    "scope": scope_name, "features": set_name, "model": model_name,
                    "rows": len(data), "base_rate": y.iloc[va].mean(),
                    "roc_auc": roc_auc_score(y.iloc[va], scores),
                    "pr_auc": average_precision_score(y.iloc[va], scores),
                })

results = pd.DataFrame(records)
summary = results.groupby(["scope", "features", "model"], sort=False).agg(
    rows=("rows", "first"),
    base_rate=("base_rate", "mean"),
    roc_auc=("roc_auc", "mean"),
    roc_sd=("roc_auc", "std"),
    pr_auc=("pr_auc", "mean"),
).reset_index()
summary["pr_lift"] = summary["pr_auc"] / summary["base_rate"]

print("=== 5-FOLD CROSS-VALIDATION, THREE FEATURE SETS, FOUR SCOPES ===")
print(summary.round(3).to_string(index=False))
print()
print("R = all 20 pre-call features; E = R without the 5 economic columns; N = E without month")
print("LR = logistic regression; GB = gradient boosting")
print("pr_lift = pr_auc divided by base_rate (1.0 means no better than a useless model)")
print("roc_sd = spread of ROC-AUC between folds")