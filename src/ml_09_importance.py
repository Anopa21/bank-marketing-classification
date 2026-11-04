"""Modelling step 9: what do the models rely on?
Permutation importance on held-out folds of the TRAINING side of the random split (the test sides are not
used), for single columns and for groups of related columns, plus logistic regression coefficients."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_common import (
    CATEGORICAL_COLS,
    CLIENT_COLS,
    ECON_COLS,
    FEATURE_SETS,
    HISTORY_COLS,
    IMAGES_DIR,
    RANDOM_STATE,
    TARGET,
    load_data,
    random_split,
)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

N_REPEATS = 5


def build_pipeline(model, feature_cols):
    """One-hot encode text columns, scale numeric columns, then fit the model."""
    categorical = [c for c in feature_cols if c in CATEGORICAL_COLS]
    numeric = [c for c in feature_cols if c not in CATEGORICAL_COLS]
    prep = ColumnTransformer([
        ("categories", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ("numbers", StandardScaler(), numeric),
    ])
    return Pipeline([("prep", prep), ("model", model)])


def permutation_drop(pipe, X_va, y_va, cols, base_roc, rng):
    """Average drop in ROC-AUC when the given columns are shuffled together among the clients."""
    drops = []
    for _ in range(N_REPEATS):
        order = rng.permutation(len(X_va))
        shuffled = X_va[cols].iloc[order].copy()
        shuffled.index = X_va.index
        X_perm = X_va.copy()
        X_perm[cols] = shuffled
        drops.append(base_roc - roc_auc_score(y_va, pipe.predict_proba(X_perm)[:, 1]))
    return float(np.mean(drops))


df = load_data()
period = pd.qcut(df["original_row"], 10, labels=False) + 1  # 1 = earliest tenth
df["era"] = pd.cut(period, bins=[0, 3, 6, 10], labels=["early", "middle", "late"])

R = FEATURE_SETS["R: pre-call"]

# Training side of the random split only; the test side is never used here
X_train, _, _, _ = random_split(df, R)
train = df.loc[X_train.index].copy()

scopes = {"all": train, "late": train[train["era"] == "late"]}
models = {
    "LR": lambda: LogisticRegression(max_iter=2000),
    "GB": lambda: HistGradientBoostingClassifier(random_state=RANDOM_STATE),
}

groups = {
    "client profile (7 columns)": CLIENT_COLS,
    "contact type and weekday": ["contact", "day_of_week"],
    "month": ["month"],
    "previous-campaign history (5 columns)": HISTORY_COLS,
    "economic indicators (5 columns)": ECON_COLS,
}
items = [(name, cols, "group") for name, cols in groups.items()]
items += [(col, [col], "column") for col in R]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
rng = np.random.default_rng(42)

records, baseline_rows = [], []
for scope_name, data in scopes.items():
    X, y = data[R], data[TARGET]
    for model_name, make_model in models.items():
        for fold, (tr, va) in enumerate(cv.split(X, y), start=1):
            pipe = build_pipeline(make_model(), R)
            pipe.fit(X.iloc[tr], y.iloc[tr])
            X_va, y_va = X.iloc[va], y.iloc[va]
            base_roc = roc_auc_score(y_va, pipe.predict_proba(X_va)[:, 1])
            baseline_rows.append({"scope": scope_name, "model": model_name, "roc_auc": base_roc})
            for name, cols, kind in items:
                records.append({
                    "scope": scope_name, "model": model_name, "kind": kind, "name": name,
                    "drop": permutation_drop(pipe, X_va, y_va, cols, base_roc, rng),
                })

print("=== BASELINE ROC-AUC OF THE MODELS BEING EXPLAINED (mean over 5 held-out folds) ===")
print(pd.DataFrame(baseline_rows).groupby(["scope", "model"])["roc_auc"].mean().round(3).to_string())
print()

results = pd.DataFrame(records)
summary = results.groupby(["kind", "name", "scope", "model"])["drop"].agg(["mean", "std"]).reset_index()
summary["key"] = summary["model"] + "_" + summary["scope"]


def show(kind, title):
    """Print mean +/- spread between folds of the ROC-AUC drop, strongest first."""
    t = summary[summary["kind"] == kind]
    means = t.pivot(index="name", columns="key", values="mean")
    sds = t.pivot(index="name", columns="key", values="std")
    order = means.sort_values("GB_all", ascending=False).index
    cols = ["GB_all", "LR_all", "GB_late", "LR_late"]
    out = pd.DataFrame(
        {c: [f"{means.loc[n, c]:.3f} +/- {sds.loc[n, c]:.3f}" for n in order] for c in cols},
        index=order,
    )
    print(title)
    print(out.to_string())
    print()


show("group", "=== GROUPS OF COLUMNS SHUFFLED TOGETHER: DROP IN ROC-AUC (mean +/- spread between folds) ===")
show("column", "=== SINGLE COLUMNS: DROP IN ROC-AUC (mean +/- spread between folds) ===")
print("GB = gradient boosting, LR = logistic regression; _all = all training clients, _late = late era only")
print("A larger drop means the model relied on that column more. Near 0 (or negative) means it did not.")
print("Related columns share credit, so a single column can look unimportant when its group matters.")
print()

# Logistic regression coefficients (all training clients)
pipe = build_pipeline(LogisticRegression(max_iter=2000), R)
pipe.fit(train[R], train[TARGET])
names = [n.replace("categories__", "").replace("numbers__", "") for n in pipe.named_steps["prep"].get_feature_names_out()]
coefs = pd.Series(pipe.named_steps["model"].coef_[0], index=names).sort_values(ascending=False)
print("=== LOGISTIC REGRESSION: STRONGEST POSITIVE COEFFICIENTS (log-odds of subscribing) ===")
print(coefs.head(8).round(3).to_string())
print()
print("=== LOGISTIC REGRESSION: STRONGEST NEGATIVE COEFFICIENTS ===")
print(coefs.tail(8).sort_values().round(3).to_string())
print()
print("Numeric columns are scaled, so a coefficient is per one standard deviation.")
print("Text columns are one-hot encoded: a coefficient is relative to the other levels, not to a fixed reference.")
print("Overlapping columns (for example previous, poutcome and pdays) share effects, so read the signs with care.")

# Chart: groups, gradient boosting, both scopes
t = summary[(summary["kind"] == "group") & (summary["model"] == "GB")]
order = t[t["scope"] == "all"].sort_values("mean")["name"].tolist()
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
for ax, scope, title in zip(
    axes, ["all", "late"], ["All training clients", "Late era only (tenths 7-10)"]
):
    s = t[t["scope"] == scope].set_index("name").loc[order]
    ax.barh(order, s["mean"], xerr=s["std"], color="steelblue", ecolor="grey", capsize=3)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Drop in ROC-AUC when the group is shuffled")
axes[0].set_ylabel("Group of columns")
fig.suptitle("What the Gradient Boosting Model Relies On (held-out folds, error bars = spread between folds)", fontsize=13)
fig.tight_layout()

path = IMAGES_DIR / "05_permutation_importance.png"
fig.savefig(path, dpi=150)
plt.close(fig)
print()
print("Chart saved to:", path)