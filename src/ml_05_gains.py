"""Modelling step 5: gains analysis.
If the bank could only call the top-ranked share of clients inside a period, what share of the
subscribers would it reach? Out-of-fold predictions on the TRAINING side of the random split;
the test sides are not used."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_common import (
    CATEGORICAL_COLS,
    ECON_COLS,
    FEATURE_SETS,
    IMAGES_DIR,
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


def capture(y_values, scores, fraction):
    """Share (%) of all subscribers found among the top-ranked `fraction` of clients."""
    y_values = np.asarray(y_values)
    order = np.argsort(-np.asarray(scores), kind="stable")
    k = int(np.ceil(fraction * len(order)))
    return 100 * y_values[order][:k].sum() / y_values.sum()


df = load_data()
df["period"] = pd.qcut(df["original_row"], 10, labels=False) + 1  # 1 = earliest tenth

R = FEATURE_SETS["R: pre-call"]
T = ["month"] + ECON_COLS  # time markers only: knows when a call happened, nothing about the client

# Training side of the random split only; the test side is never used here
X_train, _, _, _ = random_split(df, R)
train = df.loc[X_train.index].copy()
y = train[TARGET]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

models = {
    "GB": lambda: HistGradientBoostingClassifier(random_state=RANDOM_STATE),
    "LR": lambda: LogisticRegression(max_iter=2000),
}

# Every client is scored by a model that did not see that client
oof = {}
for set_name, cols in {"R": R, "T": T}.items():
    for model_name, make_model in models.items():
        pipe = build_pipeline(make_model(), cols)
        oof[f"{set_name}_{model_name}"] = pd.Series(
            cross_val_predict(pipe, train[cols], y, cv=cv, method="predict_proba")[:, 1],
            index=train.index,
        )

# 1. Top 20% of clients inside each period
FRACTION = 0.20
rows = []
for period in range(1, 11):
    mask = train["period"] == period
    row = {
        "period": period,
        "subscribers": int(y[mask].sum()),
        "base_rate_%": y[mask].mean() * 100,
    }
    for key, scores in oof.items():
        row[key] = capture(y[mask], scores[mask], FRACTION)
    rows.append(row)
by_period = pd.DataFrame(rows)

print("=== SUBSCRIBERS REACHED (%) BY CALLING THE TOP 20% OF CLIENTS, INSIDE EACH PERIOD ===")
print(by_period[["period", "subscribers", "base_rate_%", "R_GB", "T_GB", "R_LR", "T_LR"]].round(1).to_string(index=False))
print("Random selection would reach 20.0% in every row.")
print()

# 2. Several cut-offs, averaged over groups of periods (equal weight per period)
fractions = [0.10, 0.20, 0.30, 0.50]
groups = {"periods 1-6": range(1, 7), "periods 7-10": range(7, 11)}
rows2 = []
for group_name, periods in groups.items():
    for frac in fractions:
        row = {"periods": group_name, "top_share_called_%": int(frac * 100)}
        for key, scores in oof.items():
            values = [
                capture(y[train["period"] == p], scores[train["period"] == p], frac)
                for p in periods
            ]
            row[key] = float(np.mean(values))
        rows2.append(row)

print("=== SUBSCRIBERS REACHED (%) AT DIFFERENT CUT-OFFS, AVERAGED OVER PERIODS ===")
print(pd.DataFrame(rows2).round(1).to_string(index=False))
print("Random selection would reach the same % as the share called (10, 20, 30, 50).")
print()
print("R = all 20 pre-call features; T = month and economic columns only")
print("GB = gradient boosting; LR = logistic regression")
print("Periods 1-3 have only 88-146 subscribers each, so their numbers are noisy.")

# 3. Chart
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(1, 11)
width = 0.38
ax.bar(x - width / 2, by_period["R_GB"], width, color="steelblue", label="All 20 pre-call features (R)")
ax.bar(x + width / 2, by_period["T_GB"], width, color="darkorange", label="Month and economic columns only (T)")
ax.axhline(20, color="crimson", linestyle="--", label="Random selection (20%)")
ax.set_xticks(x)
ax.set_xlabel("Tenth of the date-ordered file (1 = earliest, 10 = latest)")
ax.set_ylabel("Share of subscribers reached (%)")
ax.set_title("Subscribers Reached by Calling the Top-Ranked 20% of Clients, Within Each Period\n(gradient boosting, out-of-fold predictions)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
fig.tight_layout()

path = IMAGES_DIR / "04_gains_by_period.png"
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print("Chart saved to:", path)