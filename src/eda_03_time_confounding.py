"""EDA step 3: how much does time confound the other variables? (counting only, no modelling)"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

pd.set_option("display.width", 220)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

project_root = Path(__file__).resolve().parent.parent
data_path = project_root / "data" / "bank-additional-clean.csv"
images_dir = project_root / "images"

df = pd.read_csv(data_path).sort_values("original_row").reset_index(drop=True)
df["period"] = pd.qcut(df["original_row"], 10, labels=False) + 1  # 1 = earliest tenth
df["era"] = pd.cut(
    df["period"], bins=[0, 3, 6, 10],
    labels=["early (tenths 1-3)", "middle (tenths 4-6)", "late (tenths 7-10)"],
)
month_order = ["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]

# 1. Which months fall in which tenth of the file?
month_by_period = pd.crosstab(df["month"], df["period"]).reindex(month_order)
print("=== CLIENTS PER MONTH IN EACH TENTH OF THE FILE ===")
print(month_by_period)
print()

# 2. Does the cellular advantage survive within the same era?
contact_table = df.groupby(["era", "contact"], observed=True)["subscribed"].agg(clients="count", rate="mean")
contact_table["rate_%"] = contact_table["rate"] * 100
print("=== SUBSCRIPTION RATE BY CONTACT TYPE WITHIN EACH ERA ===")
print(contact_table[["clients", "rate_%"]].round(1))
print()

# 3. How do other characteristics shift between eras?
era_summary = df.groupby("era", observed=True).agg(
    clients=("subscribed", "count"),
    rate_pct=("subscribed", lambda s: s.mean() * 100),
    cellular_pct=("contact", lambda s: (s == "cellular").mean() * 100),
    previously_contacted_pct=("previously_contacted", lambda s: s.mean() * 100),
    prior_success_pct=("poutcome", lambda s: (s == "success").mean() * 100),
    student_or_retired_pct=("job", lambda s: s.isin(["student", "retired"]).mean() * 100),
    mean_age=("age", "mean"),
)
print("=== HOW CLIENTS AND CALLS DIFFER BETWEEN ERAS ===")
print(era_summary.round(1))
print()

# 4. What would a chronological train/test split look like? (counting only)
rows = []
for frac in [0.6, 0.7, 0.8]:
    cut = int(len(df) * frac)
    train, test = df.iloc[:cut], df.iloc[cut:]
    rows.append({
        "train_share": frac,
        "train_clients": len(train),
        "test_clients": len(test),
        "train_rate_%": train["subscribed"].mean() * 100,
        "test_rate_%": test["subscribed"].mean() * 100,
        "train_cellular_%": (train["contact"] == "cellular").mean() * 100,
        "test_cellular_%": (test["contact"] == "cellular").mean() * 100,
    })
print("=== CHRONOLOGICAL SPLITS: EARLIER CALLS FOR TRAINING, LATER CALLS FOR TESTING ===")
print(pd.DataFrame(rows).round(1).to_string(index=False))
print(f"(A random split would give both sides about {df['subscribed'].mean() * 100:.1f}%.)")
print()

# 5. Chart
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

ax = axes[0]
sns.heatmap(month_by_period, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
ax.set_title("Number of Clients per Month in Each Tenth of the Date-Ordered File")
ax.set_xlabel("Tenth of the file (1 = earliest, 10 = latest)")
ax.set_ylabel("Month")

ax = axes[1]
plot_data = contact_table.reset_index()
sns.barplot(data=plot_data, x="era", y="rate_%", hue="contact", ax=ax)
ax.set_title("Subscription Rate by Contact Type Within Each Era\n(no cellular calls exist in the early era)")
ax.set_xlabel("Era (by position in the date-ordered file)")
ax.set_ylabel("Subscription rate (%)")
ax.legend(title="Contact type", loc="upper left")

fig.tight_layout()
path = images_dir / "03_time_confounding.png"
fig.savefig(path, dpi=150)
plt.close(fig)
print("Chart saved to:", path)