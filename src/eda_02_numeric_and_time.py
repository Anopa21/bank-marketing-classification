"""EDA step 2: numeric variables and change over time (no duration or campaign)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

pd.set_option("display.width", 220)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

project_root = Path(__file__).resolve().parent.parent
data_path = project_root / "data" / "bank-additional-clean.csv"
images_dir = project_root / "images"

df = pd.read_csv(data_path)
overall = df["subscribed"].mean()

# 1. Spearman correlation of each numeric variable with subscription (0/1)
numeric_cols = ["age", "previous", "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
rows = []
for col in numeric_cols:
    rho, _ = spearmanr(df[col], df["subscribed"])
    rows.append({"variable": col, "spearman_rho": rho})
corr = pd.DataFrame(rows).sort_values("spearman_rho", key=abs, ascending=False)
print("=== SPEARMAN CORRELATION WITH SUBSCRIPTION ===")
print(corr.round(3).to_string(index=False))
print()

# 2. Subscription rate by age group
age_bins = [0, 24, 34, 44, 54, 64, 100]
age_labels = ["17-24", "25-34", "35-44", "45-54", "55-64", "65+"]
df["age_group"] = pd.cut(df["age"], bins=age_bins, labels=age_labels)
age_table = df.groupby("age_group", observed=True)["subscribed"].agg(clients="count", rate="mean")
age_table["rate"] = age_table["rate"] * 100
print("=== SUBSCRIPTION RATE BY AGE GROUP ===")
print(age_table.rename(columns={"rate": "rate_%"}).round(1))
print()

# 3. Subscription rate by number of earlier contacts (before this campaign)
df["previous_group"] = pd.cut(df["previous"], bins=[-1, 0, 1, 2, 100], labels=["0", "1", "2", "3+"])
prev_table = df.groupby("previous_group", observed=True)["subscribed"].agg(clients="count", rate="mean")
prev_table["rate"] = prev_table["rate"] * 100
print("=== SUBSCRIPTION RATE BY NUMBER OF EARLIER CONTACTS ===")
print(prev_table.rename(columns={"rate": "rate_%"}).round(1))
print()

# 4. Change through the file. The file is ordered by date but has no year,
#    so we cut it into ten equal parts by row order (1 = earliest, 10 = latest).
df["period"] = pd.qcut(df["original_row"], 10, labels=False) + 1
time_table = df.groupby("period").agg(
    clients=("subscribed", "count"),
    rate=("subscribed", "mean"),
    euribor3m=("euribor3m", "mean"),
    cellular_share=("contact", lambda s: (s == "cellular").mean()),
    top_months=("month", lambda s: ", ".join(s.value_counts().index[:2])),
)
time_table["rate_%"] = time_table["rate"] * 100
time_table["cellular_%"] = time_table["cellular_share"] * 100
print("=== CHANGE THROUGH THE FILE (tenths of the date-ordered data) ===")
print(time_table[["clients", "rate_%", "euribor3m", "cellular_%", "top_months"]].round(2))
print()

# 5. Chart
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

ax = axes[0]
labels = [f"{g}\n(n={n:,})" for g, n in zip(age_table.index, age_table["clients"])]
ax.bar(labels, age_table["rate"], color="steelblue")
ax.axhline(overall * 100, color="crimson", linestyle="--")
ax.set_title("Subscription Rate by Age Group\n(dashed line = overall rate)")
ax.set_xlabel("Age group")
ax.set_ylabel("Subscription rate (%)")

ax = axes[1]
ax.plot(time_table.index, time_table["rate_%"], marker="o", color="steelblue", label="Subscription rate (%)")
ax.axhline(overall * 100, color="crimson", linestyle="--", label=f"Overall rate ({overall * 100:.1f}%)")
ax.set_xlabel("Position in the date-ordered file (1 = earliest tenth, 10 = latest tenth)")
ax.set_ylabel("Subscription rate (%)")
ax.set_xticks(range(1, 11))
ax2 = ax.twinx()
ax2.plot(time_table.index, time_table["euribor3m"], marker="s", color="darkorange", label="Mean 3-month Euribor rate (right axis)")
ax2.set_ylabel("Mean euribor3m")
ax2.grid(False)
lines = ax.get_lines() + ax2.get_lines()
ax.legend(lines, [line.get_label() for line in lines], loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
ax.set_title("Subscription Rate and Interest Rate Through the File")

fig.tight_layout()
path = images_dir / "02_age_and_time.png"
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print("Chart saved to:", path)