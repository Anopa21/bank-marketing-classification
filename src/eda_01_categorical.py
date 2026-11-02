"""EDA step 1: subscription rate by category (no duration or campaign)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", None)

project_root = Path(__file__).resolve().parent.parent
data_path = project_root / "data" / "bank-additional-clean.csv"
images_dir = project_root / "images"

df = pd.read_csv(data_path)
overall = df["subscribed"].mean()

cat_cols = [
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "poutcome",
]
month_order = ["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
day_order = ["mon", "tue", "wed", "thu", "fri"]


def rate_table(col):
    """Clients, subscribers and subscription rate (%) for each category of a column."""
    t = df.groupby(col)["subscribed"].agg(clients="count", subscribers="sum", rate="mean")
    t["rate"] = t["rate"] * 100
    if col == "month":
        t = t.reindex(month_order)
    elif col == "day_of_week":
        t = t.reindex(day_order)
    else:
        t = t.sort_values("rate", ascending=False)
    return t.rename(columns={"rate": "rate_%"})


print(f"Clients: {len(df)}, overall subscription rate: {overall * 100:.2f}%")
print()
tables = {}
for col in cat_cols:
    tables[col] = rate_table(col)
    print(f"=== {col} ===")
    print(tables[col].round(1))
    print()
print("Small groups give noisy rates, so always read the 'clients' column too.")

# Chart: six of the variables
chart_cols = ["job", "education", "poutcome", "month", "contact", "default"]
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 3, figsize=(17, 10))
for ax, col in zip(axes.flat, chart_cols):
    t = tables[col].iloc[::-1]  # reversed, so the first row is drawn at the top
    labels = [f"{idx} (n={n:,})" for idx, n in zip(t.index, t["clients"])]
    ax.barh(labels, t["rate_%"], color="steelblue")
    ax.axvline(overall * 100, color="crimson", linestyle="--")
    ax.set_title(f"Subscription rate by {col}")
    ax.set_xlabel("Subscription rate (%)")
    ax.set_ylabel(col)
fig.suptitle("Term Deposit Subscription Rate by Client and Contact Category", fontsize=14)
fig.legend(
    handles=[plt.Line2D([0], [0], color="crimson", linestyle="--", label=f"Overall rate ({overall * 100:.1f}%)")],
    loc="lower center",
)
fig.tight_layout(rect=(0, 0.04, 1, 1))

path = images_dir / "01_subscription_rate_by_category.png"
fig.savefig(path, dpi=150)
plt.close(fig)
print("Chart saved to:", path)