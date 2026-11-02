"""Step 7: investigate duplicates, 'unknown' values, zero-duration calls and extreme campaign values."""

from pathlib import Path

import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)
pd.set_option("display.max_rows", None)

project_root = Path(__file__).resolve().parent.parent
data_path = project_root / "data" / "bank-additional-full.csv"
df = pd.read_csv(data_path, sep=";")
df["y_yes"] = (df["y"] == "yes").astype(int)

# 1. Duplicates: show every copy. The row number (left) is the position in the
#    date-ordered file, so adjacent numbers mean the calls happened close together.
show_cols = ["age", "job", "education", "month", "day_of_week", "duration", "campaign", "euribor3m", "y"]
dups = df[df.duplicated(keep=False)]
print("=== ROWS INVOLVED IN DUPLICATION (every copy shown) ===")
print(f"Rows involved: {len(dups)}")
print(dups.sort_values(["age", "job", "duration"])[show_cols])
print()

# 2. Do the 'unknown' values occur in the same rows?
unknown_cols = ["default", "education", "housing", "loan", "job", "marital"]
is_unknown = df[unknown_cols] == "unknown"
print("=== OVERLAP OF 'unknown' VALUES ===")
print("housing unknown AND loan unknown:", (is_unknown["housing"] & is_unknown["loan"]).sum())
print("housing unknown only:            ", (is_unknown["housing"] & ~is_unknown["loan"]).sum())
print("loan unknown only:               ", (~is_unknown["housing"] & is_unknown["loan"]).sum())
print(f"Rows with at least one 'unknown': {is_unknown.any(axis=1).sum()} ({is_unknown.any(axis=1).mean():.1%})")
print()

# 3. Does 'unknown' carry information? Compare the subscription rate.
print("=== SUBSCRIPTION RATE: 'unknown' vs KNOWN ===")
rows = []
for col in unknown_cols:
    mask = is_unknown[col]
    rows.append({
        "column": col,
        "n_unknown": int(mask.sum()),
        "rate_if_unknown": df.loc[mask, "y_yes"].mean(),
        "rate_if_known": df.loc[~mask, "y_yes"].mean(),
    })
print(pd.DataFrame(rows).round(3).to_string(index=False))
print(f"(Overall subscription rate: {df['y_yes'].mean():.3f})")
print()

# 4. The zero-duration calls
print("=== CALLS WITH duration == 0 ===")
print(df.loc[df["duration"] == 0, ["age", "job", "month", "duration", "campaign", "y"]])
print()

# 5. Extreme campaign values
print("=== CAMPAIGN (number of contacts) ===")
print(df["campaign"].describe(percentiles=[0.5, 0.9, 0.99]).round(2))
print("Clients with campaign > 20:", (df["campaign"] > 20).sum())
print()

# 6. Subscription rate by number of contacts
bins = [0, 1, 2, 3, 5, 10, 100]
labels = ["1", "2", "3", "4-5", "6-10", "11+"]
df["campaign_group"] = pd.cut(df["campaign"], bins=bins, labels=labels)
print("=== SUBSCRIPTION RATE BY NUMBER OF CONTACTS IN THIS CAMPAIGN ===")
print(df.groupby("campaign_group", observed=True)["y_yes"].agg(clients="count", subscription_rate="mean").round(3))
