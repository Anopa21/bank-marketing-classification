"""Step 8: create a cleaned copy of the dataset. The raw file is never modified."""

from pathlib import Path

import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

project_root = Path(__file__).resolve().parent.parent
raw_path = project_root / "data" / "bank-additional-full.csv"
clean_path = project_root / "data" / "bank-additional-clean.csv"

df = pd.read_csv(raw_path, sep=";")
rows_before, cols_before = df.shape

# 1. Remember each row's position in the original file. The file is ordered by
#    date but has no year column, so row order is our only record of time.
df.insert(0, "original_row", df.index)

# 2. Remove exact duplicates (all original columns identical), keeping the first copy.
original_cols = [c for c in df.columns if c != "original_row"]
df = df.drop_duplicates(subset=original_cols, keep="first").reset_index(drop=True)

# 3. Remove accidental leading/trailing spaces in text columns (a safe precaution).
text_cols = df.select_dtypes(include=["object", "string"]).columns
for col in text_cols:
    df[col] = df[col].str.strip()

# 4. Numeric version of the target: 1 = subscribed, 0 = did not.
df["subscribed"] = (df["y"] == "yes").astype(int)

# 5. Previous-contact flags.
#    The documentation says pdays == 999 means "not previously contacted", but the
#    data show 4,110 clients with previous > 0 whose pdays is 999. So:
#      previously_contacted: based on 'previous' (earlier contacts on record)
#      pdays_recorded:       1 only when pdays holds a real number of days
#    The raw pdays column is kept unchanged.
df["previously_contacted"] = (df["previous"] > 0).astype(int)
df["pdays_recorded"] = (df["pdays"] != 999).astype(int)

# 6. Save the cleaned copy.
df.to_csv(clean_path, index=False)

print("Raw shape:    ", (rows_before, cols_before))
print("Cleaned shape:", df.shape)
print("Rows removed: ", rows_before - len(df))
print()
print("Target counts (subscribed):")
print(df["subscribed"].value_counts())
print(f"Subscription rate: {df['subscribed'].mean():.4f}")
print()
print("=== CONSISTENCY: previously_contacted vs poutcome ===")
print(pd.crosstab(df["previously_contacted"], df["poutcome"]))
print()
print("=== CONSISTENCY: previously_contacted vs pdays_recorded ===")
print(pd.crosstab(df["previously_contacted"], df["pdays_recorded"]))
print()
print("Saved to:", clean_path)