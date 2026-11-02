"""Step 6: inspect data types, missing values, duplicates, ranges and special codes."""

from pathlib import Path

import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.max_rows", None)

project_root = Path(__file__).resolve().parent.parent
data_path = project_root / "data" / "bank-additional-full.csv"
df = pd.read_csv(data_path, sep=";")

# 1. Data types: numbers or text?
print("=== DATA TYPES ===")
print(df.dtypes)
print()

# 2. Truly empty cells
print("=== EMPTY CELLS ===")
print("Total empty cells in the dataset:", df.isnull().sum().sum())
print()

# 3. Duplicate rows: identical in every column
print("=== DUPLICATE ROWS ===")
print("Fully duplicated rows:", df.duplicated().sum())
print()

# 4. Numeric columns: min, max and mean reveal impossible values
print("=== NUMERIC COLUMNS: RANGES ===")
print(df.describe().loc[["min", "max", "mean"]].T)
print()

# 5. Text columns: every distinct value, to spot typos or odd labels
text_cols = df.select_dtypes(include=["object", "string"]).columns
print("=== TEXT COLUMNS: DISTINCT VALUES ===")
for col in text_cols:
    print(f"{col}: {sorted(df[col].unique())}")
print()

# 6. Missing information may be written as the text "unknown" instead of
#    an empty cell, so count that too
unknown = (df[text_cols] == "unknown").sum()
print("=== COUNT OF THE TEXT 'unknown' PER COLUMN ===")
print(unknown[unknown > 0].sort_values(ascending=False))
print()

# 7. Special codes
print("=== SPECIAL CODES ===")
print("pdays == 999 (documented as: not previously contacted):", (df["pdays"] == 999).sum())
print("duration == 0:", (df["duration"] == 0).sum())