"""Step 5: load the raw dataset and take a first look."""

from pathlib import Path

import pandas as pd

# Show every column when printing, instead of hiding the middle ones with "..."
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# Build the path to the data file relative to this script
project_root = Path(__file__).resolve().parent.parent
data_path = project_root / "data" / "bank-additional-full.csv"

# I expect semicolon-separated values. If the shape below shows only one
# column, this assumption is wrong and we will fix it.
df = pd.read_csv(data_path, sep=";")

print("Shape (rows, columns):", df.shape)
print()
print("Column names:")
print(list(df.columns))
print()
print("First 5 rows:")
print(df.head())
print()
print("Counts in the target column 'y':")
print(df["y"].value_counts())