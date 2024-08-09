import pandas as pd
import numpy as np

DATA_FILE_PATH = "data/housing-prices.csv"
CLEANED_DATA_FILE_PATH = "data/housing-prices-cleaned.csv"

df = pd.read_csv(DATA_FILE_PATH)

# Impurify data
nan_prob = 0.05
mask = np.random.rand(*df.shape) < nan_prob
mask[:, 0] = False
# print(mask)
df = df.mask(mask)

# Using median to fill in missing data
columns = ["bedrooms", "bathrooms", "stories", "parking"]
for column in columns:
    column_median = df[column].median()
    print(f"Median for {column} is {column_median}")
    df[column] = df[column].replace("unknown", None)
    df[column] = df[column].fillna(column_median)

# Using mean to fill in missing area
area_mean = round(df["area"].mean())
print(f"Mean for area is {area_mean} (rounded)")
df["area"] = df["area"].fillna(area_mean)

# For ordinal variables, choose the mode
columns = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea",
    "furnishingstatus",
]
for column in columns:
    column_mode = df[column].mode()[0]
    print(f"Mode for {column} is {column_mode}")
    df[column] = df[column].fillna(column_mode)

df.to_csv(CLEANED_DATA_FILE_PATH)
