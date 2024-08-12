import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DATA_FILE_PATH = "data/housing-prices.csv"
CLEANED_DATA_FILE_PATH = "data/housing-prices-cleaned.csv"

df = pd.read_csv(DATA_FILE_PATH)

print(df.head())

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

# Reorder prices into price ranges
bin_count = 10
min_price = df["price"].min()
max_price = df["price"].max()
interval = (max_price - min_price) / bin_count

print(max_price, min_price)
print(
    [
        f"{lower} to {lower + interval}"
        for lower in range(int(min_price), int(max_price), int(interval))
    ]
)

# df["price"] = pd.cut(
#     df["price"],
#     bins=bin_count,
#     labels=[
#         f"{lower} - {lower + interval}"
#         for lower in range(int(min_price), int(max_price), int(interval))
#     ],
#     ordered=False,
# )
# print(df.head())

# Sort dataframe by area
df_sorted = df.sort_values(by="area")
print(df_sorted.head())

# Visualization
features = df.columns[1:]

# plt.bar(df["price"], df["area"])
# plt.show()

plt.title("Line chart of price against area")
min_row = df_sorted.iloc[0]
max_row = df_sorted.iloc[-1]
plt.plot(df_sorted["area"], df_sorted["price"])
plt.plot(
    [min_row["area"], max_row["area"]], [min_row["price"], max_row["price"]], ls="--"
)
plt.xlabel("Area")
plt.ylabel("Price")
plt.show()

# plt.scatter(df["area"], df["price"])
# plt.show()

plt.hist(df["price"])
plt.title("Distribution of prices")
plt.show()

value_map = {"unfurnished": 0, "semi-furnished": 1, "furnished": 2, "no": 0, "yes": 1}
for feature in [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea",
    "furnishingstatus",
]:
    print(df[feature])
    df[feature] = df[feature].map(value_map)
    print(df[feature])

corr = df.corr()
print(corr)
sns.heatmap(corr)
plt.title("Heatmap of all features")
plt.show()

df.to_csv(CLEANED_DATA_FILE_PATH)
