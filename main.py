import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from app.predictor import PricePredictor

DATA_FILE_PATH = "app/data/housing-prices.csv"
CLEANED_DATA_FILE_PATH = "app/data/housing-prices-cleaned.csv"

df = pd.read_csv(DATA_FILE_PATH)

print(df.head())


predictor = PricePredictor(df, "price")

columns = ["bedrooms", "bathrooms", "stories", "parking"]
predictor.sanitize(columns, PricePredictor.USE_MEDIAN)
predictor.sanitize(["area"], PricePredictor.USE_MEAN)
columns = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea",
    "furnishingstatus",
]
predictor.sanitize(columns, PricePredictor.USE_MODE)


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

# plt.scatter(df["area"], df["price"])
# plt.show()

plt.hist(df["price"])
plt.title("Distribution of prices")
plt.show()


df.to_csv(CLEANED_DATA_FILE_PATH)

# Create train test split
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns=["price"]), df["price"], test_size=0.33, random_state=42
)

# Linear regression model
# model = LinearRegression()
# model.fit(X_train, y_train)
# print(model.score(X_test, y_test))

# y_pred = model.predict(X_test)

predictor.initialize(LinearRegression())
y_pred = predictor.predict(X_test)

# KNN model
# model = KNeighborsRegressor(n_neighbors=3)
# model.fit(X_train, y_train)
# print(model.score(X_test, y_test))
