import pandas as pd
import os

from flask import Flask
from app.predictor import PricePredictor
from app.utils import impurify
from app.visualize import lineplot, heatmap, histplot, pie
from sklearn.linear_model import LinearRegression

DATA_FILE_PATH = "app/data/housing-prices.csv"
CLEANED_DATA_FILE_PATH = "app/data/housing-prices-cleaned.csv"

# create and configure the app
app = Flask(__name__, instance_relative_config=True)
app.config.from_mapping(
    SECRET_KEY="dev",
)

print(os.listdir("app/data/"))

if "LinearRegression().pkl" in os.listdir(
    "app/models/"
) and "housing-prices-cleaned.csv" in os.listdir("app/data/"):
    df = pd.read_csv(CLEANED_DATA_FILE_PATH)
    predictor = PricePredictor(
        df, target_var="price", existing_model="LinearRegression()"
    )

    print("Loaded models and data successfully!")
else:
    df = pd.read_csv(DATA_FILE_PATH)
    df = impurify(df, ignore_cols=[0])
    predictor = PricePredictor(df, target_var="price")

    # Use median to fill missing values
    columns = ["bedrooms", "bathrooms", "stories", "parking"]
    predictor.sanitize(columns, PricePredictor.USE_MEDIAN)
    # Use mean to fill missing values
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
    # Use mode to fill missing values
    predictor.sanitize(columns, PricePredictor.USE_MODE)

    # Create charts
    predictor.map_values(
        [
            "mainroad",
            "guestroom",
            "basement",
            "hotwaterheating",
            "airconditioning",
            "prefarea",
            "furnishingstatus",
        ],
        {"unfurnished": 0, "semi-furnished": 1, "furnished": 2, "no": 0, "yes": 1},
    )

    predictor.initialize(LinearRegression())
    predictor.save()

    predictor.df.to_csv(CLEANED_DATA_FILE_PATH, index=False)

if len(os.listdir("app/static/images")) != 5:
    lineplot(df, "area", "price")
    heatmap(predictor.df)
    histplot(df, "price")
    pie(df, "bedrooms")

from app import routes
