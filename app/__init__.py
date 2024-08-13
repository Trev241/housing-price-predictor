import pandas as pd

from flask import Flask
from app.predictor import PricePredictor
from app.utils import impurify
from app.visualize import lineplot, heatmap, histplot

DATA_FILE_PATH = "app/data/housing-prices.csv"
CLEANED_DATA_FILE_PATH = "app/data/housing-prices-cleaned.csv"

# create and configure the app
app = Flask(__name__, instance_relative_config=True)
app.config.from_mapping(
    SECRET_KEY="dev",
)


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
lineplot(df, "area", "price")
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
heatmap(predictor.df)
histplot(df, "price")

from app import routes
