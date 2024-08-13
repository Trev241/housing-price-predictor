import pickle
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from pathlib import Path


class PricePredictor:
    USE_MODE = "Mode"
    USE_MEAN = "Mean"
    USE_MEDIAN = "Median"

    def __init__(self, df: pd.DataFrame, target_var: str, existing_model=None) -> None:
        """
        Creates an instance of the predictor.

        :param df: The dataframe on which the model should be trained
        :param target_var: The target variable
        :param existing_mode: Load an existing model on the disk.
            If this fails, a new model will be created instead.
        """
        self.df = df
        self.target_var = target_var

        Path("app/static/images/").mkdir(parents=True, exist_ok=True)
        Path("app/models/").mkdir(parents=True, exist_ok=True)

        if existing_model:
            try:
                f = open(f"app/models/{existing_model}.pkl", "rb")
                self.model = pickle.load(f)
                f.close()
            except:
                raise Exception("Could not load model.")

    def sanitize(self, columns, using):
        """
        Sanitizes the dataframe by populating the missing values of select columns

        :param columns: Columns to be sanitized
        :param using: The method by which the column should be sanitized
        """

        for column in columns:
            if using == PricePredictor.USE_MODE:
                val = self.df[column].mode()[0]
            elif using == PricePredictor.USE_MEDIAN:
                val = round(self.df[column].mean())
            else:
                val = self.df[column].median()

            print(f"{using} for {column} is {val}")
            self.df[column] = self.df[column].fillna(val)

    def map_values(self, columns, map):
        for column in columns:
            self.df[column] = self.df[column].map(map)

    def initialize(self, model):
        """
        Initializes the model of the predictor instance.
        A call to this function will override the instance's existing model.
        """

        X = self.df.drop(columns=[self.target_var])
        y = self.df[self.target_var]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )

        model.fit(X_train, y_train)
        self.accuracy = model.score(X_test, y_test)

        # Create a scatter diagram to illustrate the model's accuracy
        y_pred = model.predict(X_test)
        plt.scatter(y_test, y_pred)
        # Create line of best fit
        y_max = max(y_test)
        plt.plot([0, y_max], [0, y_max], ls="--")
        plt.savefig(f"app/static/images/{str(model)}-scatter-accuracy.png")

        self.model = model

    def save(self):
        """Saves the model to disk"""

        f = open(f"app/models/{str(self.model)}.pkl", "wb")
        pickle.dump(self.model, f)

    def predict(self, X_input):
        """
        Predicts the value of the target variable from the given input

        :returns prediction: An array containing values of y
        """

        return self.model.predict(X_input)
