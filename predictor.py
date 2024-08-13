import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from pathlib import Path


class PricePredictor:
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

        Path("images/").mkdir(parents=True, exist_ok=True)
        Path("models/").mkdir(parents=True, exist_ok=True)

        if existing_model:
            try:
                f = open(f"models/{existing_model}.pkl", "rb")
                self.model = pickle.load(f)
                f.close()
            except:
                raise Exception("Could not load model. Creating a new one.")

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
        plt.savefig(f"images/{str(model)}-scatter-accuracy.png")

        self.model = model

    def save(self):
        """Saves the model to disk"""

        f = open(f"models/{str(self.model)}.pkl", "wb")
        pickle.dump(self.model, f)

    def predict(self, X_input):
        """
        Predicts the value of the target variable from the given input

        :returns prediction: An array containing values of y
        """

        return self.model.predict(X_input)
