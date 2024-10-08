import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def lineplot(df: pd.DataFrame, x_prop: str, y_prop: str):
    plt.clf()

    # Sort dataframe by x property
    df_sorted = df.sort_values(by=x_prop)

    plt.title(f"Line chart of {x_prop} against {y_prop}")
    min_row = df_sorted.iloc[0]
    max_row = df_sorted.iloc[-1]
    plt.plot(df_sorted[x_prop], df_sorted[y_prop])
    plt.plot(
        [min_row[x_prop], max_row[x_prop]],
        [min_row[y_prop], max_row[y_prop]],
        ls="--",
    )
    plt.xlabel(x_prop.capitalize())
    plt.ylabel(y_prop.capitalize())

    plt.savefig(f"app/static/images/{x_prop}-{y_prop}-lineplot.png")


def heatmap(df: pd.DataFrame):
    plt.clf()

    corr = df.corr()
    sns.heatmap(corr)
    plt.title("Heatmap of all features")
    plt.savefig(f"app/static/images/heatmap.png")


def histplot(df: pd.DataFrame, column):
    plt.clf()

    plt.hist(df[column])
    plt.title(f"Distribution of {column}")
    plt.xlabel(column.capitalize())
    plt.ylabel("Frequency")
    plt.savefig(f"app/static/images/{column}-hist.png")


def pie(df: pd.DataFrame, column):
    plt.clf()

    plt.title("Distribution of houses by the number of bedrooms")
    value_counts = df[column].value_counts().sort_values()
    patches, _ = plt.pie(value_counts)
    plt.legend(patches, labels=value_counts.keys(), loc="best")
    plt.savefig(f"app/static/images/{column}-pie.png")
