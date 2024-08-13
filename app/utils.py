import pandas as pd
import numpy as np


def impurify(df: pd.DataFrame, ignore_cols, chance=0.05):
    """
    Impurifies a dataframe by deleting random values

    :param df: The dataframe to impurify
    :param ignore_cols: A list of column indices to ignore
    :param chance: The probability of a value to be deleted
    """

    nan_prob = chance
    mask = np.random.rand(*df.shape) < nan_prob
    # mask[:, 0] = False
    for col in ignore_cols:
        mask[:, col] = False
    # print(mask)
    return df.mask(mask)
