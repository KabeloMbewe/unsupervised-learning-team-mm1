import time

import numpy as np
import pandas as pd
from surprise import SVD, Dataset, Reader, SVDpp, accuracy
from surprise.model_selection import GridSearchCV, cross_validate, train_test_split

from model import SVDpy

print("Loading data...")
# data = Dataset.load_builtin("ml-100k")
df = pd.read_csv("data/train.csv")
sample = df.sort_values(["userId", "movieId"], ascending=False).iloc[0:5_000_000]

data = Dataset.load_from_df(
    sample[["userId", "movieId", "rating"]], Reader(rating_scale=(1, 5))
)

algo = SVD(n_factors=800, n_epochs=5, random_state=42, verbose=True)

print("Starting cross_validate")
start = time.time()
cross_validate(algo, data, measures=["RMSE", "MAE"], cv=3, verbose=True)
duration = time.time() - start

print("Completed in {} seconds".format(round(duration, 2)))

print("Average time per epoch {} seconds".format(round(duration / 15, 1)))
