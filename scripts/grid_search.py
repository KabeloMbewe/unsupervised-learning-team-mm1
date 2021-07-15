from surprise import SVD, SVDpp
from surprise import Dataset, Reader
from surprise import accuracy
from surprise.model_selection import cross_validate, train_test_split, GridSearchCV

import numpy as np

reader = Reader(
    name=None,
    line_format="user item rating",
    sep=",",
    rating_scale=(1, 5),
    skip_lines=1,
)

print("Loading data...")
data = Dataset.load_from_file("data/train.csv", reader)

param_grid = {
    "n_factors": [400, 600, 800],
    "n_epochs": [25],
    "lr_all": [0.005, 0.007, 0.009, 0.012, 0.015, 0.018],
    "reg_all": [0.01, 0.03, 0.05, 0.07],
    "random_state": [42],
    "verbose": [True],
}

print(np.product(list(map(lambda lst: len(lst), param_grid.values()))))

print("Prepping Search")
gs = GridSearchCV(
    SVD,
    param_grid,
    measures=["rmse"],
    cv=3,
    n_jobs=1,
    joblib_verbose=2,
    # pre_dispatch=4,
)

print("Performing grid search...")
gs.fit(data)

# best RMSE score
print(gs.best_score["rmse"])

# combination of parameters that gave the best RMSE score
print(gs.best_params["rmse"])
