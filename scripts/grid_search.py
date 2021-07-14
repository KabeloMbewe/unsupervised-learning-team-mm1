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
# data = Dataset.load_from_file("data/train.csv", reader)
data = Dataset.load_builtin(name="ml-100k")

lrs = [0.004, 0.007, 0.01]  # np.arange(0.001, 0.01, 0.001)
regs = [0.01, 0.02, 0.04]  # np.arange(0.01, 0.1, 0.01)

param_grid = {
    "n_factors": [200, 400, 600, 800],
    "n_epochs": [20],
    # "init_mean": [0],
    # "init_std_dev": [0.1],
    "lr_all": lrs,  # [0.007],
    "reg_all": regs,  # [0.02],
    # "lr_bu": lrs,  # [None],
    # "lr_bi": lrs,  # [None],
    # "lr_pu": lrs,  # [None],
    # "lr_qi": lrs,  # [None],
    # "lr_yj": lrs,  # [None],
    # "reg_bu": regs,  # [None],
    # "reg_bi": regs,  # [None],
    # "reg_pu": regs,  # [None],
    # "reg_qi": regs,  # [None],
    # "reg_yj": regs,  # [None],
    "random_state": [42],
    "verbose": [False],
}

print(np.product(list(map(lambda lst: len(lst), param_grid.values()))))

print("Prepping Search")
gs = GridSearchCV(
    SVDpp,
    param_grid,
    measures=["rmse", "mae"],
    cv=3,
    n_jobs=-1,
    joblib_verbose=2,
    # pre_dispatch=4,
)

print("Performing grid search...")
gs.fit(data)

# best RMSE score
print(gs.best_score["rmse"])

# combination of parameters that gave the best RMSE score
print(gs.best_params["rmse"])
