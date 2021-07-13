from surprise import SVD, SVDpp
from surprise import Dataset, Reader
from surprise import accuracy
from surprise.model_selection import cross_validate, train_test_split, GridSearchCV

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

param_grid = {
    "n_factors": [10, 50, 100, 200, 300],
    "n_epochs": [20],
    "biased": [True, False],
    "init_mean": [0, 3.53],
    "init_std_dev": [0.1, 1.06],
    "lr_all": [0.004, 0.005, 0.006],
    "reg_all": [0.01, 0.02, 0.03],
    "lr_bu": [None],
    "lr_bi": [None],
    "lr_pu": [None],
    "lr_qi": [None],
    "reg_bu": [None],
    "reg_bi": [None],
    "reg_pu": [None],
    "reg_qi": [None],
}

gs = GridSearchCV(
    SVD, param_grid, measures=["rmse", "mae"], cv=5, n_jobs=-1, joblib_verbose=2
)

print("Performing grid search...")
gs.fit(data)

# best RMSE score
print(gs.best_score["rmse"])

# combination of parameters that gave the best RMSE score
print(gs.best_params["rmse"])
