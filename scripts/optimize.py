import sys

import comet_ml
import numpy as np
from comet_ml import Optimizer
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate

import pandas as pd

comet_ml.init(
    api_key="A33C91N4gmBuLOlY0TK57mmI0",
    project_name="movie-lens-unsup",
    workspace="profhercules",
)
print("Loading data...")
df = pd.read_csv("../cleaned/train_sample.csv").sample(n=250_000, random_state=42)
data = Dataset.load_from_df(
    df[["userId", "movieId", "rating"]], Reader(rating_scale=(0.5, 5))
)
del df

opt = Optimizer(
    sys.argv[1],
    trials=1,
    verbose=0,
    experiment_class="OfflineExperiment",
    offline_directory="./experiments_sampled",
    log_code=False,
    log_git_patch=False,
)


def fit(
    n_factors: int,
    n_epochs: int,
    init_mean: float,
    init_std_dev: float,
    lr_all: float,
    reg_all: float,
):
    algo = SVD(
        n_factors=n_factors,
        n_epochs=n_epochs,
        init_mean=init_mean,
        init_std_dev=init_std_dev,
        lr_all=lr_all,
        reg_all=reg_all,
        random_state=42,
        verbose=False,
    )

    results = cross_validate(
        algo=algo,
        data=data,
        measures=["rmse"],
        return_train_measures=True,
        cv=5,
        n_jobs=1,
        verbose=False,
    )

    return (
        results["test_rmse"].mean(),
        results["train_rmse"].mean(),
        np.array(results["fit_time"]).mean(),
        np.array(results["test_time"]).mean(),
    )


# Finally, get experiments, and train your models:
for experiment in opt.get_experiments():
    # get parameters:
    (test_rmse, train_rmse, fit_time, test_time) = fit(
        n_factors=experiment.get_parameter("n_factors"),
        n_epochs=experiment.get_parameter("n_epochs"),
        init_mean=experiment.get_parameter("init_mean"),
        init_std_dev=experiment.get_parameter("init_std_dev"),
        lr_all=experiment.get_parameter("lr_all"),
        reg_all=experiment.get_parameter("reg_all"),
    )

    experiment.log_metric("test_rmse", test_rmse)
    experiment.log_metric("train_rmse", train_rmse)
    experiment.log_metric("fit_time", fit_time)
    experiment.log_metric("test_time", test_time)
    experiment.add_tags(["SVD", "ML-full-sampled", "CV-5"])
    experiment.end()
