import sys

import comet_ml
import numpy as np
from comet_ml import Optimizer
from surprise import SVD, Dataset
from surprise.model_selection import cross_validate

comet_ml.init(
    api_key="A33C91N4gmBuLOlY0TK57mmI0",
    project_name="movie-lens-unsup",
    workspace="profhercules",
)
data = Dataset.load_builtin(name="ml-100k")

opt = Optimizer(
    sys.argv[1],
    trials=1,
    verbose=0,
    experiment_class="OfflineExperiment",
    offline_directory="./experiments_100k",
    log_code=False,
    log_git_patch=False,
)


def fit(
    n_factors: int,
    n_epochs: int,
    biased: bool,
    init_mean: float,
    init_std_dev: float,
    lr_bu: float,
    lr_bi: float,
    lr_pu: float,
    lr_qi: float,
    reg_bu: float,
    reg_bi: float,
    reg_pu: float,
    reg_qi: float,
):
    algo = SVD(
        n_factors=n_factors,
        n_epochs=n_epochs,
        biased=biased,
        init_mean=init_mean,
        init_std_dev=init_std_dev,
        lr_bu=lr_bu,
        lr_bi=lr_bi,
        lr_pu=lr_pu,
        lr_qi=lr_qi,
        reg_bu=reg_bu,
        reg_bi=reg_bi,
        reg_pu=reg_pu,
        reg_qi=reg_qi,
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


def f_beta(test_rmse, train_rmse, beta: float = 2.0):
    return (
        (1 + beta ** 2)
        * (test_rmse * train_rmse)
        / (beta ** 2 * test_rmse + train_rmse)
    )


# Finally, get experiments, and train your models:
for experiment in opt.get_experiments():
    # get parameters:
    (test_rmse, train_rmse, fit_time, test_time) = fit(
        n_factors=experiment.get_parameter("n_factors"),
        n_epochs=experiment.get_parameter("n_epochs"),
        biased=experiment.get_parameter("biased"),
        init_mean=experiment.get_parameter("init_mean"),
        init_std_dev=experiment.get_parameter("init_std_dev"),
        lr_bu=experiment.get_parameter("lr_bu"),
        lr_bi=experiment.get_parameter("lr_bi"),
        lr_pu=experiment.get_parameter("lr_pu"),
        lr_qi=experiment.get_parameter("lr_qi"),
        reg_bu=experiment.get_parameter("reg_bu"),
        reg_bi=experiment.get_parameter("reg_bi"),
        reg_pu=experiment.get_parameter("reg_pu"),
        reg_qi=experiment.get_parameter("reg_qi"),
    )

    experiment.log_metric("test_rmse", test_rmse)
    experiment.log_metric("train_rmse", train_rmse)
    experiment.log_metric("fit_time", fit_time)
    experiment.log_metric("test_time", test_time)
    experiment.add_tags(["SVD", "ML-100k", "CV-5"])
    experiment.end()
