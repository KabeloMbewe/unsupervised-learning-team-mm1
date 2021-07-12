from surprise import SVD, SVDpp
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate

reader = Reader(
    name=None,
    line_format="user item rating",
    sep=",",
    rating_scale=(1, 5),
    skip_lines=1,
)

print("Loading data...")
data = Dataset.load_from_file("data/train.csv", reader)

algo = SVD()

print("Running cross-validation")
cross_validate(
    algo=algo,
    data=data,
    measures=["RMSE", "MAE"],
    cv=5,
    return_train_measures=True,
    verbose=True,
    n_jobs=-2,
    pre_dispatch=8,
)
