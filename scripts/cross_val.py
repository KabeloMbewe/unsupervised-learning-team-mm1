from surprise import SVD, SVDpp
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate, train_test_split
import datetime
import time

print("Loading data...")
data = Dataset.load_builtin(name="ml-100k")

full_ts = data.build_full_trainset()


start = time.time()

algo = SVD(
    random_state=42,
    verbose=True,
)


cross_validate(
    algo=algo,
    data=data,
    measures=["rmse", "mae"],
    cv=3,
    n_jobs=-1,
    verbose=True,
)

duration = round(time.time() - start, 1)
print(f"Done in {duration} s")
