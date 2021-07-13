from surprise import SVD, SVDpp
from surprise import Dataset, Reader
from surprise import accuracy
from surprise.model_selection import cross_validate, train_test_split

from joblib import cpu_count, delayed, Parallel

from surprise.dump import load
import numpy as np
import pandas as pd

from typing import Callable

print("Loading model...")
# _, algo = load("model.pkl")

print("Loading prediction data...")
test = pd.read_csv("data/test.csv")

print("Predicting...")


def split_map_parallel(lst, func: Callable, n_jobs: int = -1):
    n_jobs = n_jobs if n_jobs > 0 else cpu_count()
    groups = np.array_split(lst, n_jobs)
    results = Parallel(n_jobs=n_jobs)(delayed(func)(lst_grp) for lst_grp in groups)
    return results


def preds_from_records(records):
    preds = dict()

    for uid, mid in records:
        _, _, _, est, details = algo.predict(str(uid), str(mid))

        if details["was_impossible"]:
            print(idx, "was impossible")

        pred_id = f"{uid}_{mid}"
        preds[pred_id] = est

    return preds


results = split_map_parallel(test.to_records(index=False), preds_from_records)

preds = dict()

for r in results:
    preds.update(r)


pd.Series(preds, name="rating", index=preds.keys()).to_csv(
    "submission.csv", index_label="Id"
)
