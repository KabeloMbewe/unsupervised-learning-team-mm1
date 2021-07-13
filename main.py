from surprise import SVD, SVDpp
from surprise import Dataset, Reader
from surprise import accuracy
from surprise.model_selection import cross_validate, train_test_split

from surprise.dump import load
import pandas as pd

print("Loading model...")
algo: SVD = load("model.pkl")

print("Loading prediction data...")
test = pd.read_csv("data/test.csv")

preds = dict()

print("Predicting...")
total = len(test)

for idx, (uid, mid) in enumerate(test.to_records(index=False)):
    prediction = algo.predict(uid, mid).est

    pred_id = f"{uid}_{mid}"
    preds[pred_id] = prediction

    if ((idx + 1) % 10_000) == 0:
        print("[progress]", round(idx / total * 100.0, 1), "%")

pd.Series(preds).to_csv("submission.csv")
