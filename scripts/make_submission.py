from surprise.dump import load
import pandas as pd

print("Loading model...")
_, algo = load("model.pkl")

print("Loading prediction data...")
test = pd.read_csv("data/test.csv")

preds = dict()

print("Predicting...")
total = len(test)

for idx, (uid, mid) in enumerate(test.to_records(index=False)):
    _, _, _, est, details = algo.predict(str(uid), str(mid))

    if details["was_impossible"]:
        print(idx, "Impossible")

    pred_id = f"{uid}_{mid}"
    preds[pred_id] = est

    if ((idx + 1) % 343_000) == 0:
        print("[progress]", round(idx / total * 100.0, 1), "%")

print("Producing output csv...")
pd.Series(preds, name="rating", index=preds.keys()).to_csv(
    "submission.csv", index_label="Id"
)
