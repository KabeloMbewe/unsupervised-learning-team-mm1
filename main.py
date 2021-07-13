from surprise import SVD, SVDpp
from surprise import Dataset, Reader
from surprise import accuracy
from surprise.model_selection import cross_validate, train_test_split

reader = Reader(
    name=None,
    line_format="user item rating",
    sep=",",
    rating_scale=(1, 5),
    skip_lines=1,
)

print("Loading data...")
data = Dataset.load_from_file("data/train.csv", reader)

print("Building sets...")
train_set, test_set = train_test_split(data, test_size=0.25)

algo = SVDpp(n_factors=1)

print("Fitting...")
algo.fit(train_set)

print("Testing...")
predictions = algo.test(testset)
print("RMSE:", accuracy.rmse(predictions))

# print("Running cross-validation")
# cross_validate(
#     algo=algo,
#     data=data,
#     measures=["RMSE", "MAE"],
#     cv=1,
#     return_train_measures=True,
#     verbose=True,
#     n_jobs=1,
#     pre_dispatch=2,
# )
