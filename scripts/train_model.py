from surprise import SVD
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate, train_test_split

from surprise.dump import dump


random_state = 42

reader = Reader(
    name=None,
    line_format="user item rating",
    sep=",",
    rating_scale=(0.5, 5),
    skip_lines=1,
)

print("Loading data...")
data = Dataset.load_from_file("data/train.csv", reader)

print("Building training file")
train_set = data.build_full_trainset()

del data
del reader

algo = SVD(
    n_factors=200,
    n_epochs=30,
    lr_all=0.005,
    reg_all=0.02,
    init_std_dev=0.02,
    random_state=42,
    verbose=True,
)

print("Fitting...")
algo.fit(train_set)

print("Dumping...")
dump("model.pkl", algo=algo, verbose=1)
