from surprise import SVD, SVDpp
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
    n_factors=10,
    n_epochs=46,
    biased=True,
    init_mean=3.0352,
    init_std_dev=0.3539,
    lr_bu=0.008,
    lr_bi=0.0204,
    lr_pu=0.0176,
    lr_qi=0.0173,
    reg_bu=0.0896,
    reg_bi=0.0505,
    reg_pu=0.077,
    reg_qi=0.0928,
)

print("Fitting...")
algo.fit(train_set)

print("Dumping...")
dump("model.pkl", algo=algo, verbose=1)
