import igraph as ig
import nltk
import numpy as np
import pandas as pd
import plotly.express as px
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import (
    MinMaxScaler,
    Normalizer,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)

from helpers import flatten_list

random_state = 42
stopwords = stopwords.words("english")

genome_tag_vec = pd.read_csv("cleaned/genome_tag_vec.csv")
imdb_data = pd.read_csv("cleaned/imdb_data.csv")
movies = pd.read_csv("cleaned/movies.csv")
# tags = pd.read_csv("data/tags.csv")
test = pd.read_csv("data/test.csv")
train = pd.read_csv("data/train.csv")

full_combine = movies.merge(imdb_data, on="movieId", how="left")
normal_year_mean = full_combine[full_combine.year != 0].year.mean()
full_combine["year"].replace(0, normal_year_mean, inplace=True)

# movies.year.fillna(int(movies.year.median()), inplace=True)
# movies.genres.fillna("<unknown>", inplace=True)
full_combine.title_cast.fillna("", inplace=True)
full_combine.director.fillna("", inplace=True)
full_combine.runtime.fillna(int(full_combine.runtime.median()), inplace=True)
full_combine.budget.fillna(int(full_combine.budget.median()), inplace=True)
full_combine.plot_keywords.fillna("", inplace=True)

full_combine["cast_size"] = full_combine.title_cast.str.split("|").apply(len)
full_combine["genre_count"] = full_combine.genres.str.split("|").apply(len)

movie_groups = train.groupby("movieId")
full_combine["rating_mean"] = movie_groups.rating.mean()
full_combine["rating_std"] = movie_groups.rating.std()
full_combine["rating_iqr"] = movie_groups.rating.quantile(
    0.75
) - movie_groups.rating.quantile(0.25)
full_combine["rating_count"] = movie_groups.rating.count()

full_combine.fillna(0, inplace=True)

features = []

# title_vectrz = TfidfVectorizer(min_df=20, ngram_range=(1, 3), stop_words=stopwords, norm=None)
# title_vec = title_vectrz.fit_transform(full_combine.title)
# print("Title Tokens:", len(title_vectrz.get_feature_names()))
# features.extend(title_vectrz.get_feature_names())

genre_vectrz = TfidfVectorizer(token_pattern=r"[A-z\-]+", min_df=2, norm=None)
genre_vec = genre_vectrz.fit_transform(full_combine.genres)
print("Genre Tokens:", len(genre_vectrz.get_feature_names()))
features.extend(genre_vectrz.get_feature_names())

# cast_vectrz = TfidfVectorizer(token_pattern=r"[^\|]+", min_df=50, norm=None)
# cast_vec = cast_vectrz.fit_transform(full_combine.title_cast)
# print("Cast Tokens:", len(cast_vectrz.get_feature_names()))
# features.extend(cast_vectrz.get_feature_names())

# director_vectrz = TfidfVectorizer(token_pattern=r".+", min_df=5, stop_words=['see full summary'], norm=None)
# director_vec = director_vectrz.fit_transform(full_combine.director)
# print("Director Tokens:", len(director_vectrz.get_feature_names()))
# features.extend(director_vectrz.get_feature_names())

plot_vectrz = TfidfVectorizer(
    token_pattern=r"[^\|]+", min_df=20, stop_words=stopwords, norm=None
)
plot_vec = plot_vectrz.fit_transform(full_combine.plot_keywords)
print("Plot KW Tokens:", len(plot_vectrz.get_feature_names()))
features.extend(plot_vectrz.get_feature_names())

extra_features = ["year", "rating_mean", "rating_std", "rating_iqr", "rating_count"]
features.extend(extra_features)

extra_features = full_combine[extra_features]

scaler = MinMaxScaler()
transformed = scaler.fit_transform(extra_features)
std_extra_sparse = sparse.csr_matrix(transformed)

tfidf_vecs = sparse.hstack(
    [
        # title_vec,
        genre_vec,
        # cast_vec,
        # director_vec,
        plot_vec,
    ]
).tocsr()

vecs = sparse.hstack([tfidf_vecs, std_extra_sparse])
norm = Normalizer(copy=True)
norm_vecs = norm.transform(vecs)
norm_vecs.shape

movie_indexes = full_combine.movieId
mid_to_idx = pd.Series(movie_indexes.index.values, index=movie_indexes).to_dict()

mean_movie_rating_by_movie_id = train.groupby("movieId").rating.mean()

combined_data = pd.concat([train, test])
combined_data.drop("timestamp", axis=1, inplace=True)
combined_data.sort_values(["userId", "rating"], inplace=True)
combined_data.reset_index(inplace=True, drop=True)

from joblib import Parallel, delayed
from multiprocessing import cpu_count


def applyParallel(grouped, func):
    results = Parallel(n_jobs=cpu_count())(delayed(func)(group) for _, group in grouped)
    return pd.concat(results)


def remove_diag(x):
    x_no_diag = np.ndarray.flatten(x)
    x_no_diag = np.delete(x_no_diag, range(0, len(x_no_diag), len(x) + 1), 0)
    x_no_diag = x_no_diag.reshape(len(x), len(x) - 1)
    return x_no_diag


def predict_ratings(data):
    history = data[data.rating.notna()]
    to_predict = data[data.rating.isna()]

    to_predict_idx = data.rating.isna()

    if len(history) == 1:
        print(data.userId.unique())
        mean_guesses = mean_movie_rating_by_movie_id.loc[
            to_predict.movieId.values
        ].values
        print(mean_guesses)
        data.loc[to_predict_idx, "pred"] = mean_guesses
        return data

    if len(to_predict) == 0:
        return data

    idxs = [mid_to_idx[i] for i in data.movieId]

    historic_ratings = history.rating.values

    similarity_matrix = remove_diag(cosine_similarity(norm_vecs[idxs, :]))
    sim_matrix = similarity_matrix[len(history) :, : len(history)]

    sim_totals = sim_matrix.sum(axis=1)
    ratings = np.tile(history.rating, (len(to_predict.rating), 1))
    weighted_sums = np.einsum("ij,ij->i", sim_matrix, ratings)

    data.loc[to_predict_idx, "pred"] = weighted_sums / sim_totals
    return data


print("Making parallel predictions...")
parallel_preds = applyParallel(combined_data.groupby("userId"), predict_ratings)

print(parallel_preds)


print("Making predictions...")
predictions_serial = combined_data.groupby("userId").apply(predict_ratings)

print(parallel_preds[parallel_preds != predictions_serial])
