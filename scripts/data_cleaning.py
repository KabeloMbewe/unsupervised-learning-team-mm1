import numpy as np
import pandas as pd

from helpers import runParallel

# read csv files
def clean_genome():
    genome_scores = pd.read_csv("data/genome-scores.csv")
    genome_tags = pd.read_csv("data/genome-tags.csv")

    genome_data = genome_scores.merge(genome_tags, on="tagId")

    genome_tag_vec = genome_data.pivot(
        index="movieId", columns="tag", values="relevance"
    )
    genome_tag_vec["movieId"] = genome_tag_vec.index

    cols = ["movieId"] + [col for col in genome_tag_vec.columns if col != "movieId"]
    genome_tag_vec = genome_tag_vec[cols]
    genome_tag_vec.reset_index(inplace=True, drop=True)
    genome_tag_vec.to_csv(
        "cleaned/genome_tag_vec.csv",
        index=False,
    )


def clean_imdb():
    imdb_data = pd.read_csv("data/imdb_data.csv")
    budget_to_num = lambda budget: int("".join([c for c in budget if c.isnumeric()]))

    imdb_data["budget"] = imdb_data["budget"].apply(
        lambda b: np.nan if b is np.nan or b[0] != "$" else budget_to_num(b)
    )
    imdb_data.title_cast.fillna("", inplace=True)
    imdb_data.director.fillna("", inplace=True)
    imdb_data.runtime.fillna(int(imdb_data.runtime.median()), inplace=True)
    imdb_data.budget.fillna(int(imdb_data.budget.median()), inplace=True)
    imdb_data.plot_keywords.fillna("", inplace=True)

    imdb_data.to_csv(
        "cleaned/imdb_data.csv",
        index=False,
    )


def clean_movies():
    movies = pd.read_csv("data/movies.csv")

    movies["year"] = (
        movies.title.str.strip()
        .str[-6:]
        .str.extract(r"\(([0-9]{4})\)")
        .fillna(0)
        .astype(int)
    )

    normal_year_mean = movies[movies.year != 0].year.median()
    movies["year"].replace(0, normal_year_mean, inplace=True)

    movies["title"] = movies.title.str.strip().str.replace(
        r"\(([0-9]{4})\)", "", regex=True
    )
    movies["title"] = (
        movies["title"].str.strip().str.replace(r"(.*), The$", r"The \1", regex=True)
    )

    movies["title"] = movies["title"] + movies.year.apply(lambda y: f" ({int(y)})")

    movies["genres"] = movies.genres.str.replace(
        "(no genres listed)", "unknown", regex=False
    )

    movies.to_csv(
        "cleaned/movies.csv",
        index=False,
    )


def build_full_combine():
    train = pd.read_csv("data/train.csv")
    movies = pd.read_csv("cleaned/movies.csv")
    imdb_data = pd.read_csv("cleaned/imdb_data.csv")
    genome_tag_vec = pd.read_csv("cleaned/genome_tag_vec.csv")

    full_combine = (
        movies.merge(imdb_data, on="movieId", how="left")
        .merge(genome_tag_vec, on="movieId", how="left")
        .set_index("movieId", drop=True)
    )

    normal_year_mean = full_combine[full_combine.year != 0].year.mean()
    full_combine["year"].replace(0, normal_year_mean, inplace=True)

    # movies.year.fillna(int(movies.year.median()), inplace=True)
    # movies.genres.fillna("<unknown>", inplace=True)
    full_combine.title = full_combine.title.str.strip().str.replace(
        r"(.*), The$", r"The \1", regex=True
    )
    full_combine.title_cast.fillna(" ", inplace=True)
    full_combine.director.fillna(" ", inplace=True)
    full_combine.runtime.fillna(int(full_combine.runtime.median()), inplace=True)
    full_combine.budget.fillna(int(full_combine.budget.median()), inplace=True)
    full_combine.plot_keywords.fillna(" ", inplace=True)

    full_combine["cast_size"] = full_combine.title_cast.str.split("|").apply(len)
    full_combine["genre_count"] = full_combine.genres.str.split("|").apply(len)

    movie_groups = train.set_index("movieId", drop=True).groupby("movieId")
    full_combine["rating_mean"] = movie_groups.rating.mean()
    full_combine["rating_std"] = movie_groups.rating.std().fillna(0)

    q3 = movie_groups.rating.quantile(0.75)
    q1 = movie_groups.rating.quantile(0.25)

    full_combine["rating_iqr"] = q3 - q1
    full_combine["rating_count"] = movie_groups.apply(len)

    full_combine.fillna(0, inplace=True)

    full_combine["movieId"] = full_combine.index.values
    full_combine.reset_index(inplace=True, drop=True)
    full_combine.to_csv(
        "cleaned/full_combine.csv",
        index=False,
    )


if __name__ == "__main__":
    print("Cleaning...")
    runParallel([clean_genome, clean_imdb, clean_movies])
    print("Generating full-combine file")
    build_full_combine()
    print("Done!")
