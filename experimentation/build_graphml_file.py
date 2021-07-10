import time
import datetime

import igraph as ig
import numpy as np
import pandas as pd
import plotly.express as px

from helpers import flatten_list

from functools import lru_cache


random_state = 42


def get_user_movie_graph():
    # try:
    #     graph = ig.read("generated/user_movie_graph.graphml", format="graphml")
    #     return graph
    # except:
    #     print("Building graph fresh...")

    graph = ig.Graph()

    data = pd.read_csv("data/train.csv")
    data["userId"] = data.userId.apply(lambda x: f"user_{x}")
    data["movieId"] = data.movieId.apply(lambda x: f"movie_{x}")

    print("[graph] Adding users")
    users = data.userId.unique()
    graph.add_vertices(n=users)

    print("[graph] Adding movies")
    movies = data.movieId.unique()
    graph.add_vertices(n=movies)

    print("[graph] Calculating edges...")
    edges = data[["userId", "movieId"]].to_records(index=False)
    print(f"[graph] Adding {len(edges)} edges user <-> movie")
    graph.add_edges(es=edges, attributes={"rating": data["rating"].values})

    print("Saving Graph")
    graph.write("generated/user_movie_graph.graphml", format="graphml")
    return graph


def get_movie_prop_graph():
    # try:
    #     graph = ig.read("generated/movie_prop_graph.graphml", format="graphml")
    #     return graph
    # except:
    #     print("Building graph fresh...")

    print("Building dataset...")
    movies = pd.read_csv("cleaned/movies.csv")
    imdb_data = pd.read_csv("cleaned/imdb_data.csv")

    data = movies.merge(imdb_data, on="movieId", how="left")
    data["movieId"] = data.movieId.apply(lambda x: f"movie_{x}")
    data["year"] = data.movieId.apply(lambda x: f"year_{x}")

    graph = ig.Graph()

    print("[graph] Adding movies")
    graph.add_vertices(
        n=data.movieId, attributes=data[["title", "director", "budget", "runtime"]]
    )

    data["genres"] = data.genres.str.lower().str.split("|")
    genres = list(map(lambda g: f"genre_{g}", set(flatten_list(data.genres.tolist()))))
    print("[graph] Adding genres")
    graph.add_vertices(n=genres)
    mg_map = data[["movieId", "genres"]].explode("genres")
    mg_map["genres"] = mg_map.genres.apply(lambda g: f"genre_{g}")
    graph.add_edges(mg_map.to_records(index=False))

    data["plot_keywords"] = data.plot_keywords.dropna().str.lower().str.split("|")
    plot_keywords = list(
        map(lambda kw: f"kw_{kw}", set(flatten_list(data.plot_keywords.dropna())))
    )
    print("[graph] Adding plot keywords")
    graph.add_vertices(n=plot_keywords)
    mpk_map = data[["movieId", "plot_keywords"]].explode("plot_keywords").dropna()
    mpk_map["plot_keywords"] = mpk_map.plot_keywords.apply(lambda x: f"kw_{x}")
    graph.add_edges(mpk_map.to_records(index=False))

    data["title_cast"] = data.title_cast.dropna().str.lower().str.split("|")
    title_cast = list(
        map(lambda a: f"actor_{a}", set(flatten_list(data.title_cast.dropna())))
    )
    print("[graph] Adding actors")
    graph.add_vertices(n=title_cast)
    mtc_map = data[["movieId", "title_cast"]].explode("title_cast").dropna()
    mtc_map["title_cast"] = mtc_map.title_cast.apply(lambda x: f"actor_{x}")
    graph.add_edges(mtc_map.to_records(index=False))

    years = data.year.unique()
    print("[graph] Adding years")
    graph.add_vertices(n=years)
    graph.add_edges(data[["movieId", "year"]].to_records(index=False))

    print("Saving Graph")
    graph.write("generated/movie_prop_graph.graphml", format="graphml")
    return graph

    return graph


# um_graph = get_user_movie_graph()
# # mp_graph = get_movie_prop_graph()
# # avengers = 89745
# # avengers2 = 122892
# # amazing_spiderman = 95510
# # fault_in_stars = 111921
# print(um_graph.get_shortest_paths("movie_89745", to="movie_122892"))
# # print(mp_graph.vs[62432])

# # print(um_graph.vs.find(name="user_21").neighbors())
