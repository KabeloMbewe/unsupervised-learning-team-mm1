import pandas as pd

data = pd.read_csv("data/train.csv")
data.drop("timestamp", axis=1, inplace=True)
data.sort_values(["userId", "movieId"], inplace=True)

data.to_csv("data/RATED.csv", index=False)

users = pd.Series(data.userId.unique(), name="User")
users.to_csv("data/User.csv", index=False)

movies = pd.Series(data.movieId.unique(), name="Movie")
movies.to_csv("data/Movie.csv", index=False)

print("Done")

# now use the bulk loader to load these files
print("Now run this command in the data directory:")
print(
    "redisgraph-bulk-loader UserMovieGraph --enforce-schema --nodes User.csv --nodes Movie.csv --relations RATED.csv"
)
