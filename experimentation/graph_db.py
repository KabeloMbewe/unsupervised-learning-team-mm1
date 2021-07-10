from neo4j import GraphDatabase, Transaction
import logging
from neo4j.exceptions import ServiceUnavailable

from helpers import time_steps


class GraphDB:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.define_constraints()

    def close(self):
        # Don't forget to close the driver connection when you are finished with it
        self.driver.close()

    def clear_db(self):
        with self.driver.session() as session:
            session.write_transaction(self._clear_db)

    @staticmethod
    def _clear_db(tx: Transaction):
        query = (
            # find all nodes
            "MATCH (n) "
            # delete all found nodes
            "DETACH DELETE n"
        )
        return tx.run(query)

    def create_user(self, uid):
        with self.driver.session() as session:
            result = session.write_transaction(self._create_user, uid)
            for row in result:
                print(row)

    @staticmethod
    def _create_user(tx: Transaction, uid: str):
        query = (
            # add new node
            "CREATE (u:User {uid: $uid}) "
            "RETURN u"
        )
        result = tx.run(query, uid=uid)
        return result

    def create_movie_review(self, userId, movieId, rating):
        with self.driver.session() as session:
            result = session.write_transaction(
                self._create_movie_review, userId, movieId, rating
            )

            for row in result:
                print(row)

    @staticmethod
    def _create_movie_review(tx: Transaction, uid, mid, rating):
        query = (
            # create the user
            "MERGE (u:User {userId: $uid}) "
            # create the movie
            "MERGE (m:Movie {movieId: $mid}) "
            # add relationship
            "MERGE (u)-[:RATED {rating: $rating}]->(m) "
        )
        return tx.run(query, uid=uid, mid=mid, rating=rating)

    def define_constraints(self):
        with self.driver.session() as session:
            session.run(
                "CREATE CONSTRAINT unique_uid IF NOT EXISTS ON (u:User) ASSERT u.userId IS UNIQUE;"
            )
            session.run(
                "CREATE CONSTRAINT unique_mid IF NOT EXISTS ON (m:Movie) ASSERT m.movieId IS UNIQUE;"
            )
            session.run(
                "CREATE CONSTRAINT unique_director IF NOT EXISTS ON (d:Director) ASSERT d.name IS UNIQUE;"
            )

    def batch_insert_csv(self, filepath):
        query = (
            # load csv file
            "USING PERIODIC COMMIT 5000 "
            "LOAD CSV WITH HEADERS FROM $path AS record "
            # "WITH record LIMIT 10000 "
            # transform lines
            "MERGE (user:User {userId: toInteger(record.userId)}) "
            "MERGE (movie:Movie {movieId: toInteger(record.movieId)}) "
            """ON CREATE SET 
                movie.title = record.title, 
                movie.year = toInteger(record.year), 
                movie.genres = split(record.genres, '|'),
                movie.title_cast = split(record.title_cast, '|')
            """
            "MERGE (director:Director {name: record.director}) "
            # add relationships
            "MERGE (user)-[:RATED {rating: toFloat(record.rating)}]->(movie) "
            "MERGE (director)-[:DIRECTED]->(movie) "
        )
        with self.driver.session() as session:
            session.run(query, path=filepath)


graphdb = GraphDB(uri="bolt://localhost:7687", user="neo4j", password="test1234")

time_steps(
    steps=[
        (graphdb.clear_db,),
        (graphdb.batch_insert_csv, "file:///combined.csv"),
    ]
)

graphdb.close()
