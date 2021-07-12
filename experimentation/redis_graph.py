import redis
from redisgraph import Node, Edge, Graph, Path


r = redis.Redis(host="localhost", port=6379)

graph = Graph("UserMovieGraph", r)

# query = """MATCH (root:User)-[r1:RATED]->(:Movie)<-[r2:RATED]-(u:User)
# WHERE abs(r1.rating - r2.rating) < 0.5
# RETURN root.uid, count(u) AS cnt
# ORDER BY cnt ASC"""

# community_rate_query = """
# MATCH (root:User)-[r1:RATED]->(m:Movie)<-[r2:RATED]-(u2:User)-[rt:RATED]->(mt:Movie {mid: $movieTarget})
# WHERE root.uid = $rootId AND abs(r1.rating - r2.rating) <= $threshold
# RETURN root.uid, avg(rt.rating) AS AVG_RATING, stDev(rt.rating) AS STD, count(rt) AS USERS, count(DISTINCT u2.uid)
# ORDER BY USERS
# """

community_rate_query = """
MATCH (root:User)-[r1:RATED]->(m:Movie)<-[r2:RATED]-(u2:User)-[rt:RATED]->(mt:Movie)
WHERE root.uid = $rootId AND mt.mid = $movieTarget AND NOT (root = u2) AND abs(r1.rating - r2.rating) <= $threshold
RETURN DISTINCT root.uid, u2.uid
"""

result = graph.query(
    community_rate_query,
    read_only=True,
    params={"rootId": "21", "movieTarget": "96691", "threshold": 0.5},
)

# Print resultset
result.pretty_print()

# Use parameters
# params = {"purpose": "pleasure"}
# query = """MATCH (p:person)-[v:visited {purpose:$purpose}]->(c:country)
# 		   RETURN p.name, p.age, v.purpose, c.name"""

# result = redis_graph.query(query, params)

# # Print resultset
# result.pretty_print()

# # Use query timeout to raise an exception if the query takes over 10 milliseconds
# result = redis_graph.query(query, params, timeout=10)

# # Iterate through resultset
# for record in result.result_set:
#     person_name = record[0]
#     person_age = record[1]
#     visit_purpose = record[2]
#     country_name = record[3]

# query = """MATCH p = (:person)-[:visited {purpose:"pleasure"}]->(:country) RETURN p"""

# result = redis_graph.query(query)

# # Iterate through resultset
# for record in result.result_set:
#     path = record[0]
#     print(path)


# # All done, remove graph.
# redis_graph.delete()
