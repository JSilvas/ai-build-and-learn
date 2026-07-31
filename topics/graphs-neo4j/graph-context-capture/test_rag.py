from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import time

NEO4J_URI      = "bolt://localhost:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "password"
EMBED_MODEL    = "all-MiniLM-L6-v2"

print("Loading embedding model...")
embedder = SentenceTransformer(EMBED_MODEL)

print("Connecting to Neo4j...")
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

query_text = "What have I been working on?"
print(f"Querying for: '{query_text}'")
vec = embedder.encode(query_text).tolist()

t0 = time.time()
with driver.session() as session:
    result = session.run(
        "CALL db.index.vector.queryNodes('activity_embedding', 3, $vec) "
        "YIELD node, score "
        "RETURN node.text AS text, node.timestamp AS timestamp, score",
        vec=vec
    )
    hits = [dict(r) for r in result]
elapsed = time.time() - t0

print(f"Retrieved {len(hits)} results in {elapsed:.3f}s:")
for i, hit in enumerate(hits):
    print(f"{i+1}. [{hit['timestamp']}] (score: {hit['score']:.3f}) {hit['text'][:100]}...")

driver.close()
