from qdrant_client import QdrantClient

client = QdrantClient("localhost", port=6333)

# 1. Delete the old 1024-dim collection
client.delete_collection(collection_name="rag_chunks")

# 2. Create the new 512-dim collection
from qdrant_client.http.models import Distance, VectorParams

client.create_collection(
    collection_name="rag_chunks",
    vectors_config=VectorParams(size=512, distance=Distance.COSINE),
)