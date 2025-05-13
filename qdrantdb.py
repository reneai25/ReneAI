# qdrantdb.py

from qdrant_client.http import models
import qdrant_client
import numpy as np

def create_qdrant_collection(client: qdrant_client.QdrantClient, collection_name: str, vector_size: int):
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE
        )
    )

def add_documents_to_qdrant(client: qdrant_client.QdrantClient, collection_name: str, documents: list[str], embedding_response):
    embeddings = embedding_response.embeddings

    # Prepare points
    points = [
        models.PointStruct(
            id=idx,
            vector=embedding.values,
            payload={"text": doc}
        )
        for idx, (doc, embedding) in enumerate(zip(documents, embeddings))
    ]

    # Insert into Qdrant
    client.upsert(
        collection_name=collection_name,
        points=points
    )

def query_qdrant(client: qdrant_client.QdrantClient, collection_name: str, query_vector: list[float], limit: int = 5):
    # Search for similar vectors
    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=limit
    )

    # Extract and return the text from the results
    return search_result