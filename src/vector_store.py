"""Vector store abstraction and implementations."""

from abc import ABC, abstractmethod
from typing import Dict, Generator, List, Tuple

from pinecone import Pinecone

Vector = Tuple[Dict, List[float]]  # (metadata, embedding)


class VectorStore(ABC):
    """Abstract class for a vector store."""
    @abstractmethod
    def ensure_exists(self):
        """Ensures that the vector store exists. Creates it if it doesn't."""

    @abstractmethod
    def upsert_batch(self, vectors: List[Vector]):
        """Upserts a batch of vectors."""

    def upsert(self, vectors: Generator[Vector, None, None]):
        """Upserts in batches of 100, since vector stores have a limit on upsert size."""
        batch = []
        for metadata, embedding in vectors:
            batch.append((metadata, embedding))
            if len(batch) == 100:
                self.upsert_batch(batch)
                batch = []
        if batch:
            self.upsert_batch(batch)


class PineconeVectorStore(VectorStore):
    """Vector store implementation using Pinecone."""

    def __init__(self, index_name: str, dimension: int, namespace: str):
        self.index_name = index_name
        self.dimension = dimension
        self.client = Pinecone()
        self.index = self.client.Index(self.index_name)
        self.namespace = namespace

    def ensure_exists(self):
        if self.index_name not in self.client.list_indexes().names():
            self.client.create_index(
                name=self.index_name, dimension=self.dimension, metric="cosine"
            )

    def upsert_batch(self, vectors: List[Vector]):
        pinecone_vectors = [
            (metadata.get("id", str(i)), embedding, metadata)
            for i, (metadata, embedding) in enumerate(vectors)
        ]
        self.index.upsert(vectors=pinecone_vectors, namespace=self.namespace)
