"""Vector store abstraction and implementations."""

from abc import ABC, abstractmethod
from typing import Dict, Generator, List, Tuple

import marqo
from langchain_community.vectorstores import Marqo
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone

OPENAI_EMBEDDING_SIZE = 1536
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

    @abstractmethod
    def to_langchain(self):
        """Converts the vector store to a LangChain vector store object."""


class PineconeVectorStore(VectorStore):
    """Vector store implementation using Pinecone."""

    def __init__(self, index_name: str, namespace: str, dimension: int = OPENAI_EMBEDDING_SIZE):
        self.index_name = index_name
        self.dimension = dimension
        self.client = Pinecone()
        self.index = self.client.Index(self.index_name)
        self.namespace = namespace

    def ensure_exists(self):
        if self.index_name not in self.client.list_indexes().names():
            self.client.create_index(name=self.index_name, dimension=self.dimension, metric="cosine")

    def upsert_batch(self, vectors: List[Vector]):
        pinecone_vectors = [
            (metadata.get("id", str(i)), embedding, metadata) for i, (metadata, embedding) in enumerate(vectors)
        ]
        self.index.upsert(vectors=pinecone_vectors, namespace=self.namespace)

    def to_langchain(self):
        return Pinecone.from_existing_index(
            index_name=self.index_name, embedding=OpenAIEmbeddings(), namespace=self.namespace
        )


class MarqoVectorStore(VectorStore):
    """Vector store implementation using Marqo."""

    def __init__(self, url: str, index_name: str):
        self.client = marqo.Client(url=url)
        self.index_name = index_name

    def ensure_exists(self):
        pass

    def upsert_batch(self, vectors: List[Vector]):
        # Since Marqo is both an embedder and a vector store, the embedder is already doing the upsert.
        pass

    def to_langchain(self):
        vectorstore = Marqo(client=self.client, index_name=self.index_name)

        # Monkey-patch the _construct_documents_from_results_without_score method to not expect a "metadata" field in
        # the result, and instead take the "filename" directly from the result.
        def patched_method(self, results):
            documents: List[Document] = []
            for res in results["hits"]:
                documents.append(Document(page_content=res["text"], metadata={"filename": res["filename"]}))
            return documents

        vectorstore._construct_documents_from_results_without_score = patched_method.__get__(
            vectorstore, vectorstore.__class__
        )
        return vectorstore


def build_from_args(args: dict) -> VectorStore:
    """Builds a vector store from the given command-line arguments."""
    if args.vector_store_type == "pinecone":
        return PineconeVectorStore(index_name=args.index_name, namespace=args.repo_id)
    elif args.vector_store_type == "marqo":
        return MarqoVectorStore(url=args.marqo_url, index_name=args.index_name)
    else:
        raise ValueError(f"Unrecognized vector store type {args.vector_store_type}")
