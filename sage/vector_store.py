"""Vector store abstraction and implementations."""

from abc import ABC, abstractmethod
from functools import cached_property
from typing import Dict, Generator, List, Tuple

import marqo
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_community.vectorstores import Marqo
from langchain_community.vectorstores import Pinecone as LangChainPinecone
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder

from sage.constants import TEXT_FIELD

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
    def as_retriever(self, top_k: int):
        """Converts the vector store to a LangChain retriever object."""


class PineconeVectorStore(VectorStore):
    """Vector store implementation using Pinecone."""

    def __init__(self, index_name: str, namespace: str, dimension: int, hybrid: bool = True):
        self.index_name = index_name
        self.dimension = dimension
        self.client = Pinecone()
        self.namespace = namespace
        self.hybrid = hybrid
        # The default BM25 encoder was fit in the MS MARCO dataset.
        # See https://docs.pinecone.io/guides/data/encode-sparse-vectors
        # In the future, we should fit the encoder on the current dataset. It's somewhat non-trivial for large datasets,
        # because most BM25 implementations require the entire dataset to fit in memory.
        self.bm25_encoder = BM25Encoder.default() if hybrid else None

    @cached_property
    def index(self):
        self.ensure_exists()
        index = self.client.Index(self.index_name)

        # Hack around the fact that PineconeRetriever expects the content of the chunk to be in a "text" field,
        # while PineconeHybridSearchRetrieve expects it to be in a "context" field.
        original_query = index.query

        def patched_query(*args, **kwargs):
            result = original_query(*args, **kwargs)
            for res in result["matches"]:
                res["metadata"]["context"] = res["metadata"][TEXT_FIELD]
            return result

        index.query = patched_query
        return index

    def ensure_exists(self):
        if self.index_name not in self.client.list_indexes().names():
            self.client.create_index(
                name=self.index_name,
                dimension=self.dimension,
                # See https://www.pinecone.io/learn/hybrid-search-intro/
                metric="dotproduct" if self.hybrid else "cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

    def upsert_batch(self, vectors: List[Vector]):
        pinecone_vectors = []
        for i, (metadata, embedding) in enumerate(vectors):
            vector = {"id": metadata.get("id", str(i)), "values": embedding, "metadata": metadata}
            if self.bm25_encoder:
                vector["sparse_values"] = self.bm25_encoder.encode_documents(metadata[TEXT_FIELD])
            pinecone_vectors.append(vector)

        self.index.upsert(vectors=pinecone_vectors, namespace=self.namespace)

    def as_retriever(self, top_k: int):
        if self.bm25_encoder:
            return PineconeHybridSearchRetriever(
                embeddings=OpenAIEmbeddings(),
                sparse_encoder=self.bm25_encoder,
                index=self.index,
                namespace=self.namespace,
                top_k=top_k,
                alpha=0.5,
            )

        return LangChainPinecone.from_existing_index(
            index_name=self.index_name, embedding=OpenAIEmbeddings(), namespace=self.namespace
        ).as_retriever(search_kwargs={"k": top_k})


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

    def as_retriever(self, top_k: int):
        vectorstore = Marqo(client=self.client, index_name=self.index_name)

        # Monkey-patch the _construct_documents_from_results_without_score method to not expect a "metadata" field in
        # the result, and instead take the "filename" directly from the result.
        def patched_method(self, results):
            documents: List[Document] = []
            for result in results["hits"]:
                content = result.pop(TEXT_FIELD)
                documents.append(Document(page_content=content, metadata=result))
            return documents

        vectorstore._construct_documents_from_results_without_score = patched_method.__get__(
            vectorstore, vectorstore.__class__
        )
        return vectorstore.as_retriever(search_kwargs={"k": top_k})


def build_from_args(args: dict) -> VectorStore:
    """Builds a vector store from the given command-line arguments."""
    if args.vector_store_type == "pinecone":
        dimension = args.embedding_size if "embedding_size" in args else None
        return PineconeVectorStore(
            index_name=args.index_name, namespace=args.repo_id, dimension=dimension, hybrid=args.hybrid_retrieval
        )
    elif args.vector_store_type == "marqo":
        return MarqoVectorStore(url=args.marqo_url, index_name=args.index_name)
    else:
        raise ValueError(f"Unrecognized vector store type {args.vector_store_type}")
