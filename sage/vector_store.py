"""Vector store abstraction and implementations."""

import os
import logging
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Dict, Generator, List, Optional, Tuple

import marqo
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_community.vectorstores import Marqo
from langchain_community.vectorstores import Pinecone as LangChainPinecone
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from nltk.data import find
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder

from sage.constants import TEXT_FIELD
from sage.data_manager import DataManager

Vector = Tuple[Dict, List[float]]  # (metadata, embedding)

def is_punkt_downloaded():
    try:
        find('tokenizers/punkt_tab')
        return True
    except LookupError:
        return False

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
    def as_retriever(self, top_k: int, embeddings: Embeddings):
        """Converts the vector store to a LangChain retriever object."""


class PineconeVectorStore(VectorStore):
    """Vector store implementation using Pinecone."""

    def __init__(self, index_name: str, namespace: str, dimension: int, alpha: float, bm25_cache: Optional[str] = None):
        """
        Args:
            index_name: The name of the Pinecone index to use. If it doesn't exist already, we'll create it.
            namespace: The namespace within the index to use.
            dimension: The dimension of the vectors.
            alpha: The alpha parameter for hybrid search: alpha == 1.0 means pure dense search, alpha == 0.0 means pure
                BM25, and 0.0 < alpha < 1.0 means a hybrid of the two.
            bm25_cache: The path to the BM25 encoder file. If not specified, we'll use the default BM25 (fitted on the
                MS MARCO dataset).
        """
        self.index_name = index_name
        self.dimension = dimension
        self.client = Pinecone()
        self.namespace = namespace
        self.alpha = alpha

        if alpha < 1.0:
            if bm25_cache and os.path.exists(bm25_cache):
                logging.info("Loading BM25 encoder from cache.")
                # We need nltk tokenizers for bm25 tokenization
                if is_punkt_downloaded():
                    print("punkt is already downloaded")
                else:
                    print("punkt is not downloaded")
                    # Optionally download it
                    nltk.download('punkt_tab')
                self.bm25_encoder = BM25Encoder()
                self.bm25_encoder.load(path=bm25_cache)
            else:
                logging.info("Using default BM25 encoder (fitted to MS MARCO).")
                self.bm25_encoder = BM25Encoder.default()
        else:
            self.bm25_encoder = None

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
                metric="dotproduct" if self.bm25_encoder else "cosine",
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

    def as_retriever(self, top_k: int, embeddings: Embeddings):
        if self.bm25_encoder:
            return PineconeHybridSearchRetriever(
                embeddings=embeddings,
                sparse_encoder=self.bm25_encoder,
                index=self.index,
                namespace=self.namespace,
                top_k=top_k,
                alpha=self.alpha,
            )

        return LangChainPinecone.from_existing_index(
            index_name=self.index_name, embedding=embeddings, namespace=self.namespace
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

    def as_retriever(self, top_k: int, embeddings: Embeddings = None):
        del embeddings  # Unused; The Marqo vector store is also an embedder.
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


def build_vector_store_from_args(args: dict, data_manager: Optional[DataManager] = None) -> VectorStore:
    """Builds a vector store from the given command-line arguments.

    When `data_manager` is specified and hybrid retrieval is requested, we'll use it to fit a BM25 encoder on the corpus
    of documents.
    """
    if args.vector_store_provider == "pinecone":
        bm25_cache = os.path.join(".bm25_cache", args.index_namespace, "bm25_encoder.json")

        if not os.path.exists(bm25_cache) and data_manager:
            logging.info("Fitting BM25 encoder on the corpus...")
            corpus = [content for content, _ in data_manager.walk()]
            bm25_encoder = BM25Encoder()
            bm25_encoder.fit(corpus)
            # Make sure the folder exists, before we dump the encoder.
            bm25_folder = os.path.dirname(bm25_cache)
            if not os.path.exists(bm25_folder):
                os.makedirs(bm25_folder)
            bm25_encoder.dump(bm25_cache)

        return PineconeVectorStore(
            index_name=args.pinecone_index_name,
            namespace=args.index_namespace,
            dimension=args.embedding_size if "embedding_size" in args else None,
            alpha=args.retrieval_alpha,
            bm25_cache=bm25_cache,
        )
    elif args.vector_store_provider == "marqo":
        return MarqoVectorStore(url=args.marqo_url, index_name=args.index_namespace)
    else:
        raise ValueError(f"Unrecognized vector store type {args.vector_store_provider}")
