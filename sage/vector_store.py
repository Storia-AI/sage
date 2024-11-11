"""Vector store abstraction and implementations."""

import logging
import os
from abc import ABC, abstractmethod
from enum import Enum
from functools import cached_property
from typing import Dict, Generator, List, Optional, Tuple
from uuid import uuid4

import chromadb
import faiss
import marqo
import nltk
from langchain.retrievers import EnsembleRetriever
from langchain_chroma import Chroma as LangChainChroma
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS, Marqo
from langchain_community.vectorstores import Pinecone as LangChainPinecone
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore as LangChainQdrant
from langchain_voyageai import VoyageAIEmbeddings
from nltk.data import find
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from sage.constants import TEXT_FIELD
from sage.data_manager import DataManager

Vector = Tuple[Dict, List[float]]  # (metadata, embedding)


class VectorStoreProvider(Enum):
    PINECONE = "pinecone"
    MARQO = "marqo"
    CHROMA = "chroma"
    FAISS = "faiss"
    MILVUS = "milvus"
    QDRANT = "qdrant"


def is_punkt_downloaded():
    try:
        find("tokenizers/punkt_tab")
        return True
    except LookupError:
        return False


class VectorStore(ABC):
    """Abstract class for a vector store."""

    @abstractmethod
    def ensure_exists(self):
        """Ensures that the vector store exists. Creates it if it doesn't."""

    @abstractmethod
    def upsert_batch(self, vectors: List[Vector], namespace: str):
        """Upserts a batch of vectors."""

    def upsert(self, vectors: Generator[Vector, None, None], namespace: str):
        """Upserts in batches of 100, since vector stores have a limit on upsert size."""
        batch = []
        for metadata, embedding in vectors:
            batch.append((metadata, embedding))
            if len(batch) == 100:
                self.upsert_batch(batch, namespace)
                batch = []
        if batch:
            self.upsert_batch(batch, namespace)

    @abstractmethod
    def as_retriever(self, top_k: int, embeddings: Embeddings, namespace: str):
        """Converts the vector store to a LangChain retriever object."""


class PineconeVectorStore(VectorStore):
    """Vector store implementation using Pinecone."""

    def __init__(self, index_name: str, dimension: int, alpha: float, bm25_cache: Optional[str] = None):
        """
        Args:
            index_name: The name of the Pinecone index to use. If it doesn't exist already, we'll create it.
            dimension: The dimension of the vectors.
            alpha: The alpha parameter for hybrid search: alpha == 1.0 means pure dense search, alpha == 0.0 means pure
                BM25, and 0.0 < alpha < 1.0 means a hybrid of the two.
            bm25_cache: The path to the BM25 encoder file. If not specified, we'll use the default BM25 (fitted on the
                MS MARCO dataset).
        """
        self.index_name = index_name
        self.dimension = dimension
        self.client = Pinecone()
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
                    nltk.download("punkt_tab")
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
                if TEXT_FIELD in res["metadata"]:
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

    def upsert_batch(self, vectors: List[Vector], namespace: str):
        pinecone_vectors = []
        for i, (metadata, embedding) in enumerate(vectors):
            vector = {"id": metadata.get("id", str(i)), "values": embedding, "metadata": metadata}
            if self.bm25_encoder:
                vector["sparse_values"] = self.bm25_encoder.encode_documents(metadata[TEXT_FIELD])
            pinecone_vectors.append(vector)

        self.index.upsert(vectors=pinecone_vectors, namespace=namespace)

    def as_retriever(self, top_k: int, embeddings: Embeddings, namespace: str):
        bm25_retriever = (
            BM25Retriever(
                embeddings=embeddings,
                sparse_encoder=self.bm25_encoder,
                index=self.index,
                namespace=namespace,
                top_k=top_k,
            )
            if self.bm25_encoder
            else None
        )

        dense_retriever = LangChainPinecone.from_existing_index(
            index_name=self.index_name, embedding=embeddings, namespace=namespace
        ).as_retriever(search_kwargs={"k": top_k})

        if bm25_retriever:
            return EnsembleRetriever(retrievers=[dense_retriever, bm25_retriever], weights=[self.alpha, 1 - self.alpha])
        else:
            return dense_retriever


class ChromaVectorStore(VectorStore):
    """Vector store implementation using ChromaDB"""

    def __init__(self, index_name: str, alpha: float = None, bm25_cache: Optional[str] = None):
        """
        Args:
            index_name: The name of the Chroma collection/index to use. If it doesn't exist already, we'll create it.
            alpha: The alpha parameter for hybrid search: alpha == 1.0 means pure dense search, alpha == 0.0 means pure
                BM25, and 0.0 < alpha < 1.0 means a hybrid of the two.
        """
        self.index_name = index_name
        self.alpha = alpha
        self.client = chromadb.PersistentClient()

    @cached_property
    def index(self):
        index = self.client.get_or_create_collection(self.index_name)
        return index

    def ensure_exists(self):
        pass

    def upsert_batch(self, vectors: List[Vector], namespace: str):
        del namespace

        ids = []
        embeddings = []
        metadatas = []
        documents = []

        for i, (metadata, embedding) in enumerate(vectors):
            ids.append(metadata.get("id", str(i)))
            embeddings.append(embedding)
            metadatas.append(metadata)
            documents.append(metadata[TEXT_FIELD])

        self.index.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)

    def as_retriever(self, top_k: int, embeddings: Embeddings = None, namespace: str = None):
        vector_store = LangChainChroma(
            collection_name=self.index_name, embedding_function=embeddings, client=self.client
        )

        return vector_store.as_retriever(search_kwargs={"k": top_k})


class FAISSVectorStore(VectorStore):
    """Vector store implementation using FAISS"""

    def __init__(self, index_name: str, dimension: int, embeddings: Embeddings = None):
        """
        Args:
            index_name: The name of the FAISS index to use. If it doesn't exist already, we'll create it.
            dimension: The dimension of the vectors.
            embeddings: The embedding function used to generate embeddings
        """
        self.index_name = index_name
        self.dimension = dimension
        self.embeddings = embeddings

        # check if the index exists
        if os.path.exists(self.index_name):
            # load the existing index
            self.vector_store = FAISS.load_local(
                folder_path=self.index_name, embeddings=self.embeddings, allow_dangerous_deserialization=True
            )
        # else create a new index
        else:
            self.vector_store = FAISS(
                embedding_function=self.embeddings,
                index=self.index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
            )

    @cached_property
    def index(self):
        index = faiss.IndexFlatL2(self.dimension)
        return index

    def ensure_exists(self):
        pass

    def upsert_batch(self, vectors: List[Vector], namespace: str):
        del namespace

        ids = []
        documents = []

        for i, (meta_data, embedding) in enumerate(vectors):
            ids.append(meta_data.get("id", str(i)))
            document = Document(page_content=meta_data[TEXT_FIELD], metadata=meta_data)
            documents.append(document)

        self.vector_store.add_documents(documents=documents, ids=ids)

        # saving the index after every batch upsert
        self.vector_store.save_local(self.index_name)
        print("Save Local Executed")
        logging.error("Save Local Got Executed")

    def as_retriever(self, top_k, embeddings, namespace):
        del embeddings
        del namespace

        return self.vector_store.as_retriever(search_kwards={"k": top_k})


class MilvusVectorStore(VectorStore):
    """Vector store implementation using Milvus"""

    def __init__(self, uri: str, index_name: str, embeddings: Embeddings = None):
        """
        Args:
            index_name: The name of the Milvus collection to use. If it doesn't exist already, we'll create it.
            embeddings: The embedding function used to generate embeddings
        """
        self.uri = uri
        self.index_name = index_name
        self.embeddings = embeddings

        self.vector_store = Milvus(
            embedding_function=embeddings, connection_args={"uri": self.uri}, collection_name=self.index_name
        )

    def ensure_exists(self):
        pass

    def upsert_batch(self, vectors: List[Vector], namespace: str):
        del namespace

        ids = []
        documents = []

        for i, (meta_data, embedding) in enumerate(vectors):
            ids.append(meta_data.get("id", str(i)))
            # "text" is a reserved keyword. So removing it
            page_content = meta_data[TEXT_FIELD]
            meta_data["content"] = meta_data[TEXT_FIELD]
            del meta_data[TEXT_FIELD]

            document = Document(page_content=page_content, metadata=meta_data)
            documents.append(document)

        self.vector_store.add_documents(documents=documents, ids=ids)

    def as_retriever(self, top_k, embeddings, namespace):
        del embeddings
        del namespace

        return self.vector_store.as_retriever(search_kwards={"k": top_k})


class QdrantVectorStore(VectorStore):
    """Vector store implementation using Qdrant"""

    def __init__(self, index_name: str, dimension: int, embeddings: Embeddings = None):
        """
        Args:
            index_name: The name of the Qdrant collection to use. If it doesn't exist already, we'll create it.
            embeddings: The embedding function used to generate embeddings
        """
        self.index_name = index_name
        self.dimension = dimension
        self.embeddings = embeddings
        self.client = QdrantClient(path="qdrantdb")
        self.vector_store = self.index

    @cached_property
    def index(self):
        self.ensure_exists()
        vector_store = LangChainQdrant(client=self.client, collection_name=self.index_name, embedding=self.embeddings)
        return vector_store

    def ensure_exists(self):
        if not self.client.collection_exists(self.index_name):
            self.client.create_collection(
                collection_name=self.index_name,
                vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE),
            )

    def upsert_batch(self, vectors: List[Vector], namespace: str):
        del namespace

        ids = []
        documents = []

        for i, (meta_data, embedding) in enumerate(vectors):
            ids.append(str(uuid4()))
            document = Document(page_content=meta_data[TEXT_FIELD], metadata=meta_data)
            documents.append(document)

        self.vector_store.add_documents(documents=documents, ids=ids)

    def as_retriever(self, top_k, embeddings, namespace):
        del embeddings
        del namespace

        return self.vector_store.as_retriever(search_kwards={"k": top_k})


class MarqoVectorStore(VectorStore):
    """Vector store implementation using Marqo."""

    def __init__(self, url: str, index_name: str):
        self.client = marqo.Client(url=url)
        self.index_name = index_name

    def ensure_exists(self):
        pass

    def upsert_batch(self, vectors: List[Vector], namespace: str):
        # Since Marqo is both an embedder and a vector store, the embedder is already doing the upsert.
        pass

    def as_retriever(self, top_k: int, embeddings: Embeddings = None, namespace: str = None):
        del embeddings  # Unused; The Marqo vector store is also an embedder.
        del namespace  # Unused; Unlike Pinecone, Marqo doesn't differentiate between index name and namespace.

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


def build_vector_store_from_args(
    args: dict,
    data_manager: Optional[DataManager] = None,
) -> VectorStore:
    """Builds a vector store from the given command-line arguments.

    When `data_manager` is specified and hybrid retrieval is requested, we'll use it to fit a BM25 encoder on the corpus
    of documents.
    """
    if args.embedding_provider == "openai":
        embeddings = OpenAIEmbeddings(model=args.embedding_model)
    elif args.embedding_provider == "voyage":
        embeddings = VoyageAIEmbeddings(model=args.embedding_model)
    elif args.embedding_provider == "gemini":
        embeddings = GoogleGenerativeAIEmbeddings(model=args.embedding_model)

    if args.vector_store_provider == "pinecone":
        bm25_cache = os.path.join(".bm25_cache", args.index_namespace, "bm25_encoder.json")
        if args.retrieval_alpha < 1.0 and not os.path.exists(bm25_cache) and data_manager:
            logging.info("Fitting BM25 encoder on the corpus...")
            if is_punkt_downloaded():
                print("punkt is already downloaded")
            else:
                print("punkt is not downloaded")
                # Optionally download it
                nltk.download("punkt_tab")
            corpus = [content for content, _ in data_manager.walk()]
            bm25_encoder = BM25Encoder()
            bm25_encoder.fit(corpus)
            # Make sure the folder exists, before we dump the encoder.
            bm25_folder = os.path.dirname(bm25_cache)
            if not os.path.exists(bm25_folder):
                os.makedirs(bm25_folder)
            bm25_encoder.dump(bm25_cache)

        return PineconeVectorStore(
            index_name=args.index_name,
            dimension=args.embedding_size if "embedding_size" in args else None,
            alpha=args.retrieval_alpha,
            bm25_cache=bm25_cache,
        )
    elif args.vector_store_provider == "chroma":
        return ChromaVectorStore(
            index_name=args.index_name,
        )
    elif args.vector_store_provider == "faiss":
        return FAISSVectorStore(index_name=args.index_name, dimension=args.embedding_size, embeddings=embeddings)
    elif args.vector_store_provider == "milvus":
        return MilvusVectorStore(uri=args.milvus_uri, index_name=args.index_name, embeddings=embeddings)
    elif args.vector_store_provider == "qdrant":
        return QdrantVectorStore(index_name=args.index_name, dimension=args.embedding_size, embeddings=embeddings)
    elif args.vector_store_provider == "marqo":
        return MarqoVectorStore(url=args.marqo_url, index_name=args.index_namespace)
    else:
        raise ValueError(f"Unrecognized vector store type {args.vector_store_provider}")
