import os
from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest

from sage.vector_store import (
    MarqoVectorStore,
    PineconeVectorStore,
    build_vector_store_from_args,
)

mock_vectors = [({"id": "1", "text": "example"}, [0.1, 0.2, 0.3])]
mock_namespace = "test_namespace"


class TestVectorStore:
    @pytest.fixture
    def pinecone_store(self):
        with patch("sage.vector_store.Pinecone"):
            store = PineconeVectorStore(index_name="test_index", dimension=128, alpha=0.5)
            yield store

    @pytest.fixture
    def marqo_store(self):
        with patch("marqo.Client"):
            store = MarqoVectorStore(url="http://localhost:8882", index_name="test_index")
            yield store

    @pytest.fixture
    def mock_data_manager(self):
        data_manager = MagicMock()
        data_manager.walk.return_value = [("sample content", {})]
        return data_manager

    @pytest.fixture
    def mock_nltk(self):
        with patch("nltk.data.find") as mock_find:
            mock_find.side_effect = LookupError
            yield mock_find

    @pytest.fixture
    def mock_bm25_encoder(self):
        with patch("sage.vector_store.BM25Encoder") as MockBM25Encoder:
            mock_instance = MockBM25Encoder.return_value
            mock_instance.encode_documents.return_value = [0.1, 0.2, 0.3]
            mock_instance.fit = MagicMock()
            mock_instance.dump = MagicMock()
            yield mock_instance

    def test_pinecone_vector_store_initialization(self, pinecone_store):
        assert pinecone_store.index_name == "test_index"
        assert pinecone_store.dimension == 128
        assert pinecone_store.alpha == 0.5

    def test_pinecone_vector_store_ensure_exists(self, pinecone_store):
        pinecone_store.ensure_exists()
        pinecone_store.client.create_index.assert_called_once()

    def test_pinecone_vector_store_upsert_batch(self, pinecone_store):
        pinecone_store.upsert_batch(mock_vectors, mock_namespace)
        pinecone_store.client.Index().upsert.assert_called_once()

    def test_marqo_vector_store_initialization(self, marqo_store):
        assert marqo_store.index_name == "test_index"

    def test_marqo_vector_store_ensure_exists(self, marqo_store):
        # No specific assertion as ensure_exists is a no-op
        marqo_store.ensure_exists()

    def test_marqo_vector_store_upsert_batch(self, marqo_store):
        # No specific assertion as upsert_batch is a no-op
        marqo_store.upsert_batch(mock_vectors, mock_namespace)

    def build_args(self, provider, alpha=1.0):
        if provider == "pinecone":
            return Namespace(
                vector_store_provider="pinecone",
                pinecone_index_name="test_index",
                embedding_size=128,
                retrieval_alpha=alpha,
                index_namespace="test_namespace",
            )
        elif provider == "marqo":
            return Namespace(
                vector_store_provider="marqo", marqo_url="http://localhost:8882", index_namespace="test_index"
            )

    def build_bm25_cache_path(self):
        return os.path.join(".bm25_cache", "test_namespace", "bm25_encoder.json")

    def test_builds_pinecone_vector_store_with_default_bm25_encoder(
        self, pinecone_store, mock_bm25_encoder, mock_data_manager, mock_nltk
    ):
        args = self.build_args("pinecone", alpha=0.5)
        store = build_vector_store_from_args(args, data_manager=mock_data_manager)
        assert isinstance(store, PineconeVectorStore)
        assert store.bm25_encoder is not None
        mock_bm25_encoder.fit.assert_called_once()
        mock_bm25_encoder.dump.assert_called_once_with(self.build_bm25_cache_path())

    def test_builds_pinecone_vector_store_with_cached_bm25_encoder(self, pinecone_store, mock_bm25_encoder):
        with patch("os.path.exists", return_value=True):
            args = self.build_args("pinecone", alpha=0.5)
            store = build_vector_store_from_args(args)
            assert isinstance(store, PineconeVectorStore)
            assert store.bm25_encoder is not None
            mock_bm25_encoder.load.assert_called_once_with(path=self.build_bm25_cache_path())

    def test_builds_pinecone_vector_store_without_bm25_encoder(self, pinecone_store):
        args = self.build_args("pinecone", alpha=1.0)
        store = build_vector_store_from_args(args)
        assert isinstance(store, PineconeVectorStore)
        assert store.bm25_encoder is None

    def test_builds_marqo_vector_store(self):
        args = self.build_args("marqo")
        store = build_vector_store_from_args(args)
        assert isinstance(store, MarqoVectorStore)

    def test_raises_value_error_for_unrecognized_provider(self):
        args = Namespace(vector_store_provider="unknown")
        with pytest.raises(ValueError, match="Unrecognized vector store type unknown"):
            build_vector_store_from_args(args)

    if __name__ == "__main__":
        pytest.main()
