import unittest
from unittest.mock import patch,MagicMock, call
import marqo
import pytest
from sage.embedder import MarqoEmbedder, DataManager,Chunker,GeminiBatchEmbedder,VoyageBatchEmbedder
from sage.constants import TEXT_FIELD
import os




class TestVoyageBatchEmbedder(unittest.TestCase):
    def setUp(self):
        self.data_manager = MagicMock(DataManager)
        self.chunker = MagicMock(Chunker)
        self.embedding_model = "test-model"
    
    @patch.dict(os.environ,{"VOYAGE_API_KEY":"test-api-key"})
    def test_init(self):
        embedder = VoyageBatchEmbedder(self.data_manager, self.chunker, self.embedding_model)
        self.assertEqual(embedder.data_manager, self.data_manager)
        self.assertEqual(embedder.chunker, self.chunker)
        self.assertEqual(embedder.embedding_model, self.embedding_model)
        self.assertEqual(len(embedder.embedding_data), 0)
        
        
    @patch("requests.post")
    @patch('time.sleep')
    @patch.dict(os.environ,{"VOYAGE_API_KEY":"test-api-key"})
    def test_embed_dataset(self,mock_sleep,batch_req):
        batch_req.return_value.json.return_value = {
            "data": [
                {"embedding": [0.1, 0.2, 0.3]},
                {"embedding": [0.4, 0.5, 0.6]},
                {"embedding": [0.7, 0.8, 0.9]}
            ]
        }
        self.data_manager.walk.return_value = [
            ("content1", {"metadata1": 1}),
            ("content2", {"metadata2": 2})
        ]
        self.chunker.chunk.side_effect = [
            [MagicMock(), MagicMock()],
            [MagicMock(), MagicMock(), MagicMock()]
        ]
        self.chunker.max_tokens = 1000
        embedder = VoyageBatchEmbedder(self.data_manager, self.chunker, self.embedding_model)
       
        embedder.embed_dataset(chunks_per_batch=2)
        self.assertEqual(len(embedder.embedding_data), 5)
        self.assertEqual(batch_req.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 1)
        
    def test_embeddings_are_ready(self):
        embedder = VoyageBatchEmbedder(self.data_manager, self.chunker, self.embedding_model)
        self.assertTrue(embedder.embeddings_are_ready())
        
    
    
class TestGeminiBatchEmbedder(unittest.TestCase):
    def setUp(self):
        self.data_manager = MagicMock(DataManager)
        self.chunker = MagicMock(Chunker)
        self.embedding_model = "test-model"

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test-api-key"})
    def test_init(self):
        embedder = GeminiBatchEmbedder(self.data_manager, self.chunker, self.embedding_model)
        self.assertEqual(embedder.data_manager, self.data_manager)
        self.assertEqual(embedder.chunker, self.chunker)
        self.assertEqual(embedder.embedding_model, self.embedding_model)
        self.assertEqual(len(embedder.embedding_data), 0)

    @patch('genai.embed_content')
    @patch('time.time')
    def test_embed_dataset(self, mock_time, mock_embed_content):
        mock_time.side_effect = [0, 10, 70]
        mock_embed_content.return_value = {"embedding": [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9]
        ]}
        
        self.data_manager.walk.return_value = [
            ("content1", {"metadata1": 1}),
            ("content2", {"metadata2": 2})
        ]
        self.chunker.chunk.side_effect = [
            [MagicMock(), MagicMock()],
            [MagicMock(), MagicMock(), MagicMock()]
        ]
        
        embedder = GeminiBatchEmbedder(self.data_manager, self.chunker, self.embedding_model)
        embedder.embed_dataset(chunks_per_batch=2)

        self.assertEqual(len(embedder.embedding_data), 5)
        self.assertEqual(mock_embed_content.call_count, 3)
        
    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test-api-key"})
    def test_embeddings_are_ready(self):
        embedder = GeminiBatchEmbedder(self.data_manager, self.chunker, self.embedding_model)
        self.assertTrue(embedder.embeddings_are_ready())
        
    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test-api-key"})
    def test_download_embeddings(self):
        embedder = GeminiBatchEmbedder(self.data_manager, self.chunker, self.embedding_model)
        embedder.embedding_data = [
            ({"metadata1": 1}, [0.1, 0.2, 0.3]),
            ({"metadata2": 2}, [0.4, 0.5, 0.6])
        ]
        embeddings = list(embedder.download_embeddings())
        self.assertEqual(len(embeddings), 2)
        self.assertEqual(embeddings[0][0], {"metadata1": 1})
        self.assertEqual(embeddings[0][1], [0.1, 0.2, 0.3])
        self.assertEqual(embeddings[1][0], {"metadata2": 2})
        self.assertEqual(embeddings[1][1], [0.4, 0.5, 0.6])
    
        
        
class TestMarqoEmbedder(unittest.TestCase):
    def setUp(self):
        self.data_manager = MagicMock(DataManager)
        self.chunker = MagicMock(Chunker)
        self.client = MagicMock(marqo.Client)
        self.index = MagicMock(marqo.index)
        self.index.add_documents = MagicMock()
        self.client.index.return_value = self.index
        
    def test_init(self):
        embedder = MarqoEmbedder(self.data_manager, self.chunker, "test-index", "http://localhost:8882")
        self.assertEqual(embedder.data_manager, self.data_manager)
        self.assertEqual(embedder.chunker, self.chunker)
        self.assertIsInstance(embedder.client, marqo.Client)
        self.assertIsNotNone(embedder.index)

    @patch('marqo.Client')
    def test_embed_dataset(self, mock_client):
        
        mock_client.return_value = self.client
        
        self.data_manager.walk.return_value = [
            ("content1", {"metadata1": 1}),
            ("content2", {"metadata2": 2})
        ]
        self.chunker.chunk.side_effect = [
            [MagicMock(), MagicMock()],
            [MagicMock(), MagicMock(), MagicMock()]
        ]
        
        embedder = MarqoEmbedder(self.data_manager, self.chunker, "test-index", "http://localhost:8882")
        embedder.embed_dataset(chunks_per_batch=2)
        self.assertEqual(self.index.add_documents.call_count, 3)
        self.assertEqual(self.chunker.chunk.call_count, 2)

    def test_embeddings_are_ready(self):
        embedder = MarqoEmbedder(self.data_manager, self.chunker, "test-index", "http://localhost:8882")
        self.assertTrue(embedder.embeddings_are_ready())

    def test_download_embeddings(self):
        embedder = MarqoEmbedder(self.data_manager, self.chunker, "test-index", "http://localhost:8882")
        embeddings = list(embedder.download_embeddings())
        self.assertEqual(len(embeddings), 0)

if __name__ == '__main__':
    unittest.main()