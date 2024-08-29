"""Batch embedder abstraction and implementations."""

import json
import logging
import os
from abc import ABC, abstractmethod
from collections import Counter
from typing import Dict, Generator, List, Tuple

from openai import OpenAI

from chunker import Chunk, Chunker
from repo_manager import RepoManager

Vector = Tuple[Dict, List[float]]  # (metadata, embedding)


class BatchEmbedder(ABC):
    """Abstract class for batch embedding of a repository."""

    @abstractmethod
    def embed_repo(self, chunks_per_batch: int):
        """Issues batch embedding jobs for the entire repository."""

    @abstractmethod
    def embeddings_are_ready(self) -> bool:
        """Checks whether the batch embedding jobs are done."""

    @abstractmethod
    def download_embeddings(self) -> Generator[Vector, None, None]:
        """Yields (chunk_metadata, embedding) pairs for each chunk in the repository."""


class OpenAIBatchEmbedder(BatchEmbedder):
    """Batch embedder that calls OpenAI. See https://platform.openai.com/docs/guides/batch/overview."""

    def __init__(
        self, repo_manager: RepoManager, chunker: Chunker, local_dir: str
    ):
        self.repo_manager = repo_manager
        self.chunker = chunker
        self.local_dir = local_dir
        # IDs issued by OpenAI for each batch job mapped to metadata about the chunks.
        self.openai_batch_ids = {}
        self.client = OpenAI()

    def embed_repo(self, chunks_per_batch: int):
        """Issues batch embedding jobs for the entire repository."""
        if self.openai_batch_ids:
            raise ValueError("Embeddings are in progress.")

        batch = []
        chunk_count = 0
        repo_name = self.repo_manager.repo_id.split("/")[-1]

        for filepath, content in self.repo_manager.walk():
            chunks = self.chunker.chunk(filepath, content)
            chunk_count += len(chunks)
            batch.extend(chunks)

            if len(batch) > chunks_per_batch:
                for i in range(0, len(batch), chunks_per_batch):
                    sub_batch = batch[i : i + chunks_per_batch]
                    openai_batch_id = self._issue_job_for_chunks(
                        sub_batch, batch_id=f"{repo_name}/{len(self.openai_batch_ids)}"
                    )
                    self.openai_batch_ids[openai_batch_id] = self._metadata_for_chunks(
                        sub_batch
                    )
                batch = []

        # Finally, commit the last batch.
        if batch:
            openai_batch_id = self._issue_job_for_chunks(
                batch, batch_id=f"{repo_name}/{len(self.openai_batch_ids)}"
            )
            self.openai_batch_ids[openai_batch_id] = self._metadata_for_chunks(batch)
        logging.info(
            "Issued %d jobs for %d chunks.", len(self.openai_batch_ids), chunk_count
        )

        # Save the job IDs to a file, just in case this script is terminated by mistake.
        metadata_file = os.path.join(self.local_dir, "openai_batch_ids.json")
        with open(metadata_file, "w") as f:
            json.dump(self.openai_batch_ids, f)
        logging.info("Job metadata saved at %s", metadata_file)

    def embeddings_are_ready(self) -> bool:
        """Checks whether the embeddings jobs are done (either completed or failed)."""
        if not self.openai_batch_ids:
            raise ValueError("No embeddings in progress.")
        job_ids = self.openai_batch_ids.keys()
        statuses = [self.client.batches.retrieve(job_id.strip()) for job_id in job_ids]
        are_ready = all(status.status in ["completed", "failed"] for status in statuses)
        status_counts = Counter(status.status for status in statuses)
        logging.info("Job statuses: %s", status_counts)
        return are_ready

    def download_embeddings(self) -> Generator[Vector, None, None]:
        """Yield a (chunk_metadata, embedding) pair for each chunk in the repository."""
        job_ids = self.openai_batch_ids.keys()
        statuses = [self.client.batches.retrieve(job_id.strip()) for job_id in job_ids]

        for idx, status in enumerate(statuses):
            if status.status == "failed":
                logging.error("Job failed: %s", status)
                continue

            if not status.output_file_id:
                error = self.client.files.content(status.error_file_id)
                logging.error("Job %s failed with error: %s", status.id, error.text)
                continue

            batch_metadata = self.openai_batch_ids[status.id]
            file_response = self.client.files.content(status.output_file_id)
            data = json.loads(file_response.text)["response"]["body"]["data"]
            logging.info("Job %s generated %d embeddings.", status.id, len(data))

            for datum in data:
                idx = int(datum["index"])
                metadata = batch_metadata[idx]
                embedding = datum["embedding"]
                yield (metadata, embedding)

    def _issue_job_for_chunks(self, chunks: List[Chunk], batch_id: str) -> str:
        """Issues a batch embedding job for the given chunks. Returns the job ID."""
        logging.info("*" * 100)
        logging.info("Issuing job for batch %s with %d chunks.", batch_id, len(chunks))

        # Create a .jsonl file with the batch.
        request = OpenAIBatchEmbedder._chunks_to_request(chunks, batch_id)
        input_file = os.path.join(self.local_dir, f"batch_{batch_id}.jsonl")
        OpenAIBatchEmbedder._export_to_jsonl([request], input_file)

        # Uplaod the file and issue the embedding job.
        batch_input_file = self.client.files.create(file=open(input_file, "rb"), purpose="batch")
        batch_status = self._create_batch_job(batch_input_file.id)
        logging.info("Created job with ID %s", batch_status.id)
        return batch_status.id

    def _create_batch_job(self, input_file_id: str):
        """Creates a batch embedding job for OpenAI."""
        try:
            return self.client.batches.create(
                input_file_id=input_file_id,
                endpoint="/v1/embeddings",
                completion_window="24h",  # This is the only allowed value for now.
                timeout=3 * 60,  # 3 minutes
                metadata={},
            )
        except Exception as e:
            print(
                f"Failed to create batch job with input_file_id={input_file_id}. Error: {e}"
            )
            return None

    @staticmethod
    def _export_to_jsonl(list_of_dicts: List[Dict], output_file: str):
        """Exports a list of dictionaries to a .jsonl file."""
        directory = os.path.dirname(output_file)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(output_file, "w") as f:
            for item in list_of_dicts:
                json.dump(item, f)
                f.write("\n")

    @staticmethod
    def _chunks_to_request(chunks: List[Chunk], batch_id: str):
        """Convert a list of chunks to a batch request."""
        return {
            "custom_id": batch_id,
            "method": "POST",
            "url": "/v1/embeddings",
            "body": {
                "model": "text-embedding-ada-002",
                "input": [chunk.content for chunk in chunks],
            },
        }

    @staticmethod
    def _metadata_for_chunks(chunks):
        metadata = []
        for chunk in chunks:
            filename_ascii = chunk.filename.encode("ascii", "ignore").decode("ascii")
            metadata.append(
                {
                    # Some vector stores require the IDs to be ASCII.
                    "id": f"{filename_ascii}_{chunk.start_byte}_{chunk.end_byte}",
                    "filename": chunk.filename,
                    "start_byte": chunk.start_byte,
                    "end_byte": chunk.end_byte,
                    # Note to developer: When choosing a large chunk size, you might exceed the vector store's metadata
                    # size limit. In that case, you can simply store the start/end bytes above, and fetch the content
                    # directly from the repository when needed.
                    "text": chunk.content,
                }
            )
        return metadata
