"""Runs a batch job to compute embeddings for an entire repo and stores them into a vector store."""

import argparse
import logging
import time

from chunker import UniversalChunker
from embedder import MarqoEmbedder, OpenAIBatchEmbedder
from repo_manager import RepoManager
from vector_store import build_from_args

logging.basicConfig(level=logging.INFO)

MAX_TOKENS_PER_CHUNK = 8192  # The ADA embedder from OpenAI has a maximum of 8192 tokens.
MAX_CHUNKS_PER_BATCH = 2048  # The OpenAI batch embedding API enforces a maximum of 2048 chunks per batch.
MAX_TOKENS_PER_JOB = 3_000_000  # The OpenAI batch embedding API enforces a maximum of 3M tokens processed at once.

# Note that OpenAI embedding models have fixed dimensions, however, taking a slice of them is possible.
# See "Reducing embedding dimensions" under https://platform.openai.com/docs/guides/embeddings/use-cases and
# https://platform.openai.com/docs/api-reference/embeddings/create#embeddings-create-dimensions
OPENAI_DEFAULT_EMBEDDING_SIZE = {
    "text-embedding-ada-002": 1536,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}


def _read_extensions(path):
    with open(path, "r") as f:
        return {line.strip().lower() for line in f}


def main():
    parser = argparse.ArgumentParser(description="Batch-embeds a repository")
    parser.add_argument("repo_id", help="The ID of the repository to index")
    parser.add_argument("--embedder_type", default="openai", choices=["openai", "marqo"])
    parser.add_argument(
        "--embedding_model",
        type=str,
        default=None,
        help="The embedding model. Defaults to `text-embedding-ada-002` for OpenAI and `hf/e5-base-v2` for Marqo.",
    )
    parser.add_argument(
        "--embedding_size",
        type=int,
        default=None,
        help="The embedding size to use for OpenAI; defaults to OpenAI defaults (e.g. 1536 for `text-embedding-3-small`"
        " and 3072 for `text-embedding-3-large`). Note that OpenAI allows users to reduce these default dimensions. "
        "No need to specify an embedding size for Marqo, since the embedding model determines it.",
    )
    parser.add_argument("--vector_store_type", default="pinecone", choices=["pinecone", "marqo"])
    parser.add_argument(
        "--local_dir",
        default="repos",
        help="The local directory to store the repository",
    )
    parser.add_argument(
        "--tokens_per_chunk",
        type=int,
        default=800,
        help="https://arxiv.org/pdf/2406.14497 recommends a value between 200-800.",
    )
    parser.add_argument(
        "--chunks_per_batch",
        type=int,
        default=2000,
        help="Maximum chunks per batch. We recommend 2000 for the OpenAI embedder. Marqo enforces a limit of 64.",
    )
    parser.add_argument(
        "--index_name",
        required=True,
        help="Vector store index name. For Pinecone, make sure to create it with the right embedding size.",
    )
    parser.add_argument(
        "--include",
        help="Path to a file containing a list of extensions to include. One extension per line.",
    )
    parser.add_argument(
        "--exclude",
        default="src/sample-exclude.txt",
        help="Path to a file containing a list of extensions to exclude. One extension per line.",
    )
    parser.add_argument(
        "--max_embedding_jobs",
        type=int,
        help="Maximum number of embedding jobs to run. Specifying this might result in "
        "indexing only part of the repository, but prevents you from burning through OpenAI credits.",
    )
    parser.add_argument(
        "--marqo_url",
        default="http://localhost:8882",
        help="URL for the Marqo server. Required if using Marqo as embedder or vector store.",
    )
    args = parser.parse_args()

    # Validate embedder and vector store compatibility.
    if args.embedder_type == "openai" and args.vector_store_type != "pinecone":
        parser.error("When using OpenAI embedder, the vector store type must be Pinecone.")
    if args.embedder_type == "marqo" and args.vector_store_type != "marqo":
        parser.error("When using the marqo embedder, the vector store type must also be marqo.")
    if args.embedder_type == "marqo" and args.chunks_per_batch > 64:
        args.chunks_per_batch = 64
        logging.warning("Marqo enforces a limit of 64 chunks per batch. Setting --chunks_per_batch to 64.")

    # Validate other arguments.
    if args.tokens_per_chunk > MAX_TOKENS_PER_CHUNK:
        parser.error(f"The maximum number of tokens per chunk is {MAX_TOKENS_PER_CHUNK}.")
    if args.chunks_per_batch > MAX_CHUNKS_PER_BATCH:
        parser.error(f"The maximum number of chunks per batch is {MAX_CHUNKS_PER_BATCH}.")
    if args.tokens_per_chunk * args.chunks_per_batch >= MAX_TOKENS_PER_JOB:
        parser.error(f"The maximum number of chunks per job is {MAX_TOKENS_PER_JOB}.")
    if args.include and args.exclude:
        parser.error("At most one of --include and --exclude can be specified.")

    # Set default values based on other arguments
    if args.embedding_model is None:
        args.embedding_model = "text-embedding-ada-002" if args.embedder_type == "openai" else "hf/e5-base-v2"
    if args.embedding_size is None and args.embedder_type == "openai":
        args.embedding_size = OPENAI_DEFAULT_EMBEDDING_SIZE.get(args.embedding_model)
        # No need to set embedding_size for Marqo, since the embedding model determines the embedding size.
        logging.warn("--embedding_size is ignored for Marqo embedder.")

    included_extensions = _read_extensions(args.include) if args.include else None
    excluded_extensions = _read_extensions(args.exclude) if args.exclude else None

    logging.info("Cloning the repository...")
    repo_manager = RepoManager(
        args.repo_id,
        local_dir=args.local_dir,
        included_extensions=included_extensions,
        excluded_extensions=excluded_extensions,
    )
    repo_manager.clone()

    logging.info("Issuing embedding jobs...")
    chunker = UniversalChunker(max_tokens=args.tokens_per_chunk)

    if args.embedder_type == "openai":
        embedder = OpenAIBatchEmbedder(repo_manager, chunker, args.local_dir, args.embedding_model, args.embedding_size)
    elif args.embedder_type == "marqo":
        embedder = MarqoEmbedder(
            repo_manager, chunker, index_name=args.index_name, url=args.marqo_url, model=args.embedding_model
        )
    else:
        raise ValueError(f"Unrecognized embedder type {args.embedder_type}")

    embedder.embed_repo(args.chunks_per_batch, args.max_embedding_jobs)

    if args.vector_store_type == "marqo":
        # Marqo computes embeddings and stores them in the vector store at once, so we're done.
        logging.info("Done!")
        return

    logging.info("Waiting for embeddings to be ready...")
    while not embedder.embeddings_are_ready():
        logging.info("Sleeping for 30 seconds...")
        time.sleep(30)

    logging.info("Moving embeddings to the vector store...")
    # Note to developer: Replace this with your preferred vector store.
    vector_store = build_from_args(args)
    vector_store.ensure_exists()
    vector_store.upsert(embedder.download_embeddings())
    logging.info("Done!")


if __name__ == "__main__":
    main()
