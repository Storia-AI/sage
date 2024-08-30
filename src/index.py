"""Runs a batch job to compute embeddings for an entire repo and stores them into a vector store."""

import argparse
import logging
import time

from chunker import UniversalChunker
from embedder import OpenAIBatchEmbedder, MarqoEmbedder
from repo_manager import RepoManager
from vector_store import PineconeVectorStore

logging.basicConfig(level=logging.INFO)

OPENAI_EMBEDDING_SIZE = 1536
MAX_TOKENS_PER_CHUNK = (
    8192  # The ADA embedder from OpenAI has a maximum of 8192 tokens.
)
MAX_CHUNKS_PER_BATCH = (
    2048  # The OpenAI batch embedding API enforces a maximum of 2048 chunks per batch.
)
MAX_TOKENS_PER_JOB = 3_000_000  # The OpenAI batch embedding API enforces a maximum of 3M tokens processed at once.


def _read_extensions(path):
    with open(path, "r") as f:
        return {line.strip().lower() for line in f}


def main():
    parser = argparse.ArgumentParser(description="Batch-embeds a repository")
    parser.add_argument("repo_id", help="The ID of the repository to index")
    parser.add_argument("--embedder_type", default="openai", choices=["openai", "marqo"])
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
        "--chunks_per_batch", type=int, default=2000, help="Maximum chunks per batch"
    )
    parser.add_argument(
        "--index_name", required=True, help="Vector store index name"
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
        "--max_embedding_jobs", type=int,
        help="Maximum number of embedding jobs to run. Specifying this might result in "
        "indexing only part of the repository, but prevents you from burning through OpenAI credits.",
    )
    parser.add_argument(
        "--marqo_url",
        default="http://localhost:8882",
        help="URL for the Marqo server. Required if using Marqo as embedder or vector store.",
    )
    parser.add_argument(
        "--marqo_embedding_model",
        default="hf/e5-base-v2",
        help="The embedding model to use for Marqo.",
    )
    args = parser.parse_args()

    # Validate embedder and vector store compatibility.
    if args.embedder_type == "openai" and args.vector_store_type != "pinecone":
        parser.error("When using OpenAI embedder, the vector store type must be Pinecone.")
    if args.embedder_type == "marqo" and args.vector_store_type != "marqo":
        parser.error("When using the marqo embedder, the vector store type must also be marqo.")

    # Validate other arguments.
    if args.tokens_per_chunk > MAX_TOKENS_PER_CHUNK:
        parser.error(
            f"The maximum number of tokens per chunk is {MAX_TOKENS_PER_CHUNK}."
        )
    if args.chunks_per_batch > MAX_CHUNKS_PER_BATCH:
        parser.error(
            f"The maximum number of chunks per batch is {MAX_CHUNKS_PER_BATCH}."
        )
    if args.tokens_per_chunk * args.chunks_per_batch >= MAX_TOKENS_PER_JOB:
        parser.error(f"The maximum number of chunks per job is {MAX_TOKENS_PER_JOB}.")
    if args.include and args.exclude:
        parser.error("At most one of --include and --exclude can be specified.")

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
        embedder = OpenAIBatchEmbedder(repo_manager, chunker, args.local_dir)
    elif args.embedder_type == "marqo":
        embedder = MarqoEmbedder(repo_manager,
                                 chunker,
                                 index_name=args.index_name,
                                 url=args.marqo_url,
                                 model=args.marqo_embedding_model)
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
    vector_store = PineconeVectorStore(
        index_name=args.index_name,
        dimension=OPENAI_EMBEDDING_SIZE,
        namespace=repo_manager.repo_id,
    )
    vector_store.ensure_exists()
    vector_store.upsert(embedder.download_embeddings())
    logging.info("Done!")


if __name__ == "__main__":
    main()
