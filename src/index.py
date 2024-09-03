"""Runs a batch job to compute embeddings for an entire repo and stores them into a vector store."""

import argparse
import logging
import time

from chunker import UniversalFileChunker
from data_manager import GitHubRepoManager
from embedder import build_batch_embedder_from_flags
from github import GitHubIssuesChunker, GitHubIssuesManager
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
    parser = argparse.ArgumentParser(description="Batch-embeds a GitHub repository and its issues.")
    parser.add_argument("repo_id", help="The ID of the repository to index")
    parser.add_argument("--embedder-type", default="openai", choices=["openai", "marqo"])
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="The embedding model. Defaults to `text-embedding-ada-002` for OpenAI and `hf/e5-base-v2` for Marqo.",
    )
    parser.add_argument(
        "--embedding-size",
        type=int,
        default=None,
        help="The embedding size to use for OpenAI text-embedding-3* models. Defaults to 1536 for small and 3072 for "
        "large. Note that no other OpenAI models support a dynamic embedding size, nor do models used with Marqo.",
    )
    parser.add_argument("--vector-store-type", default="pinecone", choices=["pinecone", "marqo"])
    parser.add_argument(
        "--local-dir",
        default="repos",
        help="The local directory to store the repository",
    )
    parser.add_argument(
        "--tokens-per-chunk",
        type=int,
        default=800,
        help="https://arxiv.org/pdf/2406.14497 recommends a value between 200-800.",
    )
    parser.add_argument(
        "--chunks-per-batch",
        type=int,
        default=2000,
        help="Maximum chunks per batch. We recommend 2000 for the OpenAI embedder. Marqo enforces a limit of 64.",
    )
    parser.add_argument(
        "--index-name",
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
        "--max-embedding-jobs",
        type=int,
        help="Maximum number of embedding jobs to run. Specifying this might result in "
        "indexing only part of the repository, but prevents you from burning through OpenAI credits.",
    )
    parser.add_argument(
        "--marqo-url",
        default="http://localhost:8882",
        help="URL for the Marqo server. Required if using Marqo as embedder or vector store.",
    )
    # Pass --no-index-repo in order to not index the repository.
    parser.add_argument(
        "--index-repo",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to index the repository. At least one of --index-repo and --index-issues must be True.",
    )
    # Pass --no-index-issues in order to not index the issues.
    parser.add_argument(
        "--index-issues",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to index GitHub issues. At least one of --index-repo and --index-issues must be True.",
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
    if not args.index_repo and not args.index_issues:
        parser.error("At least one of --index-repo and --index-issues must be true.")

    # Set default values based on other arguments
    if args.embedding_model is None:
        args.embedding_model = "text-embedding-ada-002" if args.embedder_type == "openai" else "hf/e5-base-v2"
    if args.embedding_size is None and args.embedder_type == "openai":
        args.embedding_size = OPENAI_DEFAULT_EMBEDDING_SIZE.get(args.embedding_model)

    ######################
    # Step 1: Embeddings #
    ######################

    # Index the repository.
    repo_embedder = None
    if args.index_repo:
        included_extensions = _read_extensions(args.include) if args.include else None
        excluded_extensions = _read_extensions(args.exclude) if args.exclude else None

        logging.info("Cloning the repository...")
        repo_manager = GitHubRepoManager(
            args.repo_id,
            local_dir=args.local_dir,
            included_extensions=included_extensions,
            excluded_extensions=excluded_extensions,
        )
        repo_manager.download()
        logging.info("Embedding the repo...")
        chunker = UniversalFileChunker(max_tokens=args.tokens_per_chunk)
        repo_embedder = build_batch_embedder_from_flags(repo_manager, chunker, args)
        repo_embedder.embed_dataset(args.chunks_per_batch, args.max_embedding_jobs)

    # Index the GitHub issues.
    issues_embedder = None
    assert args.index_issues is True
    if args.index_issues:
        logging.info("Issuing embedding jobs for GitHub issues...")
        issues_manager = GitHubIssuesManager(args.repo_id)
        issues_manager.download()
        logging.info("Embedding GitHub issues...")
        chunker = GitHubIssuesChunker(max_tokens=args.tokens_per_chunk)
        issues_embedder = build_batch_embedder_from_flags(issues_manager, chunker, args)
        issues_embedder.embed_dataset(args.chunks_per_batch, args.max_embedding_jobs)

    ########################
    # Step 2: Vector Store #
    ########################

    if args.vector_store_type == "marqo":
        # Marqo computes embeddings and stores them in the vector store at once, so we're done.
        logging.info("Done!")
        return

    if repo_embedder is not None:
        logging.info("Waiting for repo embeddings to be ready...")
        while not repo_embedder.embeddings_are_ready():
            logging.info("Sleeping for 30 seconds...")
            time.sleep(30)

        logging.info("Moving embeddings to the repo vector store...")
        repo_vector_store = build_from_args(args)
        repo_vector_store.ensure_exists()
        repo_vector_store.upsert(repo_embedder.download_embeddings())

    if issues_embedder is not None:
        logging.info("Waiting for issue embeddings to be ready...")
        while not issues_embedder.embeddings_are_ready():
            logging.info("Sleeping for 30 seconds...")
            time.sleep(30)

        logging.info("Moving embeddings to the issues vector store...")
        issues_vector_store = build_from_args(args)
        issues_vector_store.ensure_exists()
        issues_vector_store.upsert(issues_embedder.download_embeddings())

    logging.info("Done!")


if __name__ == "__main__":
    main()
