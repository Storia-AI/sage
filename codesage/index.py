"""Runs a batch job to compute embeddings for an entire repo and stores them into a vector store."""

import argparse
import logging
import os
import time

import pkg_resources

from repo2vec.chunker import UniversalFileChunker
from repo2vec.data_manager import GitHubRepoManager
from repo2vec.embedder import build_batch_embedder_from_flags
from repo2vec.github import GitHubIssuesChunker, GitHubIssuesManager
from repo2vec.vector_store import build_from_args

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

MARQO_MAX_CHUNKS_PER_BATCH = 64

OPENAI_MAX_TOKENS_PER_CHUNK = 8192  # The ADA embedder from OpenAI has a maximum of 8192 tokens.
OPENAI_MAX_CHUNKS_PER_BATCH = 2048  # The OpenAI batch embedding API enforces a maximum of 2048 chunks per batch.
OPENAI_MAX_TOKENS_PER_JOB = (
    3_000_000  # The OpenAI batch embedding API enforces a maximum of 3M tokens processed at once.
)

# Note that OpenAI embedding models have fixed dimensions, however, taking a slice of them is possible.
# See "Reducing embedding dimensions" under https://platform.openai.com/docs/guides/embeddings/use-cases and
# https://platform.openai.com/docs/api-reference/embeddings/create#embeddings-create-dimensions
OPENAI_DEFAULT_EMBEDDING_SIZE = {
    "text-embedding-ada-002": 1536,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}


def main():
    parser = argparse.ArgumentParser(description="Batch-embeds a GitHub repository and its issues.")
    parser.add_argument("repo_id", help="The ID of the repository to index")
    parser.add_argument("--embedder-type", default="marqo", choices=["openai", "marqo"])
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
    parser.add_argument("--vector-store-type", default="marqo", choices=["pinecone", "marqo"])
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
        help="Maximum chunks per batch. We recommend 2000 for the OpenAI embedder. Marqo enforces a limit of 64.",
    )
    parser.add_argument(
        "--index-name",
        default=None,
        help="Vector store index name. For Marqo, we default it to the repository name. Required for Pinecone, since "
        "it needs to be created manually on their website. In Pinecone terminology, this is *not* the namespace (which "
        "we default to the repo ID).",
    )
    parser.add_argument(
        "--include",
        help="Path to a file containing a list of extensions to include. One extension per line.",
    )
    parser.add_argument(
        "--exclude",
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
        default=False,
        help="Whether to index GitHub issues. At least one of --index-repo and --index-issues must be True. When "
        "--index-issues is set, you must also set a GITHUB_TOKEN environment variable.",
    )
    parser.add_argument(
        "--index-issue-comments",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to index the comments of GitHub issues. This is only relevant if --index-issues is set. "
        "GitHub's API for downloading comments is quite slow. Indexing solely the body of an issue seems to bring most "
        "of the gains anyway.",
    )
    args = parser.parse_args()

    # Validate embedder and vector store compatibility.
    if args.embedder_type == "openai" and args.vector_store_type != "pinecone":
        parser.error("When using OpenAI embedder, the vector store type must be Pinecone.")
    if args.embedder_type == "marqo" and args.vector_store_type != "marqo":
        parser.error("When using the marqo embedder, the vector store type must also be marqo.")
    if args.vector_store_type == "marqo":
        if not args.index_name:
            args.index_name = args.repo_id.split("/")[1]
        if "/" in args.index_name:
            parser.error("The index name cannot contain slashes when using Marqo as the vector store.")
    elif args.vector_store_type == "pinecone" and not args.index_name:
        parser.error(
            "When using Pinecone as the vector store, you must specify an index name. You can create one on "
            "the Pinecone website. Make sure to set it the right --embedding-size."
        )

    # Validate embedder parameters.
    if args.embedder_type == "marqo":
        if args.embedding_model is None:
            args.embedding_model = "hf/e5-base-v2"
        if args.chunks_per_batch is None:
            args.chunks_per_batch = MARQO_MAX_CHUNKS_PER_BATCH
        elif args.chunks_per_batch > MARQO_MAX_CHUNKS_PER_BATCH:
            args.chunks_per_batch = MARQO_MAX_CHUNKS_PER_BATCH
            logging.warning(
                f"Marqo enforces a limit of {MARQO_MAX_CHUNKS_PER_BATCH} chunks per batch. "
                "Overwriting --chunks_per_batch."
            )
    elif args.embedder_type == "openai":
        if args.tokens_per_chunk > OPENAI_MAX_TOKENS_PER_CHUNK:
            args.tokens_per_chunk = OPENAI_MAX_TOKENS_PER_CHUNK
            logging.warning(
                f"OpenAI enforces a limit of {OPENAI_MAX_TOKENS_PER_CHUNK} tokens per chunk. "
                "Overwriting --tokens_per_chunk."
            )
        if args.chunks_per_batch is None:
            args.chunks_per_batch = 2000
        elif args.chunks_per_batch > OPENAI_MAX_CHUNKS_PER_BATCH:
            args.chunks_per_batch = OPENAI_MAX_CHUNKS_PER_BATCH
            logging.warning(
                f"OpenAI enforces a limit of {OPENAI_MAX_CHUNKS_PER_BATCH} chunks per batch. "
                "Overwriting --chunks_per_batch."
            )
        if args.tokens_per_chunk * args.chunks_per_batch >= OPENAI_MAX_TOKENS_PER_JOB:
            parser.error(f"The maximum number of chunks per job is {OPENAI_MAX_TOKENS_PER_JOB}.")
        if args.embedding_model is None:
            args.embedding_model = "text-embedding-ada-002"
        if args.embedding_size is None:
            args.embedding_size = OPENAI_DEFAULT_EMBEDDING_SIZE.get(args.embedding_model)

    if args.include and args.exclude:
        parser.error("At most one of --include and --exclude can be specified.")
    if not args.include and not args.exclude:
        args.exclude = pkg_resources.resource_filename(__name__, "sample-exclude.txt")
    if not args.index_repo and not args.index_issues:
        parser.error("At least one of --index-repo and --index-issues must be true.")

    # Fail early on missing environment variables.
    if args.embedder_type == "openai" and not os.getenv("OPENAI_API_KEY"):
        parser.error("Please set the OPENAI_API_KEY environment variable.")
    if args.vector_store_type == "pinecone" and not os.getenv("PINECONE_API_KEY"):
        parser.error("Please set the PINECONE_API_KEY environment variable.")
    if args.index_issues and not os.getenv("GITHUB_TOKEN"):
        parser.error("Please set the GITHUB_TOKEN environment variable.")

    ######################
    # Step 1: Embeddings #
    ######################

    # Index the repository.
    repo_embedder = None
    if args.index_repo:
        logging.info("Cloning the repository...")
        repo_manager = GitHubRepoManager(
            args.repo_id,
            local_dir=args.local_dir,
            inclusion_file=args.include,
            exclusion_file=args.exclude,
        )
        repo_manager.download()
        logging.info("Embedding the repo...")
        chunker = UniversalFileChunker(max_tokens=args.tokens_per_chunk)
        repo_embedder = build_batch_embedder_from_flags(repo_manager, chunker, args)
        repo_jobs_file = repo_embedder.embed_dataset(args.chunks_per_batch, args.max_embedding_jobs)

    # Index the GitHub issues.
    issues_embedder = None
    if args.index_issues:
        logging.info("Issuing embedding jobs for GitHub issues...")
        issues_manager = GitHubIssuesManager(args.repo_id, index_comments=args.index_issue_comments)
        issues_manager.download()
        logging.info("Embedding GitHub issues...")
        chunker = GitHubIssuesChunker(max_tokens=args.tokens_per_chunk)
        issues_embedder = build_batch_embedder_from_flags(issues_manager, chunker, args)
        issues_jobs_file = issues_embedder.embed_dataset(args.chunks_per_batch, args.max_embedding_jobs)

    ########################
    # Step 2: Vector Store #
    ########################

    if args.vector_store_type == "marqo":
        # Marqo computes embeddings and stores them in the vector store at once, so we're done.
        logging.info("Done!")
        return

    if repo_embedder is not None:
        logging.info("Waiting for repo embeddings to be ready...")
        while not repo_embedder.embeddings_are_ready(repo_jobs_file):
            logging.info("Sleeping for 30 seconds...")
            time.sleep(30)

        logging.info("Moving embeddings to the repo vector store...")
        repo_vector_store = build_from_args(args)
        repo_vector_store.ensure_exists()
        repo_vector_store.upsert(repo_embedder.download_embeddings(repo_jobs_file))

    if issues_embedder is not None:
        logging.info("Waiting for issue embeddings to be ready...")
        while not issues_embedder.embeddings_are_ready(issues_jobs_file):
            logging.info("Sleeping for 30 seconds...")
            time.sleep(30)

        logging.info("Moving embeddings to the issues vector store...")
        issues_vector_store = build_from_args(args)
        issues_vector_store.ensure_exists()
        issues_vector_store.upsert(issues_embedder.download_embeddings(issues_jobs_file))

    logging.info("Done!")


if __name__ == "__main__":
    main()
