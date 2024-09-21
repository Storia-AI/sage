"""Utility methods to define and validate flags."""

import argparse
import logging
import os
import re
from typing import Callable

import pkg_resources
from configargparse import ArgumentParser

from sage.reranker import RerankerProvider

MARQO_MAX_CHUNKS_PER_BATCH = 64
# The ADA embedder from OpenAI has a maximum of 8192 tokens.
OPENAI_MAX_TOKENS_PER_CHUNK = 8192
# The OpenAI batch embedding API enforces a maximum of 2048 chunks per batch.
OPENAI_MAX_CHUNKS_PER_BATCH = 2048
# The OpenAI batch embedding API enforces a maximum of 3M tokens processed at once.
OPENAI_MAX_TOKENS_PER_JOB = 3_000_000

# Note that OpenAI embedding models have fixed dimensions, however, taking a slice of them is possible.
# See "Reducing embedding dimensions" under https://platform.openai.com/docs/guides/embeddings/use-cases and
# https://platform.openai.com/docs/api-reference/embeddings/create#embeddings-create-dimensions
OPENAI_DEFAULT_EMBEDDING_SIZE = {
    "text-embedding-ada-002": 1536,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}


def add_config_args(parser: ArgumentParser):
    """Adds configuration-related arguments to the parser."""
    parser.add(
        "--mode",
        choices=["local", "remote"],
        default="local",
        help="Whether to use local-only resources or call third-party providers.",
    )
    parser.add(
        "--config",
        is_config_file=True,
        help="Path to .yaml configuration file.",
    )
    args, _ = parser.parse_known_args()
    config_file = pkg_resources.resource_filename(__name__, f"configs/{args.mode}.yaml")
    parser.set_defaults(config=config_file)


def add_repo_args(parser: ArgumentParser) -> Callable:
    """Adds repository-related arguments to the parser and returns a validator."""
    parser.add("repo_id", help="The ID of the repository to index")
    parser.add("--commit-hash", help="Optional commit hash to checkout. When not provided, defaults to HEAD.")
    parser.add(
        "--local-dir",
        default="repos",
        help="The local directory to store the repository",
    )
    return validate_repo_args


def add_embedding_args(parser: ArgumentParser) -> Callable:
    """Adds embedding-related arguments to the parser and returns a validator."""
    parser.add("--embedding-provider", default="marqo", choices=["openai", "marqo"])
    parser.add(
        "--embedding-model",
        type=str,
        default=None,
        help="The embedding model. Defaults to `text-embedding-ada-002` for OpenAI and `hf/e5-base-v2` for Marqo.",
    )
    parser.add(
        "--embedding-size",
        type=int,
        default=None,
        help="The embedding size to use for OpenAI text-embedding-3* models. Defaults to 1536 for small and 3072 for "
        "large. Note that no other OpenAI models support a dynamic embedding size, nor do models used with Marqo.",
    )
    parser.add(
        "--tokens-per-chunk",
        type=int,
        default=800,
        help="https://arxiv.org/pdf/2406.14497 recommends a value between 200-800.",
    )
    parser.add(
        "--chunks-per-batch",
        type=int,
        help="Maximum chunks per batch. We recommend 2000 for the OpenAI embedder. Marqo enforces a limit of 64.",
    )
    parser.add(
        "--max-embedding-jobs",
        type=int,
        help="Maximum number of embedding jobs to run. Specifying this might result in "
        "indexing only part of the repository, but prevents you from burning through OpenAI credits.",
    )
    return validate_embedding_args


def add_vector_store_args(parser: ArgumentParser) -> Callable:
    """Adds vector store-related arguments to the parser and returns a validator."""
    parser.add("--vector-store-provider", default="marqo", choices=["pinecone", "marqo"])
    parser.add(
        "--pinecone-index-name",
        default=None,
        help="Pinecone index name. Required if using Pinecone as the vector store. If the index doesn't exist already, "
        "we will create it.",
    )
    parser.add(
        "--index-namespace",
        default=None,
        help="Index namespace for this repo. When not specified, we default it to a derivative of the repo name.",
    )
    parser.add(
        "--marqo-url",
        default="http://localhost:8882",
        help="URL for the Marqo server. Required if using Marqo as embedder or vector store.",
    )
    parser.add(
        "--hybrid-retrieval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to use a hybrid of vector DB + BM25 retrieval. When set to False, we only use vector DB "
        "retrieval. This is only relevant if using Pinecone as the vector store.",
    )
    return validate_vector_store_args


def add_indexing_args(parser: ArgumentParser) -> Callable:
    """Adds indexing-related arguments to the parser and returns a validator."""
    parser.add(
        "--include",
        help="Path to a file containing a list of extensions to include. One extension per line.",
    )
    parser.add(
        "--exclude",
        help="Path to a file containing a list of extensions to exclude. One extension per line.",
    )
    # Pass --no-index-repo in order to not index the repository.
    parser.add(
        "--index-repo",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to index the repository. At least one of --index-repo and --index-issues must be True.",
    )
    # Pass --no-index-issues in order to not index the issues.
    parser.add(
        "--index-issues",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to index GitHub issues. At least one of --index-repo and --index-issues must be True. When "
        "--index-issues is set, you must also set a GITHUB_TOKEN environment variable.",
    )
    # Pass --no-index-issue-comments in order to not index the comments of GitHub issues.
    parser.add(
        "--index-issue-comments",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to index the comments of GitHub issues. This is only relevant if --index-issues is set. "
        "GitHub's API for downloading comments is quite slow. Indexing solely the body of an issue seems to bring most "
        "of the gains anyway.",
    )
    return validate_indexing_args


def add_reranking_args(parser: ArgumentParser) -> Callable:
    """Adds reranking-related arguments to the parser."""
    parser.add("--reranker-provider", default="huggingface", choices=[r.value for r in RerankerProvider])
    parser.add(
        "--reranker-model",
        help="The reranker model name. When --reranker-provider=huggingface, we suggest choosing a model from the "
        "SentenceTransformers Cross-Encoders library https://huggingface.co/cross-encoder?sort_models=downloads#models",
    )
    # Trivial validator (nothing to check).
    return lambda _: True


def add_llm_args(parser: ArgumentParser) -> Callable:
    """Adds language model-related arguments to the parser."""
    parser.add("--llm-provider", default="ollama", choices=["openai", "anthropic", "ollama"])
    parser.add(
        "--llm-model",
        help="The LLM name. Must be supported by the provider specified via --llm-provider.",
    )
    # Trivial validator (nothing to check).
    return lambda _: True


def validate_repo_args(args):
    """Validates the configuration of the repository."""
    if not re.match(r"^[^/]+/[^/]+$", args.repo_id):
        raise ValueError("repo_id must be in the format 'owner/repo'")


def _validate_openai_embedding_args(args):
    """Validates the configuration of the OpenAI batch embedder and sets defaults."""
    if args.embedding_provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Please set the OPENAI_API_KEY environment variable.")

    if not args.embedding_model:
        args.embedding_model = "text-embedding-ada-002"

    if args.embedding_model not in OPENAI_DEFAULT_EMBEDDING_SIZE.keys():
        raise ValueError(f"Unrecognized embeddings.model={args.embedding_model}")

    if not args.embedding_size:
        args.embedding_size = OPENAI_DEFAULT_EMBEDDING_SIZE.get(args.embedding_model)

    if not args.tokens_per_chunk:
        # https://arxiv.org/pdf/2406.14497 recommends a value between 200-800.
        args.tokens_per_chunk = 800
    elif args.tokens_per_chunk > OPENAI_MAX_TOKENS_PER_CHUNK:
        args.tokens_per_chunk = OPENAI_MAX_TOKENS_PER_CHUNK
        logging.warning(
            f"OpenAI enforces a limit of {OPENAI_MAX_TOKENS_PER_CHUNK} tokens per chunk. "
            "Overwriting embeddings.tokens_per_chunk."
        )

    if not args.chunks_per_batch:
        args.chunks_per_batch = OPENAI_MAX_CHUNKS_PER_BATCH
    elif args.chunks_per_batch > OPENAI_MAX_CHUNKS_PER_BATCH:
        args.chunks_per_batch = OPENAI_MAX_CHUNKS_PER_BATCH
        logging.warning(
            f"OpenAI enforces a limit of {OPENAI_MAX_CHUNKS_PER_BATCH} chunks per batch. "
            "Overwriting embeddings.chunks_per_batch."
        )

    chunks_per_job = args.tokens_per_chunk * args.chunks_per_batch
    if chunks_per_job >= OPENAI_MAX_TOKENS_PER_JOB:
        raise ValueError(f"The maximum number of chunks per job is {OPENAI_MAX_TOKENS_PER_JOB}. Got {chunks_per_job}")


def _validate_marqo_embedding_args(args):
    """Validates the configuration of the Marqo batch embedder and sets defaults."""
    if not args.embedding_model:
        args.embedding_model = "hf/e5-base-v2"

    if not args.chunks_per_batch:
        args.chunks_per_batch = MARQO_MAX_CHUNKS_PER_BATCH
    elif args.chunks_per_batch > MARQO_MAX_CHUNKS_PER_BATCH:
        args.chunks_per_batch = MARQO_MAX_CHUNKS_PER_BATCH
        logging.warning(
            f"Marqo enforces a limit of {MARQO_MAX_CHUNKS_PER_BATCH} chunks per batch. "
            "Overwriting embeddings.chunks_per_batch."
        )


def validate_embedding_args(args):
    """Validates the configuration of the batch embedder and sets defaults."""
    if args.embedding_provider == "openai":
        _validate_openai_embedding_args(args)
    elif args.embedding_provider == "marqo":
        _validate_marqo_embedding_args(args)
    else:
        raise ValueError(f"Unrecognized --embedding-provider={args.embedding_provider}")


def validate_vector_store_args(args):
    """Validates the configuration of the vector store and sets defaults."""

    if not args.index_namespace:
        args.index_namespace = args.repo_id
        if args.commit_hash:
            args.index_namespace += "/" + args.commit_hash
        if args.vector_store_provider == "marqo":
            # Marqo doesn't allow slashes in the index namespace.
            args.index_namespace = args.index_namespace.replace("/", "_")

    if args.vector_store_provider == "marqo":
        if not args.marqo_url:
            args.marqo_url = "http://localhost:8882"
        if "/" in args.index_namespace:
            raise ValueError(f"Marqo doesn't allow slashes in --index-namespace={args.index_namespace}.")

    elif args.vector_store_provider == "pinecone":
        if not os.getenv("PINECONE_API_KEY"):
            raise ValueError("Please set the PINECONE_API_KEY environment variable.")
        if not args.pinecone_index_name:
            raise ValueError(f"Please set the vector_store.pinecone_index_name value.")


def validate_indexing_args(args):
    """Validates the indexing configuration and sets defaults."""
    if args.include and args.exclude:
        raise ValueError("At most one of indexing.include and indexing.exclude can be specified.")
    if not args.include and not args.exclude:
        args.exclude = pkg_resources.resource_filename(__name__, "sample-exclude.txt")
    if args.include and not os.path.exists(args.include):
        raise ValueError(f"Path --include={args.include} does not exist.")
    if args.exclude and not os.path.exists(args.exclude):
        raise ValueError(f"Path --exclude={args.exclude} does not exist.")
    if not args.index_repo and not args.index_issues:
        raise ValueError("Either --index_repo or --index_issues must be set to true.")
    if args.index_issues and not os.getenv("GITHUB_TOKEN"):
        raise ValueError("Please set the GITHUB_TOKEN environment variable.")
