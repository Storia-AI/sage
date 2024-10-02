"""Runs a batch job to compute embeddings for an entire repo and stores them into a vector store."""

import logging
import os
import time

import configargparse

import sage.config as sage_config
from sage.chunker import UniversalFileChunker
from sage.data_manager import GitHubRepoManager
from sage.embedder import build_batch_embedder_from_flags
from sage.github import GitHubIssuesChunker, GitHubIssuesManager
from sage.vector_store import build_vector_store_from_args

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def main():
    parser = configargparse.ArgParser(
        description="Batch-embeds a GitHub repository and its issues.", ignore_unknown_config_file_keys=True
    )
    sage_config.add_config_args(parser)

    arg_validators = [
        sage_config.add_repo_args(parser),
        sage_config.add_embedding_args(parser),
        sage_config.add_vector_store_args(parser),
        sage_config.add_indexing_args(parser),
    ]

    args = parser.parse_args()

    for validator in arg_validators:
        validator(args)

    # Additionally validate embedder and vector store compatibility.
    if args.embedding_provider == "openai" and args.vector_store_provider != "pinecone":
        parser.error("When using OpenAI embedder, the vector store type must be Pinecone.")
    if args.embedding_provider == "marqo" and args.vector_store_provider != "marqo":
        parser.error("When using the marqo embedder, the vector store type must also be marqo.")

    
    ######################
    # Step 1: Embeddings #
    ######################

    # Index the repository.
    repo_embedder = None
    if args.index_repo:
        logging.info("Cloning the repository...")
        repo_manager = GitHubRepoManager(
            args.repo_id,
            commit_hash=args.commit_hash,
            access_token=os.getenv("GITHUB_TOKEN"),
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
        issues_manager = GitHubIssuesManager(
            args.repo_id, access_token=os.getenv("GITHUB_TOKEN"), index_comments=args.index_issue_comments
        )
        issues_manager.download()
        logging.info("Embedding GitHub issues...")
        chunker = GitHubIssuesChunker(max_tokens=args.tokens_per_chunk)
        issues_embedder = build_batch_embedder_from_flags(issues_manager, chunker, args)
        issues_jobs_file = issues_embedder.embed_dataset(args.chunks_per_batch, args.max_embedding_jobs)

    ########################
    # Step 2: Vector Store #
    ########################

    if args.vector_store_provider == "marqo":
        # Marqo computes embeddings and stores them in the vector store at once, so we're done.
        logging.info("Done!")
        return

    if repo_embedder is not None:
        logging.info("Waiting for repo embeddings to be ready...")
        while not repo_embedder.embeddings_are_ready(repo_jobs_file):
            logging.info("Sleeping for 30 seconds...")
            time.sleep(30)

        logging.info("Moving embeddings to the repo vector store...")
        repo_vector_store = build_vector_store_from_args(args, repo_manager)
        repo_vector_store.ensure_exists()
        repo_vector_store.upsert(repo_embedder.download_embeddings(repo_jobs_file), namespace=args.index_namespace)

    if issues_embedder is not None:
        logging.info("Waiting for issue embeddings to be ready...")
        while not issues_embedder.embeddings_are_ready(issues_jobs_file):
            logging.info("Sleeping for 30 seconds...")
            time.sleep(30)

        logging.info("Moving embeddings to the issues vector store...")
        issues_vector_store = build_vector_store_from_args(args, issues_manager)
        issues_vector_store.ensure_exists()
        issues_vector_store.upsert(
            issues_embedder.download_embeddings(issues_jobs_file), namespace=args.index_namespace
        )

    logging.info("Done!")


if __name__ == "__main__":
    main()
