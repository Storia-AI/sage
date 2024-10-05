"""Script to call retrieval on a benchmark dataset.

Make sure to `pip install ir_measures` before running this script.
"""

import json
import logging
import os
import time

import configargparse
from dotenv import load_dotenv
from ir_measures import MAP, MRR, P, Qrel, R, Rprec, ScoredDoc, calc_aggregate, nDCG

import sage.config
from sage.data_manager import GitHubRepoManager
from sage.retriever import build_retriever_from_args

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

load_dotenv()


def main():
    parser = configargparse.ArgParser(
        description="Runs retrieval on a benchmark dataset.", ignore_unknown_config_file_keys=True
    )
    parser.add("--benchmark", required=True, help="Path to the benchmark dataset.")
    parser.add(
        "--gold-field", default="context_files", help="Field in the benchmark dataset that contains the golden answers."
    )
    parser.add(
        "--question-field", default="question", help="Field in the benchmark dataset that contains the questions."
    )
    parser.add(
        "--logs-dir",
        default=None,
        help="Path where to output predictions and metrics. Optional, since metrics are also printed to console.",
    )

    parser.add("--max-instances", default=None, type=int, help="Maximum number of instances to process.")
    sage.config.add_config_args(parser)
    sage.config.add_llm_args(parser)  # Needed for --multi-query-retriever, which rewrites the query with an LLM.
    sage.config.add_embedding_args(parser)
    sage.config.add_vector_store_args(parser)
    sage.config.add_reranking_args(parser)
    sage.config.add_repo_args(parser)
    sage.config.add_indexing_args(parser)
    args = parser.parse_args()
    sage.config.validate_vector_store_args(args)
    repo_manager = GitHubRepoManager(
        args.repo_id,
        commit_hash=args.commit_hash,
        access_token=os.getenv("GITHUB_TOKEN"),
        local_dir=args.local_dir,
        inclusion_file=args.include,
        exclusion_file=args.exclude,
    )
    repo_manager.download()
    retriever = build_retriever_from_args(args, repo_manager)

    with open(args.benchmark, "r") as f:
        benchmark = json.load(f)
    if args.max_instances is not None:
        benchmark = benchmark[: args.max_instances]

    golden_docs = []  # List of ir_measures.Qrel objects
    retrieved_docs = []  # List of ir_measures.ScoredDoc objects

    for question_idx, item in enumerate(benchmark):
        print(f"Processing question {question_idx}...")

        query_id = str(question_idx)  # Solely needed for ir_measures library.

        for golden_filepath in item[args.gold_field]:
            # All the file paths in the golden answer are equally relevant for the query (i.e. the order is irrelevant),
            # so we set relevance=1 for all of them.
            golden_docs.append(Qrel(query_id=query_id, doc_id=golden_filepath, relevance=1))

        # Make a retrieval call for the current question.
        retrieved = retriever.invoke(item[args.question_field])
        item["retrieved"] = []
        for doc_idx, doc in enumerate(retrieved):
            # The absolute value of the scores below does not affect the metrics; it merely determines the ranking of
            # the retrieved documents. The key of the score varies depending on the underlying retriever. If there's no
            # score, we use 1/(doc_idx+1) since it preserves the order of the documents.
            score = doc.metadata.get("score", doc.metadata.get("relevance_score", 1 / (doc_idx + 1)))
            retrieved_docs.append(ScoredDoc(query_id=query_id, doc_id=doc.metadata["file_path"], score=score))
            # Update the output dictionary with the retrieved documents.
            item["retrieved"].append({"file_path": doc.metadata["file_path"], "score": score})

        if "answer" in item:
            item.pop("answer")  # Makes the output file harder to read.

    print("Calculating metrics...")
    results = calc_aggregate([Rprec, P @ 1, R @ 3, nDCG @ 3, MAP, MRR], golden_docs, retrieved_docs)
    results = {str(key): value for key, value in results.items()}
    if args.logs_dir:
        if not os.path.exists(args.logs_dir):
            os.makedirs(args.logs_dir)

        out_data = {
            "data": benchmark,
            "metrics": results,
            "flags": vars(args),  # For reproducibility.
        }

        output_file = os.path.join(args.logs_dir, f"{time.time()}.json")
        with open(output_file, "w") as f:
            json.dump(out_data, f, indent=4)

    for key in sorted(results.keys()):
        print(f"{key}: {results[key]}")
    print(f"Predictions and metrics saved to {output_file}")


if __name__ == "__main__":
    main()
