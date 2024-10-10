import logging
import os
from typing import List, Optional

import anthropic
import Levenshtein
from anytree import Node, RenderTree
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.schema import BaseRetriever, Document
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_voyageai import VoyageAIEmbeddings
from pydantic import Field

from sage.data_manager import DataManager, GitHubRepoManager
from sage.llm import build_llm_via_langchain
from sage.reranker import build_reranker
from sage.vector_store import build_vector_store_from_args

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class LLMRetriever(BaseRetriever):
    """Custom Langchain retriever based on an LLM.

    Builds a representation of the folder structure of the repo, feeds it to an LLM, and asks the LLM for the most
    relevant files for a particular user query, expecting it to make decisions based solely on file names.

    Only works with Claude/Anthropic, because it's very slow (e.g. 15s for a mid-sized codebase) and we need prompt
    caching to make it usable.
    """

    repo_manager: GitHubRepoManager = Field(...)
    top_k: int = Field(...)
    all_repo_files: List[str] = Field(...)
    repo_hierarchy: str = Field(...)

    def __init__(self, repo_manager: GitHubRepoManager, top_k: int):
        super().__init__()
        self.repo_manager = repo_manager
        self.top_k = top_k

        # Best practice would be to make these fields @cached_property, but that impedes class serialization.
        self.all_repo_files = [metadata["file_path"] for metadata in self.repo_manager.walk(get_content=False)]
        self.repo_hierarchy = LLMRetriever._render_file_hierarchy(self.all_repo_files)

        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ValueError("Please set the ANTHROPIC_API_KEY environment variable for the LLMRetriever.")

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        """Retrieve relevant documents for a given query."""
        filenames = self._ask_llm_to_retrieve(user_query=query, top_k=self.top_k)
        documents = []
        for filename in filenames:
            document = Document(
                page_content=self.repo_manager.read_file(filename),
                metadata={"file_path": filename, "url": self.repo_manager.url_for_file(filename)},
            )
            documents.append(document)
        return documents

    def _ask_llm_to_retrieve(self, user_query: str, top_k: int) -> List[str]:
        """Feeds the file hierarchy and user query to the LLM and asks which files might be relevant."""
        sys_prompt = f"""
You are a retriever system. You will be given a user query and a list of files in a GitHub repository. Your task is to determine the top {top_k} files that are most relevant to the user query.
DO NOT RESPOND TO THE USER QUERY DIRECTLY. Instead, respond with full paths to relevant files that could contain the answer to the query. Say absolutely nothing else other than the file paths.

Here is the file hierarchy of the GitHub repository:

{self.repo_hierarchy}
"""
        # We are deliberately repeating the "DO NOT RESPOND TO THE USER QUERY DIRECTLY" instruction here.
        augmented_user_query = f"""
User query: {user_query}

DO NOT RESPOND TO THE USER QUERY DIRECTLY. Instead, respond with full paths to relevant files that could contain the answer to the query. Say absolutely nothing else other than the file paths.
"""
        response = LLMRetriever._call_via_anthropic_with_prompt_caching(sys_prompt, augmented_user_query)
        files_from_llm = response.content[0].text.strip().split("\n")
        validated_files = []

        for filename in files_from_llm:
            if filename not in self.all_repo_files:
                if "/" not in filename:
                    # This is most likely some natural language excuse from the LLM; skip it.
                    continue
                # Try a few heuristics to fix the filename.
                filename = LLMRetriever._fix_filename(filename, self.repo_manager.repo_id)
                if filename not in self.all_repo_files:
                    # The heuristics failed; try to find the closest filename in the repo.
                    filename = LLMRetriever._find_closest_filename(filename, self.all_repo_files)
            if filename in self.all_repo_files:
                validated_files.append(filename)
        return validated_files

    @staticmethod
    def _call_via_anthropic_with_prompt_caching(system_prompt: str, user_prompt: str) -> str:
        """Calls the Anthropic API with prompt caching for the system prompt.

        See https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching.

        We're circumventing LangChain for now, because the feature is < 1 week old at the time of writing and has no
        documentation: https://github.com/langchain-ai/langchain/pull/27087
        """
        CLAUDE_MODEL = "claude-3-5-sonnet-20240620"
        client = anthropic.Anthropic()

        system_message = {"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}
        user_message = {"role": "user", "content": user_prompt}

        response = client.beta.prompt_caching.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1024,  # The maximum number of *output* tokens to generate.
            system=[system_message],
            messages=[user_message],
        )
        # Caching information will be under `cache_creation_input_tokens` and `cache_read_input_tokens`.
        # Note that, for prompts shorter than 1024 tokens, Anthropic will not do any caching.
        logging.info("Anthropic prompt caching info: %s", response.usage)
        return response

    @staticmethod
    def _render_file_hierarchy(file_paths: List[str]) -> str:
        """Given a list of files, produces a visualization of the file hierarchy. For instance:
        folder1
            folder11
                file111.py
                file112.py
            folder12
                file121.py
        folder2
            file21.py
        """
        # The "nodepath" is the path from root to the node (e.g. huggingface/transformers/examples)
        nodepath_to_node = {}

        for path in file_paths:
            items = path.split("/")
            nodepath = ""
            parent_node = None
            for item in items:
                nodepath = f"{nodepath}/{item}"
                if nodepath in nodepath_to_node:
                    node = nodepath_to_node[nodepath]
                else:
                    node = Node(item, parent=parent_node)
                    nodepath_to_node[nodepath] = node
                parent_node = node

        root_path = f"/{file_paths[0].split('/')[0]}"
        full_render = ""
        root_node = nodepath_to_node[root_path]
        for pre, fill, node in RenderTree(root_node):
            render = "%s%s\n" % (pre, node.name)
            # Replace special lines with empty strings to save on tokens.
            render = render.replace("└", " ").replace("├", " ").replace("│", " ").replace("─", " ")
            full_render += render
        return full_render

    @staticmethod
    def _fix_filename(filename: str, repo_id: str) -> str:
        """Attempts to "fix" a filename output by the LLM.

        Common issues with LLM-generated filenames:
        - The LLM prepends an extraneous "/".
        - The LLM omits the name of the org (e.g. "transformers/README.md" instead of "huggingface/transformers/README.md").
        - The LLM omits the name of the repo (e.g. "huggingface/README.md" instead of "huggingface/transformers/README.md").
        - The LLM omits the org/repo prefix (e.g. "README.md" instead of "huggingface/transformers/README.md").
        """
        if filename.startswith("/"):
            filename = filename[1:]
        org_name, repo_name = repo_id.split("/")
        items = filename.split("/")
        if filename.startswith(org_name) and not filename.startswith(repo_id):
            new_items = [org_name, repo_name] + items[1:]
            return "/".join(new_items)
        if not filename.startswith(org_name) and filename.startswith(repo_name):
            return f"{org_name}/{filename}"
        if not filename.startswith(org_name) and not filename.startswith(repo_name):
            return f"{org_name}/{repo_name}/{filename}"
        return filename

    @staticmethod
    def _find_closest_filename(filename: str, repo_filenames: List[str], max_edit_distance: int = 10) -> Optional[str]:
        """Returns the path in the repo with smallest edit distance from `filename`. Helpful when the `filename` was
        generated by an LLM and parts of it might have been hallucinated. Returns None if the closest path is more than
        `max_edit_distance` away. In case of a tie, returns an arbitrary closest path.
        """
        distances = [(path, Levenshtein.distance(filename, path)) for path in repo_filenames]
        distances.sort(key=lambda x: x[1])
        if distances[0][1] <= max_edit_distance:
            closest_path = distances[0][0]
            return closest_path
        return None


def build_retriever_from_args(args, data_manager: Optional[DataManager] = None):
    """Builds a retriever (with optional reranking) from command-line arguments."""
    if args.llm_retriever:
        retriever = LLMRetriever(GitHubRepoManager.from_args(args), top_k=args.retriever_top_k)
    else:
        if args.embedding_provider == "openai":
            embeddings = OpenAIEmbeddings(model=args.embedding_model)
        elif args.embedding_provider == "voyage":
            embeddings = VoyageAIEmbeddings(model=args.embedding_model)
        elif args.embedding_provider == "gemini":
            embeddings = GoogleGenerativeAIEmbeddings(model=args.embedding_model)
        else:
            embeddings = None

        retriever = build_vector_store_from_args(args, data_manager).as_retriever(
            top_k=args.retriever_top_k, embeddings=embeddings, namespace=args.index_namespace
        )

    if args.multi_query_retriever:
        retriever = MultiQueryRetriever.from_llm(
            retriever=retriever, llm=build_llm_via_langchain(args.llm_provider, args.llm_model)
        )

    reranker = build_reranker(args.reranker_provider, args.reranker_model, args.reranker_top_k)
    if reranker:
        retriever = ContextualCompressionRetriever(base_compressor=reranker, base_retriever=retriever)
    return retriever
