"""A gradio app that enables users to chat with their codebase.

You must run `sage-index $GITHUB_REPO` first in order to index the codebase into a vector store.
"""

import argparse
import logging
import os

import gradio as gr
from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers import ContextualCompressionRetriever
from langchain.schema import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

import sage.vector_store as vector_store
from sage.llm import build_llm_via_langchain
from sage.reranker import build_reranker, RerankerProvider

load_dotenv()


def build_rag_chain(args):
    """Builds a RAG chain via LangChain."""
    llm = build_llm_via_langchain(args.llm_provider, args.llm_model)

    retriever_top_k = 5 if args.reranker_provider == "none" else 25
    retriever = vector_store.build_from_args(args).as_retriever(top_k=retriever_top_k)
    compressor = build_reranker(args.reranker_provider, args.reranker_model)
    if compressor:
        retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)

    # Prompt to contextualize the latest query based on the chat history.
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question which might reference context in the chat history, "
        "formualte a standalone question which can be understood without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    contextualize_q_llm = llm.with_config(tags=["contextualize_q_llm"])
    history_aware_retriever = create_history_aware_retriever(contextualize_q_llm, retriever, contextualize_q_prompt)

    qa_system_prompt = (
        f"You are my coding buddy, helping me quickly understand a GitHub repository called {args.repo_id}."
        "Assume I am an advanced developer and answer my questions in the most succinct way possible."
        "\n\n"
        "Here are some snippets from the codebase."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain


def main():
    parser = argparse.ArgumentParser(description="UI to chat with your codebase")
    parser.add_argument("repo_id", help="The ID of the repository to index")
    parser.add_argument("--llm-provider", default="ollama", choices=["openai", "anthropic", "ollama"])
    parser.add_argument(
        "--llm-model",
        help="The LLM name. Must be supported by the provider specified via --llm-provider.",
    )
    parser.add_argument("--vector-store-type", default="marqo", choices=["pinecone", "marqo"])
    parser.add_argument("--index-name", help="Vector store index name. Required for Pinecone.")
    parser.add_argument(
        "--marqo-url",
        default="http://localhost:8882",
        help="URL for the Marqo server. Required if using Marqo as embedder or vector store.",
    )
    parser.add_argument("--reranker-provider", default="huggingface", choices=[r.value for r in RerankerProvider])
    parser.add_argument(
        "--reranker-model",
        help="The reranker model name. When --reranker-provider=huggingface, we suggest choosing a model from the "
        "SentenceTransformers Cross-Encoders library https://huggingface.co/cross-encoder?sort_models=downloads#models",
    )
    parser.add_argument(
        "--share",
        default=False,
        help="Whether to make the gradio app publicly accessible.",
    )
    parser.add_argument(
        "--hybrid-retrieval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to use a hybrid of vector DB + BM25 retrieval. When set to False, we only use vector DB "
        "retrieval. This is only relevant if using Pinecone as the vector store.",
    )
    args = parser.parse_args()

    if not args.llm_model:
        if args.llm_provider == "openai":
            args.llm_model = "gpt-4"
        elif args.llm_provider == "anthropic":
            args.llm_model = "claude-3-opus-20240229"
        elif args.llm_provider == "ollama":
            args.llm_model = "llama3.1"
        else:
            raise ValueError("Please specify --llm_model")

    rag_chain = build_rag_chain(args)

    def source_md(file_path: str, url: str) -> str:
        """Formats a context source in Markdown."""
        return f"[{file_path}]({url})"

    async def _predict(message, history):
        """Performs one RAG operation."""
        history_langchain_format = []
        for human, ai in history:
            history_langchain_format.append(HumanMessage(content=human))
            history_langchain_format.append(AIMessage(content=ai))
        history_langchain_format.append(HumanMessage(content=message))

        query_rewrite = ""
        response = ""
        async for event in rag_chain.astream_events(
            {
                "input": message,
                "chat_history": history_langchain_format,
            },
            version="v1",
        ):
            if event["name"] == "retrieve_documents" and "output" in event["data"]:
                sources = [(doc.metadata["file_path"], doc.metadata["url"]) for doc in event["data"]["output"]]
                # Deduplicate while preserving the order.
                sources = list(dict.fromkeys(sources))
                response += "## Sources:\n" + "\n".join([source_md(s[0], s[1]) for s in sources]) + "\n## Response:\n"

            elif event["event"] == "on_chat_model_stream":
                chunk = event["data"]["chunk"].content

                if "contextualize_q_llm" in event["tags"]:
                    query_rewrite += chunk
                else:
                    # This is the actual response to the user query.
                    if not response:
                        logging.info(f"Query rewrite: {query_rewrite}")
                    response += chunk
                    yield response

    gr.ChatInterface(
        _predict,
        title=args.repo_id,
        examples=["What does this repo do?", "Give me some sample code."],
    ).launch(share=args.share)


if __name__ == "__main__":
    main()
