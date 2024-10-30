"""A gradio app that enables users to chat with their codebase.

You must run `sage-index $GITHUB_REPO` first in order to index the codebase into a vector store.
"""

import logging

import configargparse
import gradio as gr
from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

import sage.config as sage_config
from sage.llm import build_llm_via_langchain
from sage.retriever import build_retriever_from_args

load_dotenv()


def build_rag_chain(args):
    """Builds a RAG chain via LangChain."""
    llm = build_llm_via_langchain(args.llm_provider, args.llm_model)
    retriever = build_retriever_from_args(args)

    # Prompt to contextualize the latest query based on the chat history.
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", (
                "Given a chat history and the latest user question which might reference context in the chat history, "
                "formulate a standalone question which can be understood without the chat history. "
                "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
            )),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm.with_config(tags=["contextualize_q_llm"]),
        retriever,
        contextualize_q_prompt
    )

    qa_system_prompt = (
        f"You are my coding buddy, helping me quickly understand a GitHub repository called {args.repo_id}."
        "Assume I am an advanced developer and answer my questions in the most succinct way possible."
        "\n\nHere are some snippets from the codebase.\n\n{context}"
    )

    question_answer_chain = create_stuff_documents_chain(
        llm, ChatPromptTemplate.from_messages([("system", qa_system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")])
    )

    return create_retrieval_chain(history_aware_retriever, question_answer_chain)


def main():
    parser = configargparse.ArgParser(
        description="Batch-embeds a GitHub repository and its issues.", ignore_unknown_config_file_keys=True
    )
    parser.add("--share", default=False, help="Whether to make the gradio app publicly accessible.")

    validator = sage_config.add_all_args(parser)
    args = parser.parse_args()
    validator(args)

    rag_chain = build_rag_chain(args)

    def source_md(file_path: str, url: str) -> str:
        """Formats a context source in Markdown."""
        return f"[{file_path}]({url})"

    async def _predict(message, history):
        """Performs one RAG operation."""
        history_langchain_format = [
            (HumanMessage(content=human), AIMessage(content=ai))
            for human, ai in history
        ] + [HumanMessage(content=message)]

        response, query_rewrite = "", ""
        async for event in rag_chain.astream_events({"input": message, "chat_history": history_langchain_format}, version="v1"):
            if event["name"] == "retrieve_documents" and "output" in event["data"]:
                sources = {(doc.metadata["file_path"], doc.metadata["url"]) for doc in event["data"]["output"]}
                response += "## Sources:\n" + "\n".join(source_md(*s) for s in sources) + "\n## Response:\n"
            elif event["event"] == "on_chat_model_stream":
                chunk = event["data"]["chunk"].content
                if "contextualize_q_llm" in event["tags"]:
                    query_rewrite += chunk
                else:
                    if not response:
                        logging.info(f"Query rewrite: {query_rewrite}")
                    response += chunk
                    yield response

    gr.ChatInterface(
        _predict,
        title=f"{args.repo_id}" if args.repo_id else "GitHub Repo Chat",
        examples=["What does this repo do?", "Give me some sample code."]
    ).launch(share=args.share)


if __name__ == "__main__":
    main()
