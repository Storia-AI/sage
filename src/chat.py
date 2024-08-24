"""A gradio app that enables users to chat with their codebase.

You must run main.py first in order to index the codebase into a vector store.
"""

import argparse

from dotenv import load_dotenv

import gradio as gr
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import AIMessage, HumanMessage
from langchain_community.vectorstores import Pinecone
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from repo_manager import RepoManager

load_dotenv()


def build_rag_chain(args):
    """Builds a RAG chain via LangChain."""
    llm = ChatOpenAI(model=args.openai_model)

    vectorstore = Pinecone.from_existing_index(
        index_name=args.pinecone_index_name,
        embedding=OpenAIEmbeddings(),
        namespace=args.repo_id,
    )

    retriever = vectorstore.as_retriever()

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
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

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


def append_sources_to_response(response):
    """Given an OpenAI completion response, appends to it GitHub links of the context sources."""
    filenames = [document.metadata["filename"] for document in response["context"]]
    # Deduplicate filenames while preserving their order.
    filenames = list(dict.fromkeys(filenames))
    repo_manager = RepoManager(args.repo_id)
    github_links = [
        repo_manager.github_link_for_file(filename) for filename in filenames
    ]
    return response["answer"] + "\n\nSources:\n" + "\n".join(github_links)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UI to chat with your codebase")
    parser.add_argument("repo_id", help="The ID of the repository to index")
    parser.add_argument(
        "--openai_model", default="gpt-4", help="The OpenAI model to use for response generation"
    )
    parser.add_argument(
        "--pinecone_index_name", required=True, help="Pinecone index name"
    )
    args = parser.parse_args()

    rag_chain = build_rag_chain(args)

    def _predict(message, history):
        """Performs one RAG operation."""
        history_langchain_format = []
        for human, ai in history:
            history_langchain_format.append(HumanMessage(content=human))
            history_langchain_format.append(AIMessage(content=ai))
        history_langchain_format.append(HumanMessage(content=message))
        response = rag_chain.invoke(
            {"input": message, "chat_history": history_langchain_format}
        )
        answer = append_sources_to_response(response)
        return answer

    gr.ChatInterface(_predict,
                     title=args.repo_id,
                     description=f"Code sage for your repo: {args.repo_id}",
                     examples=["What does this repo do?", "Give me some sample code."]).launch()
