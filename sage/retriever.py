from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_voyageai import VoyageAIEmbeddings

from sage.llm import build_llm_via_langchain
from sage.reranker import build_reranker
from sage.vector_store import build_vector_store_from_args


def build_retriever_from_args(args):
    """Builds a retriever (with optional reranking) from command-line arguments."""

    if args.embedding_provider == "openai":
        embeddings = OpenAIEmbeddings(model=args.embedding_model)
    elif args.embedding_provider == "voyage":
        embeddings = VoyageAIEmbeddings(model=args.embedding_model)
    else:
        embeddings = None

    retriever = build_vector_store_from_args(args).as_retriever(
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
