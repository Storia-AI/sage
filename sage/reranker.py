import logging
import os
from enum import Enum
from typing import List, Optional

from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_cohere import CohereRerank
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.document_compressors import JinaRerank
from langchain_core.callbacks.manager import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_nvidia_ai_endpoints import NVIDIARerank
from langchain_voyageai import VoyageAIRerank
from pydantic import ConfigDict, Field

from sage.llm import build_llm_via_langchain


class RerankerProvider(Enum):
    NONE = "none"
    HUGGINGFACE = "huggingface"
    COHERE = "cohere"
    NVIDIA = "nvidia"
    JINA = "jina"
    VOYAGE = "voyage"
    # Anthropic doesn't provide an explicit reranker; we simply prompt the LLM with the user query and the content of
    # the top k documents.
    ANTHROPIC = "anthropic"


class LLMReranker(BaseDocumentCompressor):
    """Reranker that passes the user query and top N documents to a language model to order them.

    Note that Langchain's RerankLLM does not support LLMs from Anthropic.
    https://python.langchain.com/api_reference/community/document_compressors/langchain_community.document_compressors.rankllm_rerank.RankLLMRerank.html
    Also, they rely on https://github.com/castorini/rank_llm, which doesn't run on Apple Silicon (M1/M2 chips).
    """

    llm: BaseLanguageModel = Field(...)
    top_k: int = Field(...)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @property
    def prompt(self):
        return PromptTemplate.from_template(
            "Given the following query: '{query}'\n\n"
            "And these documents:\n\n{documents}\n\n"
            "Rank the documents based on their relevance to the query. "
            "Return only the document numbers in order of relevance, separated by commas. For example: 2,5,1,3,4. "
            "Return absolutely nothing else."
        )

    def compress_documents(
        self,
        documents: List[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> List[Document]:
        if len(documents) <= self.top_k:
            return documents

        doc_texts = [f"Document {i+1}:\n{doc.page_content}\n" for i, doc in enumerate(documents)]
        docs_str = "\n".join(doc_texts)

        llm_input = self.prompt.format(query=query, documents=docs_str)
        result = self.llm.predict(llm_input)

        try:
            ranked_indices = [int(idx) - 1 for idx in result.strip().split(",")][: self.top_k]
            return [documents[i] for i in ranked_indices]
        except ValueError:
            logging.warning("Failed to parse reranker output. Returning original order. LLM responded with: %s", result)
            return documents[: self.top_k]


def build_reranker(provider: str, model: Optional[str] = None, top_k: Optional[int] = 5) -> BaseDocumentCompressor:
    if provider == RerankerProvider.NONE.value:
        return None
    if provider == RerankerProvider.HUGGINGFACE.value:
        model = model or "cross-encoder/ms-marco-MiniLM-L-6-v2"
        encoder_model = HuggingFaceCrossEncoder(model_name=model)
        return CrossEncoderReranker(model=encoder_model, top_n=top_k)
    if provider == RerankerProvider.COHERE.value:
        if not os.environ.get("COHERE_API_KEY"):
            raise ValueError("Please set the COHERE_API_KEY environment variable")
        model = model or "rerank-english-v3.0"
        return CohereRerank(model=model, cohere_api_key=os.environ.get("COHERE_API_KEY"), top_n=top_k)
    if provider == RerankerProvider.NVIDIA.value:
        if not os.environ.get("NVIDIA_API_KEY"):
            raise ValueError("Please set the NVIDIA_API_KEY environment variable")
        model = model or "nvidia/nv-rerankqa-mistral-4b-v3"
        return NVIDIARerank(model=model, api_key=os.environ.get("NVIDIA_API_KEY"), top_n=top_k, truncate="END")
    if provider == RerankerProvider.JINA.value:
        if not os.environ.get("JINA_API_KEY"):
            raise ValueError("Please set the JINA_API_KEY environment variable")
        return JinaRerank(top_n=top_k)
    if provider == RerankerProvider.VOYAGE.value:
        if not os.environ.get("VOYAGE_API_KEY"):
            raise ValueError("Please set the VOYAGE_API_KEY environment variable")
        model = model or "rerank-1"
        return VoyageAIRerank(model=model, api_key=os.environ.get("VOYAGE_API_KEY"), top_k=top_k)
    if provider == RerankerProvider.ANTHROPIC.value:
        llm = build_llm_via_langchain("anthropic", model)
        return LLMReranker(llm=llm, top_k=1)
    raise ValueError(f"Invalid reranker provider: {provider}")
