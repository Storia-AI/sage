import os
from enum import Enum
from typing import Optional

from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_cohere import CohereRerank
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.document_compressors import JinaRerank
from langchain_core.documents import BaseDocumentCompressor
from langchain_nvidia_ai_endpoints import NVIDIARerank


class RerankerProvider(Enum):
    NONE = "none"
    HUGGINGFACE = "huggingface"
    COHERE = "cohere"
    NVIDIA = "nvidia"
    JINA = "jina"


def build_reranker(provider: str, model: Optional[str] = None, top_n: Optional[int] = 5) -> BaseDocumentCompressor:
    if provider == RerankerProvider.NONE.value:
        return None
    if provider == RerankerProvider.HUGGINGFACE.value:
        model = model or "cross-encoder/ms-marco-MiniLM-L-6-v2"
        encoder_model = HuggingFaceCrossEncoder(model_name=model)
        return CrossEncoderReranker(model=encoder_model, top_n=top_n)
    if provider == RerankerProvider.COHERE.value:
        if not os.environ.get("COHERE_API_KEY"):
            raise ValueError("Please set the COHERE_API_KEY environment variable")
        model = model or "rerank-english-v3.0"
        return CohereRerank(model=model, cohere_api_key=os.environ.get("COHERE_API_KEY"), top_n=top_n)
    if provider == RerankerProvider.NVIDIA.value:
        if not os.environ.get("NVIDIA_API_KEY"):
            raise ValueError("Please set the NVIDIA_API_KEY environment variable")
        model = model or "nvidia/nv-rerankqa-mistral-4b-v3"
        return NVIDIARerank(model=model, api_key=os.environ.get("NVIDIA_API_KEY"), top_n=top_n, truncate="END")
    if provider == RerankerProvider.JINA.value:
        if not os.environ.get("JINA_API_KEY"):
            raise ValueError("Please set the JINA_API_KEY environment variable")
        return JinaRerank(top_n=top_n)
    raise ValueError(f"Invalid reranker provider: {provider}")
