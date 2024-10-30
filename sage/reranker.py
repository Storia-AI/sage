import os
from enum import Enum
from typing import Optional

from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_cohere import CohereRerank
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.document_compressors import JinaRerank
from langchain_core.documents import BaseDocumentCompressor
from langchain_nvidia_ai_endpoints import NVIDIARerank
from langchain_voyageai import VoyageAIRerank


class RerankerProvider(Enum):
    NONE = "none"
    HUGGINGFACE = "huggingface"
    COHERE = "cohere"
    NVIDIA = "nvidia"
    JINA = "jina"
    VOYAGE = "voyage"


def build_reranker(provider: str, model: Optional[str] = None, top_k: int = 5) -> Optional[BaseDocumentCompressor]:
    if provider == RerankerProvider.NONE.value:
        return None

    api_key_env_vars = {
        RerankerProvider.COHERE.value: "COHERE_API_KEY",
        RerankerProvider.NVIDIA.value: "NVIDIA_API_KEY",
        RerankerProvider.JINA.value: "JINA_API_KEY",
        RerankerProvider.VOYAGE.value: "VOYAGE_API_KEY"
    }

    provider_defaults = {
        RerankerProvider.HUGGINGFACE.value: "cross-encoder/ms-marco-MiniLM-L-6-v2",
        RerankerProvider.COHERE.value: "rerank-english-v3.0",
        RerankerProvider.NVIDIA.value: "nvidia/nv-rerankqa-mistral-4b-v3",
        RerankerProvider.VOYAGE.value: "rerank-1"
    }

    model = model or provider_defaults.get(provider)

    if provider == RerankerProvider.HUGGINGFACE.value:
        encoder_model = HuggingFaceCrossEncoder(model_name=model)
        return CrossEncoderReranker(model=encoder_model, top_n=top_k)

    if provider in api_key_env_vars:
        api_key = os.getenv(api_key_env_vars[provider])
        if not api_key:
            raise ValueError(f"Please set the {api_key_env_vars[provider]} environment variable")

        if provider == RerankerProvider.COHERE.value:
            return CohereRerank(model=model, cohere_api_key=api_key, top_n=top_k)

        if provider == RerankerProvider.NVIDIA.value:
            return NVIDIARerank(model=model, api_key=api_key, top_n=top_k, truncate="END")

        if provider == RerankerProvider.JINA.value:
            return JinaRerank(top_n=top_k)

        if provider == RerankerProvider.VOYAGE.value:
            return VoyageAIRerank(model=model, api_key=api_key, top_k=top_k)

    raise ValueError(f"Invalid reranker provider: {provider}")
