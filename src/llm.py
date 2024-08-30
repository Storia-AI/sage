import os

from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


def build_llm_via_langchain(provider: str, model: str):
    """Builds a language model via LangChain."""
    if provider == "openai":
        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError("Please set the OPENAI_API_KEY environment variable.")
        return ChatOpenAI(model=model)
    elif provider == "anthropic":
        if "ANTHROPIC_API_KEY" not in os.environ:
            raise ValueError("Please set the ANTHROPIC_API_KEY environment variable.")
        return ChatAnthropic(model=model)
    elif provider == "ollama":
        return ChatOllama(model=model)
    else:
        raise ValueError(f"Unrecognized LLM provider {provider}. Contributons are welcome!")
