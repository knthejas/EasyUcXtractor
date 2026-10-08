"""Central configuration for Ollama and legacy OpenAI-compatible settings."""

import os

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen2.5:3b"


def llm_settings() -> tuple[str, str, str | None]:
    url = (os.getenv("RAG_LLM_URL") or os.getenv("OLLAMA_URL") or DEFAULT_OLLAMA_URL).strip()
    model = (os.getenv("RAG_LLM_MODEL") or os.getenv("OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL).strip()
    api_key = os.getenv("RAG_LLM_API_KEY") or None
    return url, model, api_key
