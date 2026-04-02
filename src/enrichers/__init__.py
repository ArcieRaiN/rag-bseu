"""
Пакет enrichers — обогащение данных через LLM (Ollama).

  OllamaClient   HTTP-клиент для Ollama API с retry и сбросом контекста
  LLMEnricher    Последовательное обогащение чанков метаданными
  parsers        Парсинг JSON-ответов от LLM (code fences, несбалансированные скобки)
"""

from .ollama_client import OllamaClient, OllamaConfig
from .config import EnricherConfig
from .parsers import parse_single_enrichment
from .llm_enricher import LLMEnricher

__all__ = [
    "OllamaClient",
    "OllamaConfig",
    "EnricherConfig",
    "parse_single_enrichment",
    "LLMEnricher",
]
