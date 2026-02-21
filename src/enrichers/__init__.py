"""
Пакет enrichers -- обогащение данных через LLM (Ollama).

Предоставляет OllamaClient для HTTP-взаимодействия с Ollama API,
LLMEnricher для обогащения чанков метаданными и парсеры JSON-ответов.
"""

from .client import OllamaClient, OllamaConfig
from .config import EnricherConfig
from .parsers import parse_single_enrichment
from .enrichers import LLMEnricher

__all__ = [
    "OllamaClient",
    "OllamaConfig",
    "EnricherConfig",
    "parse_single_enrichment",
    "LLMEnricher",
]
