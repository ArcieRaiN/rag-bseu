from __future__ import annotations

"""
Общие модели данных и интерфейсные структуры для RAG v2.

Вынесены в отдельный модуль, чтобы:
- избежать циклических импортов между retrieval / reranking / enrichment
- сделать типы переиспользуемыми в тестах и CLI
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import numpy as np


@dataclass
class Chunk:
    """
    Чанк документа в базе знаний.

    ВАЖНО:
    - `context` — обогащённое описание чанка (генерируется LLM на основе всего документа)
    - `text`    — оригинальный текст чанка (сырая разбивка LlamaIndex)
    """

    id: str
    context: str
    text: str
    source: str
    page: int

    # Семантические и бизнес‑метаданные, извлечённые LLM
    geo: Optional[str] = None
    metrics: Optional[List[str]] = None
    years: Optional[List[int]] = None
    time_granularity: Optional[str] = None  # e.g. "year"
    oked: Optional[str] = None

    # Дополнительные поля на будущее (чтобы не ломать интерфейсы)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnrichedQuery:
    """
    Результат обогащения пользовательского запроса.

    Используется и в retrieval, и в reranking.
    """

    query: str
    embedded_query: np.ndarray

    geo: str
    years: List[int]
    metrics: Optional[List[str]] = None
    time_granularity: Optional[str] = None
    oked: Optional[str] = None

    # Сырые данные от LLM (на случай отладки)
    raw_llm_response: Optional[str] = None


@dataclass
class ScoredChunk:
    """
    Чанк с набором скоринговых сигналов.

    ВАЖНО: hybrid_score и rerank_score логически независимы.
    """

    chunk: Chunk

    # Отдельные компоненты поиска
    semantic_score: float = 0.0
    lexical_score: float = 0.0
    metadata_score: float = 0.0

    # Сводный скор гибридного поиска (используется только для отбора top‑K)
    hybrid_score: float = 0.0

    # Скор от reranker‑а (LLM / cross‑encoder); основывается только на смысле
    rerank_score: float = 0.0


@dataclass
class RetrievalConfig:
    """
    Конфигурация весов и параметров гибридного поиска.
    """

    # Размер кандидатов перед reranking
    semantic_top_k: int = 20
    lexical_top_k: int = 20
    final_top_k: int = 10

    # Веса компонент hybrid score
    w_semantic: float = 0.55
    w_lexical: float = 0.25
    w_metadata: float = 0.20


@dataclass
class RerankConfig:
    """
    Конфигурация reranking‑этапа.
    """

    top_k: int = 3
    model_name: str = "llama2"
    temperature: float = 0.0


@dataclass
class PipelineResult:
    """
    Высокоуровневый результат пайплайна обработки запроса (без генерации ответа).

    Используется CLI для вывода top‑3 чанков.
    """

    query: str
    enriched_query: EnrichedQuery
    candidates: List[ScoredChunk]  # Top‑10 из hybrid retrieval
    top_chunks: List[ScoredChunk]  # Top‑3 после reranking

