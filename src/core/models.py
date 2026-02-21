"""
Доменные модели RAG-системы.

Содержит dataclass-ы, которые используются на всех этапах пайплайна:
от чанкинга PDF до финального вывода результатов поиска.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Sequence


@dataclass
class Chunk:
    """
    Единица хранения текста в базе знаний.

    Один чанк соответствует одной странице PDF-документа.
    Поля метаданных (geo, metrics, years и др.) заполняются
    на этапе LLM-обогащения (LLMEnricher).

    Attributes:
        id: Уникальный идентификатор чанка (формат: ``{pdf_name}::page{N}::chunk{M}``).
        context: Краткое описание содержания чанка (генерируется LLM).
        text: Полный текст страницы PDF.
        source: Имя исходного PDF-файла.
        page: Номер страницы в PDF.
        doc_id: Опциональный идентификатор документа.
        geo: Территории, упомянутые в чанке.
        metrics: Названия статистических показателей.
        years: Годы, явно упомянутые в данных.
        time_granularity: Гранулярность временного ряда (year / quarter / month).
        oked: Код ОКЭД (общегосударственный классификатор видов экономической деятельности).
        extra: Дополнительные поля, не вошедшие в основную схему.
    """
    id: str
    context: str
    text: str
    source: str
    page: int

    doc_id: Optional[str] = None

    geo: Optional[str] = None
    metrics: Optional[List[str]] = None
    years: Optional[List[int]] = None
    time_granularity: Optional[str] = None
    oked: Optional[str] = None

    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnrichedQuery:
    """
    Обогащённый пользовательский запрос.

    Содержит исходный текст запроса, его векторное представление
    и структурированные поля, извлечённые LLM для metadata-scoring.

    Attributes:
        query: Текст запроса пользователя.
        embedded_query: Векторное представление запроса (float32).
        geo: Территории, упомянутые в запросе.
        years: Годы, указанные в запросе.
        metrics: Запрашиваемые показатели.
        time_granularity: Гранулярность временного ряда.
        oked: Код ОКЭД.
        raw_llm_response: Сырой ответ LLM (для отладки).
    """
    query: str
    embedded_query: Sequence[float]

    geo: Optional[str] = None
    years: Optional[List[int]] = None
    metrics: Optional[List[str]] = None
    time_granularity: Optional[str] = None
    oked: Optional[str] = None

    raw_llm_response: Optional[str] = None


@dataclass
class ScoredChunk:
    """
    Чанк с оценками релевантности по каждому каналу поиска.

    Attributes:
        chunk: Исходный чанк из базы знаний.
        semantic_score: Оценка семантического сходства (FAISS cosine similarity).
        lexical_score: Оценка лексического совпадения (BM25).
        metadata_score: Оценка совпадения метаданных (geo, years, metrics).
        hybrid_score: Итоговая взвешенная оценка.
        rerank_score: Оценка после переранжирования (не используется в текущей версии).
    """
    chunk: Chunk

    semantic_score: float = 0.0
    lexical_score: float = 0.0
    metadata_score: float = 0.0

    hybrid_score: float = 0.0
    rerank_score: float = 0.0


@dataclass
class PipelineResult:
    """
    Результат работы QueryPipeline.

    Attributes:
        query: Исходный запрос пользователя.
        enriched_query: Обогащённый запрос с метаданными и эмбеддингом.
        candidates: Все кандидаты после гибридного поиска.
        top_chunks: Финальный Top-K чанков для вывода.
        timings: Замеры времени по этапам пайплайна (для профилирования).
    """
    query: str
    enriched_query: EnrichedQuery
    candidates: List[ScoredChunk]
    top_chunks: List[ScoredChunk]

    timings: Dict[str, float] = field(default_factory=dict)
