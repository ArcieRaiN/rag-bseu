from __future__ import annotations

"""
Высокоуровневый пайплайн обработки пользовательского запроса (PIPELINE 2–4).

ВАЖНО:
- пайплайн ОСТАНАВЛИВАЕТСЯ на Top‑3 чанках после reranking
- генерация финального ответа (LLM Answer Generation) здесь сознательно не реализована
"""

from pathlib import Path
from typing import List

from src.main.context_enrichment import QueryContextEnricher
from src.main.hybrid_search import HybridSearcher
from src.main.models import (
    PipelineResult,
    RetrievalConfig,
    RerankConfig,
)
from src.main.ollama_client import OllamaClient
from src.main.vectorizer import HashVectorizer
from src.main.semantic_search import FaissSemanticSearcher
from src.main.reranker import LLMReranker


class QueryPipelineV2:
    """
    Архитектурный фасад над:
    - QueryContextEnricher (PIPELINE 2)
    - HybridSearcher (PIPELINE 3)
    - LLMReranker (PIPELINE 4)

    Используется CLI и любыми внешними потребителями.
    """

    def __init__(
        self,
        base_dir: Path,
        *,
        ollama_client: OllamaClient | None = None,
        vector_dim: int = 256,
        retrieval_config: RetrievalConfig | None = None,
        rerank_config: RerankConfig | None = None,
    ):
        self._base_dir = Path(base_dir)

        # Настройка компонентов
        self._ollama = ollama_client or OllamaClient()
        self._vectorizer = HashVectorizer(dimension=vector_dim)

        # Vector store (FAISS + data.json)
        vector_store_dir = self._base_dir / "prepare_db" / "vector_store"
        index_path = vector_store_dir / "index.faiss"
        data_path = vector_store_dir / "data.json"

        self._semantic = FaissSemanticSearcher(index_path=index_path, data_path=data_path)
        self._retrieval_config = retrieval_config or RetrievalConfig()
        self._hybrid = HybridSearcher(
            semantic_searcher=self._semantic,
            retrieval_config=self._retrieval_config,
        )

        self._enricher = QueryContextEnricher(
            vectorizer=self._vectorizer,
            llm_client=self._ollama,
        )

        self._rerank_config = rerank_config or RerankConfig()
        self._reranker = LLMReranker(
            llm_client=self._ollama,
            config=self._rerank_config,
        )

    # -------------------- Публичный интерфейс -------------------- #

    def run(self, query: str) -> PipelineResult:
        """
        Выполнить полный пайплайн без генерации ответа:
        1. Обогатить запрос (enrichment + embedding)
        2. Получить Top‑10 кандидатов гибридным поиском
        3. Переранжировать кандидатов через LLM‑reranker до Top‑3
        """
        enriched = self._enricher.enrich(query)
        hybrid_result = self._hybrid.search(enriched)
        candidates = hybrid_result.candidates

        top_chunks = self._reranker.rerank(enriched, candidates.copy())

        return PipelineResult(
            query=query,
            enriched_query=enriched,
            candidates=candidates,
            top_chunks=top_chunks,
        )

