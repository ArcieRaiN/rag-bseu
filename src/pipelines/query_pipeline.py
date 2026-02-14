from __future__ import annotations

"""
PIPELINE: обработка пользовательского запроса (RAG + hybrid retrieval).

Этапы:
1. Query enrichment (LLM / vectorizer)
2. Hybrid search (BM25 + semantic FAISS)
"""

from pathlib import Path
import time
from typing import List

from src.core.context_enrichment import QueryContextEnricher
from src.retrieval.hybrid_search import HybridSearcher
from src.retrieval.semantic_search import FaissSemanticSearcher
from src.vectorstore.vectorizer import SentenceVectorizer
from src.core.models import PipelineResult, ScoredChunk
from src.core.config import RetrievalConfig
from src.enrichers.client import OllamaClient


class QueryPipeline:
    def __init__(
        self,
        base_dir: Path,
        *,
        llm_model: str = "llama3-chatqa:latest",
        vector_dim: int = 256,
        retrieval_config: RetrievalConfig | None = None,
    ):
        """
        Инициализация пайплайна.

        Args:
            base_dir: Корень проекта (rag-bseu)
            llm_model: Модель LLM для enrichment
            vector_dim: Размерность эмбеддингов
            retrieval_config: Настройки поиска
        """
        t0 = time.perf_counter()
        print("[INIT] QueryPipeline: инициализация...")

        self._base_dir = Path(base_dir)
        self._ollama = OllamaClient()
        self._vectorizer = SentenceVectorizer(dimension=vector_dim)

        vector_store_dir = self._base_dir / "usage" / "vector_store"
        self._semantic = FaissSemanticSearcher(
            index_path=vector_store_dir / "index.faiss",
            data_path=vector_store_dir / "data.json",
        )

        self._retrieval_config = retrieval_config or RetrievalConfig()
        self._hybrid = HybridSearcher(
            semantic_searcher=self._semantic,
            config=self._retrieval_config
        )

        self._enricher = QueryContextEnricher(
            vectorizer=self._vectorizer,
            llm_client=self._ollama,
        )

        print(f"[INIT] QueryPipeline готов за {time.perf_counter() - t0:.2f}s")

    def run(self, query: str) -> PipelineResult:
        """
        Основной метод пайплайна.

        Args:
            query: Строка запроса пользователя

        Returns:
            PipelineResult с кандидатами (без rerank)
        """
        print(f"\n[PIPELINE] Start query: {query!r}")
        t_pipeline = time.perf_counter()

        # 1. Enrichment
        t0 = time.perf_counter()
        enriched_query = self._enricher.enrich(query)
        print(f"[STEP 1] Enrichment done in {time.perf_counter() - t0:.2f}s")

        # 2. Hybrid search
        t0 = time.perf_counter()
        hybrid_result = self._hybrid.search(enriched_query)
        candidates: List[ScoredChunk] = hybrid_result.candidates
        print(f"[STEP 2] Hybrid search done in {time.perf_counter() - t0:.2f}s "
              f"(candidates={len(candidates)})")

        print(f"[PIPELINE] Finished in {time.perf_counter() - t_pipeline:.2f}s")

        return PipelineResult(
            query=query,
            enriched_query=enriched_query,
            candidates=candidates,
            top_chunks=candidates,  # теперь без rerank
        )
