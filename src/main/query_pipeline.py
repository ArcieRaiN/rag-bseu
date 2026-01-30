from __future__ import annotations

"""
Высокоуровневый пайплайн обработки пользовательского запроса (PIPELINE 2–4).
"""

from pathlib import Path
from typing import List
import time

from src.main.context_enrichment import QueryContextEnricher
from src.main.hybrid_search import HybridSearcher
from src.main.models import (
    PipelineResult,
    RetrievalConfig,
    RerankConfig,
)
from src.main.ollama_client import OllamaClient
from src.main.vectorizer import SentenceVectorizer
from src.main.semantic_search import FaissSemanticSearcher
from src.main.reranker import CrossEncoderReranker


class QueryPipelineV2:
    def __init__(
        self,
        base_dir: Path,
        *,
        ollama_client: OllamaClient | None = None,
        vector_dim: int = 256,
        retrieval_config: RetrievalConfig | None = None,
        rerank_config: RerankConfig | None = None,
    ):
        print("[INIT] QueryPipelineV2: init started")
        t0 = time.perf_counter()

        self._base_dir = Path(base_dir)
        self._ollama = ollama_client or OllamaClient()
        self._vectorizer = SentenceVectorizer(dimension=vector_dim)

        vector_store_dir = self._base_dir / "prepare_db" / "vector_store"
        index_path = vector_store_dir / "index.faiss"
        data_path = vector_store_dir / "data.json"

        print("[INIT] Loading FAISS index...")
        self._semantic = FaissSemanticSearcher(
            index_path=index_path,
            data_path=data_path,
        )

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
        self._reranker = CrossEncoderReranker(config=self._rerank_config)

        print(f"[INIT] QueryPipelineV2 ready in {time.perf_counter() - t0:.2f}s")

    # -------------------- #

    def run(self, query: str) -> PipelineResult:
        print()
        print(f"[PIPELINE] Start query: {query!r}")
        t_pipeline = time.perf_counter()

        # 1. Enrichment
        print("[STEP 1] Enrichment started")
        t0 = time.perf_counter()
        enriched = self._enricher.enrich(query)
        print(
            f"[STEP 1] Enrichment done in {time.perf_counter() - t0:.2f}s"
        )

        # 2. Hybrid search
        print("[STEP 2] Hybrid search started")
        t0 = time.perf_counter()
        hybrid_result = self._hybrid.search(enriched)
        candidates = hybrid_result.candidates
        print(
            f"[STEP 2] Hybrid search done in {time.perf_counter() - t0:.2f}s "
            f"(candidates={len(candidates)})"
        )

        # 3. Rerank
        print("[STEP 3] Rerank started")
        t0 = time.perf_counter()
        top_chunks = self._reranker.rerank(enriched, candidates.copy())
        print(
            f"[STEP 3] Rerank done in {time.perf_counter() - t0:.2f}s "
            f"(top_chunks={len(top_chunks)})"
        )

        print(
            f"[PIPELINE] Finished in {time.perf_counter() - t_pipeline:.2f}s"
        )

        return PipelineResult(
            query=query,
            enriched_query=enriched,
            candidates=candidates,
            top_chunks=top_chunks,
        )
