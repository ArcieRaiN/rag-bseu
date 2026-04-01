from __future__ import annotations

"""
PIPELINE: user query processing (RAG + hybrid retrieval).

Steps:
1. Query enrichment (embedding + regex extraction, no LLM)
2. Hybrid search (BM25 + semantic FAISS + metadata)
"""

from pathlib import Path
import time
from typing import List

from src.core.context_enrichment import QueryContextEnricher
from src.retrieval.hybrid_search import HybridSearcher
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.semantic_search import FaissSemanticSearcher
from src.vectorstore.vectorizer import SentenceVectorizer
from src.core.models import PipelineResult, ScoredChunk
from src.core.config import RetrievalConfig


class QueryPipeline:
    def __init__(self, base_dir: Path, *,
                 retrieval_config: RetrievalConfig | None = None):
        t0 = time.perf_counter()
        print("[INIT] QueryPipeline: initializing...")

        self._base_dir = Path(base_dir)
        self._vectorizer = SentenceVectorizer()

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
        )

        self._reranker = None
        if self._retrieval_config.use_reranker:
            self._reranker = CrossEncoderReranker(
                model_name=self._retrieval_config.reranker_model
            )

        print(f"[INIT] QueryPipeline ready in {time.perf_counter() - t0:.2f}s")

    def run(self, query: str) -> PipelineResult:
        t_pipeline = time.perf_counter()

        # 1. Enrichment (embedding + regex, no LLM)
        t0 = time.perf_counter()
        enriched_query = self._enricher.enrich(query)
        t_enrich = time.perf_counter() - t0

        # 2. Hybrid search
        t0 = time.perf_counter()
        hybrid_result = self._hybrid.search(enriched_query)
        candidates: List[ScoredChunk] = hybrid_result.candidates
        t_search = time.perf_counter() - t0

        # 3. Optional reranking
        t_rerank = 0.0
        if self._reranker is not None:
            t0 = time.perf_counter()
            candidates = self._reranker.rerank(
                query, candidates,
                top_k=self._retrieval_config.reranker_top_k
            )
            t_rerank = time.perf_counter() - t0

        total = time.perf_counter() - t_pipeline
        print(f"[PIPELINE] query={query!r} enrich={t_enrich:.2f}s search={t_search:.2f}s "
              f"rerank={t_rerank:.2f}s total={total:.2f}s candidates={len(candidates)}")

        return PipelineResult(
            query=query,
            enriched_query=enriched_query,
            candidates=candidates,
            top_chunks=candidates,
            timings={"enrich": t_enrich, "search": t_search, "rerank": t_rerank, "total": total},
        )
