from __future__ import annotations

"""
Hybrid Search (PIPELINE 3.1–3.6).

Комбинирует:
- Semantic Search (FAISS)
- Lexical Search (BM25‑подобный)
- Metadata Scoring
и возвращает Top‑K кандидатов по hybrid_score.
"""

from dataclasses import dataclass
from typing import List, Dict

import numpy as np

from src.main.models import (
    EnrichedQuery,
    ScoredChunk,
    RetrievalConfig,
)
from src.main.semantic_search import FaissSemanticSearcher
from src.main.lexical_search import InMemoryBM25
from src.main.metadata_scorer import MetadataScorer


@dataclass
class HybridSearchResult:
    """
    Небольшой контейнер для удобной отладки.
    """

    candidates: List[ScoredChunk]
    debug_info: Dict[str, object]


class HybridSearcher:
    """
    Высокоуровневый фасад гибридного поиска.

    ВАЖНО:
    - не знает про LLM / reranking
    - занимается только retrieval (Top‑K по hybrid_score)
    """

    def __init__(
        self,
        semantic_searcher: FaissSemanticSearcher,
        retrieval_config: RetrievalConfig,
    ):
        self._semantic = semantic_searcher
        self._config = retrieval_config
        self._metadata_scorer = MetadataScorer()

        # BM25 строим по тем же чанкам, что и FAISS
        self._lexical = InMemoryBM25(self._semantic._chunks)  # type: ignore[attr-defined]

    # -------------------- Публичный интерфейс -------------------- #

    def search(self, enriched_query: EnrichedQuery) -> HybridSearchResult:
        """
        Выполнить полный гибридный поиск и вернуть кандидатов Top‑K (по hybrid_score).
        """
        # 3.1 Semantic Search
        sem_results = self._semantic.search(
            enriched_query.embedded_query,
            top_k=self._config.semantic_top_k,
        )

        # 3.2 Lexical Search
        lex_results = self._lexical.search(
            enriched_query,
            top_k=self._config.lexical_top_k,
        )

        # Объединяем по id чанка
        merged: Dict[str, ScoredChunk] = {}

        def _key(sc: ScoredChunk) -> str:
            return sc.chunk.id

        for sc in sem_results:
            merged[_key(sc)] = sc
        for sc in lex_results:
            key = _key(sc)
            if key in merged:
                merged[key].lexical_score = sc.lexical_score
            else:
                merged[key] = sc

        candidates = list(merged.values())

        # 3.3 Metadata Scoring
        for sc in candidates:
            sc.metadata_score = self._metadata_scorer.score(sc.chunk, enriched_query)

        # 3.4 Normalization
        self._normalize_scores(candidates)

        # 3.5 Weighted Hybrid Score
        for sc in candidates:
            sc.hybrid_score = (
                self._config.w_semantic * sc.semantic_score
                + self._config.w_lexical * sc.lexical_score
                + self._config.w_metadata * sc.metadata_score
            )

        # 3.6 Top‑K Candidates
        candidates.sort(key=lambda x: x.hybrid_score, reverse=True)
        final = candidates[: self._config.final_top_k]

        debug = {
            "semantic_count": len(sem_results),
            "lexical_count": len(lex_results),
            "merged_count": len(candidates),
        }
        return HybridSearchResult(candidates=final, debug_info=debug)

    # -------------------- Вспомогательная нормализация -------------------- #

    @staticmethod
    def _normalize_scores(candidates: List[ScoredChunk]) -> None:
        """
        - semantic_score → min‑max
        - lexical_score → log + min‑max
        - metadata_score уже [0,1]
        """
        if not candidates:
            return

        sem = np.array([c.semantic_score for c in candidates], dtype=float)
        lex = np.array([c.lexical_score for c in candidates], dtype=float)

        # semantic: min‑max
        if sem.max() > sem.min():
            sem_n = (sem - sem.min()) / (sem.max() - sem.min() + 1e-9)
        else:
            sem_n = sem

        # lexical: log(1+x) + min‑max
        lex_log = np.log1p(np.maximum(lex, 0.0))
        if lex_log.max() > lex_log.min():
            lex_n = (lex_log - lex_log.min()) / (lex_log.max() - lex_log.min() + 1e-9)
        else:
            lex_n = lex_log

        for i, c in enumerate(candidates):
            c.semantic_score = float(sem_n[i])
            c.lexical_score = float(lex_n[i])
            # metadata_score оставляем как есть (уже [0,1])

