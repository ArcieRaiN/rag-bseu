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
    ScoredChunk)
from src.main.config import RetrievalConfig
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

        # 3.2b Metadata recall (важно для табличных чанков):
        # если нужный чанк не попал ни в FAISS, ни в BM25, мы всё равно можем
        # вытащить его по совпадению metrics/years/geo.
        if enriched_query.metrics:
            meta_seed = self._metadata_recall(enriched_query, top_k=max(50, self._config.final_top_k * 10))
            for sc in meta_seed:
                key = _key(sc)
                if key not in merged:
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
            "metadata_seed_count": len(meta_seed) if enriched_query.metrics else 0,
        }
        return HybridSearchResult(candidates=final, debug_info=debug)

    def _metadata_recall(self, enriched_query: EnrichedQuery, top_k: int) -> List[ScoredChunk]:
        """
        Возвращает кандидатов только по metadata_score (без FAISS/BM25).
        Это повышает recall для кейсов, где:
        - текст табличный/шумный
        - нужные термины есть в metrics/years, но плохо присутствуют в raw text/context
        """
        chunks = getattr(self._semantic, "_chunks", None)
        if not chunks:
            return []

        scored: List[ScoredChunk] = []
        for ch in chunks:
            ms = self._metadata_scorer.score(ch, enriched_query)
            if ms <= 0.0:
                continue
            scored.append(ScoredChunk(chunk=ch, metadata_score=float(ms)))

        scored.sort(key=lambda sc: sc.metadata_score, reverse=True)
        return scored[:top_k]

    # -------------------- Вспомогательная нормализация -------------------- #

    @staticmethod
    def _normalize_scores(candidates: List[ScoredChunk]) -> None:
        """
        Нормализация скоринговых компонент.

        - semantic_score → min‑max (только если разброс достаточно большой)
        - lexical_score → log + min‑max (также с порогом по разбросу)
        - metadata_score уже [0,1]
        """
        if not candidates:
            return

        sem = np.array([c.semantic_score for c in candidates], dtype=float)
        lex = np.array([c.lexical_score for c in candidates], dtype=float)

        # semantic: min‑max только при достаточном разбросе.
        # Это снижает эффект "рандомного" ранжирования, когда все скоры почти одинаковые.
        sem_range = sem.max() - sem.min()
        if sem_range > 1e-3:
            sem_n = (sem - sem.min()) / (sem_range + 1e-9)
        else:
            # Если разброс слишком мал, считаем все semantic_score равными среднему.
            sem_n = np.full_like(sem, fill_value=float(sem.mean() if sem.size else 0.0))

        # lexical: log(1+x) + min‑max при достаточном разбросе.
        lex_log = np.log1p(np.maximum(lex, 0.0))
        lex_range = lex_log.max() - lex_log.min()
        if lex_range > 1e-3:
            lex_n = (lex_log - lex_log.min()) / (lex_range + 1e-9)
        else:
            lex_n = lex_log

        for i, c in enumerate(candidates):
            c.semantic_score = float(sem_n[i])
            c.lexical_score = float(lex_n[i])
            # metadata_score оставляем как есть (уже [0,1])

